"""Memory footprint inspection and model-fit estimation for macOS unified memory.

A loaded model's weights live in GPU memory, which a process's resident set size (RSS) does not count: a Python
process holding an 8B model shows ~50 MB resident but ~18 GB of physical footprint. Everything here reads the physical
footprint (matching Activity Monitor's Memory column) through macOS's ``libproc`` API.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil


GB = 1024**3
BIG = 6 * GB  # a process this size is holding a model, or something as heavy
SWAP_OK = 2 * GB  # how far past free memory a load may go, swapping a little
SWAP_DISK = 10 * GB  # free disk below which swap can't grow
_SWAP_BURST_BYTES = 64 * 1024**2
_SWAP_HOLD_SECONDS = 20.0
_RUSAGE_INFO_V0 = 0
_PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}


class _Rusage(ctypes.Structure):
    """Subset of macOS ``rusage_info_v0`` up to ``ri_phys_footprint``."""

    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
            )
        ),
    ]


def _load_libproc() -> ctypes.CDLL | None:
    try:
        # Lives in the macOS dyld shared cache, so CDLL loads it even when no file exists on disk.
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


_libproc = _load_libproc()
_swap_lock = threading.Lock()
_last_swap_out: int | None = None
_swapping_until: float = 0.0


def footprint(pid: int) -> int | None:
    """Return a process's physical memory footprint in bytes, or ``None`` if unreadable."""
    if _libproc is None:
        return None

    rusage = _Rusage()
    if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V0, ctypes.byref(rusage)) == 0:
        return int(rusage.phys_footprint)
    return None


def _label(proc: psutil.Process) -> str:
    """Return a human-readable process label (script name for Python processes, app name otherwise)."""
    name = proc.info["name"] or "?"
    if name.lower().startswith("python"):
        cmdline = proc.info["cmdline"] or []
        script = next((arg for arg in cmdline[1:] if not arg.startswith("-")), None)
        if script:
            return f"Python ({Path(script).name})"
        return "Python"

    return name.split(" (")[0].split(":")[0].strip()


def _others(min_bytes: int = 256 * 1024**2) -> list[dict[str, Any]]:
    """Return every other readable process using at least ``min_bytes``, largest first."""
    my_pid = os.getpid()
    out: list[dict[str, Any]] = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        pid = proc.info["pid"]
        if pid == my_pid:
            continue

        try:
            mem = footprint(pid)
            if mem and mem >= min_bytes:
                out.append({"pid": pid, "name": _label(proc), "bytes": mem})
        except (psutil.Error, OSError):
            continue

    return sorted(out, key=lambda item: -item["bytes"])


def pressure() -> str:
    """Read macOS's VM memory pressure verdict: ``"normal"``, ``"warn"``, or ``"critical"``."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return _PRESSURE_LEVELS.get(int(result.stdout.strip()), "normal")
    except (OSError, ValueError, subprocess.SubprocessError):
        return "normal"


def _swapping() -> bool:
    """Return whether pages were swapped out recently, latching true for 20 s to prevent UI flicker."""
    global _last_swap_out, _swapping_until
    now = time.time()
    swap_out = psutil.swap_memory().sout

    with _swap_lock:
        prev = _last_swap_out
        _last_swap_out = swap_out
        if prev is not None and swap_out - prev > _SWAP_BURST_BYTES:
            _swapping_until = now + _SWAP_HOLD_SECONDS
        return now < _swapping_until


def snapshot() -> dict[str, Any]:
    """Return a system and process memory summary for the sidebar memory meter."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    others = _others()
    me = footprint(os.getpid()) or 0

    return {
        "total": vm.total,
        "used": vm.total - vm.available,
        "app": me,
        "others": others[:4],
        "swap": sw.used,
        "pressure": pressure(),
        "swapping": _swapping(),
    }


def _files(model: str | Path, hf_cache: Path) -> tuple[Path | None, list[Path]]:
    path = Path(model)
    if path.is_dir():
        return path, list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    cache_dir = hf_cache / f"models--{str(model).replace('/', '--')}" / "snapshots"
    snaps = sorted(cache_dir.glob("*")) if cache_dir.exists() else []
    return (snaps[-1], list(snaps[-1].glob("*.safetensors"))) if snaps else (None, [])


def weights_bytes(model: str | Path, hf_cache: Path) -> int | None:
    """Return the total weight size on disk in bytes, or ``None`` if the model is not cached locally."""
    return sum(file.stat().st_size for file in _files(model, hf_cache)[1]) or None


def streamed_bytes(model: str | Path, hf_cache: Path) -> int | None:
    """Return non-expert text-model bytes for Inkling 2 Mini (which streams experts from disk), else ``None``."""
    root, files = _files(model, hf_cache)
    if root is None:
        return None
    try:
        if json.loads((root / "config.json").read_text()).get("model_type") != "inkling2_loop":
            return None
    except (OSError, TypeError, ValueError):
        return None

    total = 0
    for file in files:
        if file.name == "mtp.safetensors":
            continue
        with open(file, "rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        total += sum(
            v["data_offsets"][1] - v["data_offsets"][0]
            for k, v in header.items()
            if k.startswith("model.llm.") and ".mlp.experts." not in k
        )
    return total


def _held_bytes(weight_bytes: float) -> float:
    return weight_bytes * 1.08 + 0.8 * GB


def fit(model: str | Path, hf_cache: Path, loaded_bytes: int = 0) -> dict[str, Any]:
    """Estimate whether ``model`` fits in RAM alongside other running processes.

    Args:
        model: Hugging Face model ID or local model directory path.
        hf_cache: Path to the local Hugging Face hub cache.
        loaded_bytes: Bytes currently held by a loaded model that will be unloaded before loading ``model``.

    Returns:
        Dictionary describing fit status and estimated memory requirements.
    """
    disk = weights_bytes(model, hf_cache)
    if disk is None:
        return {"known": False, "fits": True}

    # The engine keeps small models (under ~2.5B parameters, ~5 GB in bf16) in float32, doubling weight size;
    # activations, KV caches, and the Python runtime sit on top.
    need = _held_bytes(disk * (2 if disk < 5e9 else 1))

    # Footprints include compressed/swapped memory, so summing every process overshoots physical RAM.
    # Count other heavy processes (typically other models) and reserve a quarter of RAM for macOS and everyday apps,
    # capped by macOS's own available memory reading when that is lower.
    vm = psutil.virtual_memory()
    total = vm.total
    heavy = _others(BIG)
    used = sum(proc["bytes"] for proc in heavy)
    me = max((footprint(os.getpid()) or 0) - loaded_bytes, 0)
    free = min(total * 0.75 - used - me, vm.available + loaded_bytes)

    level = pressure()
    disk_free = shutil.disk_usage(Path.home()).free

    def _verdict(required: float) -> str | None:
        if level == "critical":
            return "macOS is already short of memory."
        if required > free + SWAP_OK:
            return "Loading it would push macOS so far into swap that the Mac could freeze."
        if required > free and disk_free < SWAP_DISK:
            return "The disk is too full for macOS to swap."
        return None

    blocked = _verdict(need)
    stream: int | None = None
    rest = streamed_bytes(model, hf_cache) if need > free or blocked else None
    if rest is not None:
        stream = int(min(max(free * 0.3, 2 * GB), 6 * GB))  # at least a layer's experts
        need = _held_bytes(rest + stream) + 2 * GB  # plus read buffers and GPU allocator working blocks
        blocked = _verdict(need)

    return {
        "known": True,
        "fits": need <= free and blocked is None,
        "blocked": blocked,
        "need": need,
        "free": max(free, 0),
        "total": total,
        "used": used,
        "others": heavy[:3],
        "stream": stream,
    }


def headroom(min_bytes: int = 3 * GB) -> bool:
    """Return whether there is room for a heavy job (e.g. a backward pass) beside what is loaded."""
    return pressure() == "normal" and psutil.virtual_memory().available >= min_bytes


def cap_gpu() -> float:
    """Cap MPS memory to current allocation plus available RAM and swap margin so OOM raises instead of stalling."""
    import torch

    torch.mps.empty_cache()
    cap = torch.mps.driver_allocated_memory() + psutil.virtual_memory().available + SWAP_OK
    torch.mps.set_per_process_memory_fraction(min(max(cap / torch.mps.recommended_max_memory(), 0.1), 2.0))
    return float(cap)
