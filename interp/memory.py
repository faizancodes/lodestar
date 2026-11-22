"""How much memory the Mac has, what's using it, and whether a model will fit next to it.

A loaded model's weights live in GPU memory, which a process's resident size doesn't count: a Python process holding
an 8B model shows ~50 MB resident but ~18 GB of footprint. So everything here reads the footprint (what Activity
Monitor's Memory column shows), through libproc.
"""
import ctypes
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

GB = 1024 ** 3
BIG = 6 * GB  # a process this size is holding a model, or something as heavy
SWAP_OK = 2 * GB  # how far past free memory a load may go, swapping a little
SWAP_DISK = 10 * GB  # free disk below which swap can't grow
try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")  # in the dyld shared cache, so it loads though no file is there
except OSError:
    _libproc = None


class _Rusage(ctypes.Structure):  # rusage_info_v0, enough to reach ri_phys_footprint
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


def footprint(pid):
    """A process's memory footprint in bytes, or None when it can't be read (another user's process, or gone)."""
    if _libproc is None:
        return None

    ru = _Rusage()
    return ru.phys_footprint if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(ru)) == 0 else None


def _label(p):
    """A name a person would recognise: Python processes by their script, the rest by their app."""
    name = p.info["name"] or "?"

    if name.lower().startswith("python"):
        script = next((a for a in (p.info["cmdline"] or [])[1:] if not a.startswith("-")), None)
        if script:
            return f"Python ({Path(script).name})"
        return "Python"

    return name.split(" (")[0].split(":")[0].strip()


def _others(min_bytes=256 * 1024 ** 2):
    """Every other readable process using at least min_bytes, biggest first."""
    me, out = os.getpid(), []

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == me:
            continue

        try:
            f = footprint(p.info["pid"])
            if f and f >= min_bytes:
                out.append({"pid": p.info["pid"], "name": _label(p), "bytes": f})
        except (psutil.Error, OSError):
            continue

    return sorted(out, key=lambda x: -x["bytes"])


def pressure():
    """macOS's own verdict: 1 normal, 2 warning, 4 critical."""
    try:
        r = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True,
                           timeout=2)
        return {1: "normal", 2: "warn", 4: "critical"}.get(int(r.stdout.strip()), "normal")
    except (OSError, ValueError, subprocess.SubprocessError):
        return "normal"


_last_swap = {"out": None, "at": 0.0, "swapping_until": 0.0}


def _swapping():
    """Whether pages went out to swap since the last look. Stays true for 20 s after, so a burst doesn't flicker."""
    now, out = time.time(), psutil.swap_memory().sout
    prev = _last_swap["out"]
    _last_swap.update(out=out, at=now)

    if prev is not None and out - prev > 64 * 1024 ** 2:
        _last_swap["swapping_until"] = now + 20

    return now < _last_swap["swapping_until"]


def snapshot():
    """For the meter beside the model picker."""
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
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


def _files(model, hf_cache):
    p = Path(model)
    if p.is_dir():
        return p, list(p.glob("*.safetensors")) or list(p.glob("*.bin"))
    snaps = sorted((hf_cache / f"models--{model.replace('/', '--')}" / "snapshots").glob("*"))
    return (snaps[-1], list(snaps[-1].glob("*.safetensors"))) if snaps else (None, [])


def weights_bytes(model, hf_cache):
    """The size of a model's weights on disk, or None when it isn't here yet (a download)."""
    return sum(f.stat().st_size for f in _files(model, hf_cache)[1]) or None


def streamed_bytes(model, hf_cache):
    """For Inkling 2 Mini, which can stream its experts from disk (interp/inkling.py): what the rest of its text model
    holds once loaded. None for every other model."""
    root, files = _files(model, hf_cache)
    try:
        if json.loads((root / "config.json").read_text()).get("model_type") != "inkling2_loop":
            return None
    except (OSError, TypeError, ValueError):
        return None

    total = 0
    for f in files:
        if f.name == "mtp.safetensors":
            continue
        with open(f, "rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items()
                     if k.startswith("model.llm.") and ".mlp.experts." not in k)
    return total


def fit(model, hf_cache, loaded_bytes=0):
    """Whether a model fits beside everything else that's running. loaded_bytes: what the model it replaces holds,
    which comes free when the new one loads."""
    disk = weights_bytes(model, hf_cache)
    if disk is None:
        return {"known": False, "fits": True}

    # the engine keeps small models (under ~2.5B parameters, ~5 GB in bf16) in float32, doubling them; then the
    # activations, caches and the Python runtime come on top
    held = lambda b: b * 1.08 + 0.8 * GB
    need = held(disk * (2 if disk < 5e9 else 1))

    # Footprints count memory that's been compressed or swapped out, so summing every process overshoots the Mac's
    # RAM. What decides whether a model fits is the other heavy processes (another model, mostly): count those, and
    # set aside a quarter of the RAM for macOS and the everyday apps.
    total = psutil.virtual_memory().total
    heavy = _others(BIG)
    used = sum(o["bytes"] for o in heavy)
    me = max((footprint(os.getpid()) or 0) - loaded_bytes, 0)
    free = total * 0.75 - used - me
    # ...but many small apps, compressed memory and the kernel can leave far less than that: on a 32 GB Mac with
    # nothing heavy open, macOS had 14 GB available against 24 GB by the count above. Take what macOS says is
    # available (plus what the model being replaced holds) when that's lower.
    free = min(free, psutil.virtual_memory().available + loaded_bytes)

    # Past a little swap, macOS can stall outright rather than slow down: a load that big, under memory pressure
    # already, or with too little disk for swap to grow is refused, with no way to load it anyway.
    level, disk_free = pressure(), shutil.disk_usage(Path.home()).free

    def verdict(need):
        if level == "critical":
            return "macOS is already short of memory."
        if need > free + SWAP_OK:
            return "Loading it would push macOS so far into swap that the Mac could freeze."
        if need > free and disk_free < SWAP_DISK:
            return "The disk is too full for macOS to swap."
        return None

    blocked, stream = verdict(need), None
    rest = streamed_bytes(model, hf_cache) if need > free or blocked else None
    if rest is not None:
        # Inkling's experts can stay on disk instead, with a GPU cache of the ones used lately: a third of what's free,
        # leaving the rest to macOS's file cache, which holds the experts read from disk
        stream = int(min(max(free * 0.3, 2 * GB), 6 * GB))  # at least a layer's experts
        need = held(rest + stream) + 2 * GB  # plus read buffers and the GPU allocator's working blocks (measured)
        blocked = verdict(need)

    return {"known": True, "fits": need <= free and blocked is None, "blocked": blocked, "need": need,
            "free": max(free, 0), "total": total, "used": used, "others": heavy[:3], "stream": stream}


def headroom(min_bytes=3 * GB):
    """Whether there's room for a heavy job (a backward pass) beside what's loaded: no pressure and some memory
    still available."""
    return pressure() == "normal" and psutil.virtual_memory().available >= min_bytes


def cap_gpu():
    """Cap this process's GPU memory at what it holds now plus what macOS has available (and the little swap a load
    may use). Past the cap, PyTorch raises an out-of-memory error, which the app reports; without it, a model or a
    backward pass that outgrows memory swaps until the whole Mac stalls."""
    import torch

    torch.mps.empty_cache()
    cap = torch.mps.driver_allocated_memory() + psutil.virtual_memory().available + SWAP_OK
    torch.mps.set_per_process_memory_fraction(min(max(cap / torch.mps.recommended_max_memory(), 0.1), 2.0))
    return cap
