"""Local HTTP API for Lodestar: JSON lookups, SSE streams, and an OpenAI-compatible ``/v1`` endpoint.

Serves the built React frontend from ``../web/dist`` alongside all interpretability endpoints.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import abl as abl_mod
from . import archview, discover, memory
from . import directions as dir_mod
from . import experiment as exp_mod
from . import jlens as jlens_mod
from . import probe as probe_mod
from .engine import Cancelled, Engine, Push
from .utils import atomic_write_json, atomic_write_text, get_logger, model_short_name


logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "web" / "dist"  # React build from `pnpm build` in web/
# Keep the app's original data directory name so previously saved models, recipes, and directions remain accessible.
DATA = Path(os.environ.get("AUTO_INTERP_DATA", Path.home() / "Library" / "Application Support" / "Auto Interp"))
DATA.mkdir(parents=True, exist_ok=True)
RECIPES = DATA / "recipes.json"
ABL_FILE = DATA / "abl.json"  # saved refusal directions + state, keyed by model id
LAST_MODEL = DATA / "last_model.txt"  # model to reopen at next launch
PROBES_FILE = DATA / "abl-probes.json"  # customized probe lists
CHATS = DATA / "chats"  # one file per saved conversation
JLENS = DATA / "jlens"  # each model's J-lens, computed once (interp/jlens.py)
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

PRESETS = [
    {"id": "Qwen/Qwen3-0.6B", "size": "1.2 GB", "note": "Fast, good for exploring"},
    {"id": "Qwen/Qwen3-1.7B", "size": "3.4 GB", "note": "Smarter, still quick"},
    {"id": "Qwen/Qwen3-4B", "size": "8 GB", "note": "Good balance of smarts and speed"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "size": "3.4 GB", "note": "A different model family"},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "size": "2.5 GB", "note": "Needs a Hugging Face login"},
    {"id": "google/gemma-3-1b-it", "size": "2 GB", "note": "Needs a Hugging Face login"},
]

EXAMPLES = {
    "Qwen3-8B": [
        {
            "name": "Gnome",
            "note": "One neuron that makes the model describe itself as a small creature (@Sauers_)",
            "prompt": "Describe the way you look.",
            "items": [{"layer": 8, "neuron": 3602, "dose": 1.0, "label": "gnome"}],
        },
        {
            "name": "Forest gnome",
            "note": "The three-neuron version from the same thread: greenish skin, pointed ears",
            "prompt": "Describe the way you look.",
            "items": [
                {"layer": 3, "neuron": 6056, "dose": -0.5, "label": "gnome a"},
                {"layer": 4, "neuron": 1658, "dose": 0.5, "label": "gnome b"},
                {"layer": 8, "neuron": 3602, "dose": 0.5, "label": "gnome"},
            ],
        },
    ],
    "Qwen3-1.7B": [
        {
            "name": "Cookies",
            "note": "Found by Describe: brings up cookies; loops above strength 0.35",
            "prompt": "What makes you happy?",
            "items": [{"layer": 26, "neuron": 2847, "dose": -0.3, "label": "cookie"}],
        },
    ],
    "Qwen3-0.6B": [
        {
            "name": "Hungry",
            "note": "Found by Describe: turns answers toward food and eating",
            "prompt": "What should I eat for dinner tonight?",
            "items": [{"layer": 9, "neuron": 2578, "dose": 1.2, "label": "munch"}],
        },
    ],
}

eng = Engine()
dirs = dir_mod.Library(DATA / "directions", eng)
probes = probe_mod.Library(DATA / "probes", eng)
cancel = threading.Event()
_state_lock = threading.Lock()
_store_lock = threading.Lock()
app = FastAPI()


def sse(gen: Iterator[dict[str, Any]]) -> StreamingResponse:
    """Wrap a synchronous event generator as a ``text/event-stream`` response.

    The generator runs end-to-end on a single dedicated thread because engine generators hold ``eng.lock`` across
    yields; stepping a plain sync generator across Starlette's threadpool could release the RLock on a different
    thread than the one that acquired it.
    """
    loop = asyncio.get_running_loop()
    out: asyncio.Queue[str | None] = asyncio.Queue()
    gone = threading.Event()

    def _put(item: str | None) -> None:
        loop.call_soon_threadsafe(out.put_nowait, item)

    def _send(event: dict[str, Any]) -> None:
        _put(f"data: {json.dumps(_finite(event), default=_np)}\n\n")

    def _work() -> None:
        cancel.clear()
        try:
            for event in gen:
                if gone.is_set():
                    break
                _send(event)
        except Cancelled:
            _send({"type": "cancelled"})
        except Exception as exc:
            logger.exception("SSE stream failed")
            _send({"type": "error", "message": str(exc)})
        finally:
            close_fn = getattr(gen, "close", None)
            if callable(close_fn):
                close_fn()
            _send({"type": "end"})
            _put(None)

    async def _run():
        threading.Thread(target=_work, daemon=True).start()
        try:
            while (chunk := await out.get()) is not None:
                yield chunk
        finally:
            gone.set()

    return StreamingResponse(_run(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _finite(obj: Any) -> Any:
    """Replace non-finite floats (NaN, Infinity) with ``None`` so JSON stays valid in browsers."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _np(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


@app.middleware("http")
async def no_store(req: Request, call_next: Any) -> Any:
    """Prevent browsers/WebKit from caching local static UI assets."""
    resp = await call_next(req)
    if not req.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def need_model() -> None:
    """Raise HTTP 409 if no model is currently loaded and ready."""
    if eng.model is None or eng.status != "ready":
        raise HTTPException(409, "Load a model first (bottom left).")


def pushes_of(body: dict[str, Any]) -> list[Push]:
    """Parse and validate enabled :class:`Push` entries from a request payload."""
    out = [Push.of(p) for p in body.get("pushes") or [] if p.get("enabled", True)]
    num_layers = eng.num_layers if eng.model is not None else 0

    for push in out:
        if push.mode not in ("dir", "erase"):
            continue
        if push.mode == "erase":
            push.dose = min(max(push.dose, 0.0), 1.0)
        if push.dir_id not in eng.dirs:
            raise HTTPException(409, f"This model has no direction called “{push.dir_id}”.")
        if not all(0 <= layer_idx < num_layers for layer_idx in push.layers) or (push.mode == "dir" and not push.layers):
            raise HTTPException(400, f"Pick layers between 0 and {num_layers - 1} for “{push.dir_id}”.")

    return out


def _reverse(push: Push) -> Push | None:
    """Compute the opposite of ``push`` for side-by-side comparison (``erase`` has no opposite)."""
    if push.mode == "erase":
        return None
    if push.mode == "mul":
        return Push(push.layer, push.neuron, 2 - push.dose, push.mode)
    return Push(push.layer, push.neuron, -push.dose, push.mode, push.dir_id, push.layers)


# ------------------------------------------------------------------ models
LOADING: dict[str, int | None] = {"bytes": None}


@app.get("/api/status")
def status() -> dict[str, Any]:
    last = LAST_MODEL.read_text().strip() if LAST_MODEL.exists() else None
    out: dict[str, Any] = {**eng.info(), "progress": eng.progress, "last": last}
    if eng.load_state:
        out["load"] = {**eng.load_state, "bytes": LOADING["bytes"]}
    return out


COPIES = DATA / "models"
copy_job: dict[str, Any] = {"src": None, "name": None, "done": 0, "total": 0, "status": "idle", "cancel": False}


def _model_files(src: Path) -> list[Path]:
    """Return regular non-hidden files in a model folder (skipping macOS ``._`` sidecars)."""
    return [f for f in sorted(src.iterdir()) if f.is_file() and not f.name.startswith(".")]


def _folder_size(folder: Path) -> int:
    return sum(f.stat().st_size for f in _model_files(folder))


def _generates(snap: Path) -> bool:
    """Return whether a cached Hugging Face model snapshot is a text-generating architecture."""
    try:
        archs = json.loads((snap / "config.json").read_text()).get("architectures") or []
    except (OSError, ValueError):
        return True
    return not archs or any(arch.endswith(("ForCausalLM", "ForConditionalGeneration")) for arch in archs)


@app.get("/api/models")
def models() -> dict[str, Any]:
    cached: list[str] = []
    if HF_CACHE.exists():
        for model_dir in sorted(HF_CACHE.glob("models--*")):
            snaps = sorted(s for s in model_dir.glob("snapshots/*") if any(s.glob("*.safetensors")))
            if snaps and _generates(snaps[-1]):
                cached.append(model_dir.name[len("models--") :].replace("--", "/"))

    copies = [
        {"path": str(d), "name": d.name, "size": _folder_size(d)}
        for d in sorted(COPIES.glob("*"))
        if d.is_dir() and not d.name.endswith(".partial") and (d / "config.json").exists()
    ]

    copied = {c["name"] for c in copies}
    local = [
        {
            "path": str(d),
            "name": d.name,
            "drive": d.parent.parent.name,
            "size": _folder_size(d),
            "copied": d.name in copied,
        }
        for d in sorted(Path("/Volumes").glob("*/models/*"))
        if (d / "config.json").exists() and any(d.glob("*.safetensors"))
    ]

    return {"presets": PRESETS, "cached": cached, "local": local, "copies": copies}


@app.get("/api/models/copy")
def copy_status() -> dict[str, Any]:
    with _state_lock:
        return {k: v for k, v in copy_job.items() if k != "cancel"}


@app.post("/api/models/copy")
async def copy_model(req: Request) -> dict[str, Any]:
    body = await req.json()
    src = Path(body["path"]).resolve()

    if not (src.is_dir() and str(src).startswith("/Volumes/") and (src / "config.json").exists()):
        raise HTTPException(400, "That isn't a model folder on an external drive.")

    files = _model_files(src)
    total = sum(f.stat().st_size for f in files)

    COPIES.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(COPIES).free
    margin = 10e9  # leave room for macOS and swap
    if free < total + margin:
        raise HTTPException(
            400,
            f"Not enough space: the copy needs {total / 1e9:.1f} GB and this Mac has {free / 1e9:.1f} GB free.",
        )

    with _state_lock:
        if copy_job["status"] == "copying":
            raise HTTPException(409, f"Already copying {copy_job['name']}.")
        copy_job.update(src=str(src), name=src.name, done=0, total=total, status="copying", cancel=False)

    def _work() -> None:
        tmp, dst = COPIES / (src.name + ".partial"), COPIES / src.name
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            for src_file in files:
                with open(src_file, "rb") as reader, open(tmp / src_file.name, "wb") as writer:
                    while chunk := reader.read(64 << 20):
                        with _state_lock:
                            if copy_job["cancel"]:
                                raise Cancelled()
                        writer.write(chunk)
                        with _state_lock:
                            copy_job["done"] += len(chunk)

            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
            with _state_lock:
                copy_job["status"] = "done"
        except Cancelled:
            shutil.rmtree(tmp, ignore_errors=True)
            with _state_lock:
                copy_job["status"] = "idle"
        except Exception as exc:
            logger.exception("Failed copying model %s", src)
            shutil.rmtree(tmp, ignore_errors=True)
            with _state_lock:
                copy_job["status"] = f"error: {exc}"

    threading.Thread(target=_work, daemon=True).start()
    return copy_status()


@app.post("/api/models/copy/cancel")
def cancel_copy() -> dict[str, bool]:
    with _state_lock:
        copy_job["cancel"] = True
    return {"ok": True}


@app.post("/api/models/remove")
async def remove_copy(req: Request) -> dict[str, bool]:
    body = await req.json()
    target = Path(body["path"]).resolve()

    if target.parent != COPIES.resolve() or not (target / "config.json").exists():
        raise HTTPException(400, "Only copies made by Lodestar can be removed here.")
    if eng.model_id == str(target):
        raise HTTPException(409, "That copy is loaded right now. Load another model first.")

    shutil.rmtree(target)
    return {"ok": True}


@app.post("/api/load")
async def load(req: Request) -> dict[str, Any]:
    body = await req.json()
    mid = body["model"].strip()

    fit = _fit(mid)
    if fit.get("blocked"):
        raise HTTPException(
            409,
            f"{Path(mid).name} won't fit in memory. {fit['blocked']} Quit some apps and try again.",
        )

    LOADING["bytes"] = memory.weights_bytes(mid, HF_CACHE)

    def _work() -> None:
        try:
            eng.load(mid, stream=fit.get("stream"))
            restore_for(mid)
            dirs.load()
            with eng.lock:
                eng.jlens = jlens_mod.load(jlens_mod.path_for(JLENS, mid), eng)
            atomic_write_text(LAST_MODEL, mid)
            # Compute automatically only when it won't crowd RAM; Inkling's waits to be asked explicitly.
            if eng.jlens is None and mid not in _jlens_stopped() and not eng.ink and memory.headroom(JLENS_ROOM):
                _jlens_start()
        except Exception as exc:
            logger.exception("Failed loading model %s", mid)
            eng.status = f"error: {exc}"
            eng.progress = None
            eng.load_state = None

    threading.Thread(target=_work, daemon=True).start()
    await asyncio.sleep(0.05)
    return status()


@app.post("/api/unload")
def unload() -> dict[str, Any]:
    """Unload the active model and forget it as the last-opened model."""
    cancel.set()
    with eng.lock:
        eng.unload()
        eng.status = "idle"
        eng.progress = None

    LAST_MODEL.unlink(missing_ok=True)
    return status()


# ------------------------------------------------------------------ memory
@app.get("/api/memory")
def memory_now() -> dict[str, Any]:
    return memory.snapshot()


def _fit(mid: str) -> dict[str, Any]:
    """Estimate fit for ``mid`` accounting for memory freed by unloading any currently loaded model."""
    loaded = 0
    if eng.model is not None:
        loaded = max((memory.footprint(os.getpid()) or 0) - memory.GB, 0)
    return memory.fit(mid, HF_CACHE, loaded)


@app.post("/api/memory/fit")
async def memory_fit(req: Request) -> dict[str, Any]:
    """Estimate whether a requested model fits alongside currently running processes."""
    return _fit((await req.json())["model"].strip())


@app.post("/api/cancel")
def do_cancel() -> dict[str, bool]:
    cancel.set()
    return {"ok": True}


# ------------------------------------------------------------------ trace
@app.post("/api/trace")
async def trace(req: Request) -> StreamingResponse:
    need_model()
    body = await req.json()
    pushes = pushes_of(body)
    pid = eng.chat_ids(body["prompt"])

    def _gen() -> Iterator[dict[str, Any]]:
        tr = eng.run_trace(pid, eng.encode(body["response"]), pushes, abl=bool(body.get("abl")))
        yield {"type": "trace", "tokens": tr["tokens"], "n_prompt": tr["n_prompt"]}

    return sse(_gen())


def _neuron_row(layer_idx: int, neuron_idx: int, _impact: float, act: float, _t: int, lens_k: int = 5) -> dict[str, Any]:
    lens = eng.lens(layer_idx, neuron_idx, lens_k)
    return {"layer": layer_idx, "neuron": neuron_idx, "act": round(act, 3), "lens": lens["up" if act >= 0 else "down"]}


def _decode_lens_rows(ids: np.ndarray, probs: np.ndarray) -> list[list[dict[str, Any]]]:
    return [
        [
            {"tok": eng.tok.decode([int(i)]), "p": round(float(p), 4)}
            for i, p in zip(ids[layer_idx], probs[layer_idx], strict=True)
        ]
        for layer_idx in range(ids.shape[0])
    ]


@app.get("/api/trace/token")
def trace_token(t: int, k: int = 24) -> dict[str, Any]:
    need_model()
    tr = eng.trace

    if tr is None or not 0 <= t < len(tr["ids"]):
        raise HTTPException(404, "no trace")

    top = eng.trace_top_neurons(t=t, k=k)
    jlens_res = eng.trace_jlens(t)
    return {
        "lens": _decode_lens_rows(tr["lens_ids"][:, t, :], tr["lens_p"][:, t, :]),
        "jlens": _decode_lens_rows(*jlens_res) if jlens_res else None,
        "neurons": [_neuron_row(*row) for row in top],
    }


# ------------------------------------------------------------------ J-lens
jlens_job: dict[str, Any] = {"model": None, "status": "idle", "done": 0, "total": 0, "elapsed": 0.0}
jlens_cancel = threading.Event()
jlens_thread: threading.Thread | None = None
JLENS_STOPPED = JLENS / "stopped.json"
JLENS_ROOM = 4 * memory.GB  # free memory its backward passes need beside the model


def _jlens_stopped() -> set[str]:
    try:
        return set(json.loads(JLENS_STOPPED.read_text()))
    except (OSError, ValueError):
        return set()


def _jlens_mark_stopped(mid: str, stopped: bool) -> None:
    with _store_lock:
        ids = _jlens_stopped() - {mid} | ({mid} if stopped else set())
        atomic_write_json(JLENS_STOPPED, sorted(ids), indent=None)


def _jlens_start() -> None:
    """Compute the loaded model's J-lens in a background thread and save it to disk."""
    global jlens_thread
    if jlens_thread is not None and jlens_thread.is_alive():
        with _state_lock:
            if jlens_job["model"] == eng.model_id:
                return
        jlens_cancel.set()
        jlens_thread.join()

    mid = eng.model_id
    jlens_cancel.clear()
    with _state_lock:
        jlens_job.update(model=mid, status="computing", done=0, total=eng.hidden, elapsed=0.0, short=False)

    def _prog(done: int, total: int, elapsed: float) -> None:
        with _state_lock:
            jlens_job.update(done=done, total=total, elapsed=round(elapsed, 1))
            if not memory.headroom(memory.GB):  # stop before backward passes push macOS into swap
                jlens_job["short"] = True
                jlens_cancel.set()

    def _work() -> None:
        try:
            jacobians = jlens_mod.compute(eng, jlens_cancel, _prog)
            jlens_mod.save(jacobians, jlens_mod.path_for(JLENS, mid))
            with eng.lock:
                if eng.model_id == mid:
                    eng.jlens = jacobians.to(eng.dev, eng.model.dtype)
            with _state_lock:
                jlens_job["status"] = "done"
        except Cancelled:
            with _state_lock:
                jlens_job["status"] = (
                    "error: Stopped: the Mac was running short of memory." if jlens_job.get("short") else "idle"
                )
        except Exception as exc:
            logger.exception("J-lens computation failed for %s", mid)
            with _state_lock:
                jlens_job["status"] = f"error: {exc}"
            _jlens_mark_stopped(mid, True)

    jlens_thread = threading.Thread(target=_work, daemon=True)
    jlens_thread.start()


@app.get("/api/jlens")
def jlens_status() -> dict[str, Any]:
    with _state_lock:
        job = dict(jlens_job) if jlens_job["model"] == eng.model_id else {"status": "idle", "done": 0, "total": 0, "elapsed": 0.0}
    return {"ready": eng.jlens is not None, **{k: v for k, v in job.items() if k not in ("model", "short")}}


@app.post("/api/jlens/compute")
def jlens_compute() -> dict[str, Any]:
    need_model()
    if not memory.headroom(JLENS_ROOM):
        raise HTTPException(
            409,
            "Not enough free memory to compute the J-lens beside the model. Quit some apps and try again.",
        )
    _jlens_mark_stopped(eng.model_id, False)
    _jlens_start()
    return jlens_status()


@app.post("/api/jlens/cancel")
def jlens_stop() -> dict[str, bool]:
    with _state_lock:
        computing_model = jlens_job["model"] if jlens_job["status"] == "computing" else None
    if computing_model:
        _jlens_mark_stopped(computing_model, True)
    jlens_cancel.set()
    return {"ok": True}


# ------------------------------------------------------------------ architecture
arch_cache: dict[str, dict[str, Any]] = {}


@app.get("/api/arch")
def arch_detail() -> dict[str, Any]:
    need_model()
    if eng.model_id not in arch_cache:
        arch_cache.clear()
        arch_cache[eng.model_id] = archview.structure(eng)
    return arch_cache[eng.model_id]


@app.get("/api/arch/neurons")
def arch_neurons(layer: int) -> dict[str, Any]:
    need_model()
    if not 0 <= layer < eng.num_layers:
        raise HTTPException(400, f"layer must be 0..{eng.num_layers - 1}")
    return {"layer": layer, "norms": archview.neuron_norms(eng, layer)}


@app.post("/api/arch/run")
async def arch_run(req: Request) -> StreamingResponse:
    """Stream a chat reply with live per-token layer frames, followed by a full traced forward pass."""
    need_model()
    text = (await req.json()).get("text", "").strip()
    if not text:
        raise HTTPException(400, "Type some text to run through the model.")
    pid = eng.chat_ids(text)

    def _gen() -> Iterator[dict[str, Any]]:
        reply: list[int] = []
        with archview.Live(eng) as live:
            for row_idx, payload in eng.generate([(pid, [])], max_new=archview.REPLY_TOKENS, cancel=cancel):
                if row_idx is None:
                    reply = payload[0]
                else:
                    yield {"type": "delta", "text": payload, "frame": live.frame()}
        yield {"type": "run", **archview.run(eng, pid + reply, len(pid))}

    return sse(_gen())


@app.get("/api/arch/acts")
def arch_acts(t: int) -> dict[str, Any]:
    need_model()
    out = archview.acts(eng, t)
    if out is None:
        raise HTTPException(404, "no run")
    return out


@app.get("/api/arch/neuron")
def arch_neuron(layer: int, n: int) -> dict[str, Any]:
    need_model()
    num_layers, num_neurons = eng.num_layers, eng.num_neurons
    if not (0 <= layer < num_layers and 0 <= n < num_neurons):
        raise HTTPException(400, f"layer must be 0..{num_layers - 1}, neuron 0..{num_neurons - 1}")
    with eng.lock:
        norm = round(float(eng.dnorm[layer, n]), 4)
    return {
        "layer": layer,
        "neuron": n,
        "norm": norm,
        "lens": eng.lens(layer, n, 10),
        "acts": archview.neuron_trace(eng, layer, n),
    }


# ------------------------------------------------------------------ neurons
@app.get("/api/neuron")
def neuron(layer: int, n: int) -> dict[str, Any]:
    need_model()
    num_layers, num_neurons = eng.num_layers, eng.num_neurons
    if not (0 <= layer < num_layers and 0 <= n < num_neurons):
        raise HTTPException(400, f"layer must be 0..{num_layers - 1}, neuron 0..{num_neurons - 1}")

    out: dict[str, Any] = {"layer": layer, "neuron": n, "lens": eng.lens(layer, n, 16)}
    tr = eng.trace
    if tr is not None:
        acts = tr["acts"][layer, :, n].astype(np.float32)
        out["trace"] = {"tokens": tr["tokens"], "n_prompt": tr["n_prompt"], "acts": np.round(acts, 3).tolist()}
    return out


@app.get("/api/examples")
def examples() -> list[dict[str, Any]]:
    need_model()
    name = model_short_name(eng.model_id)
    shipped = [
        {"name": e["name"], "note": e["about"], "prompt": (e["try"] or [""])[0], "items": [], "file": e["file"]}
        for e in exp_mod.examples("steer")
        if exp_mod.model_name(e["model"]) == exp_mod.model_name(name)
    ]
    return EXAMPLES.get(name, []) + shipped


@app.get("/api/search")
def search_word(word: str, k: int = 24, skip_last: int = 0) -> dict[str, Any]:
    """Find neurons whose down-projection pushes the model toward ``word``."""
    need_model()
    cleaned = word.strip()

    out = []
    for layer_idx, neuron_idx, score in eng.neurons_for_word(cleaned, k, skip_last):
        lens = eng.lens(layer_idx, neuron_idx, 6)
        out.append({"layer": layer_idx, "neuron": neuron_idx, "score": round(score, 3), "lens": lens["up" if score > 0 else "down"]})

    first_ids = eng.tok(" " + cleaned, add_special_tokens=False).input_ids
    piece = eng.tok.decode([first_ids[0]]).strip() if first_ids else ""
    return {"results": out, "partial": eng.last_search_partial, "piece": piece}


# ------------------------------------------------------------------ steering
@app.post("/api/steer")
async def steer(req: Request) -> StreamingResponse:
    """Stream side-by-side steered, baseline, and optionally reversed/ablated replies."""
    need_model()
    body = await req.json()
    pushes = pushes_of(body)
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in body.get("history") or []
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    pid = (
        eng.messages_ids([*history, {"role": "user", "content": body["prompt"]}])
        if history
        else eng.chat_ids(body["prompt"])
    )

    cols = body.get("columns", ["up", "base", "down"])
    inv = [rev for p in pushes if (rev := _reverse(p)) is not None]

    active = bool(abl_now().get("active"))
    ablated = {"abl": True, "up": active, "base": False, "down": False}

    rows: list[tuple[list[int], list[Push]]] = []
    flags: list[bool] = []
    for col in cols:
        rows.append((pid, {"up": pushes, "abl": [], "base": [], "down": inv}[col]))
        flags.append(ablated[col])

    abl_flags = flags if any(flags) else None
    max_new = int(body.get("max_new", 160))

    def _gen() -> Iterator[dict[str, Any]]:
        for row_idx, delta in eng.generate(rows, max_new=max_new, cancel=cancel, abl=abl_flags):
            if row_idx is None:
                break
            yield {"type": "delta", "col": cols[row_idx], "delta": delta}

    return sse(_gen())


@app.post("/api/chat")
async def chat(req: Request) -> StreamingResponse:
    """Stream one or more parallel chat completions with optional steering, ablation, continuation, and gauges."""
    need_model()
    body = await req.json()
    multi, more = "rows" in body, bool(body.get("continue"))
    specs = body["rows"] if multi else [body]

    rows: list[tuple[list[int], list[Push]]] = []
    flags: list[bool] = []
    for spec in specs:
        msgs = [
            {"role": m["role"], "content": str(m["content"])}
            for m in spec.get("messages") or []
            if m.get("role") in ("system", "user", "assistant") and str(m.get("content", "")).strip()
        ]

        if more:
            if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                raise HTTPException(400, "There's no reply to continue.")
            ids = eng.messages_ids(msgs[:-1]) + eng.encode(msgs[-1]["content"])
        elif not msgs or msgs[-1]["role"] != "user":
            raise HTTPException(400, "The conversation must end with a user message.")
        else:
            ids = eng.messages_ids(msgs)

        rows.append((ids, pushes_of(spec)))
        flags.append(bool(spec.get("abl")) and eng.abl_vec is not None)

    max_new = int(body.get("max_new", 512))
    temperature = float(body.get("temperature", 0.7))
    read = [str(d) for d in body.get("read") or [] if str(d) in eng.dirs]

    def _gen() -> Iterator[dict[str, Any]]:
        for row_idx, payload in eng.generate(
            rows,
            max_new=max_new,
            temperature=temperature,
            cancel=cancel,
            seed=int(time.time()),
            abl=flags,
            read=read or None,
        ):
            if row_idx is None:
                cut = [len(ids) >= max_new for ids in payload]
                yield {"type": "done", "cut": cut if multi else cut[0]}
                break

            ev: dict[str, Any] = {"type": "delta", "row": row_idx} if multi else {"type": "delta"}
            if read:
                ev["delta"], ev["read"] = payload
            else:
                ev["delta"] = payload
            yield ev

    return sse(_gen())


# ------------------------------------------------------------------ abliterate
def _probes() -> dict[str, list[str]]:
    custom = json.loads(PROBES_FILE.read_text()) if PROBES_FILE.exists() else {}
    return {
        "harmful": custom.get("harmful") or abl_mod.DEFAULT_HARMFUL,
        "harmless": custom.get("harmless") or abl_mod.DEFAULT_HARMLESS,
    }


def _abl_all() -> dict[str, Any]:
    return json.loads(ABL_FILE.read_text()) if ABL_FILE.exists() else {}


def _abl_save(all_data: dict[str, Any]) -> None:
    atomic_write_json(ABL_FILE, all_data, indent=1, ensure_ascii=True)


def abl_now() -> dict[str, Any]:
    """Return the stored refusal direction and state for the loaded model, or ``{}``."""
    return (_abl_all().get(eng.model_id) or {}) if eng.model_id else {}


def _scoped(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return the saved category config when refusal erasure is category-scoped, else ``None``."""
    cat = entry.get("category")
    return cat if cat and entry.get("scope") == "category" else None


def _apply(entry: dict[str, Any]) -> None:
    """Install a saved abliteration entry onto the live engine."""
    cat = _scoped(entry)
    eng.abl_of(entry["vec"], (cat or entry).get("coef", 1.0), entry.get("layer"))
    eng.gate_of(cat["gate"] if cat else None)


def _baked(entry: dict[str, Any]) -> tuple[list[float], float] | None:
    """Return ``(vec, coef)`` to orthogonalize into weights, or ``None`` if a category lacks weights."""
    cat = _scoped(entry)
    if not cat:
        return [float(x) for x in entry["vec"]], float(entry.get("coef", 1.0))
    weights = cat.get("weights")
    return ([float(x) for x in weights["vec"]], float(weights["coef"])) if weights else None


def restore_for(mid: str) -> None:
    """Re-apply a saved refusal direction after loading ``mid``."""
    entry = _abl_all().get(mid)
    if entry:
        _apply(entry)


def _cat_info(cat: dict[str, Any] | None) -> dict[str, Any] | None:
    """Format a saved category entry for the frontend without raw vectors."""
    if not cat:
        return None
    weights = cat.get("weights")
    return {k: cat.get(k) for k in ("name", "prompts", "coef", "test")} | {
        "gate_layer": cat["gate"]["layer"],
        "weights": {k: v for k, v in weights.items() if k != "vec"} if weights else None,
    }


@app.get("/api/abl")
def abl_info() -> dict[str, Any]:
    entry = abl_now()
    return {
        "model": eng.model_id,
        "probes": _probes(),
        "defaults": {"harmful": abl_mod.DEFAULT_HARMFUL, "harmless": abl_mod.DEFAULT_HARMLESS},
        "current": None
        if not entry
        else {
            "layer": entry.get("layer"),
            "coef": (_scoped(entry) or entry).get("coef", 1.0),
            "active": entry.get("active", False),
            "clean": entry.get("clean", False),
            "scope": "category" if _scoped(entry) else "all",
            "category": _cat_info(entry.get("category")),
        },
        "ready": bool(entry) and eng.abl_vec is not None,
    }


@app.post("/api/abl/probes")
async def abl_probes(req: Request) -> dict[str, list[str]]:
    body = await req.json()
    with _store_lock:
        merger = json.loads(PROBES_FILE.read_text()) if PROBES_FILE.exists() else {}
        merger.update(
            {k: [str(x).strip() for x in body.get(k) or [] if str(x).strip()] for k in ("harmful", "harmless")}
        )
        atomic_write_json(PROBES_FILE, merger, indent=1, ensure_ascii=True)
    return _probes()


@app.post("/api/abl/analyze")
async def abl_analyze(req: Request) -> StreamingResponse:
    need_model()
    body = await req.json() if req.headers.get("content-length", "0") not in ("", "0", None) else {}
    custom = {k: [str(x) for x in (body.get(k) or []) if str(x).strip()] for k in ("harmful", "harmless")}
    probes = custom if any(custom.values()) else _probes()

    def _gen() -> Iterator[dict[str, Any]]:
        final = None
        for ev in abl_mod.analyze(eng, probes["harmful"], probes["harmless"], cancel):
            if ev.get("type") == "result":
                final = ev
                ev = {k: v for k, v in ev.items() if k != "vec"}
            yield ev

        if final:
            with _store_lock:
                all_d = _abl_all()
                all_d[eng.model_id] = {
                    "vec": final["vec"],
                    "layer": final["layer"],
                    "coef": final["coef"],
                    "active": True,
                    "clean": final["clean"],
                    "ts": time.time(),
                }
                _abl_save(all_d)
                _apply(all_d[eng.model_id])

    return sse(_gen())


@app.post("/api/abl/state")
async def abl_state(req: Request) -> dict[str, Any]:
    """Toggle the saved refusal direction, switch scope, or adjust erasure strength."""
    need_model()
    body = await req.json()

    with _store_lock:
        all_d = _abl_all()
        entry = all_d.get(eng.model_id)
        if not entry:
            raise HTTPException(409, "Run the analysis first.")

        if body.get("scope") in ("all", "category"):
            if body["scope"] == "category" and not entry.get("category"):
                raise HTTPException(409, "Find a category first.")
            entry["scope"] = body["scope"]
        if "coef" in body:
            (_scoped(entry) or entry)["coef"] = max(0.1, min(2.0, float(body["coef"])))
        if "active" in body:
            entry["active"] = bool(body["active"])

        _abl_save(all_d)
        _apply(entry)

    return {
        "ok": True,
        "coef": (_scoped(entry) or entry).get("coef", 1.0),
        "scope": "category" if _scoped(entry) else "all",
        **{k: entry.get(k) for k in ("layer", "active")},
    }


@app.post("/api/abl/category")
async def abl_category(req: Request) -> StreamingResponse:
    """Fit a category-scoped refusal gate and an experimental weight-only direction."""
    need_model()
    entry = abl_now()
    if not entry or not entry.get("vec"):
        raise HTTPException(409, "Find the refusal direction first.")

    body = await req.json()
    name = str(body.get("name") or "").strip()[:60] or "Category"
    prompts = [str(p).strip() for p in body.get("prompts") or [] if str(p).strip()]
    probes = _probes()

    def _gen() -> Iterator[dict[str, Any]]:
        final = None
        for ev in abl_mod.category(
            eng, prompts, probes["harmful"], probes["harmless"], entry["vec"], float(entry.get("coef", 1.0)), cancel
        ):
            if ev.get("type") == "result":
                final = ev
                ev = {k: v for k, v in ev.items() if k != "gate"} | {
                    "weights": {k: v for k, v in ev["weights"].items() if k != "vec"}
                }
            yield ev

        if final:
            with _store_lock:
                all_d = _abl_all()
                e = all_d[eng.model_id]
                test = {k: final[k] for k in ("inside", "outside", "base_in", "base_out", "gate_test", "gate_acc")}
                e["category"] = {
                    "name": name,
                    "prompts": prompts,
                    "gate": final["gate"],
                    "coef": final["coef"],
                    "weights": final["weights"],
                    "test": test,
                    "ts": time.time(),
                }
                e["scope"], e["active"] = "category", True
                _abl_save(all_d)
                _apply(e)

    return sse(_gen())


@app.post("/api/abl/category/suggest")
async def abl_category_suggest(req: Request) -> StreamingResponse:
    """Ask the loaded model to write novel requests belonging to a named category."""
    need_model()
    body = await req.json()
    name = str(body.get("name") or "").strip()[:60]
    if not name:
        raise HTTPException(400, "Name the category first.")
    have = [str(p) for p in body.get("have") or []]
    return sse(abl_mod.suggest(eng, name, have, cancel=cancel))


@app.post("/api/abl/test")
async def abl_test(req: Request) -> StreamingResponse:
    """Compare plain and ablated replies on test probes and measure refusal rates and fluency cost."""
    need_model()
    if eng.abl_vec is None:
        raise HTTPException(409, "Analyse the model first: there is no direction to erase yet.")
    body = await req.json()
    probes = [str(p).strip() for p in (body.get("probes") or []) if str(p).strip()][:4] or abl_mod.EXAMPLE_PROMPTS

    n_probes = len(probes)
    rows: list[tuple[list[int], list[Push]]] = []
    for prompt in probes:
        pid = eng.chat_ids(prompt)
        rows.extend([(pid, []), (pid, [])])

    flags = [i % 2 == 1 for i in range(2 * n_probes)]

    def _gen() -> Iterator[dict[str, Any]]:
        texts = [""] * (2 * n_probes)
        for row_idx, delta in eng.generate(rows, max_new=160, temperature=0, cancel=cancel, abl=flags):
            if row_idx is None:
                break
            texts[row_idx] += delta
            yield {"type": "delta", "probe": row_idx // 2, "kind": "base" if row_idx % 2 == 0 else "abl", "delta": delta}

        opened = eng.last_gate[1::2] if eng.last_gate else None
        verdicts = abl_mod.refused(texts)
        nll0 = float(eng.reference_nll([[]], cancel, abl=[False])[0])
        nll1 = nll0 if eng.abl_gate is not None else float(eng.reference_nll([[]], cancel, abl=[True])[0])
        yield {"type": "result", "refused": [verdicts[i::2] for i in (0, 1)], "nll": [nll0, nll1], "gate": opened}

    return sse(_gen())


export_job: dict[str, Any] = {
    "kind": None,
    "name": None,
    "dst": None,
    "status": "idle",
    "done": 0,
    "total": 0,
    "msg": "",
    "cancel": False,
}


def _clean_name(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "", str(raw or "")).strip(" .")


def _start_export(kind: str, name: str, **edits: Any) -> dict[str, Any]:
    """Bake an edited copy of the active model into ``COPIES/name`` on a background thread."""
    with _state_lock:
        if export_job["status"] == "working":
            raise HTTPException(409, f"Already writing {export_job['name']}.")
        if eng.ink:
            raise HTTPException(409, "Inkling can't be saved as an edited copy: use it live here or through the API.")
        export_job.update(kind=kind, name=name, dst=None, status="working", done=0, total=0, msg="Starting…", cancel=False)

    def _work() -> None:
        tmp, dst = COPIES / (name + ".partial"), COPIES / name
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            def _prog(done: int, total: int, msg: str) -> None:
                with _state_lock:
                    if export_job["cancel"]:
                        raise Cancelled()
                    export_job.update(done=done, total=total, msg=msg)

            abl_mod.bake(eng, tmp, _prog, cancel, **edits)
            with _state_lock:
                if export_job["cancel"]:
                    raise Cancelled()

            # Replace any prior export only after the new copy finishes writing cleanly.
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
            with _state_lock:
                export_job.update(dst=str(dst), status="done", msg="")
        except Cancelled:
            shutil.rmtree(tmp, ignore_errors=True)
            with _state_lock:
                export_job["status"] = "idle"
        except Exception as exc:
            logger.exception("Model export failed for %s", name)
            shutil.rmtree(tmp, ignore_errors=True)
            with _state_lock:
                export_job["status"] = f"error: {exc}"

    threading.Thread(target=_work, daemon=True).start()
    return abl_export_status()


@app.get("/api/abl/export")
def abl_export_status() -> dict[str, Any]:
    with _state_lock:
        return {k: v for k, v in export_job.items() if k != "cancel"}


@app.post("/api/abl/export")
async def abl_export(req: Request) -> dict[str, Any]:
    need_model()
    if eng.abl_vec is None:
        raise HTTPException(409, "Analyse the model first: there is no direction to bake in yet.")

    entry = abl_now()
    if not entry or not entry.get("vec"):
        raise HTTPException(409, "No saved direction for this model: analyse first.")

    body = await req.json()
    baked = _baked(entry)
    if baked is None:
        raise HTTPException(409, "This category has no version for the weights yet: find it again.")
    vec, coef = baked
    if not _scoped(entry):
        coef = max(0.1, min(2.0, float(body.get("coef", coef))))

    base = model_short_name(eng.model_id)
    name = _clean_name(body.get("name")) or f"{base}-abliterated" + ("" if coef == 1.0 else f"-{coef:g}")
    return _start_export("abl", name, abl=(vec, coef))


@app.post("/api/steer/export")
async def steer_export(req: Request) -> dict[str, Any]:
    """Bake a steering recipe (and optionally active refusal erasure) into a new model copy."""
    need_model()
    body = await req.json()
    raw = body.get("items") or []

    if any(i.get("kind") == "dir" and i.get("op") == "erase" for i in raw):
        raise HTTPException(
            409,
            "An erased direction can't be built into the weights yet. Switch it to a push or turn it off first.",
        )

    items = [
        {"layer": int(i["layer"]), "neuron": int(i["neuron"]), "dose": float(i["dose"])}
        for i in raw
        if i.get("kind") != "dir" and i.get("neuron") is not None and i.get("mode", "add") == "add"
    ]
    items += [
        {"kind": "dir", "dir": str(i["dir"]), "layers": [int(layer_idx) for layer_idx in i["layers"]], "dose": float(i["dose"])}
        for i in raw
        if i.get("kind") == "dir"
    ]

    pushes_of({"pushes": items})
    if not items:
        raise HTTPException(409, "Turn on some neurons first: there is nothing to build in yet.")

    entry = abl_now()
    abl = _baked(entry) if body.get("abl") and entry.get("active") and entry.get("vec") else None

    base = model_short_name(eng.model_id)
    name = _clean_name(body.get("name")) or f"{base}-steered"
    return _start_export(
        "steer",
        name,
        steer=items,
        abl=abl,
        fit_prompts=discover.DEFAULT_PROMPTS[:6],
        recipe=str(body.get("recipe") or ""),
    )


@app.post("/api/abl/export/cancel")
def abl_export_cancel() -> dict[str, bool]:
    with _state_lock:
        export_job["cancel"] = True
    return {"ok": True}


# ------------------------------------------------------------------ tuning
@app.get("/api/tune/defaults")
def tune_defaults() -> dict[str, list[str]]:
    return {"prompts": discover.DEFAULT_PROMPTS}


@app.post("/api/tune/draft")
async def tune_draft(req: Request) -> StreamingResponse:
    need_model()
    body = await req.json()
    return sse(discover.draft_targets(eng, body["behavior"], body["prompts"], int(body.get("max_new", 72)), cancel))


@app.post("/api/tune/search")
async def tune_search(req: Request) -> StreamingResponse:
    need_model()
    body = await req.json()
    settings = body.get("settings", {})
    budget = discover.budget_for(eng.n_params)
    if settings.get("candidates"):
        budget["n_candidates"] = int(settings["candidates"])

    return sse(
        discover.search(
            eng,
            body["pairs"],
            behavior=body.get("behavior") or None,
            **budget,
            max_neurons=int(settings.get("max_neurons", 6)),
            doses=tuple(float(d) for d in settings.get("doses", (0.25, 0.5, 1.0))),
            contrast=float(settings.get("contrast", 0.5)),
            fluency_budget=float(settings.get("fluency_budget", 0.35)),
            max_new=int(settings.get("max_new", 72)),
            focus=[w.strip() for w in re.split(r"[,\n]", settings.get("focus") or "") if w.strip()],
            focus_weight=float(settings.get("focus_weight", 5.0)),
            cancel=cancel,
        )
    )


@app.post("/api/tune/refine")
async def tune_refine(req: Request) -> StreamingResponse:
    """Refine the active neuron recipe from user feedback on a steered reply."""
    need_model()
    body = await req.json()
    recipe = [
        {"layer": int(i["layer"]), "neuron": int(i["neuron"]), "dose": float(i["dose"]), "lens": i.get("lens") or []}
        for i in body.get("recipe") or []
        if i.get("kind") != "dir"
    ]

    if not recipe:
        raise HTTPException(409, "Turn on some neurons first: there is nothing to fix yet.")
    if not (body.get("feedback") or "").strip():
        raise HTTPException(400, "Say what's wrong with the reply.")

    big = eng.n_params > 4e9
    budget = discover.budget_for(eng.n_params)
    others = list(body.get("prompts") or discover.DEFAULT_PROMPTS)[: 2 if big else 4]

    return sse(
        discover.refine(
            eng,
            recipe,
            feedback=body["feedback"],
            prompt=body["prompt"],
            bad=body.get("bad") or "",
            behavior=body.get("behavior") or None,
            prompts=others,
            abl=bool(abl_now().get("active")),
            target_tokens=budget["target_tokens"] or 48,
            n_new=8 if big else 12,
            rounds=2 if big else 3,
            cancel=cancel,
        )
    )


# ------------------------------------------------------------------ directions
def _dir_row(entry: dict[str, Any]) -> dict[str, Any]:
    direction = eng.dirs.get(entry["id"])
    metrics = entry.get("metrics") or {}
    auc = metrics.get("auc") or []
    band = entry.get("band") or []
    return {k: entry.get(k) for k in ("id", "name", "kind", "band", "created", "pos", "neg")} | {
        "ready": direction is not None,
        "auc": round(min(auc[layer_idx] for layer_idx in band), 2) if auc and band else None,
    }


@app.get("/api/directions")
def directions_list() -> dict[str, Any]:
    need_model()
    return {"directions": [_dir_row(e) for e in dirs.index()], "layers": eng.num_layers}


@app.get("/api/directions/defaults")
def directions_defaults() -> dict[str, Any]:
    return {"example": dir_mod.EXAMPLE}


@app.post("/api/directions/extract")
async def directions_extract(req: Request) -> StreamingResponse:
    need_model()
    body = await req.json()
    return sse(dir_mod.extract(eng, dirs, body, cancel))


@app.post("/api/directions/import")
async def directions_import(req: Request) -> dict[str, Any]:
    need_model()
    body = await req.json()
    path = Path(str(body.get("path") or "")).expanduser()

    if not path.is_file():
        raise HTTPException(400, f"No file at {path}.")

    try:
        added = dirs.import_npz(path, cancel)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    return {"added": [_dir_row(e) for e in added]}


@app.delete("/api/directions/{did}")
def directions_delete(did: str) -> dict[str, bool]:
    need_model()
    dirs.delete(did)
    return {"ok": True}


# ------------------------------------------------------------------ probes and experiments
def _probe(pid: str) -> dict[str, Any]:
    try:
        return probes.get(pid)
    except KeyError as exc:
        raise HTTPException(404, "No such probe for this model.") from exc


@app.get("/api/probes")
def probes_list() -> dict[str, Any]:
    need_model()
    return {"probes": probes.index(), "examples": exp_mod.examples()}


@app.get("/api/probes/{pid}")
def probes_get(pid: str) -> dict[str, Any]:
    need_model()
    return _probe(pid)


@app.delete("/api/probes/{pid}")
def probes_delete(pid: str) -> dict[str, bool]:
    need_model()
    _probe(pid)
    probes.delete(pid)
    return {"ok": True}


@app.post("/api/probes/run")
async def probes_run(req: Request) -> StreamingResponse:
    """Build a probe from two sets of prompts (``{name, a: {label, prompts}, b: {...}}``)."""
    need_model()
    spec = await req.json()
    try:
        probe_mod.parse_spec(spec)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return sse(probe_mod.run(eng, probes, spec, cancel))


@app.post("/api/probes/{pid}/score")
async def probes_score(pid: str, req: Request) -> dict[str, Any]:
    """Score a new prompt through a saved probe: ``{prompt} -> {p, z, path}``."""
    need_model()
    _probe(pid)
    body = await req.json()
    vec = probes.vectors(pid)
    return await run_in_threadpool(probe_mod.score, eng, vec, str(body.get("prompt", "")))


@app.post("/api/probes/{pid}/experiment")
async def probes_export(pid: str, req: Request) -> dict[str, str]:
    """Write a saved probe as an experiment file in Downloads: ``{about, try} -> {path}``."""
    need_model()
    body = await req.json()
    exp = exp_mod.from_probe(_probe(pid), body.get("about", ""), body.get("try") or [])
    out = Path.home() / "Downloads" / f"{pid}.lodestar.json"
    atomic_write_json(out, exp, indent=1, ensure_ascii=False)
    return {"path": str(out)}


@app.get("/api/experiments/{name}")
def experiments_example(name: str) -> dict[str, Any]:
    """Return one of the shipped experiments by file name or keyword (e.g. ``"hive"``)."""
    try:
        return exp_mod.find_example(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/experiments/fetch")
async def experiments_fetch(req: Request) -> dict[str, Any] | None:
    """Fetch an experiment from a URL or from the macOS clipboard when ``{clipboard: true}``."""
    body = await req.json()
    url = str(body.get("url") or "")
    if body.get("clipboard"):
        url = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        if not exp_mod.is_link(url):
            return None
    try:
        return await run_in_threadpool(exp_mod.fetch, url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/experiments/steer")
async def experiments_steer(req: Request) -> dict[str, Any]:
    """Convert a steering experiment's pushes into recipe items for the loaded model."""
    need_model()
    exp = await req.json()
    try:
        items, mismatched = await run_in_threadpool(exp_mod.open_steer, eng, dirs, exp)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"items": items, "mismatched": mismatched}


@app.post("/api/recipes/experiment")
async def recipes_export(req: Request) -> dict[str, str]:
    """Write the current steering setup as an experiment file in Downloads."""
    need_model()
    body = await req.json()
    name = str(body.get("name") or "Steering").strip()
    try:
        exp = exp_mod.from_steer(eng, body.get("items") or [], name, body.get("about", ""), body.get("try") or [])
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    out = Path.home() / "Downloads" / f"{dir_mod.make_id(name, set())}.lodestar.json"
    atomic_write_json(out, exp, indent=1, ensure_ascii=False)
    return {"path": str(out)}


@app.post("/api/experiments/run")
async def experiments_run(req: Request) -> StreamingResponse:
    """Run an experiment file: build its probe and check results against its ``expect`` ranges."""
    need_model()
    exp = await req.json()
    try:
        exp_mod.validate(exp)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if "steer" in exp:
        raise HTTPException(400, "This is a steering experiment: open it beside Saved.")
    return sse(exp_mod.run(eng, probes, exp, cancel))


# ------------------------------------------------------------------ recipes
def _recipes() -> list[dict[str, Any]]:
    return json.loads(RECIPES.read_text()) if RECIPES.exists() else []


@app.get("/api/recipes")
def recipes() -> list[dict[str, Any]]:
    return _recipes()


@app.post("/api/recipes")
async def save_recipe(req: Request) -> dict[str, Any]:
    recipe = await req.json()
    recipe.setdefault("id", f"r{int(time.time() * 1000)}")
    recipe.setdefault("created", time.time())
    recipe.setdefault("model", eng.model_id)

    with _store_lock:
        all_recipes = [item for item in _recipes() if item["id"] != recipe["id"]] + [recipe]
        atomic_write_json(RECIPES, all_recipes, indent=1, ensure_ascii=True)
    return recipe


@app.delete("/api/recipes/{rid}")
def delete_recipe(rid: str) -> dict[str, bool]:
    with _store_lock:
        atomic_write_json(RECIPES, [item for item in _recipes() if item["id"] != rid], indent=1, ensure_ascii=True)
    return {"ok": True}


# ------------------------------------------------------------------ chats
def _chat_file(cid: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid):
        raise HTTPException(400, "Bad chat id.")
    return CHATS / f"{cid}.json"


def _summary(chat_doc: dict[str, Any]) -> dict[str, Any]:
    return {k: chat_doc.get(k) for k in ("id", "title", "model", "updated")} | {
        "turns": sum(m["role"] == "user" for m in chat_doc["messages"])
    }


@app.get("/api/chats")
def chats() -> list[dict[str, Any]]:
    """Return saved conversations sorted newest first."""
    out: list[dict[str, Any]] = []
    for path in CHATS.glob("*.json"):
        try:
            out.append(_summary(json.loads(path.read_text())))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda c: -(c["updated"] or 0))


@app.get("/api/chats/{cid}")
def chat_get(cid: str) -> dict[str, Any]:
    path = _chat_file(cid)
    if not path.exists():
        raise HTTPException(404, "That chat is gone.")
    return json.loads(path.read_text())


@app.post("/api/chats")
async def chat_save(req: Request) -> dict[str, Any]:
    """Atomically save or overwrite a conversation along with its active steering settings."""
    chat_doc = await req.json()
    path = _chat_file(chat_doc.get("id", ""))
    chat_doc["updated"] = time.time()
    chat_doc.setdefault("model", eng.model_id)
    atomic_write_json(path, chat_doc, indent=None, ensure_ascii=False)
    return _summary(chat_doc)


@app.delete("/api/chats/{cid}")
def chat_delete(cid: str) -> dict[str, bool]:
    _chat_file(cid).unlink(missing_ok=True)
    return {"ok": True}


# ------------------------------------------------------------------ OpenAI-compatible API
serving: dict[str, Any] = {"pushes": [], "log": deque(maxlen=12), "active": 0}


@app.post("/api/serve")
async def serve_set(req: Request) -> dict[str, bool]:
    """Update the active steering recipe mirrored by the OpenAI-compatible ``/v1`` endpoint."""
    body = await req.json()
    with _state_lock:
        serving["pushes"] = [p for p in body.get("pushes") or [] if isinstance(p, dict)]
    return {"ok": True}


@app.get("/api/serve")
def serve_get() -> dict[str, Any]:
    with _state_lock:
        log_copy = list(serving["log"])
        active = serving["active"]
    return {"models": _api_models(), "log": log_copy, "active": active}


def _api_models() -> list[dict[str, str]]:
    """Return ``[{"id": ..., "about": ...}]`` served by ``/v1/models`` (active configuration first)."""
    if eng.model is None or eng.status != "ready":
        return []

    name = model_short_name(eng.model_id)
    out = [{"id": name, "about": "As set in Lodestar"}, {"id": f"{name}-plain", "about": "Plain"}]
    entry = abl_now()
    if entry.get("vec") is not None:
        cat = _scoped(entry)
        about = f"Refusals erased for {cat['name']}" if cat else "Refusals erased"
        out.append({"id": f"{name}-abliterated", "about": about})
    return out


def _oai_error(status_code: int, msg: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": kind, "param": None, "code": None}}, status_code=status_code)


def _text(content: Any) -> str:
    """Flatten OpenAI message content (plain string or content-part list) into a string."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return "" if content is None else str(content)


@app.get("/v1/models")
def api_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": 0, "owned_by": "lodestar"} for m in _api_models()],
    }


@app.post("/v1/chat/completions")
async def api_chat(req: Request) -> Any:
    try:
        body = await req.json()
    except ValueError:
        return _oai_error(400, "The body isn't JSON.")

    available = _api_models()
    if not available:
        return _oai_error(503, "No model is loaded in Lodestar.", "server_error")

    model_map = {m["id"]: m for m in available}
    want = str(body.get("model") or "")
    first = available[0]["id"]
    if want and want not in model_map:
        return _oai_error(404, f"No model called {want!r}. This server has: {', '.join(model_map)}.", "not_found_error")
    if body.get("tools"):
        return _oai_error(400, "Tool calling isn't supported.")
    if int(body.get("n") or 1) != 1:
        return _oai_error(400, "Only n=1 is supported.")

    msgs: list[dict[str, str]] = []
    for raw_msg in body.get("messages") or []:
        role = {"developer": "system"}.get(raw_msg.get("role"), raw_msg.get("role"))
        if role not in ("system", "user", "assistant"):
            return _oai_error(400, f"Messages with role {raw_msg.get('role')!r} aren't supported.")
        text = _text(raw_msg.get("content"))
        if text.strip():
            msgs.append({"role": role, "content": text})
    if not msgs or msgs[-1]["role"] != "user":
        return _oai_error(400, "The conversation must end with a user message.")

    mid = want or first
    if mid == first:
        with _state_lock:
            pushes = list(serving["pushes"])
        erase = bool(abl_now().get("active"))
    elif mid == f"{first}-plain":
        pushes, erase = [], False
    else:
        pushes, erase = [], True

    try:
        row = (eng.messages_ids(msgs), pushes_of({"pushes": pushes}))
    except HTTPException as exc:
        return _oai_error(exc.status_code, str(exc.detail))
    except (ValueError, KeyError) as exc:
        return _oai_error(400, str(exc))

    max_new = max(1, min(int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024), 8192))
    temp = float(body.get("temperature") if body.get("temperature") is not None else 0.7)
    seed = int(body["seed"]) if body.get("seed") is not None else int(time.time())
    raw_stops = body.get("stop") or []
    stops = [raw_stops] if isinstance(raw_stops, str) else [str(x) for x in raw_stops if x]
    erase = erase and eng.abl_vec is not None

    rid, made = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    stop_ev = threading.Event()

    def _run() -> Iterator[str | tuple[str, int]]:
        with _state_lock:
            serving["active"] += 1
        t0, text, n_toks, finish = time.time(), "", 0, "stop"
        gen = eng.generate([row], max_new=max_new, temperature=temp, cancel=stop_ev, seed=seed, abl=[erase])

        try:
            for row_idx, delta in gen:
                if row_idx is None:
                    n_toks = len(delta[0])
                    finish = "length" if n_toks >= max_new else "stop"
                    break

                cut = None
                combined = text + delta
                for stop_seq in stops:
                    idx = combined.find(stop_seq, max(0, len(text) - len(stop_seq)))
                    if idx >= 0 and (cut is None or idx < cut):
                        cut = idx
                if cut is not None:
                    if cut > len(text):
                        yield combined[len(text) : cut]
                    text, finish = combined[:cut], "stop"
                    n_toks = len(eng.encode(text))
                    break

                text = combined
                yield delta
        except Cancelled:
            finish, n_toks = "stop", len(eng.encode(text))
        finally:
            gen.close()
            with _state_lock:
                serving["active"] -= 1
                serving["log"].appendleft(
                    {
                        "at": time.time(),
                        "model": mid,
                        "prompt": len(row[0]),
                        "tokens": n_toks,
                        "seconds": round(time.time() - t0, 1),
                        "finish": finish,
                        "stream": bool(body.get("stream")),
                    }
                )

        yield (finish, n_toks)

    def _usage(completion_tokens: int) -> dict[str, int]:
        prompt_tokens = len(row[0])
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    if not body.get("stream"):
        def _whole() -> tuple[str, tuple[str, int]]:
            parts: list[str] = []
            end = ("stop", 0)
            for item in _run():
                if isinstance(item, tuple):
                    end = item
                else:
                    parts.append(item)
            return "".join(parts), end

        text, (finish, n_toks) = await run_in_threadpool(_whole)
        return {
            "id": rid,
            "object": "chat.completion",
            "created": made,
            "model": mid,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": _usage(n_toks),
        }

    def _chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        payload = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": made,
            "model": mid,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | tuple[str, int] | Exception] = asyncio.Queue()

    def _work() -> None:
        try:
            for item in _run():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            logger.exception("Streaming /v1/chat/completions failed")
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    async def _events():
        threading.Thread(target=_work, daemon=True).start()
        try:
            yield _chunk({"role": "assistant", "content": ""})
            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    yield "data: " + json.dumps({"error": {"message": str(item), "type": "server_error"}}) + "\n\n"
                    break
                if isinstance(item, tuple):
                    finish, n_toks = item
                    yield _chunk({}, finish)
                    if (body.get("stream_options") or {}).get("include_usage"):
                        usage_chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": made,
                            "model": mid,
                            "choices": [],
                            "usage": _usage(n_toks),
                        }
                        yield "data: " + json.dumps(usage_chunk) + "\n\n"
                    break
                if item:
                    yield _chunk({"content": item})
            yield "data: [DONE]\n\n"
        finally:
            stop_ev.set()

    return StreamingResponse(_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ ui
@app.get("/", response_model=None)
def index() -> Response:
    if not (UI / "index.html").exists():
        return HTMLResponse(
            "<p>The UI isn’t built yet: run <code>pnpm install && pnpm build</code> in web/, then reopen.</p>"
        )
    return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=UI, check_dir=False), name="ui")


@app.exception_handler(HTTPException)
async def http_err(_req: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
