"""Local HTTP API for the Lodestar window. Streams (server-sent events) for anything that generates text or runs a
search; plain JSON for lookups. Serves the UI built in ../web/dist."""
import json
import math
import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import abl as abl_mod
from . import directions as dir_mod
from . import discover
from . import memory
from .engine import Cancelled, Engine, Push

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "web" / "dist"  # the React build: `pnpm build` in web/ (scripts/build_app.sh does it)
# the data folder keeps the app's first name, so models, recipes and directions saved before the rename stay put
DATA = Path(os.environ.get("AUTO_INTERP_DATA", Path.home() / "Library" / "Application Support" / "Auto Interp"))
DATA.mkdir(parents=True, exist_ok=True)
RECIPES = DATA / "recipes.json"
ABL_FILE = DATA / "abl.json"  # saved refusal directions + state, keyed by model id
LAST_MODEL = DATA / "last_model.txt"  # the model to open again at the next launch
PROBES_FILE = DATA / "abl-probes.json"  # customised probe lists
CHATS = DATA / "chats"  # one file per saved conversation
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

PRESETS = [
    {"id": "Qwen/Qwen3-0.6B", "size": "1.2 GB", "note": "Fast, good for exploring"},
    {"id": "Qwen/Qwen3-1.7B", "size": "3.4 GB", "note": "Smarter, still quick"},
    {"id": "Qwen/Qwen3-4B", "size": "8 GB", "note": "Good balance of smarts and speed"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "size": "3.4 GB", "note": "A different model family"},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "size": "2.5 GB", "note": "Needs a Hugging Face login"},
    {"id": "google/gemma-3-1b-it", "size": "2 GB", "note": "Needs a Hugging Face login"},
]

# Known neurons per model (matched on the model's folder or repo name), so a first try is one click.
EXAMPLES = {
    "Qwen3-8B": [
        {"name": "Gnome", "note": "One neuron that makes the model describe itself as a small creature (@Sauers_)",
         "prompt": "Describe the way you look.", "items": [{"layer": 8, "neuron": 3602, "dose": 1.0, "label": "gnome"}]},
        {"name": "Forest gnome", "note": "The three-neuron version from the same thread: greenish skin, pointed ears",
         "prompt": "Describe the way you look.",
         "items": [{"layer": 3, "neuron": 6056, "dose": -0.5, "label": "gnome a"},
                   {"layer": 4, "neuron": 1658, "dose": 0.5, "label": "gnome b"},
                   {"layer": 8, "neuron": 3602, "dose": 0.5, "label": "gnome"}]},
    ],
    "Qwen3-1.7B": [
        {"name": "Cookies", "note": "Found by Describe: brings up cookies; loops above strength 0.35",
         "prompt": "What makes you happy?", "items": [{"layer": 26, "neuron": 2847, "dose": -0.3, "label": "cookie"}]},
    ],
    "Qwen3-0.6B": [
        {"name": "Hungry", "note": "Found by Describe: turns answers toward food and eating",
         "prompt": "What should I eat for dinner tonight?", "items": [{"layer": 9, "neuron": 2578, "dose": 1.2, "label": "munch"}]},
    ],
}

eng = Engine()
dirs = dir_mod.Library(DATA / "directions", eng)
cancel = threading.Event()
app = FastAPI()


def sse(gen):
    """Wraps an event generator as a text/event-stream; errors and cancellations become events."""
    def run():
        cancel.clear()

        try:
            for ev in gen:
                yield f"data: {json.dumps(_finite(ev), default=_np)}\n\n"
        except Cancelled:
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
        except Exception as e:  # surface the message in the UI rather than a dead stream
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(run(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _finite(o):
    """Browsers reject NaN and Infinity in JSON; send them as null."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@app.middleware("http")
async def no_store(req, call_next):  # the UI is local and edited in place; never serve it stale
    resp = await call_next(req)
    if not req.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def need_model():
    if eng.model is None or eng.status != "ready":
        raise HTTPException(409, "Load a model first (bottom left).")


def pushes_of(body):
    out = [Push.of(p) for p in body.get("pushes") or [] if p.get("enabled", True)]
    L = len(eng.layers) if eng.model is not None else 0

    for p in out:
        if p.mode not in ("dir", "erase"):
            continue

        if p.mode == "erase":
            p.dose = min(max(p.dose, 0.0), 1.0)  # 1 = fully back to ordinary text; past that it overshoots
        if p.dir_id not in eng.dirs:
            raise HTTPException(409, f"This model has no direction called “{p.dir_id}”.")
        if not all(0 <= l < L for l in p.layers) or (p.mode == "dir" and not p.layers):
            raise HTTPException(400, f"Pick layers between 0 and {L - 1} for “{p.dir_id}”.")

    return out


def _reverse(p):
    """The opposite of a push: add and direction doses flip sign, a multiplier mirrors around 1 (ablate x0 <-> double
    x2). Erasing has no opposite: exaggerating the component at every layer compounds into noise, so it's dropped."""
    if p.mode == "erase":
        return None
    if p.mode == "mul":
        return Push(p.layer, p.neuron, 2 - p.dose, p.mode)
    return Push(p.layer, p.neuron, -p.dose, p.mode, p.dir_id, p.layers)


# ------------------------------------------------------------------ models
LOADING = {"bytes": None}  # the size of the weights being loaded, when known (it isn't before a download)


@app.get("/api/status")
def status():
    last = LAST_MODEL.read_text().strip() if LAST_MODEL.exists() else None
    out = {**eng.info(), "progress": eng.progress, "last": last}

    if eng.load_state:  # for the loading scene: the stage, and how much of each layer has arrived
        out["load"] = {**eng.load_state, "bytes": LOADING["bytes"]}

    return out


# Copies of drive models on this Mac's internal disk. A USB 3 stick reads at ~0.4 GB/s against ~3 GB/s for
# the internal SSD, and reading the weights is nearly all of a big model's load time.
COPIES = DATA / "models"
copy_job = {"src": None, "name": None, "done": 0, "total": 0, "status": "idle", "cancel": False}


def _model_files(src):
    """Files that make up a model folder, skipping macOS ._ sidecar files and the download cache."""
    return [f for f in sorted(src.iterdir()) if f.is_file() and not f.name.startswith(".")]


def _folder_size(d):
    return sum(f.stat().st_size for f in _model_files(d))


def _generates(snap):
    """Whether a downloaded model writes text, rather than only classifying it (a zero-shot or a rejection
    classifier, which other tools leave in the same cache)."""
    try:
        archs = json.loads((snap / "config.json").read_text()).get("architectures") or []
    except (OSError, ValueError):
        return True  # can't tell: list it, and let loading say if it doesn't work

    return not archs or any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs)


@app.get("/api/models")
def models():
    cached = []
    if HF_CACHE.exists():
        for d in sorted(HF_CACHE.glob("models--*")):
            snaps = sorted(s for s in d.glob("snapshots/*") if any(s.glob("*.safetensors")))
            if snaps and _generates(snaps[-1]):
                cached.append(d.name[len("models--"):].replace("--", "/"))

    # models saved as plain folders on external drives: /Volumes/<drive>/models/<name>
    copies = [{"path": str(d), "name": d.name, "size": _folder_size(d)}
              for d in sorted(COPIES.glob("*")) if d.is_dir() and not d.name.endswith(".partial")
              and (d / "config.json").exists()]

    copied = {c["name"] for c in copies}
    local = [{"path": str(d), "name": d.name, "drive": d.parent.parent.name, "size": _folder_size(d), "copied": d.name in copied}
             for d in sorted(Path("/Volumes").glob("*/models/*"))
             if (d / "config.json").exists() and any(d.glob("*.safetensors"))]

    return {"presets": PRESETS, "cached": cached, "local": local, "copies": copies}


@app.get("/api/models/copy")
def copy_status():
    return {k: v for k, v in copy_job.items() if k != "cancel"}


@app.post("/api/models/copy")
async def copy_model(req: Request):
    body = await req.json()
    src = Path(body["path"])

    if not (src.is_dir() and str(src).startswith("/Volumes/") and (src / "config.json").exists()):
        raise HTTPException(400, "That isn't a model folder on an external drive.")
    if copy_job["status"] == "copying":
        raise HTTPException(409, f"Already copying {copy_job['name']}.")

    files = _model_files(src)
    total = sum(f.stat().st_size for f in files)

    COPIES.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(COPIES).free
    margin = 10e9  # leave room for the system and swap
    if free < total + margin:
        raise HTTPException(400, f"Not enough space: the copy needs {total / 1e9:.1f} GB and this Mac has {free / 1e9:.1f} GB free.")

    copy_job.update(src=str(src), name=src.name, done=0, total=total, status="copying", cancel=False)

    def work():
        tmp, dst = COPIES / (src.name + ".partial"), COPIES / src.name

        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            for f in files:
                with open(f, "rb") as a, open(tmp / f.name, "wb") as b:
                    while chunk := a.read(64 << 20):
                        if copy_job["cancel"]:
                            raise Cancelled()
                        b.write(chunk)
                        copy_job["done"] += len(chunk)

            tmp.rename(dst)  # only a finished copy ever appears under its real name
            copy_job["status"] = "done"
        except Cancelled:
            shutil.rmtree(tmp, ignore_errors=True)
            copy_job["status"] = "idle"
        except Exception as e:
            traceback.print_exc()
            shutil.rmtree(tmp, ignore_errors=True)
            copy_job["status"] = f"error: {e}"

    threading.Thread(target=work, daemon=True).start()
    return copy_status()


@app.post("/api/models/copy/cancel")
def cancel_copy():
    copy_job["cancel"] = True
    return {"ok": True}


@app.post("/api/models/remove")
async def remove_copy(req: Request):
    body = await req.json()
    d = Path(body["path"]).resolve()

    if d.parent != COPIES.resolve() or not (d / "config.json").exists():
        raise HTTPException(400, "Only copies made by Lodestar can be removed here.")
    if eng.model_id == str(d):
        raise HTTPException(409, "That copy is loaded right now. Load another model first.")

    shutil.rmtree(d)
    return {"ok": True}


@app.post("/api/load")
async def load(req: Request):
    body = await req.json()
    mid = body["model"].strip()

    LOADING["bytes"] = memory.weights_bytes(mid, HF_CACHE)

    def work():
        try:
            eng.load(mid)
            restore_for(mid)
            dirs.load()
            LAST_MODEL.write_text(mid)
        except Exception as e:
            traceback.print_exc()
            eng.status = f"error: {e}"
            eng.progress = None
            eng.load_state = None

    threading.Thread(target=work, daemon=True).start()
    time.sleep(0.05)
    return status()


@app.post("/api/unload")
def unload():
    """Free the model's memory. It isn't opened again at the next launch either."""
    cancel.set()

    with eng.lock:  # waits for a stream that's running to see the cancel and stop
        eng.unload()
        eng.status = "idle"
        eng.progress = None

    LAST_MODEL.unlink(missing_ok=True)
    return status()


# ------------------------------------------------------------------ memory
@app.get("/api/memory")
def memory_now():
    return memory.snapshot()


@app.post("/api/memory/fit")
async def memory_fit(req: Request):
    """Whether a model fits beside what's already running, asked before loading it."""
    mid = (await req.json())["model"].strip()

    # loading a different model unloads this one first, so what it holds (all but the runtime's ~1 GB) comes free
    loaded = 0
    if eng.model is not None and eng.model_id != mid:
        loaded = max((memory.footprint(os.getpid()) or 0) - memory.GB, 0)

    return memory.fit(mid, HF_CACHE, loaded)


@app.post("/api/cancel")
def do_cancel():
    cancel.set()
    return {"ok": True}


# ------------------------------------------------------------------ trace
@app.post("/api/trace")
async def trace(req: Request):
    need_model()
    body = await req.json()
    pushes = pushes_of(body)
    pid = eng.chat_ids(body["prompt"])

    def gen():
        tr = eng.run_trace(pid, eng.encode(body["response"]), pushes, abl=bool(body.get("abl")))
        yield {"type": "trace", "tokens": tr["tokens"], "n_prompt": tr["n_prompt"]}

    return sse(gen())


def _neuron_row(l, n, _impact, act, _t, lens_k=5):
    lens = eng.lens(l, n, lens_k)
    return {"layer": l, "neuron": n, "act": round(act, 3), "lens": lens["up" if act >= 0 else "down"]}


@app.get("/api/trace/token")
def trace_token(t: int, k: int = 24):
    need_model()
    tr = eng.trace

    if tr is None or not 0 <= t < len(tr["ids"]):
        raise HTTPException(404, "no trace")

    top = eng.trace_top_neurons(t=t, k=k)
    ids, ps = tr["lens_ids"][:, t, :], tr["lens_p"][:, t, :]

    return {
        "lens": [[{"tok": eng.tok.decode([int(i)]), "p": round(float(p), 4)} for i, p in zip(ids[l], ps[l])]
                 for l in range(ids.shape[0])],
        "neurons": [_neuron_row(*x) for x in top],
    }


# ------------------------------------------------------------------ neurons
@app.get("/api/neuron")
def neuron(layer: int, n: int):
    need_model()
    L, N = eng.dnorm.shape

    if not (0 <= layer < L and 0 <= n < N):
        raise HTTPException(400, f"layer must be 0..{L - 1}, neuron 0..{N - 1}")

    out = {"layer": layer, "neuron": n, "lens": eng.lens(layer, n, 16)}

    tr = eng.trace
    if tr is not None:
        a = tr["acts"][layer, :, n].astype(np.float32)
        out["trace"] = {"tokens": tr["tokens"], "n_prompt": tr["n_prompt"], "acts": np.round(a, 3).tolist()}

    return out


@app.get("/api/examples")
def examples():
    need_model()
    return EXAMPLES.get(eng.model_id.rstrip("/").split("/")[-1], [])


@app.get("/api/search")
def search_word(word: str, k: int = 24, skip_last: int = 0):
    """Neurons that push the model toward a word."""
    need_model()

    out = []
    for l, n, sc in eng.neurons_for_word(word.strip(), k, skip_last):
        lens = eng.lens(l, n, 6)
        out.append({"layer": l, "neuron": n, "score": round(sc, 3), "lens": lens["up" if sc > 0 else "down"]})

    first = eng.tok(" " + word.strip(), add_special_tokens=False).input_ids[0]
    return {"results": out, "partial": eng.last_search_partial, "piece": eng.tok.decode([first]).strip()}


# ------------------------------------------------------------------ steering
@app.post("/api/steer")
async def steer(req: Request):
    """Side by side: the pushes as given, the unsteered model, and (optionally) every push reversed. With `history`,
    the conversation so far (user and assistant turns) comes before the prompt."""
    need_model()
    body = await req.json()
    pushes = pushes_of(body)
    history = [{"role": m["role"], "content": str(m["content"])} for m in body.get("history") or []
               if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    pid = eng.messages_ids(history + [{"role": "user", "content": body["prompt"]}]) if history else eng.chat_ids(body["prompt"])

    cols = body.get("columns", ["up", "base", "down"])
    inv = [q for q in map(_reverse, pushes) if q is not None]

    # "abl" erases the model's refusal direction with no neuron pushes; when ablation is on, the changed column
    # carries it too and the normal/opposite columns stay untouched
    active = bool(abl_now().get("active"))
    ablated = {"abl": True, "up": active, "base": False, "down": False}

    rows, flags = [], []
    for c in cols:
        rows.append((pid, {"up": pushes, "abl": [], "base": [], "down": inv}[c]))
        flags.append(ablated[c])

    abl_flags = flags if any(flags) else None

    def gen():
        for r, x in eng.generate(rows, max_new=int(body.get("max_new", 160)), cancel=cancel, abl=abl_flags):
            if r is None:
                break
            yield {"type": "delta", "col": cols[r], "delta": x}

    return sse(gen())


@app.post("/api/chat")
async def chat(req: Request):
    """One assistant turn of a conversation, with the neurons that are on and (optionally) the refusal direction
    erased. With `rows`, several versions of the conversation at once, each with its own pushes and erasure (Compare):
    they decode in one batch and every delta says which row it belongs to. With `continue`, each conversation ends in
    the reply so far, and the model carries on writing it. The last event says which replies ran out of length."""
    need_model()
    body = await req.json()
    multi, more = "rows" in body, bool(body.get("continue"))
    specs = body["rows"] if multi else [body]

    rows, flags = [], []
    for spec in specs:
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in spec.get("messages") or []
                if m.get("role") in ("system", "user", "assistant") and str(m.get("content", "")).strip()]

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

    def gen():
        for r, x in eng.generate(rows, max_new=max_new, temperature=float(body.get("temperature", 0.7)),
                                 cancel=cancel, seed=int(time.time()), abl=flags):
            if r is None:
                cut = [len(ids) >= max_new for ids in x]  # wrote to the limit without reaching its end
                yield {"type": "done", "cut": cut if multi else cut[0]}
                break
            yield {"type": "delta", "row": r, "delta": x} if multi else {"type": "delta", "delta": x}

    return sse(gen())


# ------------------------------------------------------------------ abliterate
def _probes():
    custom = json.loads(PROBES_FILE.read_text()) if PROBES_FILE.exists() else {}
    return {"harmful": custom.get("harmful") or abl_mod.DEFAULT_HARMFUL,
            "harmless": custom.get("harmless") or abl_mod.DEFAULT_HARMLESS}


def _abl_all():
    return json.loads(ABL_FILE.read_text()) if ABL_FILE.exists() else {}


def _abl_save(all_d):
    ABL_FILE.write_text(json.dumps(all_d, indent=1))


def abl_now():
    """The stored direction + state for the currently loaded model ({} when none)."""
    return (_abl_all().get(eng.model_id) or {}) if eng.model_id else {}


def restore_for(mid):
    """Re-apply a saved refusal direction after a model loads, so the state feels like a modified model."""
    d = _abl_all().get(mid)
    if d:
        eng.abl_of(d["vec"], d.get("coef", 1.0), d.get("layer"))


@app.get("/api/abl")
def abl_info():
    entry = abl_now()
    entry = entry if entry else {}
    return {"model": eng.model_id, "probes": _probes(),
            "defaults": {"harmful": abl_mod.DEFAULT_HARMFUL, "harmless": abl_mod.DEFAULT_HARMLESS},
            "current": None if not entry else {"layer": entry.get("layer"), "coef": entry.get("coef", 1.0),
                                               "active": entry.get("active", False), "clean": entry.get("clean", False)},
            "ready": bool(entry) and eng.abl_vec is not None}


@app.post("/api/abl/probes")
async def abl_probes(req: Request):
    body = await req.json()
    merger = json.loads(PROBES_FILE.read_text()) if PROBES_FILE.exists() else {}
    merger.update({k: [str(x).strip() for x in body.get(k) or [] if str(x).strip()]
                   for k in ("harmful", "harmless")})
    PROBES_FILE.write_text(json.dumps(merger, indent=1))
    return _probes()


@app.post("/api/abl/analyze")
async def abl_analyze(req: Request):
    need_model()
    body = await req.json() if req.headers.get("content-length", "0") not in ("", "0", None) else {}
    probes = {k: [str(x) for x in (body.get(k) or []) if str(x).strip()] for k in ("harmful", "harmless")}
    probes = {k: v for k, v in probes.items()} if any(probes.values()) else _probes()

    def gen():
        final = None
        for ev in abl_mod.analyze(eng, probes["harmful"], probes["harmless"], cancel):
            if ev.get("type") == "result":
                final = ev
                ev = {k: v for k, v in ev.items() if k != "vec"}  # the direction itself stays here
            yield ev

        if final:
            all_d = _abl_all()
            all_d[eng.model_id] = {"vec": final["vec"], "layer": final["layer"], "coef": final["coef"],
                                   "active": True, "clean": final["clean"], "ts": time.time()}
            _abl_save(all_d)
            eng.abl_of(final["vec"], final["coef"], final["layer"])

    return sse(gen())


@app.post("/api/abl/state")
async def abl_state(req: Request):
    """Turn the saved direction on/off or re-tune its strength. Requires an analyzed model."""
    need_model()
    body = await req.json()
    all_d = _abl_all()
    entry = all_d.get(eng.model_id)

    if not entry:
        raise HTTPException(409, "Run the analysis first.")

    if "coef" in body:
        entry["coef"] = max(0.1, min(2.0, float(body["coef"])))
    if "active" in body:
        entry["active"] = bool(body["active"])

    _abl_save(all_d)
    eng.abl_of(entry["vec"], entry.get("coef", 1.0), entry.get("layer"))
    return {"ok": True, **{k: entry.get(k) for k in ("layer", "coef", "active")}}


@app.post("/api/abl/test")
async def abl_test(req: Request):
    """Generate on a few probes, plain beside ablated, then measure refusal rates and the fluency cost."""
    need_model()
    if eng.abl_vec is None:
        raise HTTPException(409, "Analyse the model first: there is no direction to erase yet.")
    body = await req.json()
    probes = [str(p).strip() for p in (body.get("probes") or []) if str(p).strip()][:4] or abl_mod.EXAMPLE_PROMPTS

    n = len(probes)
    rows = []
    for p in probes:
        pid = eng.chat_ids(p)
        rows += [(pid, []), (pid, [])]

    fl = [i % 2 == 1 for i in range(2 * n)]

    def gen():
        texts = [""] * (2 * n)
        for r, dx in eng.generate(rows, max_new=160, temperature=0, cancel=cancel, abl=fl):
            if r is None:
                break
            texts[r] += dx
            yield {"type": "delta", "probe": r // 2, "kind": "base" if r % 2 == 0 else "abl", "delta": dx}

        refused = abl_mod.refused(texts)
        nll0 = float(eng.reference_nll([[]], cancel, abl=[False])[0])
        nll1 = float(eng.reference_nll([[]], cancel, abl=[True])[0])
        yield {"type": "result", "refused": [refused[i::2] for i in (0, 1)], "nll": [nll0, nll1]}

    return sse(gen())


# saved copies of edited weights (refusals erased, steering built in), made from the currently loaded model
export_job = {"kind": None, "name": None, "dst": None, "status": "idle", "done": 0, "total": 0, "msg": "",
              "cancel": False}


def _clean_name(raw):
    return re.sub(r"[^A-Za-z0-9._ -]+", "", str(raw or "")).strip(" .")


def _start_export(kind, name, **edits):
    """Write an edited copy of the loaded model into COPIES/name on a background thread; see abl_mod.bake."""
    if export_job["status"] == "working":
        raise HTTPException(409, f"Already writing {export_job['name']}.")

    export_job.update(kind=kind, name=name, dst=None, status="working", done=0, total=0, msg="Starting…", cancel=False)

    def work():
        tmp, dst = COPIES / (name + ".partial"), COPIES / name

        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            def prog(done, total, msg):
                if export_job["cancel"]:
                    raise Cancelled()
                export_job.update(done=done, total=total, msg=msg)

            abl_mod.bake(eng, tmp, prog, cancel, **edits)
            if export_job["cancel"]:
                raise Cancelled()

            tmp.rename(dst)
            export_job.update(dst=str(dst), status="done", msg="")
        except Cancelled:
            shutil.rmtree(tmp, ignore_errors=True)
            export_job["status"] = "idle"
        except Exception as e:
            traceback.print_exc()
            shutil.rmtree(tmp, ignore_errors=True)
            export_job["status"] = f"error: {e}"

    threading.Thread(target=work, daemon=True).start()
    return abl_export_status()


@app.get("/api/abl/export")
def abl_export_status():
    return {k: v for k, v in export_job.items() if k != "cancel"}


@app.post("/api/abl/export")
async def abl_export(req: Request):
    need_model()
    if eng.abl_vec is None:
        raise HTTPException(409, "Analyse the model first: there is no direction to bake in yet.")

    entry = abl_now()
    if not entry or not entry.get("vec"):
        raise HTTPException(409, "No saved direction for this model: analyse first.")

    body = await req.json()
    coef = max(0.1, min(2.0, float(body.get("coef", entry.get("coef", 1.0)))))

    base = eng.model_id.rstrip("/").split("/")[-1]
    name = _clean_name(body.get("name")) or f"{base}-abliterated" + ("" if coef == 1.0 else f"-{coef:g}")
    vec = [float(x) for x in entry["vec"]]

    return _start_export("abl", name, abl=(vec, coef))


@app.post("/api/steer/export")
async def steer_export(req: Request):
    """Save the model with a recipe built into its weights, and its refusals erased too if that's switched on."""
    need_model()
    body = await req.json()
    raw = body.get("items") or []

    if any(i.get("kind") == "dir" and i.get("op") == "erase" for i in raw):
        raise HTTPException(409, "An erased direction can't be built into the weights yet. Switch it to a push or "
                                 "turn it off first.")

    items = [{"layer": int(i["layer"]), "neuron": int(i["neuron"]), "dose": float(i["dose"])}
             for i in raw if i.get("kind") != "dir" and i.get("neuron") is not None and i.get("mode", "add") == "add"]
    items += [{"kind": "dir", "dir": str(i["dir"]), "layers": [int(l) for l in i["layers"]], "dose": float(i["dose"])}
              for i in raw if i.get("kind") == "dir"]

    pushes_of({"pushes": items})  # the directions must exist for this model
    if not items:
        raise HTTPException(409, "Turn on some neurons first: there is nothing to build in yet.")

    entry = abl_now()
    abl = (entry["vec"], float(entry.get("coef", 1.0))) if body.get("abl") and entry.get("active") and entry.get("vec") else None

    base = eng.model_id.rstrip("/").split("/")[-1]
    name = _clean_name(body.get("name")) or f"{base}-steered"
    return _start_export("steer", name, steer=items, abl=abl, fit_prompts=discover.DEFAULT_PROMPTS[:6],
                         recipe=str(body.get("recipe") or ""))


@app.post("/api/abl/export/cancel")
def abl_export_cancel():
    export_job["cancel"] = True
    return {"ok": True}


# ------------------------------------------------------------------ tuning
@app.get("/api/tune/defaults")
def tune_defaults():
    return {"prompts": discover.DEFAULT_PROMPTS}


@app.post("/api/tune/draft")
async def tune_draft(req: Request):
    need_model()
    body = await req.json()
    return sse(discover.draft_targets(eng, body["behavior"], body["prompts"], int(body.get("max_new", 72)), cancel))


@app.post("/api/tune/search")
async def tune_search(req: Request):
    need_model()
    body = await req.json()
    s = body.get("settings", {})
    budget = discover.budget_for(eng.n_params)  # sized to the model; "Neurons to test" in Options overrides
    if s.get("candidates"):
        budget["n_candidates"] = int(s["candidates"])

    return sse(discover.search(
        eng, body["pairs"], behavior=body.get("behavior") or None, **budget, max_neurons=int(s.get("max_neurons", 6)),
        doses=tuple(float(d) for d in s.get("doses", (0.25, 0.5, 1.0))), contrast=float(s.get("contrast", 0.5)),
        fluency_budget=float(s.get("fluency_budget", 0.35)), max_new=int(s.get("max_new", 72)),
        focus=[w.strip() for w in re.split(r"[,\n]", s.get("focus") or "") if w.strip()],
        focus_weight=float(s.get("focus_weight", 5.0)), cancel=cancel))


@app.post("/api/tune/refine")
async def tune_refine(req: Request):
    """Fix the recipe that's on from a complaint about one of its replies (see discover.refine)."""
    need_model()
    body = await req.json()
    recipe = [{"layer": int(i["layer"]), "neuron": int(i["neuron"]), "dose": float(i["dose"]), "lens": i.get("lens") or []}
              for i in body.get("recipe") or [] if i.get("kind") != "dir"]

    if not recipe:
        raise HTTPException(409, "Turn on some neurons first: there is nothing to fix yet.")
    if not (body.get("feedback") or "").strip():
        raise HTTPException(400, "Say what's wrong with the reply.")

    big = eng.n_params > 4e9
    budget = discover.budget_for(eng.n_params)
    others = [p for p in body.get("prompts") or discover.DEFAULT_PROMPTS][: 2 if big else 4]

    return sse(discover.refine(
        eng, recipe, feedback=body["feedback"], prompt=body["prompt"], bad=body.get("bad") or "",
        behavior=body.get("behavior") or None, prompts=others, abl=bool(abl_now().get("active")),
        target_tokens=budget["target_tokens"] or 48, n_new=8 if big else 12, rounds=2 if big else 3, cancel=cancel))


# ------------------------------------------------------------------ directions
def _dir_row(e):
    d = eng.dirs.get(e["id"])
    m = e.get("metrics") or {}
    auc = m.get("auc") or []
    band = e.get("band") or []

    return {k: e.get(k) for k in ("id", "name", "kind", "band", "created", "pos", "neg")} | {
        "ready": d is not None, "auc": round(min(auc[l] for l in band), 2) if auc and band else None}


@app.get("/api/directions")
def directions_list():
    need_model()
    return {"directions": [_dir_row(e) for e in dirs.index()], "layers": len(eng.layers)}


@app.get("/api/directions/defaults")
def directions_defaults():
    return {"example": dir_mod.EXAMPLE}


@app.post("/api/directions/extract")
async def directions_extract(req: Request):
    need_model()
    body = await req.json()
    return sse(dir_mod.extract(eng, dirs, body, cancel))


@app.post("/api/directions/import")
async def directions_import(req: Request):
    need_model()
    body = await req.json()
    path = Path(str(body.get("path") or "")).expanduser()

    if not path.is_file():
        raise HTTPException(400, f"No file at {path}.")

    try:
        added = dirs.import_npz(path, cancel)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))

    return {"added": [_dir_row(e) for e in added]}


@app.delete("/api/directions/{did}")
def directions_delete(did: str):
    need_model()
    dirs.delete(did)
    return {"ok": True}


# ------------------------------------------------------------------ recipes
def _recipes():
    return json.loads(RECIPES.read_text()) if RECIPES.exists() else []


@app.get("/api/recipes")
def recipes():
    return _recipes()


@app.post("/api/recipes")
async def save_recipe(req: Request):
    r = await req.json()
    r.setdefault("id", f"r{int(time.time() * 1000)}")
    r.setdefault("created", time.time())
    r.setdefault("model", eng.model_id)

    allr = [x for x in _recipes() if x["id"] != r["id"]] + [r]
    RECIPES.write_text(json.dumps(allr, indent=1))
    return r


@app.delete("/api/recipes/{rid}")
def delete_recipe(rid: str):
    RECIPES.write_text(json.dumps([x for x in _recipes() if x["id"] != rid], indent=1))
    return {"ok": True}


# ------------------------------------------------------------------ chats
def _chat_file(cid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid):
        raise HTTPException(400, "Bad chat id.")
    return CHATS / f"{cid}.json"


def _summary(c):
    return {k: c.get(k) for k in ("id", "title", "model", "updated")} | {"turns": sum(m["role"] == "user" for m in c["messages"])}


@app.get("/api/chats")
def chats():
    """Saved conversations, newest first."""
    out = []
    for f in CHATS.glob("*.json"):
        try:
            out.append(_summary(json.loads(f.read_text())))
        except (ValueError, KeyError):
            continue

    return sorted(out, key=lambda c: -(c["updated"] or 0))


@app.get("/api/chats/{cid}")
def chat_get(cid: str):
    f = _chat_file(cid)
    if not f.exists():
        raise HTTPException(404, "That chat is gone.")
    return json.loads(f.read_text())


@app.post("/api/chats")
async def chat_save(req: Request):
    """Save a conversation with the settings, switches and recipe it ran under; the same id overwrites it."""
    c = await req.json()
    f = _chat_file(c.get("id", ""))

    c["updated"] = time.time()
    c.setdefault("model", eng.model_id)

    CHATS.mkdir(exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, ensure_ascii=False))
    tmp.replace(f)  # a crash mid-write leaves the old copy, not half a file
    return _summary(c)


@app.delete("/api/chats/{cid}")
def chat_delete(cid: str):
    _chat_file(cid).unlink(missing_ok=True)
    return {"ok": True}


# ------------------------------------------------------------------ ui
@app.get("/")
def index():
    if not (UI / "index.html").exists():
        return HTMLResponse("<p>The UI isn’t built yet: run <code>pnpm install && pnpm build</code> in web/, then reopen.</p>")
    return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=UI, check_dir=False), name="ui")


@app.exception_handler(HTTPException)
async def http_err(req, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
