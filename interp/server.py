"""Local HTTP API for the Auto Interp window. Streams (server-sent events) for anything that generates text or runs a
search; plain JSON for lookups. Serves the UI from ../ui."""
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import abl as abl_mod
from . import discover
from .engine import Cancelled, Engine, Push

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
DATA = Path(os.environ.get("AUTO_INTERP_DATA", Path.home() / "Library" / "Application Support" / "Auto Interp"))
DATA.mkdir(parents=True, exist_ok=True)
RECIPES = DATA / "recipes.json"
ABL_FILE = DATA / "abl.json"  # saved refusal directions + state, keyed by model id
PROBES_FILE = DATA / "abl-probes.json"  # customised probe lists
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
    return [Push.of(p) for p in body.get("pushes") or [] if p.get("enabled", True)]


# ------------------------------------------------------------------ models
@app.get("/api/status")
def status():
    return {**eng.info(), "progress": eng.progress}


# Copies of drive models on this Mac's internal disk. A USB 3 stick reads at ~0.4 GB/s against ~3 GB/s for
# the internal SSD, and reading the weights is nearly all of a big model's load time.
COPIES = DATA / "models"
copy_job = {"src": None, "name": None, "done": 0, "total": 0, "status": "idle", "cancel": False}


def _model_files(src):
    """Files that make up a model folder, skipping macOS ._ sidecar files and the download cache."""
    return [f for f in sorted(src.iterdir()) if f.is_file() and not f.name.startswith(".")]


def _folder_size(d):
    return sum(f.stat().st_size for f in _model_files(d))


@app.get("/api/models")
def models():
    cached = []
    if HF_CACHE.exists():
        for d in sorted(HF_CACHE.glob("models--*")):
            if any(d.glob("snapshots/*/*.safetensors")):
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
        raise HTTPException(400, "Only copies made by Auto Interp can be removed here.")
    if eng.model_id == str(d):
        raise HTTPException(409, "That copy is loaded right now. Load another model first.")
    shutil.rmtree(d)
    return {"ok": True}


@app.post("/api/load")
async def load(req: Request):
    body = await req.json()
    mid = body["model"].strip()

    def work():
        try:
            eng.load(mid)
            restore_for(mid)
        except Exception as e:
            traceback.print_exc()
            eng.status = f"error: {e}"
            eng.progress = None

    threading.Thread(target=work, daemon=True).start()
    time.sleep(0.05)
    return status()


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
    pid = eng.chat_ids(body["prompt"], system=body.get("system") or None)

    def gen():
        given = body.get("response")
        if given:
            rid = eng.encode(given)
        else:
            rid = None
            for r, x in eng.generate([(pid, pushes)], max_new=int(body.get("max_new", 120)),
                                     temperature=float(body.get("temperature", 0)), cancel=cancel):
                if r is None:
                    rid = x[0]
                else:
                    yield {"type": "delta", "delta": x}
        yield {"type": "stage", "label": "Tracing every neuron"}
        tr = eng.run_trace(pid, rid, pushes, abl=bool(body.get("abl")))
        lens = tr["lens_ids"][:, :, 0]  # [L, T] top-1 token per layer per position
        vocab = {}
        lens_txt = [[vocab.setdefault(int(t), eng.tok.decode([int(t)])) for t in row] for row in lens]
        top = eng.trace_top_neurons(k=48)
        yield {
            "type": "trace", "tokens": tr["tokens"], "n_prompt": tr["n_prompt"],
            "mlp": np.round(tr["mlp_norm"], 2), "attn": None if tr["attn_norm"] is None else np.round(tr["attn_norm"], 2),
            "resid": np.round(tr["resid_norm"], 1), "lens": lens_txt, "lens_p": np.round(tr["lens_p"][:, :, 0], 3),
            "tok_lp": np.round(tr["tok_lp"], 3), "top": [_neuron_row(*x) for x in top],
        }

    return sse(gen())


def _neuron_row(l, n, impact, act, t, lens_k=5):
    lens = eng.lens(l, n, lens_k)
    return {"layer": l, "neuron": n, "impact": round(impact, 4), "act": round(act, 3), "t": t,
            "lens": lens["up" if act >= 0 else "down"]}


@app.get("/api/trace/token")
def trace_token(t: int, k: int = 24):
    need_model()
    tr = eng.trace
    if tr is None or not 0 <= t < len(tr["ids"]):
        raise HTTPException(404, "no trace")
    top = eng.trace_top_neurons(t=t, k=k)
    ids, ps = tr["lens_ids"][:, t, :], tr["lens_p"][:, t, :]
    return {
        "t": t, "token": tr["tokens"][t],
        "next": tr["tokens"][t + 1] if t + 1 < len(tr["tokens"]) else None,
        "next_lp": float(tr["tok_lp"][t]) if t < len(tr["tok_lp"]) else None,
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
    out = {"layer": layer, "neuron": n, "lens": eng.lens(layer, n, 16),
           "down_norm": float(eng.dnorm[layer, n]), "resid_p50": float(eng.resid_p50[layer]),
           "unit": float(eng.unit[layer, n])}
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


@app.post("/api/read")
async def read(req: Request):
    """A neuron's activation over arbitrary text (as a user turn when chat is true)."""
    need_model()
    body = await req.json()
    ids = eng.chat_ids(body["text"]) if body.get("chat") else eng.tok(body["text"]).input_ids
    a = eng.read(int(body["layer"]), int(body["neuron"]), ids, pushes_of(body))
    return {"tokens": eng.token_strs(ids), "acts": np.round(a, 3).tolist()}


# ------------------------------------------------------------------ steering
@app.post("/api/steer")
async def steer(req: Request):
    """Side by side: the pushes as given, the unsteered model, and (optionally) every push reversed."""
    need_model()
    body = await req.json()
    pushes = pushes_of(body)
    pid = eng.chat_ids(body["prompt"], system=body.get("system") or None, history=body.get("history"))
    cols = body.get("columns", ["up", "base", "down"])
    # reversed: add pushes flip sign; a multiplier mirrors around 1 (ablate x0 <-> double x2)
    inv = [Push(p.layer, p.neuron, -p.dose if p.mode == "add" else 2 - p.dose, p.mode) for p in pushes]
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
        for r, x in eng.generate(rows, max_new=int(body.get("max_new", 160)),
                                 temperature=float(body.get("temperature", 0)), cancel=cancel, abl=abl_flags):
            if r is None:
                break
            yield {"type": "delta", "col": cols[r], "delta": x}

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
    return _abl_all().get(eng.model_id) if eng.model_id else {}


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
    """Turn the saved direction on/off, re-tune its strength or select a layer. Requires an analyzed model."""
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
    if "layer" in body and body["layer"] != entry.get("layer"):
        raise HTTPException(400, "The saved direction is tied to its layer; analyse again to use another layer.")
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
        for r, dx in eng.generate(rows, max_new=int(body.get("max_new", 160)), temperature=0, cancel=cancel, abl=fl):
            if r is None:
                break
            texts[r] += dx
            yield {"type": "delta", "probe": r // 2, "kind": "base" if r % 2 == 0 else "abl", "delta": dx}
        refused = [abl_mod.refusalish(t) for t in texts]
        nll0 = float(eng.reference_nll([[]], cancel, abl=[False])[0])
        nll1 = float(eng.reference_nll([[]], cancel, abl=[True])[0])
        yield {"type": "result", "probes": probes, "texts": [t.strip() for t in texts],
               "refused": [refused[i::2] for i in (0, 1)],
               "refused_rate": [sum(refused[i::2]) / n for i in (0, 1)], "nll": [nll0, nll1]}

    return sse(gen())


# saved copies of abliterated weights, made from the currently loaded model
export_job = {"name": None, "dst": None, "status": "idle", "done": 0, "total": 0, "msg": "", "cancel": False}


@app.get("/api/abl/export")
def abl_export_status():
    return {k: v for k, v in export_job.items() if k != "cancel"}


@app.post("/api/abl/export")
async def abl_export(req: Request):
    need_model()
    if eng.abl_vec is None:
        raise HTTPException(409, "Analyse the model first: there is no direction to bake in yet.")
    if export_job["status"] == "working":
        raise HTTPException(409, f"Already writing {export_job['name']}.")
    entry = abl_now()
    if not entry or not entry.get("vec"):
        raise HTTPException(409, "No saved direction for this model: analyse first.")
    body = await req.json()
    coef = max(0.1, min(2.0, float(body.get("coef", entry.get("coef", 1.0)))))
    base = eng.model_id.rstrip("/").split("/")[-1]
    name = f"{base}-abliterated" + ("" if coef == 1.0 else f"-{coef:g}")
    vec = [float(x) for x in (body.get("vec") or entry["vec"])]
    export_job.update(name=name, dst=None, status="working", done=0, total=0, msg="Starting…", cancel=False)

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

            abl_mod.bake(eng, vec, coef, tmp, prog, cancel)
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


# ------------------------------------------------------------------ ui
@app.get("/")
def index():
    return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=UI), name="ui")


@app.exception_handler(HTTPException)
async def http_err(req, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
