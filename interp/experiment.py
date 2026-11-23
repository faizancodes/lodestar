"""Experiments: a probe or a steering setup written down as one file, so anyone with Lodestar can run it and check
they get the same result. A probe file (`*.lodestar.json`) holds:

    {
      "lodestar": "experiment/1",
      "name": "Inkling knows when it doesn't know",
      "about": "One or two sentences on what it shows.",
      "model": "Inkling-2-Mini",              # the model folder or repo name it was run on
      "probe": {"a": {"label": "Made up", "prompts": ["...", {"text": "...", "group": "bridge"}]},
                "b": {"label": "Real", "prompts": [...]}},
      "try": ["Who designed the Kessington Bridge in Cleveland?"],   # questions to ask it once it's built
      "expect": {"separation": [0.95, 1.0], "new_groups": [0.93, 1.0], "words_only": [0.0, 0.75]}
    }

`expect` names metrics from interp/probe.py with the range a run should land in.

A steering file holds `steer` in place of `probe`: the exact pushes, so another copy of the model gets the same
change, not just the same slider positions.

    "steer": [{"layer": 34, "neuron": 70770, "add": 302.1, "weight": 1.84},   # add 302.1 to its activation
              {"layer": 8, "neuron": 3602, "times": 0},                         # multiply it (0 = ablate)
              {"direction": "hawk", "layers": [14, 15, 16], "op": "push", "dose": 4}],
    "directions": {"hawk": {"name": "Hawk", "band": [14, 15, 16], "mag": [...], "unit": "<base64 float32 [L, H]>"}}

`add` is in the neuron's own activation units, not the app's dose, so it doesn't depend on how the app scales its
sliders. `weight` is the length of the neuron's output weights: if a copy's differs, it isn't the same neuron and
the file says so. A direction travels with the file whole.

Run one in the app (Open, beside Saved, takes a file or a link; so does ⌘V with a link copied), or
from a terminal against a running Lodestar:

    uv run python -m interp.experiment path/to/file.lodestar.json [--port 8792]
    uv run python -m interp.experiment https://github.com/you/repo/blob/main/x.lodestar.json
"""
import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import directions as dir_mod
from . import probe as probe_mod

FORMAT = "experiment/1"
EXAMPLES = Path(__file__).resolve().parent.parent / "experiments"
METRICS = ("separation", "new_groups", "words_only", "words_only_new_groups", "shuffled", "caught", "layer")


def model_name(model_id):
    return str(model_id or "").rstrip("/").split("/")[-1].lower()


def validate(exp):
    if not isinstance(exp, dict) or exp.get("lodestar") != FORMAT:
        raise ValueError(f'Not a Lodestar experiment: the file needs "lodestar": "{FORMAT}".')
    if "steer" in exp:
        return _validate_steer(exp)
    probe_mod.parse_spec({"name": exp.get("name"), **(exp.get("probe") or {})})
    for k, v in (exp.get("expect") or {}).items():
        if k not in METRICS or not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"expect.{k}: give one of {', '.join(METRICS)} with a [low, high] range.")
    return exp


def check(expect, metrics):
    """Each expectation against what this run got: [{metric, want, got, ok}]; a metric this run has no value for
    (no groups, say) fails."""
    out = []
    for k, (lo, hi) in (expect or {}).items():
        got = metrics.get(k)
        out.append({"metric": k, "want": [lo, hi], "got": got, "ok": got is not None and lo <= got <= hi})
    return out


def run(eng, lib, exp, cancel=None):
    """Build the experiment's probe, then check it against `expect`. Same events as probe.run, with `checks` and
    `same_model` added to the result."""
    validate(exp)
    same = model_name(exp.get("model")) == model_name(eng.model_id) if exp.get("model") else None
    spec = {"name": exp.get("name"), **exp["probe"]}

    for ev in probe_mod.run(eng, lib, spec, cancel):
        if ev["type"] == "result":
            ev = {**ev, "checks": check(exp.get("expect"), ev["probe"]["metrics"]), "same_model": same}
        yield ev


def from_probe(entry, about="", tries=()):
    """A saved probe written as an experiment, expecting the separation numbers it got to within 0.03."""
    m = entry["metrics"]
    expect = {k: [round(max(0.0, m[k] - 0.03), 2), round(min(1.0, m[k] + 0.03), 2)]
              for k in ("separation", "new_groups", "words_only") if k in m}
    side = lambda s: {"label": entry["labels"][1 - s], "prompts": [
        {"text": it["text"], "group": it["group"]} if it["group"] else it["text"] for it in entry["items"] if it["side"] == s]}

    return {"lodestar": FORMAT, "name": entry["name"], "about": about, "model": entry["model"].rstrip("/").split("/")[-1],
            "probe": {"a": side(1), "b": side(0)}, "try": list(tries), "expect": expect}


def examples(kind="probe"):
    """The experiments that ship with Lodestar (experiments/ in the repo), of one kind: "probe" or "steer"."""
    out = []
    for f in sorted(EXAMPLES.glob("*.lodestar.json")):
        e = json.loads(f.read_text())
        if ("steer" in e) == (kind == "steer"):
            out.append({"file": f.name, "name": e.get("name"), "about": e.get("about", ""), "model": e.get("model"),
                        "try": e.get("try") or []})
    return out


# ------------------------------------------------------------------ steering
def _validate_steer(exp):
    dirs = exp.get("directions") or {}
    if not isinstance(exp["steer"], list) or not exp["steer"]:
        raise ValueError("steer: give at least one push.")
    for i, p in enumerate(exp["steer"]):
        if "direction" in p:
            if p["direction"] not in dirs:
                raise ValueError(f"steer[{i}]: the file has no direction called {p['direction']!r} under directions.")
        elif not ("layer" in p and "neuron" in p and ("add" in p or "times" in p)):
            raise ValueError(f"steer[{i}]: a neuron push needs layer, neuron and add (or times).")
    return exp


def from_steer(eng, items, name, about="", tries=()):
    """The setup on now (the app's recipe items) written as an experiment, with each neuron's push as the exact
    activation it adds and every direction it uses carried whole."""
    pushes, dirs = [], {}
    for it in items:
        if it.get("kind") == "dir":
            d = eng.dirs.get(it["dir"])
            if d is None:
                raise ValueError(f"No direction called {it['dir']!r} for this model.")
            pushes.append({"direction": it["dir"], "layers": list(it["layers"]), "op": it.get("op", "push"),
                           "dose": float(it["dose"])})
            dirs[it["dir"]] = {"name": it.get("name") or it["dir"], "band": d["band"], "mag": d["mag"],
                               "unit": base64.b64encode(d["unit"].float().cpu().numpy().tobytes()).decode()}
            continue

        l, n, dose = int(it["layer"]), int(it["neuron"]), float(it["dose"])
        p = {"layer": l, "neuron": n}
        if it.get("mode") == "mul":
            p["times"] = dose
        else:
            p["add"] = round(dose * float(eng.unit[l, n]), 4)
        p["weight"] = round(float(eng.dnorm[l, n]), 5)
        if it.get("label"):
            p["label"] = it["label"]
        pushes.append(p)

    return {"lodestar": FORMAT, "name": name, "about": about, "model": str(eng.model_id).rstrip("/").split("/")[-1],
            "steer": pushes, "directions": dirs, "try": list(tries)}


def open_steer(eng, lib, exp):
    """A steering experiment as the app's recipe items for the loaded model. Directions the model doesn't have yet are
    added to its library. Returns (items, mismatched): mismatched names each neuron whose weights differ here."""
    validate(exp)
    L, N = eng.dnorm.shape
    ids = {}

    for did, d in (exp.get("directions") or {}).items():
        unit = np.frombuffer(base64.b64decode(d["unit"]), np.float32)
        if unit.size % L:
            raise ValueError(f"The direction {d.get('name', did)!r} was made on a model with a different shape.")
        unit = unit.reshape(L, -1)
        if unit.shape[1] != eng.hidden:
            raise ValueError(f"The direction {d.get('name', did)!r} was made on a model with a different shape.")

        have = eng.dirs.get(did)
        if have is not None and np.allclose(have["unit"].float().cpu().numpy(), unit, atol=1e-6):
            ids[did] = did
            continue
        new = dir_mod.make_id(did, {e["id"] for e in lib.index()})
        lib.save({"id": new, "name": d.get("name") or did, "kind": "experiment", "band": d.get("band") or []},
                 unit, np.asarray(d["mag"], np.float32))
        ids[did] = new

    items, mismatched = [], []
    for p in exp["steer"]:
        if "direction" in p:
            d = exp["directions"][p["direction"]]
            items.append({"kind": "dir", "dir": ids[p["direction"]], "name": d.get("name"),
                          "layers": [int(l) for l in p["layers"]], "op": p.get("op", "push"), "dose": float(p["dose"])})
            continue

        l, n = int(p["layer"]), int(p["neuron"])
        if not (0 <= l < L and 0 <= n < N):
            raise ValueError(f"L{l} N{n} isn't in this model, which has {L} layers of {N} neurons.")
        w = float(eng.dnorm[l, n])
        if p.get("weight") and abs(w / p["weight"] - 1) > 0.01:
            mismatched.append(f"L{l} N{n}")

        it = {"layer": l, "neuron": n}
        if "times" in p:
            it |= {"dose": float(p["times"]), "mode": "mul"}
        else:
            it["dose"] = float(p["add"]) / float(eng.unit[l, n])
        if p.get("label"):
            it["label"] = p["label"]
        items.append(it)

    return items, mismatched


# ------------------------------------------------------------------ opening one from a link
MAX_BYTES = 64 << 20  # a file carrying several directions of a large model is a few MB


def is_link(text):
    """Whether text is a link to an experiment: a .lodestar.json file anywhere, or a gist."""
    text = text.strip()
    return bool(re.fullmatch(r"https?://\S+", text)) and bool(
        re.search(r"\.lodestar\.json([?#]|$)", text) or text.startswith("https://gist.github.com/"))


def raw_url(url):
    """The address of the file itself for a link someone shared: a GitHub file page or a gist becomes its raw text."""
    url = url.strip()
    if not re.match(r"https?://", url):
        raise ValueError("That isn't a link: it should start with https://.")
    if m := re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/(?:blob|raw)/(.+?)/?(?:[?#].*)?", url):
        return f"https://raw.githubusercontent.com/{m[1]}/{m[2]}"
    if m := re.fullmatch(r"https://gist\.github\.com/([^/]+)/([0-9a-f]+)/?(?:[?#].*)?", url):
        return f"https://gist.github.com/{m[1]}/{m[2]}/raw"
    return url


def fetch(url):
    """Download and check an experiment from a link."""
    req = urllib.request.Request(raw_url(url), headers={"user-agent": "Lodestar"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise ValueError(f"The link answered {e.code}, so there's no file to open there.") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ValueError("Couldn't reach that link. Check you're online and the address is right.") from e
    if len(body) > MAX_BYTES:
        raise ValueError("That file is too large to be an experiment.")
    try:
        exp = json.loads(body)
    except ValueError as e:
        raise ValueError("That link isn't a Lodestar experiment: it doesn't lead to a .lodestar.json file.") from e
    return validate(exp)


# ------------------------------------------------------------------ running one from a terminal
def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return urllib.request.urlopen(req)


def _events(resp):
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: "):
            yield json.loads(line[6:])


def main():
    ap = argparse.ArgumentParser(description="Run a Lodestar experiment against the running app.")
    ap.add_argument("file", help="a .lodestar.json file, or a link to one")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    try:
        exp = fetch(args.file) if re.match(r"https?://", args.file) else validate(json.loads(Path(args.file).read_text()))
    except ValueError as e:
        sys.exit(f"error: {e}")
    if "steer" in exp:
        return main_steer(args.port, exp)

    result = None
    for ev in _events(_post(args.port, "/api/experiments/run", exp)):
        if ev["type"] == "stage":
            print(ev["label"], flush=True)
        elif ev["type"] == "progress" and (ev["done"] % 25 == 0 or ev["done"] == ev["total"]):
            print(f"  {ev['done']}/{ev['total']}", flush=True)
        elif ev["type"] == "error":
            sys.exit(f"error: {ev['message']}")
        elif ev["type"] == "result":
            result = ev

    if result is None:
        sys.exit("The run stopped before it finished.")

    p = result["probe"]
    print(f"\n{p['name']}: layer {p['metrics']['layer']}")
    for k in METRICS[:-1]:
        if k in p["metrics"]:
            print(f"  {k:<22} {p['metrics'][k]}")
    if result["same_model"] is False:
        print(f"\nNote: this file was made on {exp['model']}; the loaded model is different, so the ranges may not apply.")

    for q in exp.get("try") or []:
        s = json.load(_post(args.port, f"/api/probes/{p['id']}/score", {"prompt": q}))
        answer = "".join(e.get("delta", "") for e in _events(_post(args.port, "/api/steer", {
            "prompt": q, "columns": ["base"], "pushes": [], "max_new": 90})))
        print(f"\n{q}\n  {p['labels'][0]}: {s['p']:.0%}\n  {answer.strip()}")

    failed = [c for c in result["checks"] if not c["ok"]]
    for c in result["checks"]:
        print(f"{'✓' if c['ok'] else '✗'} {c['metric']} {c['got']} (expected {c['want'][0]}–{c['want'][1]})")
    sys.exit(1 if failed else 0)


def main_steer(port, exp):
    """Turn the file's pushes on in the running app and ask its `try` questions, steered and unsteered."""
    try:
        r = json.load(_post(port, "/api/experiments/steer", exp))
    except urllib.error.HTTPError as e:
        sys.exit(f"error: {json.load(e).get('detail', e)}")

    print(f"{exp['name']}: {len(r['items'])} pushes")
    for it in r["items"]:
        what = f"{it['name']} at layers {it['layers']}" if it.get("kind") == "dir" else f"L{it['layer']} N{it['neuron']}"
        print(f"  {what}  dose {it['dose']:+.3f}")
    for m in r["mismatched"]:
        print(f"✗ {m} has different weights here, so it isn't the same neuron.")

    for q in exp.get("try") or []:
        out = {"up": "", "base": ""}
        for e in _events(_post(port, "/api/steer", {"prompt": q, "columns": ["up", "base"], "pushes": r["items"],
                                                     "max_new": 120})):
            if e.get("type") == "delta":
                out[e["col"]] += e["delta"]
        print(f"\n{q}\n  steered:   {out['up'].strip()}\n  unchanged: {out['base'].strip()}")

    sys.exit(1 if r["mismatched"] else 0)


if __name__ == "__main__":
    main()
