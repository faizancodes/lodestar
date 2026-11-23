"""Experiments: a probe written down as one file, so anyone with Lodestar can run it and check they get the same
result. A file (`*.lodestar.json`) holds:

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

`expect` names metrics from interp/probe.py with the range a run should land in. Run one in the app (Probe, then
Open), or from a terminal against a running Lodestar:

    uv run python -m interp.experiment path/to/file.lodestar.json [--port 8792]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import probe as probe_mod

FORMAT = "experiment/1"
EXAMPLES = Path(__file__).resolve().parent.parent / "experiments"
METRICS = ("separation", "new_groups", "words_only", "words_only_new_groups", "shuffled", "caught", "layer")


def model_name(model_id):
    return str(model_id or "").rstrip("/").split("/")[-1].lower()


def validate(exp):
    if not isinstance(exp, dict) or exp.get("lodestar") != FORMAT:
        raise ValueError(f'Not a Lodestar experiment: the file needs "lodestar": "{FORMAT}".')
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


def examples():
    """The experiments that ship with Lodestar (experiments/ in the repo)."""
    out = []
    for f in sorted(EXAMPLES.glob("*.lodestar.json")):
        e = json.loads(f.read_text())
        out.append({"file": f.name, "name": e.get("name"), "about": e.get("about", ""), "model": e.get("model")})
    return out


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
    ap.add_argument("file")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    exp = validate(json.loads(Path(args.file).read_text()))

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


if __name__ == "__main__":
    main()
