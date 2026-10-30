"""Exploration helpers, exec'd into repl.py's namespace."""
import json
import time


def vec(name, l, k):
    return k * D[name + "_mag"][l] * D[name][l]


def cfg_vecs(cfgs, n_rows_each):
    """cfgs: list of configs, each {layer: [(concept, k), ...]} (or None for unsteered). Every config gets
    n_rows_each rows. Returns {layer: [vec or None per row]}."""
    layers = sorted({l for c in cfgs if c for l in c})
    out = {l: [] for l in layers}
    for c in cfgs:
        for l in layers:
            v = None
            for name, k in (c or {}).get(l, []):
                v = vec(name, l, k) if v is None else v + vec(name, l, k)
            out[l] += [v] * n_rows_each
    return out


def band(name, k, lo=14, hi=22, step=2):
    """Spread a dose over a band of layers: k / n_layers per layer."""
    ls = list(range(lo, hi + 1, step))
    return {l: [(name, k / len(ls))] for l in ls}


def merge(*cs):
    out = {}
    for c in cs:
        for l, v in c.items():
            out.setdefault(l, []).extend(v)
    return out


LOG = OUT / "explore.jsonl"


def run(prompts, cfgs, labels, max_new=200, system=None, show=600, chunk=12):
    """Generate every prompt under every config (batched together); print and log."""
    t = time.time()
    P = [p for _ in cfgs for p in prompts]
    V = cfg_vecs(cfgs, len(prompts))
    texts = generate_chunked(eng, steer, P, V or None, max_new=max_new, chunk=chunk, system=system)
    with open(LOG, "a") as f:
        for i, txt in enumerate(texts):
            lab = labels[i // len(prompts)]
            f.write(json.dumps({"label": lab, "prompt": P[i], "system": system, "text": txt}, ensure_ascii=False) + "\n")
    for i, p in enumerate(prompts):
        print(f"\n######## {p}")
        for j, lab in enumerate(labels):
            print(f"--- [{lab}] " + texts[j * len(prompts) + i][:show].replace("\n", " "))
    print(f"\n({time.time() - t:.0f}s)")
    return texts
