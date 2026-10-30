"""Step 2: steer with each direction across layers and doses on held-out questions (no system prompt); judge the
stance and coherence of every reply with the unsteered model; save sweep.json.

usage: python sweep.py [concept ...]
"""
import sys
import time

import numpy as np

from concepts import N_TRAIN, PERSONAS, QUESTIONS
from lib import COHERENCE, OUT, Steer, generate_chunked, judge, load, ref_nll, save_json

LAYERS = [10, 16, 22]
KS = [-4, -2, -1, 1, 2, 4]
QS = QUESTIONS[N_TRAIN:][:6]
MAX_NEW = 90

names = sys.argv[1:] or list(PERSONAS)
D = np.load(OUT / "directions.npz")
eng = load()
steer = Steer(eng)
t0 = time.time()

base = generate_chunked(eng, steer, QS, None, max_new=MAX_NEW)
results = {}
for name in names:
    u, mag = D[name], D[name + "_mag"]
    rubric = PERSONAS[name][2]
    rows = [(l, k, q) for l in LAYERS for k in KS for q in range(len(QS))]
    prompts = [QS[q] for _, _, q in rows]
    texts = []
    for l in LAYERS:  # one layer per generation pass keeps the hook simple
        sub = [(ll, k, q) for ll, k, q in rows if ll == l]
        vecs = [k * mag[l] * u[l] for _, k, _ in sub]
        texts += generate_chunked(eng, steer, [QS[q] for *_, q in sub], {l: vecs}, max_new=MAX_NEW)
    stance = judge(eng, list(zip(prompts, texts)), rubric)
    coh = judge(eng, list(zip(prompts, texts)), COHERENCE)
    s0 = judge(eng, list(zip(QS, base)), rubric)
    c0 = judge(eng, list(zip(QS, base)), COHERENCE)
    nll = {}
    for l in LAYERS:
        nll[l] = ref_nll(eng, steer, {l: [k * mag[l] * u[l] for k in [0] + KS]}).tolist()
    table = {}
    for l in LAYERS:
        for k in KS:
            ix = [i for i, (ll, kk, _) in enumerate(rows) if ll == l and kk == k]
            table[f"{l}:{k}"] = {"stance": float(stance[ix].mean()), "coh": float(coh[ix].mean())}
    results[name] = {"base_stance": float(s0.mean()), "base_coh": float(c0.mean()), "table": table, "nll": nll,
                     "samples": [{"layer": l, "k": k, "q": QS[q], "text": t, "stance": float(s), "coh": float(c)}
                                 for (l, k, q), t, s, c in zip(rows, texts, stance, coh)]}
    print(f"\n== {name}  base stance {s0.mean():.2f} coh {c0.mean():.2f}  [{time.time() - t0:.0f}s]")
    print("layer " + " ".join(f"{k:>11}" for k in KS))
    for l in LAYERS:
        print(f"{l:5} " + " ".join(f"{table[f'{l}:{k}']['stance']:5.2f}/{table[f'{l}:{k}']['coh']:4.1f}" for k in KS))
    save_json(f"sweep_{'_'.join(names)}.json", {"base": base, "qs": QS, "results": results})
print(f"done {time.time() - t0:.0f}s")
