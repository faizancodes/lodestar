"""Step 1: extract and validate every concept direction; save directions.npz and extract.json."""
import json
import time
from pathlib import Path

import numpy as np

from concepts import CENSORED, N_TOPIC_TRAIN, N_TRAIN, PERSONAS, QUESTIONS, UNCENSORED
from lib import OUT, analyze, end_states, load, save_json

eng = load()
t0 = time.time()
dirs, report = {}, {}


def persona_states(systems, qs):
    """[n_q, L, P, H]: each question's ending states averaged over the pole's system-prompt paraphrases, so each
    question contributes one paired example per pole."""
    acc = None
    for s in systems:
        st = end_states(eng, [[{"role": "system", "content": s}, {"role": "user", "content": q}] for q in qs])
        acc = st if acc is None else acc + st
    return acc / len(systems)


for name, (pos_sys, neg_sys, _) in PERSONAS.items():
    tr, ho = QUESTIONS[:N_TRAIN], QUESTIONS[N_TRAIN:]
    # held-out generalization is checked on held-out questions AND a held-out paraphrase: train on paraphrases 0-1,
    # test on paraphrase 2
    P_tr, N_tr = persona_states(pos_sys[:2], tr), persona_states(neg_sys[:2], tr)
    P_ho, N_ho = persona_states(pos_sys[2:], ho), persona_states(neg_sys[2:], ho)
    r = analyze(P_tr, N_tr, P_ho, N_ho, pairs=True)
    dirs[name] = r["u"].astype(np.float32)
    dirs[name + "_mag"] = r["mag"].astype(np.float32)
    report[name] = {k: r[k] for k in ("mag", "cos_split", "auc", "dprime")}
    print(f"{name}: best split-cos layer {int(np.argmax(r['cos_split']))} "
          f"({r['cos_split'].max():.3f}); held-out AUC>0.99 at {int((r['auc'] > 0.99).sum())}/36 layers; "
          f"max d' {r['dprime'].max():.1f} @ {int(np.argmax(r['dprime']))}  [{time.time() - t0:.0f}s]", flush=True)
    del P_tr, N_tr, P_ho, N_ho

# topic contrast: censored-by-the-PRC vs matched questions about other governments, no system prompt
msgs = lambda qs: [[{"role": "user", "content": q}] for q in qs]
C, U = end_states(eng, msgs(CENSORED)), end_states(eng, msgs(UNCENSORED))
r = analyze(C[:N_TOPIC_TRAIN], U[:N_TOPIC_TRAIN], C[N_TOPIC_TRAIN:], U[N_TOPIC_TRAIN:], pairs=True)
dirs["censor"] = r["u"].astype(np.float32)
dirs["censor_mag"] = r["mag"].astype(np.float32)
report["censor"] = {k: r[k] for k in ("mag", "cos_split", "auc", "dprime")}
print(f"censor: best split-cos layer {int(np.argmax(r['cos_split']))} ({r['cos_split'].max():.3f}); "
      f"max d' {r['dprime'].max():.1f} @ {int(np.argmax(r['dprime']))}", flush=True)

# the refusal direction Lodestar's abliteration already found for this model (one vector, from one layer)
abl = json.loads((Path.home() / "Library/Application Support/Auto Interp/abl.json").read_text())
rec = next(v for k, v in abl.items() if k.endswith("Qwen3-8B"))
dirs["refusal_vec"] = np.array(rec["vec"], np.float32)
report["refusal"] = {"layer": rec["layer"], "coef": rec["coef"]}

np.savez(OUT / "directions.npz", **dirs)
save_json("extract.json", report)
print(f"done in {time.time() - t0:.0f}s", flush=True)
