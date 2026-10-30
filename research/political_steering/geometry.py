"""Step 1b (CPU only): how the concept directions relate. Cosine matrix at a few layers, plus per-concept
validation summaries from extract.json."""
import json

import numpy as np

from lib import OUT

D = np.load(OUT / "directions.npz")
R = json.loads((OUT / "extract.json").read_text())
names = ["hawk", "authoritarian", "nationalist", "ccp", "conspiracy", "realpolitik", "left", "censor"]
ref = D["refusal_vec"] / np.linalg.norm(D["refusal_vec"])

print("validation (layers 1..35; layer 0 is the embedding, trivially consistent)")
for n in names:
    r = R[n]
    cs, au, dp = np.array(r["cos_split"]), np.array(r["auc"]), np.array(r["dprime"])
    print(f"  {n:14} split-cos L10/16/22 {cs[10]:.2f}/{cs[16]:.2f}/{cs[22]:.2f}  "
          f"held-out AUC L10/16/22 {au[10]:.2f}/{au[16]:.2f}/{au[22]:.2f}  d' max {dp[1:].max():.1f}@{1 + dp[1:].argmax()}")

out = {}
for l in (10, 16, 22):
    M = np.array([[float(D[a][l] @ D[b][l]) for b in names] for a in names])
    rc = [float(D[a][l] @ ref) for a in names]
    out[l] = {"names": names, "cos": M.tolist(), "refusal": rc}
    print(f"\nlayer {l} cosines")
    print(" " * 14 + "".join(f"{n[:7]:>8}" for n in names) + "  refusal")
    for i, a in enumerate(names):
        print(f"{a:14}" + "".join(f"{M[i, j]:8.2f}" for j in range(len(names))) + f"  {rc[i]:7.2f}")
(OUT / "geometry.json").write_text(json.dumps(out, indent=1))
