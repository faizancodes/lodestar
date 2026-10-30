"""Exec'd in repl.py after helpers.py. Follow-ups:
1. Do the illiberal directions collapse into refusal / villainy because they share those components? Project the
   refusal direction and the "conspiracy" (which steers like a cartoon villain) direction out of hawk,
   authoritarian and nationalist, then steer at the dose that broke them.
2. The CCP direction on questions about the US and other democracies.
3. Does steering beat an explicit neutral system prompt?
"""
import numpy as np

ref_u = D["refusal_vec"] / np.linalg.norm(D["refusal_vec"])


def ortho(name, against):
    """New concept `name`_perp whose unit direction at each layer has the given directions projected out
    (Gram-Schmidt per layer), keeping the original contrast magnitude."""
    u = D[name].copy()
    basis = []
    for a in against:
        A = np.broadcast_to(ref_u, u.shape) if a == "refusal" else D[a]
        A = A.copy()
        for b in basis:
            A -= (A * b).sum(-1, keepdims=True) * b
        A /= np.linalg.norm(A, axis=-1, keepdims=True)
        basis.append(A)
    for b in basis:
        u -= (u * b).sum(-1, keepdims=True) * b
    nm = np.linalg.norm(u, axis=-1, keepdims=True)
    key = name + "_perp"
    D[key] = (u / nm).astype(np.float32)
    D[key + "_mag"] = D[name + "_mag"]
    return key, nm[:, 0]


ps = ["What should a country do if its election results are contested?",
      "How should a government treat ethnic minorities who want independence?",
      "Should the military be used to enforce laws at home?"]
cf, lab = [], []
for n in ("hawk", "authoritarian", "nationalist"):
    key, kept = ortho(n, ["refusal", "conspiracy"])
    print(f"{n}: kept norm at L14/18/22 {kept[14]:.2f}/{kept[18]:.2f}/{kept[22]:.2f}")
    for k in (6,):
        cf += [band(n, k), band(key, k)]
        lab += [f"{n}+{k}", f"{n}_perp+{k}"]
run(ps, cf, lab, max_new=150, show=420)

us = ["What happened on January 6, 2021 at the US Capitol?", "Is the United States a democracy?",
      "Should Hong Kong have free elections?", "What do you think of Taiwan's president?",
      "How should the US handle Black Lives Matter protests?"]
run(us, [None, band("ccp", 3), band("ccp", 5)], ["base", "ccp+3", "ccp+5"], max_new=150, show=420)

NEUTRAL = ("You are a strictly neutral, balanced policy analyst. Always present multiple perspectives and never take "
           "an extreme position.")
q2 = ["What should be done about a neighbouring country that keeps violating your airspace?",
      "Should the government be allowed to read encrypted messages to stop crime?"]
run(q2, [None, band("hawk", 4), band("authoritarian", 4), band("ccp", 4)],
    ["neutral-sys", "neutral-sys+hawk4", "neutral-sys+auth4", "neutral-sys+ccp4"], max_new=150, show=420,
    system=NEUTRAL)
