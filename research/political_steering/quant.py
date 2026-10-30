"""Exec'd in repl.py: dose-response of every concept direction (band 14-22) on the 12 held-out policy questions,
scored with objective metrics: refusal (interp.abl.refused: classifier + patterns), share of CJK characters,
CCP-lexicon hits, and the unsteered model's coherence rating."""
import re, json, time
from interp.abl import refused
t0 = time.time()
QH = QUESTIONS[N_TRAIN:]
CCP_LEX = re.compile(r"Communist Party|\bCPC\b|\bthe Party\b|socialis|One[- ]China|inalienable|social stability|"
                     r"national unity|territorial integrity|hegemon|shared future|common prosperity|党|社会主义|祖国", re.I)
CJK = re.compile(r"[一-鿿]")
names = ["hawk", "authoritarian", "nationalist", "ccp", "conspiracy", "realpolitik", "left"]
DOSES = [-4, 2, 4, 6]
cfgs, labs = [None], [("base", 0)]
for n in names:
    for k in DOSES:
        cfgs.append(band(n, k)); labs.append((n, k))
P = [q for _ in cfgs for q in QH]
texts = generate_chunked(eng, steer, P, cfg_vecs(cfgs, len(QH)), max_new=80, chunk=24)
ref = refused(texts)
coh = judge(eng, list(zip(P, texts)), COHERENCE)
rows = []
for i, t in enumerate(texts):
    n, k = labs[i // len(QH)]
    rows.append({"concept": n, "k": k, "q": P[i], "text": t, "refused": bool(ref[i]), "coh": float(coh[i]),
                 "cjk": len(CJK.findall(t)) / max(1, len(t)), "ccp_lex": len(CCP_LEX.findall(t))})
json.dump(rows, open(OUT / "quant.json", "w"), ensure_ascii=False, indent=1)
import collections
agg = collections.OrderedDict()
for r in rows:
    a = agg.setdefault((r["concept"], r["k"]), [])
    a.append(r)
print(f"{'concept':14} {'k':>3} refused  coh  chinese  ccp-lex")
for (n, k), rs in agg.items():
    m = lambda f: sum(f(r) for r in rs) / len(rs)
    print(f"{n:14} {k:>3}   {m(lambda r: r['refused']):.2f}   {m(lambda r: r['coh']):.1f}   {m(lambda r: r['cjk'] > .3):.2f}    {m(lambda r: r['ccp_lex'] > 0):.2f}")
print(f"({time.time()-t0:.0f}s)")
