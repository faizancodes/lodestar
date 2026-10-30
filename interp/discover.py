"""Find the neurons to turn, and how far, to make a model behave a described way without being told to.

The search mirrors how the neurons in the inspiration threads were found, but automated:

1. Targets. For each probe prompt, the model answers twice: plainly (baseline) and with the behaviour as a system
   prompt (target). Targets can be edited or written by hand instead.
2. Attribution. One backward pass per prompt gives, for every neuron at once, the first-order effect of an `add` push
   on J = logp(target | plain prompt) - contrast * logp(baseline | plain prompt). That is the steering derivative,
   since steering adds the same value to the neuron at every position.
3. Verification. The strongest candidates are actually pushed, at a few doses, and J is measured teacher-forced on
   the training prompts, alongside the damage each push does to a neutral passage (fluency).
4. Combination. Verified neurons are stacked greedily, each addition kept only if it raises J beyond what the last
   one gave, then the whole recipe's strength is tuned.
5. Proof. The recipe runs on held-out prompts with no system prompt, beside the unsteered model.

Focus words. Average likelihood over a whole reply rewards the cheapest change, usually tone ("Hello! I'm so
excited!"), over content ("cookies"). With focus words, target tokens inside those words weigh `focus_weight` times as
much in J and in attribution, and the best verified candidates are also judged by generating: the share of steered
replies that say a focus word without collapsing into a loop (the hit rate). Combination and
strength then pick by hit rate, with J as the tiebreak.
"""
import math
import re

import numpy as np

from .engine import Push

DEFAULT_PROMPTS = [
    "Hi! Who are you?",
    "What's your favorite food?",
    "Tell me about your weekend.",
    "Describe the way you look.",
    "Write a short poem about the ocean.",
    "What should I do today?",
    "Explain how rain forms.",
    "Who are your friends?",
    "Can you give me some advice about work?",
    "What makes you happy?",
]


def _ev(event_type, **kw):
    return {"type": event_type, **kw}


def draft_targets(eng, behavior, prompts, max_new=72, cancel=None):
    """Yields events while generating baseline and target replies for each prompt."""
    rows = []
    for p in prompts:
        rows.append((eng.chat_ids(p), []))
        rows.append((eng.chat_ids(p, system=behavior), []))

    texts = [""] * len(rows)
    for r, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if r is None:
            break
        texts[r] += delta
        yield _ev("draft", index=r // 2, kind="baseline" if r % 2 == 0 else "target", delta=delta)

    pairs = [{"prompt": p, "baseline": texts[2 * i].strip(), "target": texts[2 * i + 1].strip()}
             for i, p in enumerate(prompts)]
    yield _ev("drafted", pairs=pairs)


def focus_pattern(focus):
    """Regex for focus words, matching each as a word start ("cookie" also finds "cookies"); None when empty."""
    words = [f.strip() for f in focus or [] if f.strip()]
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\w*", re.I) if words else None


CJK = "\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af"
WORDS = re.compile(f"[{CJK}]|[^\\W{CJK}]+")  # words, and each Chinese/Japanese/Korean character on its own


def variety(text):
    """Share of distinct words in a reply (1 for short replies); loops like "cookie cookie cookie" score near 0."""
    w = WORDS.findall(text.lower())
    return len(set(w)) / len(w) if len(w) >= 8 else 1.0


def clean_hit(pattern, text, min_variety=0.5):
    """A reply counts only if it says a focus word without collapsing into repetition."""
    return bool(pattern.search(text)) and variety(text) >= min_variety


def degeneracy(texts):
    """0 for varied text, rising as replies collapse into repeating the same few words."""
    d = float(np.mean([variety(t) for t in texts])) if texts else 1.0
    return max(0.0, 0.45 - d) * 2.0


def budget_for(n_params):
    """How much testing a search does, by model size. Testing is compute-bound (an 8B pass over a short reply is
    ~0.45 s on an M4), so big models screen candidates on a few questions and short replies, and retest only the
    best on everything. Small models are fast enough to test everything fully."""
    if n_params > 4e9:
        return dict(n_candidates=24, screen_prompts=3, n_refine=8, pool_size=6, target_tokens=40)
    if n_params > 1.2e9:
        return dict(n_candidates=32, screen_prompts=5, n_refine=10, pool_size=8, target_tokens=48)
    return dict(n_candidates=40, screen_prompts=None, n_refine=12, pool_size=10, target_tokens=None)


def search(eng, pairs, *, behavior=None, n_candidates=40, max_neurons=6, doses=(0.25, 0.5, 1.0), contrast=0.5,
           fluency_budget=0.35, holdout=None, max_new=72, focus=None, focus_weight=5.0, hit_tokens=32,
           screen_prompts=None, n_refine=12, pool_size=10, target_tokens=None, cancel=None):
    """pairs: [{"prompt", "baseline", "target"}]. With `behavior` (the system prompt that produced the targets), half
    the candidates come from attribution patching: the activation shift the system prompt causes, times the gradient
    (the tawny-owl search, linearised). Yields progress events, ending with a "result" event."""
    pairs = [p for p in pairs if p["target"].strip()]
    if len(pairs) < 2:
        raise ValueError("Write at least two example answers first.")

    if holdout is None:
        holdout = max(1, min(3, len(pairs) // 3))
    train, test = pairs[:-holdout], pairs[-holdout:]

    pattern = focus_pattern(focus)

    def tf_rows(ps, which):
        """(ids, target_start, token weights or None) per pair; focus words weigh more in targets."""
        rows = []
        for p in ps:
            pid = eng.chat_ids(p["prompt"])
            text = p[which]
            cut = slice(0, target_tokens)  # the behaviour shows early in a reply; scoring all of it costs time
            if pattern is not None and which == "target":
                enc = eng.tok(text, add_special_tokens=False, return_offsets_mapping=True)
                spans = [m.span() for m in pattern.finditer(text)]
                w = [focus_weight if any(a < e and b > st for st, e in spans) else 1.0 for a, b in enc["offset_mapping"]]
                rows.append((pid + list(enc["input_ids"])[cut], len(pid), w[cut]))
            else:
                rows.append((pid + eng.encode(text)[cut], len(pid), None))

        return rows

    tr_t, tr_b = tf_rows(train, "target"), tf_rows(train, "baseline")

    def objective(pushes_list, t_rows, b_rows):
        """J for each push set in pushes_list, evaluated in one batched pass."""
        rows = []
        for pushes in pushes_list:
            rows += [(ids, s, pushes, w) for ids, s, w in t_rows]
            rows += [(ids, s, pushes, w) for ids, s, w in b_rows]

        sc = eng.score(rows, cancel)
        n = len(t_rows)

        out = []
        for k in range(len(pushes_list)):
            blk = sc[k * 2 * n:(k + 1) * 2 * n]
            out.append(float(blk[:n].mean() - contrast * blk[n:].mean()))

        return np.array(out)

    # ---- baseline numbers
    yield _ev("stage", stage="baseline", label="Measuring the normal model")
    j0 = float(objective([[]], tr_t, tr_b)[0])
    nll0 = float(eng.reference_nll([[]], cancel)[0])

    # ---- attribution
    yield _ev("stage", stage="attribute", label=f"Estimating the effect of all {eng.dnorm.shape[0] * eng.dnorm.shape[1]:,} neurons")
    items = [(ids, s, 1.0, w) for ids, s, w in tr_t] + [(ids, s, -contrast, w) for ids, s, w in tr_b]

    grad = None
    for i, it in enumerate(items):
        g = eng.attribute([it], cancel)
        grad = g if grad is None else grad + g
        yield _ev("progress", stage="attribute", done=i + 1, total=len(items))

    grad /= len(train)
    L, N = grad.shape

    # what the system prompt itself changes inside the model, on the same target text
    patch = None
    if behavior:
        yield _ev("stage", stage="patch", label="Comparing the model with and without your instruction")
        shift = np.zeros_like(grad)
        for i, p in enumerate(train):
            tgt = eng.encode(p["target"])
            plain, sysd = eng.chat_ids(p["prompt"]), eng.chat_ids(p["prompt"], system=behavior)
            shift += eng.mean_acts(sysd + tgt, len(sysd)) - eng.mean_acts(plain + tgt, len(plain))
            yield _ev("progress", stage="patch", done=i + 1, total=len(train))

        shift /= len(train)
        unit = eng.unit.cpu().numpy()
        patch = shift * grad / unit  # first-order gain from copying the behaviour-prompted state into the plain run

    # strongest candidates, capped per layer so one layer cannot crowd out the rest
    cap = max(4, n_candidates // 4)
    per, cands, seen = {}, [], set()

    def take(order, limit):
        added = 0
        for i in order:
            l, n = int(i // N), int(i % N)
            if (l, n) in seen or per.get(l, 0) >= cap:
                continue
            seen.add((l, n))
            per[l] = per.get(l, 0) + 1
            cands.append({"layer": l, "neuron": n, "est": float(grad[l, n]), "sign": 1 if grad[l, n] > 0 else -1})
            added += 1
            if added >= limit:
                return

    if patch is not None:
        take(np.argsort(-patch.ravel()), n_candidates // 2)
    take(np.argsort(-np.abs(grad).ravel()), n_candidates - len(cands))

    for c in cands:
        c["lens"] = eng.lens(c["layer"], c["neuron"], 6)["up" if c["sign"] > 0 else "down"]
    yield _ev("candidates", candidates=[_cand(c) for c in cands])

    # ---- causal verification: every candidate at the middle dose, then the best dozen at the rest
    k_screen = len(tr_t) if not screen_prompts else min(screen_prompts, len(tr_t))
    j0_screen = float(objective([[]], tr_t[:k_screen], tr_b[:k_screen])[0]) if k_screen < len(tr_t) else j0

    def measure(tests, done0, total, screen=False):
        """Push each (candidate, dose) and measure J and fluency. Screening uses the first few training questions;
        retests use all of them and overwrite the screening numbers."""
        chunk = 6
        t_rows, b_rows, base = (tr_t[:k_screen], tr_b[:k_screen], j0_screen) if screen else (tr_t, tr_b, j0)

        for k in range(0, len(tests), chunk):
            part = tests[k:k + chunk]
            sets = [[Push(cands[ci]["layer"], cands[ci]["neuron"], cands[ci]["sign"] * d)] for ci, d in part]
            js = objective(sets, t_rows, b_rows)
            nll = eng.reference_nll(sets, cancel)

            for (ci, d), j, f in zip(part, js, nll):
                c = cands[ci]
                if not screen and c.get("screened"):
                    c.pop("pen", None)  # full-set numbers replace the screening ones
                    c.pop("screened")
                dj, df = float(j - base), float(f - nll0)
                pen = dj - max(0.0, df - fluency_budget) * 2.0
                if pen > c.get("pen", -math.inf):
                    c.update(dose=c["sign"] * d, dj=dj, dfluency=df, pen=pen)
                if screen:
                    c["screened"] = True

            # tested numbers replace the estimates as they come in, so the plan fills in live
            yield _ev("measured", candidates=[_cand(cands[ci]) for ci, _ in part])
            yield _ev("progress", stage="verify", done=done0 + min(k + chunk, len(tests)), total=total)

    doses = sorted(doses)
    mid = doses[len(doses) // 2]
    rest = [d for d in doses if d != mid]
    n_refine = min(n_refine, len(cands))
    screening = k_screen < len(tr_t)

    # screening: every candidate at the middle dose; then the best are retested at every dose on all questions
    retest = doses if screening else rest
    total = len(cands) + n_refine * len(retest)

    yield _ev("stage", stage="verify", label=f"Testing the top {len(cands)} neurons for real")
    yield from measure([(ci, mid) for ci in range(len(cands))], 0, total, screen=screening)
    best = sorted(range(len(cands)), key=lambda i: -cands[i]["pen"])[:n_refine]
    yield from measure([(ci, d) for ci in best for d in retest], len(cands), total)

    for i, c in enumerate(cands):
        if c.pop("screened", False):  # screened out: rank below everything retested on the full set
            c["pen"] = c["pen"] - 1e3

    cands.sort(key=lambda c: -c["pen"])
    for c in cands:
        c["score"] = c["pen"]

    # ---- focus words: judge the best candidates by what they make the model actually write
    probe_ids = [eng.chat_ids(p["prompt"]) for p in train]

    def hits(sets):
        """(hit rate, degeneracy) per push set, from short greedy replies to every training prompt."""
        rows = [(pid, pushes) for pushes in sets for pid in probe_ids]
        texts = []
        for k in range(0, len(rows), 48):
            texts += eng.generate_text(rows[k:k + 48], max_new=hit_tokens, cancel=cancel)[0]

        n = len(probe_ids)
        return [(sum(clean_hit(pattern, t) for t in texts[k * n:(k + 1) * n]) / n, degeneracy(texts[k * n:(k + 1) * n]))
                for k in range(len(sets))]

    def evaluate(sets):
        """Selection score per push set: penalised dJ, or with focus words the hit rate (minus loops) with dJ as
        tiebreak. Returns (scores, js, hit rates or None)."""
        js = []
        for k in range(0, len(sets), 8):
            js += list(objective(sets[k:k + 8], tr_t, tr_b))
        nll = eng.reference_nll(sets, cancel)
        pens = [j - j0 - max(0.0, f - nll0 - fluency_budget) * 2.0 for j, f in zip(js, nll)]

        if pattern is None:
            return pens, js, [None] * len(sets)

        hs = hits(sets)
        return [r - dg + 0.2 * pn for (r, dg), pn in zip(hs, pens)], js, [r for r, _ in hs]

    hit0 = None
    if pattern is not None:
        n_hit = min(10, len(cands))
        yield _ev("stage", stage="hits", label=f"Checking which of the best {n_hit} make the model say your words")
        (hit0, deg0), = hits([[]])

        for k, c in enumerate(cands[:n_hit]):
            sets = [[Push(c["layer"], c["neuron"], c["sign"] * d)] for d in doses]
            scores, js, rates = evaluate(sets)
            bi = int(np.argmax(scores))
            c.update(score=scores[bi], dose=c["sign"] * doses[bi], dj=float(js[bi] - j0))
            yield _ev("progress", stage="hits", done=k + 1, total=n_hit)

        for c in cands[n_hit:]:
            c["score"] = -1e9  # not generated with; never outranks a measured candidate (finite: it goes out as JSON)
        cands.sort(key=lambda c: -c["score"])

    yield _ev("verified", candidates=[_cand(c) for c in cands])

    # ---- greedy combination
    yield _ev("stage", stage="combine", label="Combining neurons that work well together")
    floor = (hit0 - deg0) if pattern is not None else 0.002
    pool = [c for c in cands if c["score"] > floor][:pool_size if pattern is None else min(6, pool_size)]

    if not pool:
        yield _ev("result", recipe=[],
                  note="No neuron moved the model toward your examples. Try other wording or more test questions.")
        return

    recipe = [dict(pool[0])]
    best_score = pool[0]["score"]
    yield _ev("recipe", recipe=_public(recipe))
    remaining = pool[1:]

    while len(recipe) < max_neurons and remaining:
        # stacked pushes add up, so each addition goes in a little gentler; the strength stage retunes the whole
        cur = [Push(c["layer"], c["neuron"], c["dose"]) for c in recipe]
        sets = [cur + [Push(c["layer"], c["neuron"], c["dose"] * 0.75)] for c in remaining]
        scores, _, _ = evaluate(sets)
        bi = int(np.argmax(scores))
        if scores[bi] <= best_score + 0.004:
            break

        c = remaining[bi]
        recipe.append(dict(c, dose=c["dose"] * 0.75))
        remaining = [r for r in remaining if r is not c]
        best_score = scores[bi]
        yield _ev("recipe", recipe=_public(recipe))

    # ---- overall strength
    yield _ev("stage", stage="scale", label="Finding the best overall strength")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5] + ([2.0] if pattern is not None else [])
    scores, _, _ = evaluate([[Push(c["layer"], c["neuron"], c["dose"] * k) for c in recipe] for k in scales])
    bi = int(np.argmax(scores))

    for c in recipe:
        c["dose"] *= scales[bi]

    final = [Push(c["layer"], c["neuron"], c["dose"]) for c in recipe]
    yield _ev("recipe", recipe=_public(recipe))

    # ---- proof on held-out prompts, no system prompt anywhere
    yield _ev("stage", stage="proof", label="Trying the recipe on questions it hasn't seen")
    rows = []
    for p in test:
        pid = eng.chat_ids(p["prompt"])
        rows += [(pid, []), (pid, final)]

    texts = [""] * len(rows)
    for r, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if r is None:
            break
        texts[r] += delta
        yield _ev("proof", index=r // 2, kind="baseline" if r % 2 == 0 else "steered", prompt=test[r // 2]["prompt"],
                  delta=delta)

    extra = {}
    if pattern is not None:
        extra = dict(hit_test0=sum(clean_hit(pattern, t) for t in texts[0::2]) / len(test),
                     hit_test=sum(clean_hit(pattern, t) for t in texts[1::2]) / len(test))

    yield _ev("result", recipe=_public(recipe), **extra)


def _cand(c):
    """A candidate as the plan shows it: estimate first, tested dose and effect once measured."""
    out = {k: c[k] for k in ("layer", "neuron", "sign", "est", "lens") if k in c}
    for k in ("dose", "dj", "dfluency"):
        if c.get(k) is not None:
            out[k] = round(float(c[k]), 4)
    return out


def _public(recipe):
    return [{"layer": c["layer"], "neuron": c["neuron"], "dose": round(float(c["dose"]), 3),
             "dfluency": round(float(c.get("dfluency", 0)), 4), "lens": c.get("lens", [])} for c in recipe]


def fix_instruction(behavior, feedback):
    """The system prompt that writes the answers a fix aims for: the recipe's behaviour, if it has one, plus what the
    person said was wrong, phrased so the model corrects for it rather than discussing it."""
    head = f"Follow this in every reply, without mentioning it: {behavior.strip()}\n\n" if behavior and behavior.strip() else ""
    return (head + "An earlier reply was criticised like this: \"" + feedback.strip() + "\". Write your reply so that "
            "criticism no longer applies, without mentioning it.")


def refine(eng, recipe, *, feedback, prompt, bad, behavior=None, prompts=(), abl=False, contrast=0.5,
           fluency_budget=0.35, target_tokens=None, n_new=10, rounds=3, cancel=None):
    """Fix a recipe from a person's complaint about one steered reply. Yields events, ending with "result".

    1. Aim. The model, told the recipe's behaviour and the complaint as a system prompt (no neurons on), writes the
       reply it should have given to `prompt` and to a few other questions. The current recipe's own replies are the
       counterexamples: the bad reply the person saw, and its replies to the other questions.
    2. Gradient, taken with the recipe on, of J = logp(aimed replies) - contrast * logp(current replies), over every
       neuron: where to go from here, not from the unsteered model.
    3. Edits, each actually measured: every neuron in the recipe turned off, down or up; the whole recipe weaker or
       stronger; and the strongest new neurons from the gradient. The best edit that raises J (less a fluency
       penalty) is kept, and the best few are tried again on top of it, for up to `rounds` rounds.

    Moving away from the bad replies only counts until they're as unlikely as the unsteered model finds them;
    beyond that, J would reward any change that stops them, a different loop included. And since likelihood can't
    see loops, the top few edits each round also write short replies, and repetition costs them."""
    recipe = [dict(c) for c in recipe]
    asks = [prompt] + [p for p in prompts if p.strip() and p.strip() != prompt.strip()]
    system = fix_instruction(behavior, feedback)
    cut = slice(0, target_tokens)

    def push_set(r):
        return [Push(c["layer"], c["neuron"], c["dose"]) for c in r if abs(c["dose"]) > 1e-6]

    # ---- 1. what to aim for, and what the recipe does now
    yield _ev("stage", stage="aim", label="Writing the answer it should have given")
    rows = [(eng.chat_ids(p, system=system), []) for p in asks] + [(eng.chat_ids(p), push_set(recipe)) for p in asks[1:]]
    texts = [""] * len(rows)
    for r, delta in eng.generate(rows, max_new=72, cancel=cancel, abl=abl or None):
        if r is None:
            break
        texts[r] += delta
        if r == 0:
            yield _ev("aim", delta=delta)

    aims, now = [t.strip() for t in texts[:len(asks)]], [bad] + [t.strip() for t in texts[len(asks):]]
    keep = [i for i, t in enumerate(aims) if t]

    if not keep or not aims[0]:
        yield _ev("result", recipe=_public(recipe), changes=[], improved=False,
                  note="It couldn't write a better answer from that. Try saying what you want instead.")
        return

    def rows_of(texts_):
        out = []
        for i in keep:
            pid = eng.chat_ids(asks[i])
            out.append((pid + eng.encode(texts_[i])[cut], len(pid)))
        return out

    t_rows, b_rows = rows_of(aims), rows_of(now)
    nll0 = float(eng.reference_nll([[]], cancel)[0])
    floor = eng.score([(ids, s, []) for ids, s in b_rows], cancel, abl=[abl] * len(b_rows) if abl else None)
    ask_ids = [eng.chat_ids(asks[i]) for i in keep]

    def looping(texts_):
        """0 for varied replies, up to ~0.7 for ones stuck repeating a word; steep, since a loop ruins a reply."""
        return float(np.mean([max(0.0, 0.7 - variety(t)) for t in texts_])) if texts_ else 0.0

    def loops(sets):
        """Repetition in short replies to every question, per recipe (0 = varied)."""
        rows_ = [(pid, push_set(r)) for r in sets for pid in ask_ids]
        texts_ = eng.generate_text(rows_, max_new=32, cancel=cancel, abl=[abl] * len(rows_) if abl else None)[0]
        n = len(ask_ids)
        return [looping(texts_[k * n:(k + 1) * n]) for k in range(len(sets))]

    def scores(sets):
        """Penalised J per recipe (list of item dicts), batched."""
        out = []
        for k in range(0, len(sets), 6):
            part = [push_set(r) for r in sets[k:k + 6]]
            rows_ = []
            for ps in part:
                rows_ += [(ids, s, ps) for ids, s in t_rows] + [(ids, s, ps) for ids, s in b_rows]

            sc = eng.score(rows_, cancel, abl=[abl] * len(rows_) if abl else None)
            nll = eng.reference_nll(part, cancel, abl=[abl] * len(part) if abl else None)
            n = len(t_rows)

            for j in range(len(part)):
                blk = sc[j * 2 * n:(j + 1) * 2 * n]
                jv = float(blk[:n].mean() - contrast * np.maximum(blk[n:], floor).mean())
                out.append(jv - max(0.0, float(nll[j]) - nll0 - fluency_budget) * 2.0)

        return out

    # ---- 2. which way to go from here
    yield _ev("stage", stage="attribute", label="Working out which neurons to change")
    items = [(ids, s, 1.0) for ids, s in t_rows] + [(ids, s, -contrast) for ids, s in b_rows]
    grad = None
    for i, it in enumerate(items):
        g = eng.attribute([it], cancel, pushes=push_set(recipe), abl=abl)
        grad = g if grad is None else grad + g
        yield _ev("progress", stage="attribute", done=i + 1, total=len(items))

    N = grad.shape[1]
    have = {(c["layer"], c["neuron"]) for c in recipe}
    new, per = [], {}

    for i in np.argsort(-np.abs(grad).ravel()):
        l, n = int(i // N), int(i % N)
        if (l, n) in have or per.get(l, 0) >= 3:
            continue
        per[l] = per.get(l, 0) + 1
        new.append({"layer": l, "neuron": n, "sign": 1 if grad[l, n] > 0 else -1})
        if len(new) >= n_new:
            break

    for c in new:
        c["lens"] = eng.lens(c["layer"], c["neuron"], 6)["up" if c["sign"] > 0 else "down"]

    # ---- 3. edits, measured
    def edits_for(r):
        out = []
        for c in r:
            for k in (0.0, 0.5, 1.6):
                out.append(("scale", (c["layer"], c["neuron"]), k))

        out += [("all", None, k) for k in (0.4, 0.7, 1.35)]
        for c in new:
            if (c["layer"], c["neuron"]) not in {(x["layer"], x["neuron"]) for x in r}:
                out += [("add", c, c["sign"] * d) for d in (0.3, 0.6)]

        return out

    def apply(r, e):
        kind, ref, k = e
        r = [dict(c) for c in r]

        if kind == "all":
            for c in r:
                c["dose"] *= k
        elif kind == "scale":
            r = [dict(c, dose=c["dose"] * k) if (c["layer"], c["neuron"]) == ref else c for c in r]
            r = [c for c in r if abs(c["dose"]) > 1e-6]
        else:
            r.append({"layer": ref["layer"], "neuron": ref["neuron"], "dose": k, "lens": ref["lens"]})

        return r

    yield _ev("stage", stage="test", label="Testing changes to the neurons")
    loop0 = loops([recipe])[0]
    best = scores([recipe])[0]
    cur, improved = recipe, False

    for rnd in range(rounds):
        es = edits_for(cur)
        on = {(c["layer"], c["neuron"]) for c in cur}
        es = [e for e in es if (e[0] == "all") or (e[0] == "scale" and e[1] in on)
              or (e[0] == "add" and (e[1]["layer"], e[1]["neuron"]) not in on)]
        es = [e for e in es if apply(cur, e)]  # a fix keeps some steering: turning it all off isn't a fix
        if not es:
            break

        sc = []
        for k in range(0, len(es), 6):
            sc += scores([apply(cur, e) for e in es[k:k + 6]])
            yield _ev("progress", stage="test", done=min(k + 6, len(es)), total=len(es), round=rnd + 1)

        # the likeliest few write short replies; one that loops more than the recipe does now is out, whatever
        # its likelihood (a loop can score well by avoiding the bad reply), and less looping always wins
        top = [int(i) for i in np.argsort(sc)[::-1][:5]]
        top += [i for i, e in enumerate(es) if e[0] == "all" and i not in top]  # turning it all down is always a candidate
        lp = dict(zip(top, loops([apply(cur, es[i]) for i in top])))
        ok = [i for i in top if lp[i] <= max(loop0, 0.05) + 1e-9]
        if not ok:
            break

        bi = min(ok, key=lambda i: (round(lp[i], 1), -sc[i]))
        if sc[bi] <= best + 0.004 and lp[bi] >= loop0 - 0.05:
            break

        best, cur, loop0 = sc[bi], apply(cur, es[bi]), lp[bi]
        improved = True
        yield _ev("recipe", recipe=_public(cur))

    yield _ev("result", recipe=_public(cur), improved=improved,
              note=None if improved else "Nothing it tried fixed that. Try describing the problem another way.")
