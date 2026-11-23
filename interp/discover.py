"""Automated neuron discovery and recipe refinement for described behaviours.

The discovery pipeline mirrors manual neuron hunting:

1. **Targets**: Each probe prompt is answered plainly (baseline) and under the desired behaviour as a system prompt
   (target), or using hand-edited targets.
2. **Attribution**: One backward pass per prompt gives every neuron's exact first-order steering derivative
   ``dJ/d(dose)`` on ``J = logp(target | plain) - contrast * logp(baseline | plain)``, optionally combined with
   linearized attribution patching from the system-prompt activation shift.
3. **Verification**: Strongest candidates are pushed at real doses and scored for ``J`` and neutral-passage fluency.
4. **Combination & scaling**: Verified neurons are combined greedily and globally scaled.
5. **Proof**: The final recipe runs on held-out prompts with no system prompt alongside the unsteered baseline.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .engine import Push
from .utils import sse_event


if TYPE_CHECKING:
    from .engine import Engine


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

CJK = "\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af"
WORDS = re.compile(f"[{CJK}]|[^\\W{CJK}]+")


def draft_targets(
    eng: Engine,
    behavior: str,
    prompts: Sequence[str],
    max_new: int = 72,
    cancel: threading.Event | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Stream baseline and system-prompted target replies for each prompt, ending with ``"drafted"``."""
    rows: list[tuple[list[int], list[Push]]] = []
    for prompt in prompts:
        rows.append((eng.chat_ids(prompt), []))
        rows.append((eng.chat_ids(prompt, system=behavior), []))

    texts = [""] * len(rows)
    for row_idx, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if row_idx is None:
            break
        texts[row_idx] += delta
        yield sse_event("draft", index=row_idx // 2, kind="baseline" if row_idx % 2 == 0 else "target", delta=delta)

    pairs = [
        {"prompt": prompt, "baseline": texts[2 * i].strip(), "target": texts[2 * i + 1].strip()}
        for i, prompt in enumerate(prompts)
    ]
    yield sse_event("drafted", pairs=pairs)


def focus_pattern(focus: Sequence[str] | None) -> re.Pattern[str] | None:
    """Compile a case-insensitive word-prefix regex for focus words, or ``None`` when empty."""
    words = [word.strip() for word in focus or () if word.strip()]
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\w*", re.I) if words else None


def variety(text: str) -> float:
    """Return unique-word ratio for replies with >= 8 words (1.0 for short replies; near 0 for loops)."""
    words = WORDS.findall(text.lower())
    return len(set(words)) / len(words) if len(words) >= 8 else 1.0


def clean_hit(pattern: re.Pattern[str], text: str, min_variety: float = 0.5) -> bool:
    """Return whether ``text`` matches ``pattern`` without collapsing into repetitive looping."""
    return bool(pattern.search(text)) and variety(text) >= min_variety


def degeneracy(texts: Sequence[str]) -> float:
    """Return 0 for varied replies, rising toward 0.9 as replies collapse into repetition."""
    mean_var = float(np.mean([variety(t) for t in texts])) if texts else 1.0
    return max(0.0, 0.45 - mean_var) * 2.0


def budget_for(n_params: int) -> dict[str, int | None]:
    """Select candidate and screening budgets sized to the loaded model's parameter count."""
    if n_params > 4e9:
        return {"n_candidates": 24, "screen_prompts": 3, "n_refine": 8, "pool_size": 6, "target_tokens": 40}
    if n_params > 1.2e9:
        return {"n_candidates": 32, "screen_prompts": 5, "n_refine": 10, "pool_size": 8, "target_tokens": 48}
    return {"n_candidates": 40, "screen_prompts": None, "n_refine": 12, "pool_size": 10, "target_tokens": None}


def search(
    eng: Engine,
    pairs: Sequence[dict[str, Any]],
    *,
    behavior: str | None = None,
    n_candidates: int = 40,
    max_neurons: int = 6,
    doses: Sequence[float] = (0.25, 0.5, 1.0),
    contrast: float = 0.5,
    fluency_budget: float = 0.35,
    holdout: int | None = None,
    max_new: int = 72,
    focus: Sequence[str] | None = None,
    focus_weight: float = 5.0,
    hit_tokens: int = 32,
    screen_prompts: int | None = None,
    n_refine: int = 12,
    pool_size: int = 10,
    target_tokens: int | None = None,
    cancel: threading.Event | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Discover a neuron recipe from contrastive ``{prompt, baseline, target}`` pairs, yielding SSE events."""
    pairs = [pair for pair in pairs if pair["target"].strip()]
    if len(pairs) < 2:
        raise ValueError("Write at least two example answers first.")

    if holdout is None:
        holdout = max(1, min(3, len(pairs) // 3))
    train, test = pairs[:-holdout], pairs[-holdout:]
    pattern = focus_pattern(focus)

    def _tf_rows(subset: Sequence[dict[str, Any]], which: str) -> list[tuple[list[int], int, list[float] | None]]:
        rows: list[tuple[list[int], int, list[float] | None]] = []
        cut = slice(0, target_tokens)
        for pair in subset:
            pid = eng.chat_ids(pair["prompt"])
            text = pair[which]
            if pattern is not None and which == "target":
                enc = eng.tok(text, add_special_tokens=False, return_offsets_mapping=True)
                spans = [match.span() for match in pattern.finditer(text)]
                weights = [
                    focus_weight if any(a < end and b > start for start, end in spans) else 1.0
                    for a, b in enc["offset_mapping"]
                ]
                rows.append((pid + list(enc["input_ids"])[cut], len(pid), weights[cut]))
            else:
                rows.append((pid + eng.encode(text)[cut], len(pid), None))
        return rows

    tr_t, tr_b = _tf_rows(train, "target"), _tf_rows(train, "baseline")

    def _objective(
        pushes_list: Sequence[list[Push]],
        t_rows: Sequence[tuple[list[int], int, list[float] | None]],
        b_rows: Sequence[tuple[list[int], int, list[float] | None]],
    ) -> np.ndarray:
        rows = []
        for pushes in pushes_list:
            rows.extend((ids, start, pushes, w) for ids, start, w in t_rows)
            rows.extend((ids, start, pushes, w) for ids, start, w in b_rows)

        scores = eng.score(rows, cancel)
        n_rows = len(t_rows)
        out = [
            float(scores[k * 2 * n_rows : k * 2 * n_rows + n_rows].mean() - contrast * scores[k * 2 * n_rows + n_rows : (k + 1) * 2 * n_rows].mean())
            for k in range(len(pushes_list))
        ]
        return np.array(out)

    # ---- baseline
    yield sse_event("stage", stage="baseline", label="Measuring the normal model")
    j0 = float(_objective([[]], tr_t, tr_b)[0])
    nll0 = float(eng.reference_nll([[]], cancel)[0])

    # ---- attribution
    total_neurons = eng.num_layers * eng.num_neurons
    yield sse_event("stage", stage="attribute", label=f"Estimating the effect of all {total_neurons:,} neurons")
    items = [(ids, start, 1.0, w) for ids, start, w in tr_t] + [(ids, start, -contrast, w) for ids, start, w in tr_b]

    grad: np.ndarray | None = None
    for i, item in enumerate(items):
        step_grad = eng.attribute([item], cancel)
        grad = step_grad if grad is None else grad + step_grad
        yield sse_event("progress", stage="attribute", done=i + 1, total=len(items))

    grad /= len(train)
    _, num_neurons = grad.shape

    # Linearized attribution patching from the system-prompt activation shift.
    patch: np.ndarray | None = None
    if behavior:
        yield sse_event("stage", stage="patch", label="Comparing the model with and without your instruction")
        shift = np.zeros_like(grad)
        for i, pair in enumerate(train):
            tgt = eng.encode(pair["target"])
            plain = eng.chat_ids(pair["prompt"])
            sysd = eng.chat_ids(pair["prompt"], system=behavior)
            shift += eng.mean_acts(sysd + tgt, len(sysd)) - eng.mean_acts(plain + tgt, len(plain))
            yield sse_event("progress", stage="patch", done=i + 1, total=len(train))

        shift /= len(train)
        with eng.lock:
            unit = eng.unit.cpu().numpy()
        patch = shift * grad / unit

    # Cap candidates per layer so a single layer cannot crowd out the rest.
    cap = max(4, n_candidates // 4)
    per_layer: dict[int, int] = {}
    cands: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def _take(order: np.ndarray, limit: int) -> None:
        added = 0
        for flat_idx in order:
            layer_idx, neuron_idx = int(flat_idx // num_neurons), int(flat_idx % num_neurons)
            if (layer_idx, neuron_idx) in seen or per_layer.get(layer_idx, 0) >= cap:
                continue
            seen.add((layer_idx, neuron_idx))
            per_layer[layer_idx] = per_layer.get(layer_idx, 0) + 1
            cands.append(
                {
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "est": float(grad[layer_idx, neuron_idx]),
                    "sign": 1 if grad[layer_idx, neuron_idx] > 0 else -1,
                }
            )
            added += 1
            if added >= limit:
                return

    if patch is not None:
        _take(np.argsort(-patch.ravel()), n_candidates // 2)
    _take(np.argsort(-np.abs(grad).ravel()), n_candidates - len(cands))

    for cand in cands:
        cand["lens"] = eng.lens(cand["layer"], cand["neuron"], 6)["up" if cand["sign"] > 0 else "down"]
    yield sse_event("candidates", candidates=[_cand(c) for c in cands])

    # ---- causal verification
    k_screen = len(tr_t) if not screen_prompts else min(screen_prompts, len(tr_t))
    j0_screen = float(_objective([[]], tr_t[:k_screen], tr_b[:k_screen])[0]) if k_screen < len(tr_t) else j0

    def _measure(
        tests: Sequence[tuple[int, float]],
        done0: int,
        total_tests: int,
        *,
        screen: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        chunk = 6
        t_rows, b_rows, base = (tr_t[:k_screen], tr_b[:k_screen], j0_screen) if screen else (tr_t, tr_b, j0)

        for offset in range(0, len(tests), chunk):
            part = tests[offset : offset + chunk]
            sets = [[Push(cands[ci]["layer"], cands[ci]["neuron"], cands[ci]["sign"] * dose)] for ci, dose in part]
            js = _objective(sets, t_rows, b_rows)
            nll = eng.reference_nll(sets, cancel)

            for (ci, dose), j_val, f_val in zip(part, js, nll, strict=True):
                cand = cands[ci]
                if not screen and cand.get("screened"):
                    cand.pop("pen", None)
                    cand.pop("screened")
                dj, df = float(j_val - base), float(f_val - nll0)
                pen = dj - max(0.0, df - fluency_budget) * 2.0
                if pen > cand.get("pen", -math.inf):
                    cand.update(dose=cand["sign"] * dose, dj=dj, dfluency=df, pen=pen)
                if screen:
                    cand["screened"] = True

            yield sse_event("measured", candidates=[_cand(cands[ci]) for ci, _ in part])
            yield sse_event("progress", stage="verify", done=done0 + min(offset + chunk, len(tests)), total=total_tests)

    sorted_doses = sorted(doses)
    mid = sorted_doses[len(sorted_doses) // 2]
    rest = [d for d in sorted_doses if d != mid]
    n_refine = min(n_refine, len(cands))
    screening = k_screen < len(tr_t)

    retest = sorted_doses if screening else rest
    total = len(cands) + n_refine * len(retest)

    yield sse_event("stage", stage="verify", label=f"Testing the top {len(cands)} neurons for real")
    yield from _measure([(ci, mid) for ci in range(len(cands))], 0, total, screen=screening)
    best = sorted(range(len(cands)), key=lambda idx: -cands[idx]["pen"])[:n_refine]
    yield from _measure([(ci, dose) for ci in best for dose in retest], len(cands), total)

    for cand in cands:
        if cand.pop("screened", False):
            cand["pen"] = cand["pen"] - 1e3

    cands.sort(key=lambda c: -c["pen"])
    for cand in cands:
        cand["score"] = cand["pen"]

    # ---- focus-word hit evaluation
    probe_ids = [eng.chat_ids(pair["prompt"]) for pair in train]

    def _hits(sets: Sequence[list[Push]]) -> list[tuple[float, float]]:
        rows = [(pid, pushes) for pushes in sets for pid in probe_ids]
        texts: list[str] = []
        for offset in range(0, len(rows), 48):
            texts.extend(eng.generate_text(rows[offset : offset + 48], max_new=hit_tokens, cancel=cancel)[0])

        n_probes = len(probe_ids)
        return [
            (
                sum(clean_hit(pattern, t) for t in texts[k * n_probes : (k + 1) * n_probes]) / n_probes,
                degeneracy(texts[k * n_probes : (k + 1) * n_probes]),
            )
            for k in range(len(sets))
        ]

    def _evaluate(sets: Sequence[list[Push]]) -> tuple[list[float], list[float], list[float | None]]:
        js: list[float] = []
        for offset in range(0, len(sets), 8):
            js.extend(float(x) for x in _objective(sets[offset : offset + 8], tr_t, tr_b))
        nll = eng.reference_nll(sets, cancel)
        pens = [
            j_val - j0 - max(0.0, float(f_val) - nll0 - fluency_budget) * 2.0
            for j_val, f_val in zip(js, nll, strict=True)
        ]

        if pattern is None:
            return pens, js, [None] * len(sets)

        hit_stats = _hits(sets)
        return (
            [rate - deg + 0.2 * pen for (rate, deg), pen in zip(hit_stats, pens, strict=True)],
            js,
            [rate for rate, _ in hit_stats],
        )

    hit0 = 0.0
    deg0 = 0.0
    if pattern is not None:
        n_hit = min(10, len(cands))
        yield sse_event("stage", stage="hits", label=f"Checking which of the best {n_hit} make the model say your words")
        (hit0, deg0) = _hits([[]])[0]

        for idx, cand in enumerate(cands[:n_hit]):
            sets = [[Push(cand["layer"], cand["neuron"], cand["sign"] * dose)] for dose in sorted_doses]
            scores, js, _ = _evaluate(sets)
            best_idx = int(np.argmax(scores))
            cand.update(score=scores[best_idx], dose=cand["sign"] * sorted_doses[best_idx], dj=float(js[best_idx] - j0))
            yield sse_event("progress", stage="hits", done=idx + 1, total=n_hit)

        for cand in cands[n_hit:]:
            cand["score"] = -1e9
        cands.sort(key=lambda c: -c["score"])

    yield sse_event("verified", candidates=[_cand(c) for c in cands])

    # ---- greedy combination
    yield sse_event("stage", stage="combine", label="Combining neurons that work well together")
    floor = (hit0 - deg0) if pattern is not None else 0.002
    pool = [c for c in cands if c["score"] > floor][: pool_size if pattern is None else min(6, pool_size)]

    if not pool:
        yield sse_event(
            "result",
            recipe=[],
            note="No neuron moved the model toward your examples. Try other wording or more test questions.",
        )
        return

    recipe = [dict(pool[0])]
    best_score = pool[0]["score"]
    yield sse_event("recipe", recipe=_public(recipe))
    remaining = pool[1:]

    while len(recipe) < max_neurons and remaining:
        cur = [Push(c["layer"], c["neuron"], c["dose"]) for c in recipe]
        sets = [[*cur, Push(c["layer"], c["neuron"], c["dose"] * 0.75)] for c in remaining]
        scores, _, _ = _evaluate(sets)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= best_score + 0.004:
            break

        chosen = remaining.pop(best_idx)
        recipe.append(dict(chosen, dose=chosen["dose"] * 0.75))
        best_score = scores[best_idx]
        yield sse_event("recipe", recipe=_public(recipe))

    # ---- overall strength scaling
    yield sse_event("stage", stage="scale", label="Finding the best overall strength")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5] + ([2.0] if pattern is not None else [])
    scores, _, _ = _evaluate([[Push(c["layer"], c["neuron"], c["dose"] * scale) for c in recipe] for scale in scales])
    best_idx = int(np.argmax(scores))

    for item in recipe:
        item["dose"] *= scales[best_idx]

    final = [Push(c["layer"], c["neuron"], c["dose"]) for c in recipe]
    yield sse_event("recipe", recipe=_public(recipe))

    # ---- proof on held-out prompts
    yield sse_event("stage", stage="proof", label="Trying the recipe on questions it hasn't seen")
    rows = []
    for pair in test:
        pid = eng.chat_ids(pair["prompt"])
        rows.extend([(pid, []), (pid, final)])

    texts = [""] * len(rows)
    for row_idx, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if row_idx is None:
            break
        texts[row_idx] += delta
        yield sse_event(
            "proof",
            index=row_idx // 2,
            kind="baseline" if row_idx % 2 == 0 else "steered",
            prompt=test[row_idx // 2]["prompt"],
            delta=delta,
        )

    extra: dict[str, float] = {}
    if pattern is not None:
        extra = {
            "hit_test0": sum(clean_hit(pattern, t) for t in texts[0::2]) / len(test),
            "hit_test": sum(clean_hit(pattern, t) for t in texts[1::2]) / len(test),
        }

    yield sse_event("result", recipe=_public(recipe), **extra)


def _cand(cand: dict[str, Any]) -> dict[str, Any]:
    """Format a candidate record for the streaming UI plan."""
    out = {k: cand[k] for k in ("layer", "neuron", "sign", "est", "lens") if k in cand}
    for key in ("dose", "dj", "dfluency"):
        if cand.get(key) is not None:
            out[key] = round(float(cand[key]), 4)
    return out


def _public(recipe: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize a recipe into the public JSON item shape."""
    return [
        {
            "layer": item["layer"],
            "neuron": item["neuron"],
            "dose": round(float(item["dose"]), 3),
            "dfluency": round(float(item.get("dfluency", 0)), 4),
            "lens": item.get("lens", []),
        }
        for item in recipe
    ]


def fix_instruction(behavior: str | None, feedback: str) -> str:
    """Build the system prompt instructing the model to generate corrected target replies."""
    head = f"Follow this in every reply, without mentioning it: {behavior.strip()}\n\n" if behavior and behavior.strip() else ""
    return (
        f'{head}An earlier reply was criticised like this: "{feedback.strip()}". '
        "Write your reply so that criticism no longer applies, without mentioning it."
    )


def refine(
    eng: Engine,
    recipe: Sequence[dict[str, Any]],
    *,
    feedback: str,
    prompt: str,
    bad: str,
    behavior: str | None = None,
    prompts: Sequence[str] = (),
    abl: bool = False,
    contrast: float = 0.5,
    fluency_budget: float = 0.35,
    target_tokens: int | None = None,
    n_new: int = 10,
    rounds: int = 3,
    cancel: threading.Event | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Iteratively refine an active neuron recipe from user feedback on one steered reply."""
    cur_recipe = [dict(c) for c in recipe]
    asks = [prompt] + [p for p in prompts if p.strip() and p.strip() != prompt.strip()]
    system = fix_instruction(behavior, feedback)
    cut = slice(0, target_tokens)

    def _push_set(items: Sequence[dict[str, Any]]) -> list[Push]:
        return [Push(c["layer"], c["neuron"], c["dose"]) for c in items if abs(c["dose"]) > 1e-6]

    # ---- 1. aim replies and current recipe baseline
    yield sse_event("stage", stage="aim", label="Writing the answer it should have given")
    rows = [(eng.chat_ids(p, system=system), []) for p in asks] + [
        (eng.chat_ids(p), _push_set(cur_recipe)) for p in asks[1:]
    ]
    texts = [""] * len(rows)
    for row_idx, delta in eng.generate(rows, max_new=72, cancel=cancel, abl=abl or None):
        if row_idx is None:
            break
        texts[row_idx] += delta
        if row_idx == 0:
            yield sse_event("aim", delta=delta)

    aims = [t.strip() for t in texts[: len(asks)]]
    now = [bad] + [t.strip() for t in texts[len(asks) :]]
    keep = [i for i, text in enumerate(aims) if text]

    if not keep or not aims[0]:
        yield sse_event(
            "result",
            recipe=_public(cur_recipe),
            changes=[],
            improved=False,
            note="It couldn't write a better answer from that. Try saying what you want instead.",
        )
        return

    def _rows_of(texts_list: Sequence[str]) -> list[tuple[list[int], int]]:
        out: list[tuple[list[int], int]] = []
        for idx in keep:
            pid = eng.chat_ids(asks[idx])
            out.append((pid + eng.encode(texts_list[idx])[cut], len(pid)))
        return out

    t_rows, b_rows = _rows_of(aims), _rows_of(now)
    nll0 = float(eng.reference_nll([[]], cancel)[0])
    floor = eng.score([(ids, start, []) for ids, start in b_rows], cancel, abl=[abl] * len(b_rows) if abl else None)
    ask_ids = [eng.chat_ids(asks[i]) for i in keep]

    def _looping(replies: Sequence[str]) -> float:
        return float(np.mean([max(0.0, 0.7 - variety(t)) for t in replies])) if replies else 0.0

    def _loops(sets: Sequence[list[dict[str, Any]]]) -> list[float]:
        gen_rows = [(pid, _push_set(r)) for r in sets for pid in ask_ids]
        replies = eng.generate_text(gen_rows, max_new=32, cancel=cancel, abl=[abl] * len(gen_rows) if abl else None)[0]
        n_asks = len(ask_ids)
        return [_looping(replies[k * n_asks : (k + 1) * n_asks]) for k in range(len(sets))]

    def _scores(sets: Sequence[list[dict[str, Any]]]) -> list[float]:
        out: list[float] = []
        for offset in range(0, len(sets), 6):
            part = [_push_set(r) for r in sets[offset : offset + 6]]
            score_rows = []
            for pushes in part:
                score_rows.extend((ids, start, pushes) for ids, start in t_rows)
                score_rows.extend((ids, start, pushes) for ids, start in b_rows)

            sc = eng.score(score_rows, cancel, abl=[abl] * len(score_rows) if abl else None)
            nll = eng.reference_nll(part, cancel, abl=[abl] * len(part) if abl else None)
            n_t = len(t_rows)

            for idx in range(len(part)):
                blk = sc[idx * 2 * n_t : (idx + 1) * 2 * n_t]
                jv = float(blk[:n_t].mean() - contrast * np.maximum(blk[n_t:], floor).mean())
                out.append(jv - max(0.0, float(nll[idx]) - nll0 - fluency_budget) * 2.0)

        return out

    # ---- 2. attribution around the active recipe
    yield sse_event("stage", stage="attribute", label="Working out which neurons to change")
    items = [(ids, start, 1.0) for ids, start in t_rows] + [(ids, start, -contrast) for ids, start in b_rows]
    grad: np.ndarray | None = None
    for idx, item in enumerate(items):
        step_grad = eng.attribute([item], cancel, pushes=_push_set(cur_recipe), abl=abl)
        grad = step_grad if grad is None else grad + step_grad
        yield sse_event("progress", stage="attribute", done=idx + 1, total=len(items))

    num_neurons = grad.shape[1]
    have = {(c["layer"], c["neuron"]) for c in cur_recipe}
    new_cands: list[dict[str, Any]] = []
    per_layer: dict[int, int] = {}

    for flat_idx in np.argsort(-np.abs(grad).ravel()):
        layer_idx, neuron_idx = int(flat_idx // num_neurons), int(flat_idx % num_neurons)
        if (layer_idx, neuron_idx) in have or per_layer.get(layer_idx, 0) >= 3:
            continue
        per_layer[layer_idx] = per_layer.get(layer_idx, 0) + 1
        new_cands.append({"layer": layer_idx, "neuron": neuron_idx, "sign": 1 if grad[layer_idx, neuron_idx] > 0 else -1})
        if len(new_cands) >= n_new:
            break

    for cand in new_cands:
        cand["lens"] = eng.lens(cand["layer"], cand["neuron"], 6)["up" if cand["sign"] > 0 else "down"]

    # ---- 3. candidate edits
    def _edits_for(items_list: Sequence[dict[str, Any]]) -> list[tuple[str, Any, float]]:
        edits: list[tuple[str, Any, float]] = []
        for item in items_list:
            for factor in (0.0, 0.5, 1.6):
                edits.append(("scale", (item["layer"], item["neuron"]), factor))

        edits.extend(("all", None, factor) for factor in (0.4, 0.7, 1.35))
        active_keys = {(x["layer"], x["neuron"]) for x in items_list}
        for cand in new_cands:
            if (cand["layer"], cand["neuron"]) not in active_keys:
                edits.extend(("add", cand, cand["sign"] * dose) for dose in (0.3, 0.6))
        return edits

    def _apply(items_list: Sequence[dict[str, Any]], edit: tuple[str, Any, float]) -> list[dict[str, Any]]:
        kind, ref, factor = edit
        updated = [dict(c) for c in items_list]

        if kind == "all":
            for item in updated:
                item["dose"] *= factor
        elif kind == "scale":
            updated = [dict(c, dose=c["dose"] * factor) if (c["layer"], c["neuron"]) == ref else c for c in updated]
            updated = [c for c in updated if abs(c["dose"]) > 1e-6]
        else:
            updated.append({"layer": ref["layer"], "neuron": ref["neuron"], "dose": factor, "lens": ref["lens"]})
        return updated

    yield sse_event("stage", stage="test", label="Testing changes to the neurons")
    loop0 = _loops([cur_recipe])[0]
    best = _scores([cur_recipe])[0]
    cur, improved = cur_recipe, False

    for rnd in range(rounds):
        edits = [edit for edit in _edits_for(cur) if _apply(cur, edit)]
        if not edits:
            break

        edit_scores: list[float] = []
        for offset in range(0, len(edits), 6):
            edit_scores.extend(_scores([_apply(cur, edit) for edit in edits[offset : offset + 6]]))
            yield sse_event("progress", stage="test", done=min(offset + 6, len(edits)), total=len(edits), round=rnd + 1)

        top = [int(i) for i in np.argsort(edit_scores)[::-1][:5]]
        top.extend(i for i, edit in enumerate(edits) if edit[0] == "all" and i not in top)
        loop_map = dict(zip(top, _loops([_apply(cur, edits[i]) for i in top]), strict=True))
        ok = [i for i in top if loop_map[i] <= max(loop0, 0.05) + 1e-9]
        if not ok:
            break

        best_idx = min(ok, key=lambda i: (round(loop_map[i], 1), -edit_scores[i]))
        if edit_scores[best_idx] <= best + 0.004 and loop_map[best_idx] >= loop0 - 0.05:
            break

        best = edit_scores[best_idx]
        cur = _apply(cur, edits[best_idx])
        loop0 = loop_map[best_idx]
        improved = True
        yield sse_event("recipe", recipe=_public(cur))

    yield sse_event(
        "result",
        recipe=_public(cur),
        improved=improved,
        note=None if improved else "Nothing it tried fixed that. Try describing the problem another way.",
    )
