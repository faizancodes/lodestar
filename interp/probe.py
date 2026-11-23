"""Probes: can you tell two kinds of prompt apart from the model's state, before it writes anything?

Give two sets of prompts (made-up questions and real ones, say). Lodestar reads the residual stream after every layer
at the tokens that end each prompt, one prompt at a time, and asks how well one direction separates the two sets. The
checks are built in, so a result can be trusted and repeated:

1. Separation, nested: 5-fold cross-validation where the layer is chosen inside each training fold (4-fold inner),
   never on the prompts it is scored on. AUC over the out-of-fold scores.
2. New groups: prompts can carry a group (a category). Each group is held out in turn and scored by a probe trained
   on everything else, so the number says whether the probe works on kinds of prompt it never saw.
3. Words only: the same tests with a classifier that sees only the prompt's tokens (words and pairs of words, plus
   how fragmented they are), no model state. If it does about as well, the probe is reading the wording.
4. Shuffled labels: the nested test with the labels shuffled, which should land near 0.5.
5. Caught: of side A, the share scored above 90% of side B (a 10% false-alarm rate).

The probe itself is a difference of means at the layer the nested folds picked most, with a logistic calibration
fitted on out-of-fold scores, so the live meter shows an honest probability. The state map places every prompt at
every layer: x along that layer's A-minus-B direction (each prompt placed by a direction trained without its fold, so
the gap shown is out-of-sample), y along the largest other variation.

States are read one prompt at a time: a batch left-pads shorter prompts, and on some models (Inkling) the padding
moves the states enough to change a score by several standard deviations.

Each model's probes live in DATA/probes/<model>/ as <id>.json (spec, metrics, map) and <id>.npz (vectors).
"""
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

from .directions import make_id, slug
from .engine import Cancelled
from .utils import atomic_write_json


FOLDS, INNER, SHUFFLES, SEED = 5, 4, 10, 0
MIN_PER_SIDE = 12


def _ev(t, **kw):
    return {"type": t, **kw}


# ------------------------------------------------------------------ statistics (numpy only)
def auc(s, y):
    """Area under the ROC curve: the chance a random A prompt scores above a random B one (ties count half)."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    order = s.argsort(kind="mergesort")
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = (np.bincount(inv, np.arange(1, len(s) + 1)) / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def folds(y, k, seed):
    """Stratified k-fold splits: [(train, test), ...]."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        f[idx] = np.arange(len(idx)) % k
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def dom(Xtr, ytr, Xte):
    """Difference-of-means direction per layer from the training rows; the test rows' scores [L, n_te], in units of
    the training rows' spread along it."""
    mu = Xtr.mean(0)
    d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
    sd = np.einsum("nlh,lh->ln", Xtr - mu, d).std(1, keepdims=True) + 1e-9
    return np.einsum("nlh,lh->ln", Xte - mu, d) / sd


def pick_layer(Xtr, ytr, seed):
    s = np.zeros((Xtr.shape[1], len(ytr)))
    for a, b in folds(ytr, INNER, seed):
        s[:, b] = dom(Xtr[a], ytr[a], Xtr[b])
    layers = range(1, Xtr.shape[1]) if Xtr.shape[1] > 1 else range(1)  # 0 is the embedding: same for every prompt end
    return max(layers, key=lambda layer_idx: auc(s[layer_idx], ytr))


def nested(X, y, splits, seed=SEED):
    """Out-of-sample scores for every test row, each from the layer its own training rows picked."""
    out, picked = np.full(len(y), np.nan), []
    for tr, te in splits:
        layer_idx = pick_layer(X[tr], y[tr], seed)
        picked.append(int(layer_idx))
        out[te] = dom(X[tr], y[tr], X[te])[layer_idx]
    return out, picked


def logistic(F, y, C=1.0, iters=25):
    """L2 logistic regression by Newton's method, solved through the n x n system so wide features stay cheap. The
    last column of F should be a constant (its weight is the intercept, lightly penalized). Returns the weights."""
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(F @ w, -30, 30)))
        g = C * F.T @ (p - y) + w
        W = np.clip(p * (1 - p), 1e-6, None) * C
        # (I + F' W F)^-1 g = g - F' (W^-1 + F F')^-1 F g
        step = g - F.T @ np.linalg.solve(np.diag(1 / W) + F @ F.T, F @ g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def word_features(tok, texts):
    """The prompt's tokens and pairs of tokens (present or not), and two counts: tokens, and tokens beyond one per
    word (how fragmented the words are, which flags made-up names)."""
    ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    grams = [set(x) | set(itertools.pairwise(x)) for x in ids]
    vocab = {g: i for i, g in enumerate(sorted({g for gs in grams for g in gs}, key=str))}
    B = np.zeros((len(texts), len(vocab)), np.float32)
    for r, gs in enumerate(grams):
        B[r, [vocab[g] for g in gs]] = 1
    counts = np.array([[len(x), len(x) - len(t.split())] for x, t in zip(ids, texts, strict=True)], np.float32)
    return B, counts


def words_only(B, counts, y, splits):
    """Out-of-sample scores from the words-only classifier (features fitted on each training split's vocabulary)."""
    out = np.full(len(y), np.nan)
    for tr, te in splits:
        keep = B[tr].any(0)
        mu, sd = counts[tr].mean(0), counts[tr].std(0) + 1e-6

        def _features(idx, keep=keep, mu=mu, sd=sd):
            return np.hstack([B[idx][:, keep], (counts[idx] - mu) / sd, np.full((len(idx), 1), 10.0)])

        out[te] = _features(te) @ logistic(_features(tr), y[tr].astype(float))
    return out


def calibrate(z, y):
    """p = sigmoid(a z + b) fitted on out-of-fold scores."""
    w = logistic(np.stack([z, np.full(len(z), 10.0)], 1), y.astype(float), C=100.0)
    return float(w[0]), float(w[1] * 10)


# ------------------------------------------------------------------ the spec
def parse_spec(spec):
    """{name, a: {label, prompts}, b: {label, prompts}}; a prompt is a string or {text, group}. Returns the name, the
    two labels and the items [{text, side, group}] with side 1 for A."""
    name = (spec.get("name") or "").strip() or "Probe"
    items, labels = [], []
    for side, key in ((1, "a"), (0, "b")):
        s = spec.get(key) or {}
        labels.append((s.get("label") or ("A" if side else "B")).strip())
        for p in s.get("prompts") or []:
            text, group = (p, None) if isinstance(p, str) else (p.get("text", ""), p.get("group"))
            if text.strip():
                items.append({"text": text.strip(), "side": side, "group": (group or "").strip() or None})

    for side, label in ((1, labels[0]), (0, labels[1])):
        if sum(it["side"] == side for it in items) < MIN_PER_SIDE:
            raise ValueError(f"Give at least {MIN_PER_SIDE} prompts for “{label}”.")
    return name, labels, items


# ------------------------------------------------------------------ running one
def read_states(eng, texts, cancel=None):
    """[n, L+1, H] float32: the residual after every layer, averaged over the tokens that end each prompt, read one
    prompt at a time. Yields progress events, then returns the array (generator return value)."""
    P = eng.prompt_end_len()
    out = []
    for i, t in enumerate(texts):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        out.append(np.asarray(eng.prompt_end_states([t], n_pos=P), np.float32)[0].mean(1))
        yield _ev("progress", stage="read", done=i + 1, total=len(texts))
    return np.stack(out)


def analyze(X, y, groups, B, counts):
    """Every check in the module docstring. Returns (metrics, nested scores, layer)."""
    kf = folds(y, FOLDS, SEED)
    s_cv, picked = nested(X, y, kf)
    w_cv = words_only(B, counts, y, kf)
    m = {"separation": auc(s_cv, y), "words_only": auc(w_cv, y), "layers_picked": picked}

    named = sorted({g for g in groups if g})
    held = [g for g in named if len(set(y[groups == g])) == 2]
    if len(held) >= 3:
        lo = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in held]
        s_lo, _ = nested(X, y, lo)
        w_lo = words_only(B, counts, y, lo)
        inside = np.isin(groups, held)
        m["new_groups"] = auc(s_lo[inside], y[inside])
        m["words_only_new_groups"] = auc(w_lo[inside], y[inside])
        m["per_group"] = {g: round(auc(s_lo[groups == g], y[groups == g]), 3) for g in held}

    rng = np.random.default_rng(SEED)
    null = []
    for i in range(SHUFFLES):
        perm = rng.permutation(y)
        s, _ = nested(X, perm, folds(perm, FOLDS, i), seed=i)
        null.append(auc(s, perm))
    m["shuffled"] = float(np.mean(null))

    thr = np.percentile(s_cv[y == 0], 90)
    m["caught"] = float((s_cv[y == 1] > thr).mean())

    layer = max(set(picked), key=lambda layer_idx: (picked.count(layer_idx), layer_idx))
    m["layer"] = int(layer)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, s_cv, layer


def state_map(X, y):
    """Per layer: every prompt's place (x out-of-fold along the A-minus-B direction, y along the largest other
    variation), and the vectors that place a new prompt the same way."""
    n, L1, H = X.shape
    kf = folds(y, FOLDS, SEED)
    mu, d, e = (np.zeros((L1, H), np.float32) for _ in range(3))
    sx, sy = np.ones(L1, np.float32), np.ones(L1, np.float32)
    pts = np.zeros((L1, n, 2), np.float32)
    prev = None

    for layer_idx in range(L1):
        x = X[:, layer_idx]
        mu[layer_idx] = x.mean(0)
        dl = x[y == 1].mean(0) - x[y == 0].mean(0)
        d[layer_idx] = dl / (np.linalg.norm(dl) + 1e-9)

        xo = np.zeros(n)
        for tr, te in kf:
            dd = x[tr][y[tr] == 1].mean(0) - x[tr][y[tr] == 0].mean(0)
            xo[te] = (x[te] - x[tr].mean(0)) @ (dd / (np.linalg.norm(dd) + 1e-9))

        c = x - mu[layer_idx]
        r = c - np.outer(c @ d[layer_idx], d[layer_idx])
        e[layer_idx] = np.linalg.svd(r, full_matrices=False)[2][0]
        yo = c @ e[layer_idx]
        if prev is not None and yo.std() > 1e-6 and prev.std() > 1e-6 and np.corrcoef(yo, prev)[0, 1] < 0:
            e[layer_idx], yo = -e[layer_idx], -yo
        prev = yo

        sx[layer_idx], sy[layer_idx] = max(xo.std(), 1e-6), max(yo.std(), 1e-6)
        pts[layer_idx] = np.stack([xo / sx[layer_idx], yo / sy[layer_idx]], 1)

    return pts, {"mu": mu, "d": d, "e": e, "sx": sx, "sy": sy}


def run(eng, lib, spec, cancel=None):
    """Read, check and save a probe. Yields events and ends with `result`: the saved probe (as `Library.get`)."""
    name, labels, items = parse_spec(spec)
    texts = [it["text"] for it in items]

    yield _ev("stage", stage="read", label="Reading the model's state at the end of each prompt")
    X = yield from read_states(eng, texts, cancel)

    yield _ev("stage", stage="check", label="Checking it")
    y = np.array([it["side"] for it in items])
    groups = np.array([it["group"] or "" for it in items])
    B, counts = word_features(eng.tok, texts)
    metrics, _s_cv, layer = analyze(X, y, groups, B, counts)

    yield _ev("stage", stage="map", label="Mapping every layer")
    pts, vec = state_map(X, y)

    # the live probe: all prompts, at the layer the folds picked, calibrated on out-of-fold scores at that layer
    oof = np.zeros(len(y))
    for tr, te in folds(y, FOLDS, SEED):
        oof[te] = dom(X[tr], y[tr], X[te])[layer]
    a, b = calibrate(oof, y)
    sd = float(((X[:, layer] - vec["mu"][layer]) @ vec["d"][layer]).std() + 1e-9)

    entry = {"id": make_id(name, {e["id"] for e in lib.index()}), "name": name, "labels": labels,
             "model": eng.model_id, "created": time.time(), "metrics": metrics,
             "items": items,
             "map": np.round(pts, 2).tolist()}
    lib.save(entry, {**vec, "sd": np.float32(sd), "a": np.float32(a), "b": np.float32(b), "layer": np.int32(layer)})
    yield _ev("result", probe=entry)


def score(eng, probe_vec, text):
    """A new prompt: the probability it reads as side A, and its place on the map at every layer."""
    st = np.asarray(eng.prompt_end_states([text], n_pos=eng.prompt_end_len()), np.float32)[0].mean(1)
    c = st - probe_vec["mu"]
    xs = (c * probe_vec["d"]).sum(1)
    layer_idx = int(probe_vec["layer"])
    z = xs[layer_idx] / float(probe_vec["sd"])
    p = 1 / (1 + np.exp(-(float(probe_vec["a"]) * z + float(probe_vec["b"]))))
    path = np.stack([xs / probe_vec["sx"], (c * probe_vec["e"]).sum(1) / probe_vec["sy"]], 1)
    return {"p": round(float(p), 4), "z": round(float(z), 3), "path": np.round(path, 2).tolist()}


# ------------------------------------------------------------------ saved probes
class Library:
    def __init__(self, root, eng):
        self.root, self.eng = Path(root), eng

    @property
    def dir(self):
        return self.root / slug(self.eng.model_id)

    def _f(self, pid, ext):
        if not re.fullmatch(r"[a-z0-9-]{1,60}", pid):
            raise KeyError(pid)
        return self.dir / f"{pid}.{ext}"

    def index(self):
        """Saved probes for the loaded model, newest first, without their items and map."""
        if not self.dir.exists():
            return []
        out = [json.loads(f.read_text()) for f in self.dir.glob("*.json")]
        return sorted(({k: e[k] for k in ("id", "name", "labels", "created", "metrics")} for e in out),
                      key=lambda e: -e["created"])

    def get(self, pid):
        f = self._f(pid, "json")
        if not f.exists():
            raise KeyError(pid)
        return json.loads(f.read_text())

    def vectors(self, pid):
        with np.load(self._f(pid, "npz")) as archive:
            return dict(archive)

    def save(self, entry, vec):
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._f(entry["id"], "npz"), **vec)
        atomic_write_json(self._f(entry["id"], "json"), entry, indent=None, ensure_ascii=False)

    def delete(self, pid):
        for ext in ("json", "npz"):
            self._f(pid, ext).unlink(missing_ok=True)
