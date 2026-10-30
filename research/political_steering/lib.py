"""Political concept directions in Qwen3-8B: extraction, validation, steering, judging.

Builds on interp.engine (the model, chat templating, batching, KV-cached generation) and follows interp.abl's
recipe for a direction: difference in means of the residual stream at the tokens the chat template puts after the
request, normalized per token and averaged, with split-half agreement to check it is real.

Two kinds of contrast:
- persona: the same question under two opposite system prompts (where the model "decides" its stance), as in
  Contrastive Activation Addition (Rimsky et al. 2024) and the Tune view's system-prompt targets.
- topic: two matched sets of questions with no system prompt (as abliteration does for refusal), e.g. questions the
  Chinese government censors against matched questions about other governments.

Steering adds k * mag * u at one layer's output, at every position, where u is the unit direction and mag the mean
size of the contrast at that layer, so k = 1 moves the residual by exactly the average persona difference.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from interp.engine import Engine

MODEL = str(Path.home() / "Library/Application Support/Auto Interp/models/Qwen3-8B")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def load():
    eng = Engine()
    eng.load(MODEL)
    eng.n_pos = eng.prompt_end_len()
    return eng


# ------------------------------------------------------------------ reading
def end_states(eng, msgs_list, budget=6144):
    """Residual stream after every decoder layer at the template's ending tokens, normalized-then-not: returns
    [n, L, P, H] float32 numpy, index l = output of layer l (the final norm is not applied)."""
    P = eng.n_pos
    ids_list = [eng.messages_ids(m) for m in msgs_list]
    out = [None] * len(ids_list)
    L = len(eng.layers)
    for bi in eng._batches([(x,) for x in ids_list], budget):
        seqs = [ids_list[i] for i in bi]
        T = max(map(len, seqs))
        pad = eng.tok.pad_token_id
        ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=eng.dev)
        mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=eng.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        cap = {}
        hs = [eng.layers[l].register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -P:].float().cpu()))
            for l in range(L)]
        try:
            with torch.inference_mode():
                eng.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        st = torch.stack([cap[l] for l in range(L)], 1).numpy()  # [B, L, P, H]
        for j, i in enumerate(bi):
            out[i] = st[j]
    return np.stack(out)


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9), n[..., 0]


def diff_dir(pos, neg):
    """interp.abl.diff_dirs on two sets: per-token unit differences averaged. pos, neg: [n, L, P, H].
    Returns unit [L, H] and the mean contrast size [L] (norm of the mean difference, averaged over tokens)."""
    unit_p, mag_p = _unit(pos.mean(0) - neg.mean(0))
    unit, _ = _unit(unit_p.mean(1))
    return unit, mag_p.mean(1)


def proj(states, u):
    """Mean over the ending tokens of each prompt's projection on u: states [n, L, P, H], u [L, H] -> [n, L]."""
    return np.einsum("nlph,lh->nlp", states, u).mean(-1)


def auc(a, b):
    """P(score of an a > score of a b), ties half: a and b are 1-D."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def analyze(pos, neg, pos_ho, neg_ho, pairs=True):
    """Split-half agreement and held-out separation per layer. With pairs (persona contrasts, where pos[i] and
    neg[i] share a question), the halves split by question."""
    n = len(pos)
    idx = np.arange(n)
    a, b = idx[: n // 2], idx[n // 2:]
    ua, _ = diff_dir(pos[a], neg[a] if pairs else neg)
    ub, _ = diff_dir(pos[b], neg[b] if pairs else neg)
    u, mag = diff_dir(pos, neg)
    cos = (ua * ub).sum(-1)
    pp, pn = proj(pos_ho, u), proj(neg_ho, u)
    L = u.shape[0]
    aucs = np.array([auc(pp[:, l], pn[:, l]) for l in range(L)])
    # separation in units of pooled std: how far apart the held-out sets sit along u
    d = (pp.mean(0) - pn.mean(0)) / np.sqrt(0.5 * (pp.var(0) + pn.var(0)) + 1e-9)
    return {"u": u, "mag": mag, "cos_split": cos, "auc": aucs, "dprime": d}


# ------------------------------------------------------------------ steering
class Steer:
    """Adds alpha[r] * vec to layer `layer`'s output for each batch row r (at every position) while active."""

    def __init__(self, eng):
        self.eng = eng
        self.h = None
        self.add = None  # [B, H] on device

    def set(self, layer, vecs):
        """vecs: list (per row) of numpy [H] or None. Several (layer, vec) can be combined by passing a dict."""
        self.clear()
        specs = layer if isinstance(layer, dict) else {layer: vecs}
        self.h = []
        for l, vs in specs.items():
            H = self.eng.layers[0].mlp.down_proj.out_features
            add = torch.stack([torch.zeros(H) if v is None else torch.as_tensor(np.asarray(v), dtype=torch.float32)
                               for v in vs]).to(self.eng.dev, self.eng.model.dtype)

            def hook(m, i, o, add=add):
                h = o[0] if isinstance(o, tuple) else o
                h = h + add[: h.shape[0], None, :]
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

            self.h.append(self.eng.layers[l].register_forward_hook(hook))

    def clear(self):
        for h in self.h or []:
            h.remove()
        self.h = None


def generate(eng, steer, prompts, layer_vecs, max_new=120, system=None, temperature=0.0):
    """prompts: list of str (one per row); layer_vecs: {layer: [vec or None per row]} or None. Greedy by default."""
    rows = [(eng.chat_ids(p, system=system), []) for p in prompts]
    if layer_vecs:
        steer.set(layer_vecs, None)
    try:
        texts, _ = eng.generate_text(rows, max_new=max_new, temperature=temperature)
    finally:
        steer.clear()
    return [t.strip() for t in texts]


def generate_chunked(eng, steer, prompts, vecs_by_layer, max_new=120, chunk=16, **kw):
    out = []
    for s in range(0, len(prompts), chunk):
        sl = slice(s, s + chunk)
        lv = {l: v[sl] for l, v in vecs_by_layer.items()} if vecs_by_layer else None
        out += generate(eng, steer, prompts[sl], lv, max_new=max_new, **kw)
    return out


def ref_nll(eng, steer, layer_vecs):
    """Mean NLL of interp.engine's neutral reference passage under each row's steering: fluency cost."""
    from interp.engine import REFERENCE_TEXT
    ids = eng.tok(REFERENCE_TEXT).input_ids
    n = len(next(iter(layer_vecs.values())))
    steer.set(layer_vecs, None)
    try:
        return -eng.score([(ids, 1, [])] * n)
    finally:
        steer.clear()


# ------------------------------------------------------------------ judging
DIGITS = [str(i) for i in range(1, 8)]


def judge(eng, items, rubric, budget=6144):
    """The unsteered model as a judge: expected rating 1..7 from its next-token distribution over the digits.
    items: list of (question, reply). rubric: what 1 and 7 mean."""
    dig = [eng.tok(d, add_special_tokens=False).input_ids[0] for d in DIGITS]
    rows = []
    for q, a in items:
        p = (f"Here is a question and a reply.\n\nQUESTION: {q}\n\nREPLY: {a[:1800]}\n\n"
             f"Rate the REPLY on this scale: {rubric}\nAnswer with a single digit from 1 to 7 and nothing else.")
        rows.append(eng.chat_ids(p))
    out = np.zeros(len(rows))
    for bi in eng._batches([(x,) for x in rows], budget):
        seqs = [rows[i] for i in bi]
        T = max(map(len, seqs))
        pad = eng.tok.pad_token_id
        ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=eng.dev)
        mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=eng.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            lg = eng.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                           logits_to_keep=1).logits[:, -1].float()
        p = torch.softmax(lg[:, dig], -1).cpu().numpy()
        for j, i in enumerate(bi):
            out[i] = float((p[j] * np.arange(1, 8)).sum())
    return out


COHERENCE = ("1 = gibberish, broken or endlessly repeating text; 4 = understandable but rambling or off-topic; "
             "7 = fluent, coherent and on-topic.")


def save_json(name, obj):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(type(x))
    (OUT / name).write_text(json.dumps(obj, indent=1, default=conv, ensure_ascii=False))
