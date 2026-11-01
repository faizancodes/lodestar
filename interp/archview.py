"""The Architecture page's data: the loaded model's real structure, every weight tensor with its shape, and what text
does as it passes through, layer by layer, head by head and neuron by neuron.

structure(eng) describes the model once (cached per model): its config, the embedding and unembedding, and for every
layer the blocks it's made of, their tensors, each attention head's output weight norm and each neuron's.
run(eng, ids) sends text through and keeps, for every layer and token, how big the residual stream is, how much
attention and the MLP each write into it, how much each head writes, every neuron's activation, and the logit lens.
"""
import base64
import re

import numpy as np
import torch

LENS_K = 3
MAX_TOKENS = 192
REPLY_TOKENS = 96


def _cfg(eng):
    c = eng.model.config
    return c.get_text_config() if hasattr(c, "get_text_config") else c


def attention_of(layer):
    """(name, module, its output projection) for a layer's attention block, whatever the family calls it."""
    for name in ("self_attn", "linear_attn", "attn", "attention", "mixer"):
        m = getattr(layer, name, None)
        if m is None:
            continue
        for p in ("o_proj", "out_proj", "dense", "c_proj"):
            out = getattr(m, p, None)
            if isinstance(out, torch.nn.Linear):
                return name, m, out
        return name, m, None
    return None, None, None


def _head_dim(eng, mod, kind):
    c = _cfg(eng)
    if kind == "linear":
        return getattr(mod, "head_v_dim", None) or getattr(c, "linear_value_head_dim", None)
    return getattr(mod, "head_dim", None) or getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads


def _tensors(mod, prefix=""):
    out = []
    for n, p in mod.named_parameters(prefix=prefix, recurse=True):
        w = p.detach()
        out.append({"name": n, "shape": list(w.shape), "params": int(w.numel()),
                    "rms": round(float(w.float().pow(2).mean().sqrt()), 5)})
    return out


def structure(eng):
    c = _cfg(eng)
    kinds = eng.arch()["attention"]
    emb = eng.model.get_input_embeddings()
    rope = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
    theta = getattr(c, "rope_theta", None) or (rope.get("rope_theta") if isinstance(rope, dict) else None)

    layers = []
    with torch.no_grad():
        for l, layer in enumerate(eng.layers):
            kind = kinds[l]
            aname, amod, out = attention_of(layer)
            hd = _head_dim(eng, amod, kind) if amod is not None else None
            heads = []
            if out is not None and hd:
                W = out.weight.float()  # [H, heads * head_dim]
                n = W.shape[1] // hd
                heads = [round(float(x), 4) for x in W.view(W.shape[0], n, hd).norm(dim=(0, 2))]

            blocks = []
            for name, child in layer.named_children():
                t = _tensors(child, name)
                role = "attention" if name == aname else "mlp" if name == "mlp" else "norm" if "norm" in name else "other"
                blocks.append({"name": name, "role": role, "type": type(child).__name__,
                               "params": sum(x["params"] for x in t), "tensors": t})

            layers.append({"index": l, "attention": kind, "heads": heads, "head_dim": hd, "blocks": blocks,
                           "params": sum(b["params"] for b in blocks),
                           "resid": round(float(eng.resid_p50[l]), 3)})

        emb_t = _tensors(emb, "embed_tokens")
        norm_t = _tensors(eng.final_norm, "norm")
        head_t = [] if getattr(c, "tie_word_embeddings", False) else _tensors(eng.lm_head, "lm_head")

    kv = getattr(c, "num_key_value_heads", None)
    return {
        "model": eng.model_id,
        "type": getattr(c, "model_type", ""),
        "dtype": str(eng.model.dtype).replace("torch.", ""),
        "params": eng.n_params,
        "hidden": int(c.hidden_size),
        "neurons": int(eng.dnorm.shape[1]),
        "vocab": int(eng.lm_head.weight.shape[0]),
        "activation": getattr(c, "hidden_act", None) or getattr(c, "hidden_activation", None),
        "context": getattr(c, "max_position_embeddings", None),
        "rope_theta": theta,
        "norm_eps": getattr(c, "rms_norm_eps", None) or getattr(c, "layer_norm_eps", None),
        "heads": getattr(c, "num_attention_heads", None), "kv_heads": kv,
        "linear_heads": getattr(c, "linear_num_value_heads", None),
        "linear_k_heads": getattr(c, "linear_num_key_heads", None),
        "tied": bool(getattr(c, "tie_word_embeddings", False)),
        "experts": getattr(c, "num_experts", None) or None,
        "embedding": {"tensors": emb_t, "params": sum(t["params"] for t in emb_t)},
        "final_norm": {"tensors": norm_t, "params": sum(t["params"] for t in norm_t)},
        "unembedding": {"tensors": head_t, "params": sum(t["params"] for t in head_t)},
        "layers": layers,
    }


def neuron_norms(eng, l):
    """Each neuron's down-projection column norm in layer l, the size of what it writes per unit of activation."""
    return [round(float(x), 4) for x in eng.dnorm[l].cpu()]


def run(eng, ids, n_prompt=None):
    """One forward pass over ids (the prompt, then n_prompt onward the model's reply to it), keeping what each layer
    did at every token. Returns the summary sent to the page; the neuron activations stay on eng.flow for acts()."""
    ids = list(ids)
    n_prompt = len(ids) if n_prompt is None else n_prompt
    if len(ids) > MAX_TOKENS:  # keep the end: the reply, and as much of the prompt before it as fits
        n_prompt = max(0, n_prompt - (len(ids) - MAX_TOKENS))
        ids = ids[-MAX_TOKENS:]
    L = len(eng.layers)
    heads, attn_w, mlp_w, handles = {}, {}, {}, []

    def pre(l, hd):
        def hook(mod, inp):
            x = inp[0][0].float()  # [T, heads * head_dim]
            heads[l] = x.view(x.shape[0], -1, hd).norm(dim=-1).cpu()
        return hook

    def write(store, l):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[l] = o[0].float().norm(dim=-1).cpu()
        return hook

    kinds = eng.arch()["attention"]
    with eng.lock:
        for l, layer in enumerate(eng.layers):
            _, amod, out = attention_of(layer)
            if out is not None:
                hd = _head_dim(eng, amod, kinds[l])
                if hd:
                    handles.append(out.register_forward_pre_hook(pre(l, hd)))
                handles.append(out.register_forward_hook(write(attn_w, l)))
            handles.append(layer.mlp.down_proj.register_forward_hook(write(mlp_w, l)))

        eng._cap_acts, eng._cap_resid = {}, {}
        try:
            with torch.inference_mode():
                eng.model(input_ids=torch.tensor([ids], device=eng.dev), use_cache=False, logits_to_keep=1)
                resid = [eng._cap_resid[l].float().norm(dim=-1).cpu() for l in range(L)]
                lens_ids, lens_p = [], []
                for l in range(L):
                    p = torch.softmax(eng.lm_head(eng.final_norm(eng._cap_resid[l])).float(), -1)
                    top = p.topk(LENS_K, dim=-1)
                    lens_ids.append(top.indices.cpu().numpy())
                    lens_p.append(top.values.cpu().numpy())
                    del p
            acts = torch.stack([eng._cap_acts[l] for l in range(L)]).half().numpy()  # [L, T, N]
        finally:
            eng._cap_acts = eng._cap_resid = None
            for h in handles:
                h.remove()

    eng.flow = {"ids": ids, "acts": acts}
    r2 = lambda x: [round(float(v), 2) for v in x]
    tokens = eng.token_strs(ids)
    dec = {}
    def word(i):
        if i not in dec:
            dec[i] = eng.tok.decode([int(i)])
        return dec[i]

    return {
        "tokens": tokens,
        "n_prompt": n_prompt,
        "special": [i in eng.tok.all_special_ids or bool(re.fullmatch(r"<\|?/?\w+\|?>", t)) for i, t in zip(ids, tokens)],
        "resid": [r2(x) for x in resid],
        "attn": [r2(attn_w[l]) if l in attn_w else [] for l in range(L)],
        "mlp": [r2(mlp_w[l]) for l in range(L)],
        "heads": [[r2(h) for h in heads[l].tolist()] if l in heads else [] for l in range(L)],
        "lens": [[[{"tok": word(i), "p": round(float(p), 3)} for i, p in zip(lens_ids[l][t], lens_p[l][t])]
                  for t in range(len(ids))] for l in range(L)],
    }


def acts(eng, t):
    """Every neuron's impact at token t of the last run (activation x its down-column norm / the layer's typical
    residual size, as Steer ranks neurons), signed and quantised to int8 per layer, base64: [L, N] row-major.
    Returns None before any run."""
    f = getattr(eng, "flow", None)
    if f is None or not 0 <= t < len(f["ids"]):
        return None
    a = torch.from_numpy(f["acts"][:, t, :].astype(np.float32))  # [L, N]
    imp = a * (eng.dnorm / eng.resid_p50[:, None]).cpu()
    peak = imp.abs().amax(dim=1).clamp(min=1e-8)  # [L]
    q = (imp / peak[:, None] * 127).round().clamp(-127, 127).to(torch.int8).numpy()
    top = []
    flat = imp.abs().flatten()
    for i in flat.topk(min(24, flat.numel())).indices.tolist():
        l, n = divmod(i, imp.shape[1])
        top.append({"layer": l, "neuron": n, "impact": round(float(imp[l, n]), 3), "act": round(float(a[l, n]), 3)})
    return {"layers": imp.shape[0], "neurons": imp.shape[1], "peak": [round(float(x), 4) for x in peak],
            "q": base64.b64encode(q.tobytes()).decode(), "top": top}


def neuron_trace(eng, l, n):
    """Neuron n of layer l's activation at every token of the last run, or None."""
    f = getattr(eng, "flow", None)
    if f is None:
        return None
    return [round(float(x), 3) for x in f["acts"][l, :, n].astype(np.float32)]
