"""Architecture view data: model structure, parameter tensors, and per-token layer/head/neuron activity.

``structure(eng)`` describes the loaded model once (cached per model): its configuration, embedding and unembedding
tensors, and each layer's blocks, attention head output norms, and residual stream scale.
``run(eng, ids)`` executes one forward pass and records per-layer/per-token residual norms, attention/MLP write norms,
attention head write norms, MLP neuron activations, and the logit lens.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .engine import SparseActs
from .utils import text_config_of, unwrap_tensor


if TYPE_CHECKING:
    from .engine import Engine


LENS_K = 3
MAX_TOKENS = 192
REPLY_TOKENS = 96
LIVE_LENS = 8  # layers the logit lens reads at each token while the model writes
_SPECIAL_TAG_RE = re.compile(r"<\|?/?\w+\|?>")


def _round2(values: Iterable[float]) -> list[float]:
    return [round(float(v), 2) for v in values]


def attention_of(layer: torch.nn.Module) -> tuple[str | None, torch.nn.Module | None, torch.nn.Linear | None]:
    """Return ``(attr_name, module, output_projection)`` for a layer's attention block across model families."""
    for name in ("self_attn", "linear_attn", "attn", "attention", "mixer"):
        mod = getattr(layer, name, None)
        if mod is None:
            continue
        for proj_name in ("o_proj", "out_proj", "dense", "c_proj"):
            out = getattr(mod, proj_name, None)
            if isinstance(out, torch.nn.Linear):
                return name, mod, out
        return name, mod, None
    return None, None, None


def _head_dim(eng: Engine, mod: torch.nn.Module, kind: str) -> int | None:
    cfg = text_config_of(eng.model.config)
    if kind == "linear":
        return getattr(mod, "head_v_dim", None) or getattr(cfg, "linear_value_head_dim", None)
    return (
        getattr(mod, "head_dim", None)
        or getattr(cfg, "head_dim", None)
        or cfg.hidden_size // cfg.num_attention_heads
    )


def _tensors(mod: torch.nn.Module, prefix: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, param in mod.named_parameters(prefix=prefix, recurse=True):
        weight = param.detach()
        out.append(
            {
                "name": name,
                "shape": list(weight.shape),
                "params": int(weight.numel()),
                "rms": round(float(weight.float().pow(2).mean().sqrt()), 5),
            }
        )
    return out


def structure(eng: Engine) -> dict[str, Any]:
    """Inspect the loaded model's architecture, parameter counts, and per-head output weight norms."""
    cfg = text_config_of(eng.model.config)
    kinds = eng.arch()["attention"]
    emb = eng.model.get_input_embeddings()
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    theta = getattr(cfg, "rope_theta", None) or (rope.get("rope_theta") if isinstance(rope, dict) else None)

    layers: list[dict[str, Any]] = []
    with eng.lock, torch.no_grad():  # Weight norms run on GPU; serialize device access.
        for layer_idx, layer in enumerate(eng.layers):
            kind = kinds[layer_idx]
            attn_name, attn_mod, out_proj = attention_of(layer)
            head_dim = _head_dim(eng, attn_mod, kind) if attn_mod is not None else None
            heads: list[float] = []
            if out_proj is not None and head_dim:
                weight = out_proj.weight.float()  # [H, heads * head_dim]
                n_heads = weight.shape[1] // head_dim
                heads = [round(float(x), 4) for x in weight.view(weight.shape[0], n_heads, head_dim).norm(dim=(0, 2))]

            blocks: list[dict[str, Any]] = []
            for name, child in layer.named_children():
                tensors = _tensors(child, name)
                if name == attn_name:
                    role = "attention"
                elif name == "mlp":
                    role = "mlp"
                elif "norm" in name:
                    role = "norm"
                else:
                    role = "other"
                blocks.append(
                    {
                        "name": name,
                        "role": role,
                        "type": type(child).__name__,
                        "params": sum(t["params"] for t in tensors),
                        "tensors": tensors,
                    }
                )

            layers.append(
                {
                    "index": layer_idx,
                    "attention": kind,
                    "heads": heads,
                    "head_dim": head_dim,
                    "blocks": blocks,
                    "params": sum(b["params"] for b in blocks),
                    "resid": round(float(eng.resid_p50[layer_idx]), 3),
                }
            )

        emb_t = _tensors(emb, "embed_tokens")
        norm_t = _tensors(eng.final_norm, "norm")
        head_t = [] if getattr(cfg, "tie_word_embeddings", False) else _tensors(eng.lm_head, "lm_head")

    return {
        "model": eng.model_id,
        "type": getattr(cfg, "model_type", ""),
        "dtype": str(eng.model.dtype).replace("torch.", ""),
        "params": eng.n_params,
        "hidden": int(cfg.hidden_size),
        "neurons": eng.num_neurons,
        "vocab": int(eng.lm_head.weight.shape[0]),
        "activation": getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None),
        "context": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": theta,
        "norm_eps": getattr(cfg, "rms_norm_eps", None) or getattr(cfg, "layer_norm_eps", None),
        "heads": getattr(cfg, "num_attention_heads", None),
        "kv_heads": getattr(cfg, "num_key_value_heads", None),
        "linear_heads": getattr(cfg, "linear_num_value_heads", None),
        "linear_k_heads": getattr(cfg, "linear_num_key_heads", None),
        "tied": bool(getattr(cfg, "tie_word_embeddings", False)),
        "experts": getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None) or None,
        "embedding": {"tensors": emb_t, "params": sum(t["params"] for t in emb_t)},
        "final_norm": {"tensors": norm_t, "params": sum(t["params"] for t in norm_t)},
        "unembedding": {"tensors": head_t, "params": sum(t["params"] for t in head_t)},
        "layers": layers,
    }


def neuron_norms(eng: Engine, layer: int) -> list[float]:
    """Return each neuron's down-projection column norm in ``layer``."""
    with eng.lock:
        norms = eng.dnorm[layer].cpu()
    return [round(float(x), 4) for x in norms]


def run(eng: Engine, ids: Sequence[int], n_prompt: int | None = None) -> dict[str, Any]:
    """Run one forward pass over ``ids`` and record per-layer/per-token activity.

    Stores raw neuron activations on ``eng.flow`` for subsequent :func:`acts` and :func:`neuron_trace` queries.
    """
    token_ids = list(ids)
    n_prompt = len(token_ids) if n_prompt is None else n_prompt
    if len(token_ids) > MAX_TOKENS:
        # Keep the tail: the reply, plus as much of the prompt before it as fits.
        n_prompt = max(0, n_prompt - (len(token_ids) - MAX_TOKENS))
        token_ids = token_ids[-MAX_TOKENS:]

    num_layers = eng.num_layers
    heads: dict[int, torch.Tensor] = {}
    attn_w: dict[int, torch.Tensor] = {}
    mlp_w: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _pre_hook(layer_idx: int, head_dim: int):
        def _hook(_mod: torch.nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
            x = inp[0][0].float()  # [T, heads * head_dim]
            heads[layer_idx] = x.view(x.shape[0], -1, head_dim).norm(dim=-1).cpu()

        return _hook

    def _write_hook(store: dict[int, torch.Tensor], layer_idx: int):
        def _hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            store[layer_idx] = unwrap_tensor(out)[0].float().norm(dim=-1).cpu()

        return _hook

    kinds = eng.arch()["attention"]
    with eng.lock:
        try:
            for layer_idx, layer in enumerate(eng.layers):
                _, attn_mod, out_proj = attention_of(layer)
                if out_proj is not None:
                    head_dim = _head_dim(eng, attn_mod, kinds[layer_idx])
                    if head_dim:
                        handles.append(eng.hook_sub(layer_idx, out_proj, _pre_hook(layer_idx, head_dim), pre=True))
                    handles.append(eng.hook_sub(layer_idx, out_proj, _write_hook(attn_w, layer_idx)))
                handles.append(eng.hook_sub(layer_idx, eng.mlp_writer(layer_idx), _write_hook(mlp_w, layer_idx)))

            eng._cap_acts = {}
            eng._cap_resid = {}
            eng._cap_read = {} if eng.ink else None
            with torch.inference_mode():
                eng.model(input_ids=torch.tensor([token_ids], device=eng.dev), use_cache=False, logits_to_keep=1)
                resid = [eng._cap_resid[layer_idx].float().norm(dim=-1).cpu() for layer_idx in range(num_layers)]
                lens_ids: list[np.ndarray] = []
                lens_probs: list[np.ndarray] = []
                for layer_idx in range(num_layers):
                    probs = torch.softmax(eng._head(eng.final_norm(eng.lens_state(layer_idx))), dim=-1)
                    top = probs.topk(LENS_K, dim=-1)
                    lens_ids.append(top.indices.cpu().numpy())
                    lens_probs.append(top.values.cpu().numpy())
                    del probs
            if eng.ink:
                captured_acts: Any = SparseActs([eng._cap_acts[layer_idx] for layer_idx in range(num_layers)], eng.num_neurons)
            else:
                captured_acts = torch.stack([eng._cap_acts[layer_idx] for layer_idx in range(num_layers)]).half().numpy()
        finally:
            eng._cap_acts = None
            eng._cap_resid = None
            eng._cap_read = None
            for handle in handles:
                handle.remove()

    eng.flow = {"ids": token_ids, "acts": captured_acts}
    tokens = eng.token_strs(token_ids)
    decoded: dict[int, str] = {}

    def _word(tok_id: int) -> str:
        key = int(tok_id)
        if key not in decoded:
            decoded[key] = eng.tok.decode([key])
        return decoded[key]

    return {
        "tokens": tokens,
        "n_prompt": n_prompt,
        "special": [
            tok_id in eng.tok.all_special_ids or bool(_SPECIAL_TAG_RE.fullmatch(tok_str))
            for tok_id, tok_str in zip(token_ids, tokens, strict=True)
        ],
        "resid": [_round2(x) for x in resid],
        "attn": [_round2(attn_w[layer_idx]) if layer_idx in attn_w else [] for layer_idx in range(num_layers)],
        "mlp": [_round2(mlp_w[layer_idx]) for layer_idx in range(num_layers)],
        "heads": [
            [_round2(head_vals) for head_vals in heads[layer_idx].tolist()] if layer_idx in heads else []
            for layer_idx in range(num_layers)
        ],
        "lens": [
            [
                [
                    {"tok": _word(i), "p": round(float(p), 3)}
                    for i, p in zip(lens_ids[layer_idx][t], lens_probs[layer_idx][t], strict=True)
                ]
                for t in range(len(token_ids))
            ]
            for layer_idx in range(num_layers)
        ],
    }


def acts(eng: Engine, t: int) -> dict[str, Any] | None:
    """Return every neuron's signed int8-quantized impact at token ``t`` of the last run, or ``None``."""
    flow = getattr(eng, "flow", None)
    if flow is None or not 0 <= t < len(flow["ids"]):
        return None
    raw_acts = torch.from_numpy(flow["acts"][:, t, :].astype(np.float32))  # [L, N]
    with eng.lock:
        scale = (eng.dnorm / eng.resid_p50[:, None]).cpu()
    impact = raw_acts * scale
    peak = impact.abs().amax(dim=1).clamp(min=1e-8)  # [L]
    quantized = (impact / peak[:, None] * 127).round().clamp(-127, 127).to(torch.int8).numpy()
    top: list[dict[str, Any]] = []
    flat = impact.abs().flatten()
    for idx in flat.topk(min(24, flat.numel())).indices.tolist():
        layer_idx, neuron_idx = divmod(idx, impact.shape[1])
        top.append(
            {
                "layer": layer_idx,
                "neuron": neuron_idx,
                "impact": round(float(impact[layer_idx, neuron_idx]), 3),
                "act": round(float(raw_acts[layer_idx, neuron_idx]), 3),
            }
        )
    return {
        "layers": impact.shape[0],
        "neurons": impact.shape[1],
        "peak": [round(float(x), 4) for x in peak],
        "q": base64.b64encode(quantized.tobytes()).decode(),
        "top": top,
    }


def neuron_trace(eng: Engine, layer: int, neuron: int) -> list[float] | None:
    """Return neuron ``neuron`` of ``layer``'s activation at every token of the last run, or ``None``."""
    flow = getattr(eng, "flow", None)
    if flow is None:
        return None
    return [round(float(x), 3) for x in flow["acts"][layer, :, neuron].astype(np.float32)]


class Live:
    """Context manager capturing per-layer activity at the newest position during streaming generation.

    Hooks stay on the device during generation and :meth:`frame` transfers a compact summary per emitted piece.
    """

    def __init__(self, eng: Engine, k: int = 32) -> None:
        self.eng = eng
        self.k = k
        self.handles: list[Any] = []
        num_layers = eng.num_layers
        self.heads: list[torch.Tensor | None] = [None] * num_layers
        self.attn: list[torch.Tensor | None] = [None] * num_layers
        self.mlp: list[torch.Tensor | None] = [None] * num_layers
        self.acts: list[torch.Tensor | None] = [None] * num_layers

    def __enter__(self) -> Live:
        eng = self.eng
        eng.lock.acquire()
        try:
            kinds = eng.arch()["attention"]

            def _keep(store: list[torch.Tensor | None], layer_idx: int, extract):
                def _hook(_mod: torch.nn.Module, inp: tuple, out: Any = None) -> None:
                    store[layer_idx] = extract(inp, out)

                return _hook

            for layer_idx, layer in enumerate(eng.layers):
                _, attn_mod, out_proj = attention_of(layer)
                if out_proj is not None:
                    head_dim = _head_dim(eng, attn_mod, kinds[layer_idx])
                    if head_dim:
                        self.handles.append(
                            eng.hook_sub(
                                layer_idx,
                                out_proj,
                                _keep(
                                    self.heads,
                                    layer_idx,
                                    lambda inp, _out, hd=head_dim: inp[0][0, -1].float().view(-1, hd).norm(dim=-1),
                                ),
                                pre=True,
                            )
                        )
                    self.handles.append(
                        eng.hook_sub(
                            layer_idx,
                            out_proj,
                            _keep(
                                self.attn,
                                layer_idx,
                                lambda _inp, out: unwrap_tensor(out)[0, -1].float().norm(),
                            ),
                        )
                    )
                down = eng.mlp_writer(layer_idx)
                if not eng.ink:
                    self.handles.append(
                        down.register_forward_pre_hook(
                            _keep(self.acts, layer_idx, lambda inp, _out: inp[0][0, -1].float())
                        )
                    )
                self.handles.append(
                    eng.hook_sub(
                        layer_idx,
                        down,
                        _keep(self.mlp, layer_idx, lambda _inp, out: out[0, -1].float().norm()),
                    )
                )
            eng._cap_resid = {}
            if eng.ink:
                eng._cap_acts = {}
                eng._cap_read = {}
        except BaseException:
            self._cleanup()
            raise
        return self

    def _cleanup(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.eng._cap_resid = None
        if self.eng.ink:
            self.eng._cap_acts = None
            self.eng._cap_read = None
        self.eng.lock.release()

    def __exit__(self, *exc: Any) -> None:
        self._cleanup()

    def frame(self) -> dict[str, Any]:
        """Read the latest forward pass's residual, attention, MLP, top neurons, and sampled logit lens."""
        eng = self.eng
        num_layers = eng.num_layers
        with torch.inference_mode():
            if eng.ink:
                dense_acts = torch.zeros(num_layers, eng.num_neurons, device=eng.dev)
                for layer_idx in range(num_layers):
                    e_idx, e_val = eng._cap_acts[layer_idx]
                    units = e_val.shape[-1]
                    ids = e_idx[-1].long()[:, None] * units + torch.arange(units)
                    dense_acts[layer_idx, ids.reshape(-1).to(eng.dev)] = e_val[-1].reshape(-1).float().to(eng.dev)
                self.acts = list(dense_acts)
            imp = torch.stack(self.acts) * (eng.dnorm / eng.resid_p50[:, None]).float()  # [L, N]
            peak = imp.abs().amax(dim=1).clamp(min=1e-8)
            top_k = min(self.k, imp.shape[1])
            idx = imp.abs().topk(top_k, dim=1).indices
            val = imp.gather(1, idx) / peak[:, None]
            resid = torch.stack([eng._cap_resid[layer_idx][-1] for layer_idx in range(num_layers)])  # [L, H]
            if num_layers <= 1:
                lens_at = [0] if num_layers == 1 else []
            else:
                lens_at = sorted({round(i * (num_layers - 1) / (LIVE_LENS - 1)) for i in range(LIVE_LENS)})
            if lens_at:
                read = torch.stack([eng.lens_state(layer_idx)[-1] for layer_idx in lens_at])
                lens = eng._head(eng.final_norm(read)).argmax(dim=-1)
            else:
                lens = torch.empty(0, dtype=torch.long)
            zero = torch.zeros((), device=eng.dev)
            heads = [h for h in self.heads if h is not None]
            sizes = [0 if h is None else h.numel() for h in self.heads]
            flat_tensor = torch.cat(
                [
                    val.flatten(),
                    peak,
                    resid.float().norm(dim=-1),
                    torch.stack([zero if a is None else a for a in self.attn]),
                    torch.stack(self.mlp),
                    *heads,
                ]
            )
            flat = flat_tensor.cpu().tolist()
            idx_list = idx.cpu().tolist()
            lens_list = lens.cpu().tolist()

        cursor = 0

        def _take(count: int) -> list[float]:
            nonlocal cursor
            cursor += count
            return flat[cursor - count : cursor]

        val_flat = _take(num_layers * top_k)
        peak_list = _take(num_layers)
        resid_norms = _take(num_layers)
        attn_list = _take(num_layers)
        mlp_list = _take(num_layers)
        words = dict(zip(lens_at, (eng.tok.decode([tok]) for tok in lens_list), strict=True))
        return {
            "resid": _round2(resid_norms),
            "attn": _round2(attn_list),
            "mlp": _round2(mlp_list),
            "heads": [_round2(_take(size)) for size in sizes],
            "top": [
                [
                    [n, round(v, 3)]
                    for n, v in zip(idx_list[layer_idx], val_flat[layer_idx * top_k : (layer_idx + 1) * top_k], strict=True)
                ]
                for layer_idx in range(num_layers)
            ],
            "peak": [round(p, 4) for p in peak_list],
            "lens": [words.get(layer_idx) for layer_idx in range(num_layers)],
        }
