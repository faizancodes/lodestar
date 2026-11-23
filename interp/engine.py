"""Live model engine for Lodestar: tracing, neuron reading, steering, teacher-forced scoring, and attribution.

A "neuron" is one unit at the input of a layer's MLP ``down_proj`` (where ``silu(gate) * up`` lives), so reading,
adding, and multiplying are exact. Supports any Llama-style causal decoder in ``transformers`` (Qwen, Llama, Mistral,
Gemma, SmolLM, etc.) whose decoder layers expose ``mlp.down_proj``.

Doses for ``add`` pushes are scaled by the layer's median residual norm divided by the neuron's down-projection column
norm, so dose ``1.0`` shifts the residual stream by roughly one typical residual norm at any layer of any model.
``mul`` pushes scale the activation (``0`` ablates).

A "direction" is a vector in the residual stream (a concept's difference in means; see :mod:`interp.directions`). A
``dir`` push adds ``dose / len(layers)`` times the direction's contrast magnitude to each layer's output; ``erase``
clamps the stream's projection along the direction back toward the mean residual of ordinary prompts.
"""

from __future__ import annotations

import gc
import os
import re
import threading
import time
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from . import inkling, memory
from .utils import empty_device_cache, get_logger, text_config_of, unwrap_tensor


# transformers 5 moves weights onto the device from a thread pool; on Apple GPUs concurrent copies can race in
# torch's Metal shader cache and corrupt memory. Single-threaded loading avoids the race.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

logger = get_logger(__name__)

LAYER_RE = re.compile(r"\blayers\.(\d+)\.")  # decoder layer weight names, e.g. "model.layers.13.mlp.down_proj.weight"

REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


@dataclass
class Push:
    """Single steering intervention on either an MLP neuron or a residual-stream direction."""

    layer: int | None  # None for a direction push, which carries its own layers
    neuron: int | None  # None = every neuron in the layer's MLP (mul only)
    dose: float
    mode: str = "add"  # "add" | "mul" | "dir" | "erase"
    dir_id: str | None = None
    layers: tuple[int, ...] = ()  # dir: layers spread over; erase: source layer for the direction

    @staticmethod
    def of(spec: dict[str, Any]) -> Push:
        """Build a :class:`Push` from a JSON API item dictionary."""
        if spec.get("kind") == "dir":
            op = "erase" if spec.get("op") == "erase" else "dir"
            return Push(
                None,
                None,
                float(spec["dose"]),
                op,
                str(spec["dir"]),
                tuple(int(layer_idx) for layer_idx in spec.get("layers") or ()),
            )
        neuron = None if spec.get("neuron") is None else int(spec["neuron"])
        return Push(int(spec["layer"]), neuron, float(spec["dose"]), spec.get("mode", "add"))


class Cancelled(Exception):
    """Raised when an active streaming, scoring, or search operation is cancelled."""


class SparseActs:
    """Sparse neuron activations ``[L, T, N]`` for mixture-of-experts models.

    Stores only the routed experts per token: ``idx [L, T, K]`` (expert indices) and
    ``val [L, T, K, units]`` (expert unit activations). Supports ``acts[:, t, :]`` and ``acts[l, :, n]``.
    """

    def __init__(self, per_layer: Sequence[tuple[torch.Tensor, torch.Tensor]], num_neurons: int) -> None:
        self.idx = torch.stack([idx for idx, _ in per_layer]).numpy()  # [L, T, K] int16
        self.val = torch.stack([val for _, val in per_layer]).numpy()  # [L, T, K, units] float16
        self.units = int(self.val.shape[-1])
        self.shape = (int(self.idx.shape[0]), int(self.idx.shape[1]), int(num_neurons))

    def _ids(self, idx: np.ndarray) -> np.ndarray:
        return idx.astype(np.int64)[..., None] * self.units + np.arange(self.units)

    def __getitem__(self, key: tuple[Any, Any, Any]) -> np.ndarray:
        a, b, c = key
        if isinstance(b, (int, np.integer)) and a == slice(None) and c == slice(None):
            num_layers = self.shape[0]
            out = np.zeros((num_layers, self.shape[2]), np.float16)
            np.put_along_axis(out, self._ids(self.idx[:, b]).reshape(num_layers, -1), self.val[:, b].reshape(num_layers, -1), 1)
            return out
        if isinstance(a, (int, np.integer)) and b == slice(None) and isinstance(c, (int, np.integer)):
            expert_idx, unit_idx = divmod(int(c), self.units)
            return (self.val[a, :, :, unit_idx] * (self.idx[a] == expert_idx)).sum(-1).astype(np.float16)
        raise IndexError("SparseActs supports acts[:, t, :] and acts[l, :, n]")

    def mean(self, start: int) -> np.ndarray:
        """Compute every neuron's mean activation over positions ``start:`` as ``[L, N]`` float32."""
        num_layers, seq_len, num_neurons = self.shape
        out = np.zeros((num_layers, num_neurons), np.float32)
        for layer_idx in range(num_layers):
            np.add.at(
                out[layer_idx],
                self._ids(self.idx[layer_idx, start:]).ravel(),
                self.val[layer_idx, start:].astype(np.float32).ravel(),
            )
        return out / max(seq_len - start, 1)

    def top(
        self,
        scale: np.ndarray,
        k: int,
        t: int | None = None,
        start: int = 0,
    ) -> list[tuple[int, int, float, float, int]]:
        """Return top-``k`` neurons by impact: ``[(layer, neuron, signed_impact, activation, token)]``."""
        num_layers = self.shape[0]
        if t is not None:
            ids = self._ids(self.idx[:, t]).reshape(num_layers, -1)
            acts = self.val[:, t].reshape(num_layers, -1).astype(np.float32)
            imp = acts * np.take_along_axis(scale, ids, 1)
            flat = np.abs(imp).ravel()
            top_k = min(k, flat.size)
            sel = np.argpartition(-flat, top_k - 1)[:top_k] if top_k < flat.size else np.arange(flat.size)
            sel = sel[np.argsort(-flat[sel])]
            return [
                (int(i // ids.shape[1]), int(ids.ravel()[i]), float(imp.ravel()[i]), float(acts.ravel()[i]), t)
                for i in sel
            ]

        cands: list[tuple[float, int, int]] = []
        for layer_idx in range(num_layers):
            ids = torch.from_numpy(self._ids(self.idx[layer_idx, start:]).reshape(-1))
            imp = (
                torch.from_numpy(self.val[layer_idx, start:].astype(np.float32).reshape(-1))
                * torch.from_numpy(scale[layer_idx])[ids]
            )
            best = torch.zeros(self.shape[2]).scatter_reduce(0, ids, imp.abs(), "amax")
            top = best.topk(min(k, self.shape[2]))
            cands.extend((float(val), layer_idx, int(n)) for val, n in zip(top.values, top.indices, strict=True))
        cands.sort(reverse=True)

        out: list[tuple[int, int, float, float, int]] = []
        for _, layer_idx, neuron_idx in cands[:k]:
            series = self[layer_idx, :, neuron_idx].astype(np.float32)
            tok_pos = start + int(np.abs(series[start:]).argmax())
            out.append(
                (
                    layer_idx,
                    neuron_idx,
                    float(series[tok_pos] * scale[layer_idx, neuron_idx]),
                    float(series[tok_pos]),
                    tok_pos,
                )
            )
        return out


class Engine:
    """Thread-safe local language model runtime with hooks for steering, ablation, tracing, and attribution."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model: Any = None
        self.tok: Any = None
        self.model_id: str | None = None
        self.dev: str = "cpu"
        self.n_params: int = 0
        self.layers: Any = ()
        self.final_norm: Any = None
        self.lm_head: Any = None
        self.backbone: Any = None
        self.dnorm: torch.Tensor | None = None
        self.resid_p50: torch.Tensor | None = None
        self.unit: torch.Tensor | None = None
        self.end_ids: set[int] = set()
        self.last_search_partial: bool = False

        self.status: str = "idle"
        self.progress: float | None = None  # 0..1 while loading, else None
        # Loading progress metadata for the UI loading scene: {"stage", "layers", "lit"}.
        self.load_state: dict[str, Any] | None = None
        self.trace: dict[str, Any] | None = None
        self.flow: dict[str, Any] | None = None  # Architecture view's last forward pass
        self.jlens: torch.Tensor | None = None  # [L, H, H] on device once computed or loaded

        self._push_state: dict[Any, Any] | None = None
        self.dirs: dict[str, dict[str, Any]] = {}

        self.abl_vec: torch.Tensor | None = None  # [H] unit refusal direction in model dtype
        self.abl_vec_f: torch.Tensor | None = None  # [H] float32 copy for weight edits
        self.abl_coef: float = 1.0
        self.abl_layer: int | None = None
        self._abl_mask: torch.Tensor | None = None  # [B] bool per-row ablation mask
        self._abl_any: bool = False
        # Optional category gate: {"layer": int, "vec": Tensor[H], "bias": float}.
        self.abl_gate: dict[str, Any] | None = None
        self.last_gate: list[bool | None] | None = None

        self._deltas: list[torch.Tensor] | None = None
        self._cap_acts: dict[int, Any] | None = None
        self._cap_resid: dict[int, torch.Tensor] | None = None
        self._cap_read: dict[int, torch.Tensor] | None = None
        self.ink: dict[str, Any] | None = None
        self.reply_prefix: list[int] = []
        self.softcap: float | None = None
        self._read: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        self._read_buf: torch.Tensor | None = None
        self._handles: list[Any] = []

    @property
    def num_layers(self) -> int:
        """Number of decoder layers in the loaded model."""
        return len(self.layers)

    @property
    def num_neurons(self) -> int:
        """Number of MLP intermediate neurons per layer."""
        return int(self.dnorm.shape[1])

    @property
    def hidden(self) -> int:
        """Residual stream width (hidden dimension) of the loaded model."""
        return int(self.lm_head.weight.shape[1])

    @property
    def hidden_size(self) -> int:
        """Alias for :attr:`hidden`."""
        return self.hidden

    # ------------------------------------------------------------------ loading
    def load(self, model_id: str, stream: int | None = None) -> dict[str, Any]:
        """Load ``model_id`` onto MPS, CUDA, or CPU, install hooks, and calibrate layer norms.

        Args:
            model_id: Hugging Face model ID or local model directory path.
            stream: Optional GPU cache size in bytes for streaming Inkling 2 Mini experts from disk.
        """
        with self.lock:
            if self.model_id == model_id and self.model is not None:
                return self.info()
            self.status = f"loading {model_id}"
            self.load_state = {"stage": "open", "layers": self._layer_count(model_id), "lit": None}
            self.progress = 0.0
            t0 = time.perf_counter()
            took: dict[str, float] = {}
            self.unload()
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            if dev == "mps":
                memory.cap_gpu()

            # Load the tokenizer alongside the model weights instead of sequentially.
            tok_box: dict[str, Any] = {}

            def _load_tok() -> None:
                try:
                    tok_box["tok"] = AutoTokenizer.from_pretrained(model_id)
                except Exception as exc:
                    tok_box["err"] = exc

            tok_thread = threading.Thread(target=_load_tok, daemon=True)
            tok_thread.start()
            self.progress = 0.04

            # Load directly onto the target device to avoid holding two copies of the weights in memory.
            ink: dict[str, Any] | None = None
            prev_hook = hf_logging.set_tqdm_hook(self._weights_progress_hook)
            short: str | None = None
            try:
                if inkling.is_inkling(model_id):
                    model, ink = inkling.load(
                        model_id,
                        dev,
                        on_layer=self._experts_arrived,
                        cache_bytes=stream if dev == "mps" else None,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=dev)
            except MemoryError as exc:
                short = str(exc)  # re-raised below after freeing any half-loaded weights held by traceback
            finally:
                hf_logging.set_tqdm_hook(prev_hook)

            if short is not None:
                gc.collect()
                empty_device_cache()
                raise RuntimeError(short)

            tok_thread.join()
            if "err" in tok_box:
                raise RuntimeError(f"Couldn't load the tokenizer for {model_id}: {tok_box['err']}") from tok_box["err"]
            if "tok" not in tok_box:
                raise RuntimeError(f"Couldn't load the tokenizer for {model_id}")
            tok = tok_box["tok"]

            took["weights"] = time.perf_counter() - t0
            self.progress = 0.9
            self.load_state = {**self.load_state, "stage": "measure"}

            n_params = int(ink["n_params"]) if ink else sum(p.numel() for p in model.parameters())
            if n_params < 2.5e9:
                # Small models fit comfortably in fp32, keeping gradients and lenses precise.
                model = model.float()
            model = model.eval()
            model.requires_grad_(False)

            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            self.tok = tok
            self.model = model
            self.dev = dev
            self.model_id = model_id
            self.n_params = n_params
            self.ink = ink
            self.softcap = getattr(text_config_of(model.config), "final_logit_softcapping", None)
            self.lm_head = model.get_output_embeddings()
            if ink:
                self.layers = ink["layers"]
                self.final_norm = ink["text"].norm
                self.backbone = ink["text"]
                self.reply_prefix = tok.convert_tokens_to_ids(list(inkling.REPLY_PREFIX))
                self._handles.extend(ink["handles"])
                for layer in ink["text"].layers:
                    layer.mlp.control = self._moe_control
                self._handles.append(ink["text"].embed_tokens.register_forward_hook(self._embed_hook))
            else:
                self.layers = self._find_layers(model)
                self.final_norm = self._find_norm(model)
                self.backbone = getattr(model, model.base_model_prefix)
                self.reply_prefix = []

            for idx, layer in enumerate(self.layers):
                if not ink:
                    self._handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._mlp_hook(idx)))
                self._handles.append(layer.register_forward_hook(self._resid_hook(idx)))

            if not ink:
                self._use_builtin_steering(model_id)

            t1 = time.perf_counter()
            with torch.no_grad():
                if ink:
                    norms = [layer.mlp.store.down_norms() for layer in ink["text"].layers]
                    self.dnorm = torch.stack([norms[ink["tracker"].physical(v)] for v in range(len(self.layers))])
                else:
                    self.dnorm = torch.stack([layer.mlp.down_proj.weight.float().norm(dim=0) for layer in self.layers])
            took["norms"] = time.perf_counter() - t1

            t1 = time.perf_counter()
            self.resid_p50 = self._measure_resid_norms()
            took["reference pass"] = time.perf_counter() - t1

            self.unit = self.resid_p50[:, None] / self.dnorm.clamp(min=1e-6)  # [L, N] activation per unit dose
            self.end_ids = self._end_ids()

            took["total"] = time.perf_counter() - t0
            took["setup"] = took["total"] - took["weights"]
            logger.info(
                "loaded %s: %s",
                model_id,
                ", ".join(f"{name} {secs:.1f}s" for name, secs in took.items()),
            )

            self.progress = None
            self.load_state = None
            self.status = "ready"
            return self.info()

    @staticmethod
    def _layer_count(model_id: str) -> int | None:
        """Read the model's layer count from its config before loading weights, or ``None`` if unavailable."""
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            return getattr(text_config_of(cfg), "num_hidden_layers", None)
        except Exception:
            return None

    def _experts_arrived(self, layer_idx: int, total_layers: int) -> None:
        """Track Inkling packed-expert loading progress and abort if macOS hits critical memory pressure."""
        if memory.pressure() == "critical":
            raise MemoryError("Stopped loading: the Mac ran short of memory. Quit some apps and try again.")
        if self.load_state is not None:
            lit = self.load_state.get("lit") or [0.0] * total_layers
            if len(lit) == total_layers:
                lit[layer_idx] = 1.0
            self.load_state = {**self.load_state, "stage": "weights", "layers": total_layers, "lit": lit}
        self.progress = 0.5 + 0.4 * (layer_idx + 1) / total_layers

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits from final-normed states, applying logit softcapping when configured."""
        logits = self.lm_head(hidden).float()
        return torch.tanh(logits / self.softcap) * self.softcap if self.softcap else logits

    def _experts(self, layer: int) -> Any:
        """Return the packed expert store backing virtual ``layer`` of an Inkling model."""
        return self.ink["text"].layers[self.ink["tracker"].physical(layer)].mlp.store

    def _down_col(self, layer: int, neuron: int) -> torch.Tensor:
        """Return neuron ``neuron``'s down-projection column ``[H]`` in the model's dtype."""
        if self.ink:
            return self._experts(layer).down_columns([neuron])[:, 0].to(self.model.dtype)
        return self.layers[layer].mlp.down_proj.weight[:, neuron]

    def _down_scores(self, layers: Sequence[int], dirs: Sequence[torch.Tensor]) -> dict[int, torch.Tensor]:
        """Compute ``dirs[i] @ down_proj`` for each layer in ``layers``: ``{layer: [N] float32}``."""
        out: dict[int, torch.Tensor] = {}
        if not self.ink:
            for i, layer_idx in enumerate(layers):
                down_w = self.layers[layer_idx].mlp.down_proj.weight
                out[layer_idx] = (dirs[i].to(down_w.dtype) @ down_w).float()
            return out

        by_phys: dict[int, list[tuple[int, int]]] = {}
        for i, layer_idx in enumerate(layers):
            by_phys.setdefault(self.ink["tracker"].physical(layer_idx), []).append((i, layer_idx))
        for group in by_phys.values():
            scores = self._experts(group[0][1]).down_scores(torch.stack([dirs[i] for i, _ in group]).float())
            for j, (_, layer_idx) in enumerate(group):
                out[layer_idx] = scores[j]
        return out

    def hook_sub(self, layer: int, sub: torch.nn.Module, fn: Any, pre: bool = False) -> Any:
        """Register a hook on ``sub`` scoped to virtual ``layer`` (filtering looped Inkling passes)."""
        if self.ink:
            return self.layers[layer].hook(sub, fn, pre)
        return sub.register_forward_pre_hook(fn) if pre else sub.register_forward_hook(fn)

    def mlp_writer(self, layer: int) -> torch.nn.Module:
        """Return the module whose output is layer ``layer``'s MLP residual contribution."""
        return self.layers[layer].module.mlp if self.ink else self.layers[layer].mlp.down_proj

    def lens_state(self, layer: int) -> torch.Tensor:
        """Return the captured hidden state at ``layer`` as the output head reads it: ``[T, H]``."""
        return (self._cap_read or self._cap_resid)[layer]

    def _moe_control(self, layer_idx: int) -> Any:
        """Return per-pass steering/attribution/capture instructions for Inkling's packed MoE layers."""
        state = self._push_state.get(layer_idx) if self._push_state else None
        delta = self._deltas[layer_idx] if self._deltas is not None else None
        cap = self._keep_moe_acts if self._cap_acts is not None else None
        if state is None and delta is None and cap is None:
            return None
        add, mul = state if state is not None else (None, None)
        return inkling.MoeControl(mul=mul, add=add, delta=delta, capture=cap)

    def _keep_moe_acts(self, layer_idx: int, idx: torch.Tensor, acts: torch.Tensor) -> None:
        self._cap_acts[layer_idx] = (idx[0].to(torch.int16).cpu(), acts[0].cpu())

    def _embed_hook(self, _mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> torch.Tensor | None:
        if self._abl_any and self.abl_vec is not None:
            mask = self._abl_mask.to(out.dtype)[:, None, None]
            vec = self.abl_vec.to(out.dtype)
            return out - mask * self.abl_coef * (out @ vec)[:, :, None] * vec
        return None

    def _all_hidden(self, **kwargs: Any) -> Sequence[torch.Tensor]:
        """Return embedding plus per-layer hidden states ``[L + 1]`` (using hooks for looped Inkling layers)."""
        if not self.ink:
            return self.model(**kwargs, output_hidden_states=True).hidden_states

        got: dict[int, torch.Tensor] = {}

        def _make_grab(key: int):
            def _hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
                got[key] = unwrap_tensor(out)

            return _hook

        handles = [self.ink["text"].embed_tokens.register_forward_hook(_make_grab(-1))]
        handles.extend(layer.register_forward_hook(_make_grab(layer_idx)) for layer_idx, layer in enumerate(self.layers))
        try:
            self.model(**{**kwargs, "logits_to_keep": 1})
        finally:
            for handle in handles:
                handle.remove()
        return [got[layer_idx] for layer_idx in range(-1, len(self.layers))]

    def _weights_progress_hook(self, factory: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Mirror transformers' weight-loading progress bar into ``self.progress`` and ``self.load_state``."""
        bar = factory(*args, **kwargs)
        if kwargs.get("desc") != "Loading weights":
            return bar

        eng = self
        names = [x[0] if isinstance(x, tuple) else x for x in list(args[0])] if args else []

        def _layer_of(name: Any) -> int | None:
            if isinstance(name, str) and (match := LAYER_RE.search(name)):
                return int(match.group(1))
            return None

        per: dict[int, int] = {}
        for name in names:
            layer_idx = _layer_of(name)
            if layer_idx is not None:
                per[layer_idx] = per.get(layer_idx, 0) + 1

        num_layers = max(per) + 1 if per else (eng.load_state or {}).get("layers")
        got = [0] * (num_layers or 0)
        eng.load_state = {**(eng.load_state or {}), "stage": "weights", "layers": num_layers, "lit": [0.0] * (num_layers or 0)}

        class Watched:
            def __iter__(self) -> Iterator[Any]:
                total = getattr(bar, "total", None) or len(names)
                for idx, item in enumerate(bar, 1):
                    if total:
                        eng.progress = 0.04 + 0.86 * idx / total

                    layer_idx = _layer_of(item[0] if isinstance(item, tuple) else item)
                    if layer_idx is not None and layer_idx < len(got) and per.get(layer_idx):
                        got[layer_idx] += 1
                        eng.load_state["lit"][layer_idx] = round(got[layer_idx] / per[layer_idx], 2)
                    yield item

            def __getattr__(self, key: str) -> Any:
                return getattr(bar, key)

            def __enter__(self) -> Watched:
                bar.__enter__()
                return self

            def __exit__(self, *exc: Any) -> Any:
                return bar.__exit__(*exc)

        return Watched()

    def _use_builtin_steering(self, model_id: str) -> None:
        """Swap fitted portable steering biases for exact ``down_proj`` hooks when loading a Lodestar-baked copy."""
        spec = getattr(self.model.config, "lodestar_steering", None)
        if not spec:
            return

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        local = Path(model_id) / spec["file"]
        exact = load_file(str(local if local.exists() else hf_hub_download(model_id, spec["file"])))

        with torch.no_grad():
            for layer_idx in spec["layers"]:
                self.layers[layer_idx + 1].self_attn.o_proj.bias.zero_()
                bias = exact[f"layer.{layer_idx}"].to(self.dev, self.model.dtype)
                self._handles.append(
                    self.layers[layer_idx].mlp.down_proj.register_forward_hook(
                        lambda _m, _i, out, b=bias: out + b
                    )
                )

    def unload(self) -> None:
        """Remove all hooks, clear model references and cached state, and release device memory."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        self.model = None
        self.tok = None
        self.model_id = None
        self.ink = None
        self.n_params = 0
        self.layers = ()
        self.final_norm = None
        self.lm_head = None
        self.backbone = None
        self.dnorm = None
        self.resid_p50 = None
        self.unit = None
        self.reply_prefix = []
        self.softcap = None
        self.end_ids = set()
        self.trace = None
        self.flow = None
        self.jlens = None
        self.dirs = {}
        self.clear_abl()
        self.last_gate = None

        gc.collect()  # hooks and Inkling's layer wrappers can form reference cycles
        empty_device_cache()

    @staticmethod
    def _find_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
        """Locate the list of decoder layers exposing ``mlp.down_proj``."""
        for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
            obj: Any = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            if len(obj) > 0 and hasattr(obj[0], "mlp") and hasattr(obj[0].mlp, "down_proj"):
                return obj

        raise RuntimeError("unsupported architecture: no layers with mlp.down_proj found")

    @staticmethod
    def _find_norm(model: torch.nn.Module) -> torch.nn.Module:
        """Locate the model's final layer normalization module."""
        for path in ("model.norm", "model.language_model.norm", "transformer.ln_f"):
            obj: Any = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue

        raise RuntimeError("no final norm found")

    def _end_ids(self) -> set[int]:
        ids = {self.tok.eos_token_id}
        for tok_str in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", *inkling.END_TOKENS):
            tok_id = self.tok.convert_tokens_to_ids(tok_str)
            if isinstance(tok_id, int) and tok_id != self.tok.unk_token_id:
                ids.add(tok_id)

        return {i for i in ids if i is not None}

    def info(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the engine's current load status and model dimensions."""
        if self.model is None or self.status != "ready":
            return {"loaded": False, "status": self.status}

        return {
            "loaded": True,
            "status": self.status,
            "model": self.model_id,
            "device": self.dev,
            "dtype": str(self.model.dtype).replace("torch.", ""),
            "params": self.n_params,
            "layers": self.num_layers,
            "neurons": self.num_neurons,
            "hidden": self.hidden_size,
            "arch": self.arch(),
        }

    def arch(self) -> dict[str, Any]:
        """Summarize the loaded model's attention layout and vocabulary for the Home diagram."""
        cfg = text_config_of(self.model.config)
        num_layers = self.num_layers
        types = getattr(cfg, "layer_types", None) or ["full_attention"] * num_layers
        if self.ink:
            types = [types[self.ink["tracker"].physical(v)] for v in range(num_layers)]
        return {
            "heads": getattr(cfg, "num_attention_heads", None),
            "kv_heads": getattr(cfg, "num_key_value_heads", None),
            "vocab": int(self.lm_head.weight.shape[0]),
            "tied": bool(getattr(cfg, "tie_word_embeddings", False)),
            "experts": getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None) or None,
            "attention": ["linear" if "linear" in str(t) else "full" for t in types[:num_layers]],
        }

    # ------------------------------------------------------------------ ablation & gating
    def abl_of(self, vec: Sequence[float] | np.ndarray, coef: float, layer: int | None = None) -> None:
        """Store a unit refusal direction to erase from every layer's residual stream."""
        if self.model is None:
            raise RuntimeError("no model loaded")

        with self.lock:
            vec_f = torch.tensor([float(x) for x in vec], device=self.dev, dtype=torch.float32)
            vec_f = vec_f / vec_f.norm().clamp(min=1e-8)
            self.abl_vec = vec_f.to(self.model.dtype)
            self.abl_vec_f = vec_f
            self.abl_coef = float(coef)
            self.abl_layer = layer

    def clear_abl(self) -> None:
        """Remove any active refusal direction and category gate from the engine."""
        self.abl_vec = None
        self.abl_vec_f = None
        self._abl_mask = None
        self.abl_layer = None
        self._abl_any = False
        self.abl_gate = None

    def gate_of(self, gate: dict[str, Any] | None) -> None:
        """Restrict refusal erasure to prompts passing ``gate`` (``{layer, vec, bias}``), or clear when ``None``."""
        if gate is None:
            self.abl_gate = None
            return
        with self.lock:
            vec = torch.tensor([float(x) for x in gate["vec"]], device=self.dev, dtype=torch.float32)
            self.abl_gate = {
                "layer": int(gate["layer"]),
                "vec": vec / vec.norm().clamp(min=1e-8),
                "bias": float(gate["bias"]),
            }

    @contextmanager
    def temporary_ablation(self) -> Iterator[None]:
        """Save and restore the engine's refusal direction and gate across an exploratory search block."""
        prev_abl = (self.abl_vec_f.tolist(), self.abl_coef, self.abl_layer) if self.abl_vec is not None else None
        prev_gate = self.abl_gate
        self.abl_gate = None
        try:
            yield
        finally:
            if prev_abl is not None:
                self.abl_of(*prev_abl)
            else:
                self.clear_abl()
            self.abl_gate = prev_gate

    def _left_pad_batch(self, seqs: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pad token sequences into ``(input_ids, attention_mask, position_ids)`` on ``self.dev``."""
        max_len = max(len(seq) for seq in seqs)
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (max_len - len(seq)) + list(seq) for seq in seqs], device=self.dev)
        mask = torch.tensor([[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs], device=self.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    def prompt_head_len(self) -> int:
        """Number of leading tokens inserted by the chat template before the user's words."""
        seq_a = self.chat_ids("What is the capital of France?")
        seq_b = self.chat_ids("tell me a joke")
        count = 0
        while count < min(len(seq_a), len(seq_b)) - 1 and seq_a[count] == seq_b[count]:
            count += 1
        return count

    def _words_mean(
        self,
        seqs: Sequence[Sequence[int]],
        upto: int | None = None,
    ) -> torch.Tensor:
        """Mean residual stream over each prompt's own words (excluding chat template head and tail).

        Returns ``[B, L+1, H]`` float32 for all layers, or ``[B, H]`` at hidden state ``upto`` with early exit.
        """
        head = self.prompt_head_len()
        tail = self.prompt_end_len()
        ids, mask, pos = self._left_pad_batch(seqs)
        seq_len = ids.shape[1]
        words = torch.zeros(len(seqs), seq_len, device=self.dev)
        for row_idx, seq in enumerate(seqs):
            start, end = seq_len - len(seq) + head, seq_len - tail
            if end <= start:  # fallback if nothing lies between head and tail
                start, end = seq_len - len(seq), seq_len
            words[row_idx, start:end] = 1.0 / (end - start)

        def _pool(hidden: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bt,bth->bh", words, hidden.float())

        if upto is None:
            with torch.inference_mode():
                hs = self._all_hidden(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=pos,
                    use_cache=False,
                    logits_to_keep=1,
                )
                return torch.stack([_pool(h) for h in hs], dim=1)

        captured: dict[str, torch.Tensor] = {}

        class _EarlyStop(Exception):
            pass

        def _capture_hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            captured["h"] = _pool(unwrap_tensor(out))
            raise _EarlyStop()

        handle = self.layers[upto - 1].register_forward_hook(_capture_hook)
        try:
            with torch.inference_mode():
                self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
        except _EarlyStop:
            pass
        finally:
            handle.remove()
        return captured["h"]

    def prompt_word_means(
        self,
        prompts: Sequence[str],
        cancel: threading.Event | None = None,
        budget: int = 6144,
    ) -> np.ndarray:
        """Compute per-layer word-averaged residuals for ``prompts`` in batches: ``[n, L+1, H]`` float32 CPU."""
        with self.lock:
            ids_list = [self.chat_ids(prompt) for prompt in prompts]
            out: list[torch.Tensor] = []
            for batch_idx in self._batches([(seq,) for seq in ids_list], budget):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                out.append(self._words_mean([ids_list[i] for i in batch_idx]).cpu())
            return torch.cat(out).numpy()

    def gate_scores(self, seqs: Sequence[Sequence[int]]) -> list[float]:
        """Compute signed gate projections for templated token sequences (positive opens the gate)."""
        gate = self.abl_gate
        saved = self._push_state, self._abl_any
        self._push_state, self._abl_any = None, False
        try:
            hidden = self._words_mean(seqs, upto=gate["layer"])
        finally:
            self._push_state, self._abl_any = saved
        return (hidden @ gate["vec"] - gate["bias"]).tolist()

    def _gated(self, seqs: Sequence[Sequence[int]], flags: list[bool] | None) -> list[bool] | None:
        """Resolve per-row ablation flags through ``self.abl_gate`` and record ``self.last_gate``."""
        if flags is None or self.abl_gate is None or self.abl_vec is None:
            self.last_gate = None
            return flags
        active_idx = [i for i, flag in enumerate(flags) if flag]
        opened = (
            dict(zip(active_idx, (score > 0 for score in self.gate_scores([seqs[i] for i in active_idx])), strict=True))
            if active_idx
            else {}
        )
        self.last_gate = [opened.get(i) for i in range(len(flags))]
        return [bool(opened.get(i)) for i in range(len(flags))]

    @staticmethod
    def _abl_flags(abl: bool | Sequence[bool] | None, n: int) -> tuple[list[bool] | None, bool]:
        """Normalize a per-row ablation flag specification (``bool``, sequence of bools, or ``None``)."""
        if abl is None:
            return None, False
        flags = [bool(abl)] * n if isinstance(abl, bool) else [bool(x) for x in abl]
        return flags, any(flags)

    def prompt_end_len(self) -> int:
        """Number of trailing chat-template tokens after the user's message (clamped to 1..12)."""
        seq_a = self.chat_ids("What is the capital of France?")
        seq_b = self.chat_ids("tell me a joke")
        count = 0
        while count < min(len(seq_a), len(seq_b)) and seq_a[-1 - count] == seq_b[-1 - count]:
            count += 1
        return max(1, min(count, 12))

    def prompt_end_states(
        self,
        prompts: Sequence[str],
        n_pos: int = 1,
        cancel: threading.Event | None = None,
        budget: int = 6144,
    ) -> np.ndarray:
        """Capture residual streams at the last ``n_pos`` tokens of each prompt: ``[n, L+1, n_pos, H]`` float16 CPU."""
        with self.lock:
            ids_list = [self.chat_ids(prompt) for prompt in prompts]
            out: list[torch.Tensor] = []

            for batch_idx in self._batches([(seq,) for seq in ids_list], budget):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()

                ids, mask, pos = self._left_pad_batch([ids_list[i] for i in batch_idx])
                with torch.inference_mode():
                    hs = self._all_hidden(
                        input_ids=ids,
                        attention_mask=mask,
                        position_ids=pos,
                        use_cache=False,
                    )
                    last = torch.stack([h[:, -n_pos:, :] for h in hs], dim=1)  # [B, L+1, n_pos, H]
                    out.append(last.half().cpu())
                    del hs, last

            if not out:
                return np.zeros((0, self.num_layers + 1, n_pos, self.hidden_size), np.float16)
            return torch.cat(out).numpy()

    # ------------------------------------------------------------------ hooks
    def _mlp_hook(self, layer_idx: int):
        def _hook(_mod: torch.nn.Module, inp: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor] | None:
            hidden = inp[0]
            if self._cap_acts is not None:
                self._cap_acts[layer_idx] = hidden[0].detach().float().cpu()

            changed = False
            layer_push = self._push_state.get(layer_idx) if self._push_state else None
            if layer_push is not None:
                add, mul = layer_push
                if mul is not None:
                    hidden = hidden * mul[:, None, :].to(hidden.dtype)
                if add is not None:
                    hidden = hidden + add[:, None, :].to(hidden.dtype)
                changed = True

            if self._deltas is not None:
                hidden = hidden + self._deltas[layer_idx].to(hidden.dtype)
                changed = True

            return (hidden,) if changed else None

        return _hook

    def _resid_hook(self, layer_idx: int):
        def _hook(
            mod: torch.nn.Module,
            inp: tuple,
            out: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor] | None:
            hidden = unwrap_tensor(out)
            changed = False
            state = self._push_state
            offset = state["dir"].get(layer_idx) if state and "dir" in state else None

            if offset is not None:
                hidden = hidden + offset[:, None, :]
                changed = True

            if self._abl_any and self.abl_vec is not None:
                mask = self._abl_mask.to(hidden.dtype)[:, None, None]
                vec = self.abl_vec
                proj = (hidden @ vec)[:, :, None] * vec
                hidden = hidden - mask * self.abl_coef * proj
                changed = True

            for coef, vec, ref in (state.get("erase") or ()) if state else ():
                # Clamp along ``vec`` toward ordinary text rather than 0, avoiding pushing toward the opposite pole.
                proj = hidden @ vec - ref[layer_idx]
                hidden = hidden - coef[:, None, None] * proj[:, :, None] * vec
                changed = True

            if self._cap_resid is not None:
                self._cap_resid[layer_idx] = hidden.detach()[0]
                if self._cap_read is not None:
                    self._cap_read[layer_idx] = inkling.readout(self.ink["text"], mod, inp, hidden.detach()[:1])[0]

            read_spec = self._read.get(layer_idx) if self._read else None
            if read_spec is not None:
                units, ref, scale, cols = read_spec
                self._read_buf[:, cols] += (hidden[:, -1].float() @ units.T - ref) * scale

            if changed:
                return (hidden,) if isinstance(out, tuple) else hidden
            return None

        return _hook

    def _measure_resid_norms(self) -> torch.Tensor:
        ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
        self._cap_resid = {}
        try:
            with torch.inference_mode():
                self.model(input_ids=ids, use_cache=False)
            return torch.tensor(
                [float(self._cap_resid[layer_idx][1:].float().norm(dim=-1).median()) for layer_idx in range(self.num_layers)],
                device=self.dev,
            )
        finally:
            self._cap_resid = None

    def _make_push_state(self, row_pushes: Sequence[Sequence[Push] | None], batch_size: int) -> dict[Any, Any] | None:
        """Compile per-row :class:`Push` lists into vectorized per-layer intervention tensors."""
        num_neurons = self.num_neurons
        state: dict[Any, Any] = {}
        dir_offsets: dict[int, torch.Tensor] = {}
        erase_specs: dict[tuple[str | None, int], list[torch.Tensor]] = {}

        for row_idx, pushes in enumerate(row_pushes):
            for push in pushes or ():
                if push.mode in ("dir", "erase"):
                    direction = self.dirs.get(push.dir_id)
                    if direction is None:
                        raise ValueError(f"No direction called {push.dir_id!r} for this model.")

                    if push.mode == "dir":
                        for layer_idx in push.layers:
                            offset = dir_offsets.get(layer_idx)
                            if offset is None:
                                offset = dir_offsets[layer_idx] = torch.zeros(
                                    batch_size,
                                    direction["unit"].shape[1],
                                    device=self.dev,
                                    dtype=torch.float32,
                                )
                            offset[row_idx] += (push.dose / len(push.layers) * direction["mag"][layer_idx]) * direction["unit"][layer_idx]
                    else:
                        src_layer = push.layers[0] if push.layers else direction["band"][len(direction["band"]) // 2]
                        key = (push.dir_id, src_layer)
                        if key not in erase_specs:
                            vec = direction["unit"][src_layer]
                            ref = (
                                direction["neutral"] @ vec
                                if direction.get("neutral") is not None
                                else torch.zeros(self.num_layers, device=self.dev)
                            )
                            dtype = self.model.dtype
                            erase_specs[key] = [torch.zeros(batch_size, device=self.dev), vec.to(dtype), ref.to(dtype)]
                        erase_specs[key][0][row_idx] += push.dose
                    continue

                add, mul = state.get(push.layer, (None, None))
                if push.mode == "mul":
                    if mul is None:
                        mul = torch.ones(batch_size, num_neurons, device=self.dev)
                    if push.neuron is None:
                        mul[row_idx] *= push.dose
                    else:
                        mul[row_idx, push.neuron] *= push.dose
                else:
                    if push.neuron is None:
                        raise ValueError("whole-layer pushes must use mode 'mul'")
                    if self.ink:
                        # Experts only run for routed tokens, so apply the push onto the MLP residual output.
                        if add is None:
                            add = torch.zeros(batch_size, self.hidden, device=self.dev)
                        add[row_idx] += float(push.dose * self.unit[push.layer, push.neuron]) * self._down_col(
                            push.layer, push.neuron
                        ).float()
                    else:
                        if add is None:
                            add = torch.zeros(batch_size, num_neurons, device=self.dev)
                        add[row_idx, push.neuron] += push.dose * self.unit[push.layer, push.neuron]

                state[push.layer] = (add, mul)

        if dir_offsets:
            state["dir"] = {layer_idx: off.to(self.model.dtype) for layer_idx, off in dir_offsets.items()}
        if erase_specs:
            state["erase"] = [(coef.to(self.model.dtype), vec, ref) for coef, vec, ref in erase_specs.values()]

        return state or None

    # ------------------------------------------------------------------ tokens
    def chat_ids(self, prompt: str, system: str | None = None) -> list[int]:
        """Format a single user prompt (with optional system message) into chat-templated token IDs."""
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.messages_ids(msgs)

    def messages_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        """Apply the tokenizer's chat template (with fallback for templates lacking system-role support)."""
        if self.tok.chat_template is None:
            text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
            return self.tok(text).input_ids

        try:
            extra = {"reasoning_effort": "none"} if self.ink else {}
            enc = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                **extra,
            )
        except Exception:
            if messages and messages[0]["role"] == "system" and len(messages) > 1:
                sys_msg = messages[0]["content"]
                rest = [dict(m) for m in messages[1:]]
                rest[0]["content"] = f"{sys_msg}\n\n{rest[0]['content']}"
                return self.messages_ids(rest)
            raise

        return list(enc["input_ids"] if hasattr(enc, "keys") else enc) + list(self.reply_prefix)

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` without adding special tokens."""
        return self.tok(text, add_special_tokens=False).input_ids

    def token_strs(self, ids: Sequence[int]) -> list[str]:
        """Decode display strings per token while keeping multi-byte UTF-8 characters intact."""
        out: list[str] = []
        prev = ""

        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            if full.endswith("\ufffd") and i < len(ids) - 1:
                out.append("")
                continue
            out.append(full[len(prev) :] if full.startswith(prev) else self.tok.decode([ids[i]]))
            prev = full

        return out

    # ------------------------------------------------------------------ generation
    def _read_plan(
        self,
        dir_ids: Sequence[str],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compile per-layer projection matrices for live direction gauges."""
        by_layer: dict[int, list[tuple[torch.Tensor, float, float, int]]] = {}
        for col_idx, did in enumerate(dir_ids):
            direction = self.dirs.get(did)
            if direction is None:
                raise ValueError(f"No direction called {did!r} for this model.")

            band = direction["band"] or list(range(self.num_layers // 3, 2 * self.num_layers // 3))
            for layer_idx in band:
                vec = direction["unit"][layer_idx]
                ref = float(direction["neutral"][layer_idx] @ vec) if direction.get("neutral") is not None else 0.0
                by_layer.setdefault(layer_idx, []).append((vec, ref, 1.0 / (direction["mag"][layer_idx] * len(band)), col_idx))

        return {
            layer_idx: (
                torch.stack([item[0] for item in items]),
                torch.tensor([item[1] for item in items], device=self.dev),
                torch.tensor([item[2] for item in items], device=self.dev),
                torch.tensor([item[3] for item in items], device=self.dev),
            )
            for layer_idx, items in by_layer.items()
        }

    def generate(
        self,
        rows: Sequence[tuple[Sequence[int], Sequence[Push]]],
        max_new: int = 160,
        temperature: float = 0.0,
        cancel: threading.Event | None = None,
        seed: int = 0,
        abl: bool | Sequence[bool] | None = None,
        read: Sequence[str] | None = None,
    ) -> Generator[tuple[int | None, Any], None, None]:
        """Batched streaming generation with per-row pushes, optional ablation, and optional direction gauges.

        Yields ``(row_index, text_delta)`` (or ``(row_index, (text_delta, readings))`` when ``read`` is set)
        and terminates with ``(None, generated_token_ids_per_row)``.
        """
        with self.lock:
            batch_size = len(rows)
            seqs = [row[0] for row in rows]
            flags, _ = self._abl_flags(abl, batch_size)
            flags = self._gated(seqs, flags)
            any_abl = bool(flags) and any(flags)

            ids, mask, pos = self._left_pad_batch(seqs)
            done = [False] * batch_size
            out_ids: list[list[int]] = [[] for _ in range(batch_size)]
            sent = [""] * batch_size
            held: list[list[list[float]]] = [[] for _ in range(batch_size)]
            gen = torch.Generator(device="cpu").manual_seed(seed)

            self._push_state = self._make_push_state([row[1] for row in rows], batch_size)
            if any_abl:
                self._abl_mask = torch.tensor(flags, device=self.dev)
                self._abl_any = True
            if read:
                self._read = self._read_plan(read)

            def _step(**kwargs: Any) -> Any:
                if read:
                    self._read_buf = torch.zeros(batch_size, len(read), device=self.dev)
                return self.model(**kwargs)

            try:
                with torch.inference_mode():
                    out = _step(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)

                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()

                        logits = out.logits[:, -1].float()
                        if temperature > 0:
                            probs = torch.softmax(logits / temperature, dim=-1).cpu()
                            nxt = torch.multinomial(probs, 1, generator=gen)[:, 0].to(self.dev)
                        else:
                            nxt = logits.argmax(dim=-1)

                        vals = self._read_buf.cpu().tolist() if read else None
                        for row_idx in range(batch_size):
                            if done[row_idx]:
                                continue

                            tok_id = int(nxt[row_idx])
                            if tok_id in self.end_ids:
                                done[row_idx] = True
                                continue

                            out_ids[row_idx].append(tok_id)
                            if read:
                                held[row_idx].append([round(x, 3) for x in vals[row_idx]])

                            text = self.tok.decode(out_ids[row_idx], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                delta = text[len(sent[row_idx]) :]
                                yield row_idx, (delta, held[row_idx]) if read else delta
                                sent[row_idx], held[row_idx] = text, []

                        if all(done):
                            break

                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
                        pos = pos[:, -1:] + 1
                        out = _step(
                            input_ids=nxt[:, None],
                            attention_mask=mask,
                            position_ids=pos,
                            past_key_values=out.past_key_values,
                            use_cache=True,
                            logits_to_keep=1,
                        )
            finally:
                self._read = None
                self._read_buf = None
                self._push_state = None
                self._abl_mask = None
                self._abl_any = False

            yield None, out_ids

    def generate_text(
        self,
        rows: Sequence[tuple[Sequence[int], Sequence[Push]]],
        max_new: int = 160,
        temperature: float = 0.0,
        cancel: threading.Event | None = None,
        abl: bool | Sequence[bool] | None = None,
        seed: int = 0,
    ) -> tuple[list[str], list[list[int]]]:
        """Run :meth:`generate` to completion and return ``(decoded_texts, token_id_lists)``."""
        final: list[list[int]] = []
        for row_idx, payload in self.generate(rows, max_new, temperature, cancel, seed=seed, abl=abl):
            if row_idx is None:
                final = payload
        return [self.tok.decode(ids, skip_special_tokens=True) for ids in final], final

    # ------------------------------------------------------------------ tracing
    def run_trace(
        self,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        pushes: Sequence[Push] | None = None,
        topk_lens: int = 5,
        abl: bool = False,
    ) -> dict[str, Any]:
        """Run one forward pass over ``prompt_ids + response_ids``, caching activations and logit lens."""
        with self.lock:
            ids = list(prompt_ids) + list(response_ids)
            num_layers = self.num_layers
            self._cap_acts = {}
            self._cap_resid = {}
            self._cap_read = {} if self.ink else None

            self._push_state = self._make_push_state([pushes or []], 1)
            if abl and self._gated([list(prompt_ids)], [True])[0]:
                self._abl_mask = torch.ones(1, dtype=torch.bool, device=self.dev)
                self._abl_any = True

            try:
                with torch.inference_mode():
                    self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)

                    lens_ids: list[np.ndarray] = []
                    lens_probs: list[np.ndarray] = []
                    for layer_idx in range(num_layers):
                        resid_layer = self.lens_state(layer_idx)
                        logits = self._head(self.final_norm(resid_layer))
                        probs = torch.softmax(logits, dim=-1)
                        top = probs.topk(topk_lens, dim=-1)
                        lens_ids.append(top.indices.cpu().numpy())
                        lens_probs.append(top.values.cpu().numpy())
                        del logits, probs

                if self.ink:
                    acts: Any = SparseActs([self._cap_acts[layer_idx] for layer_idx in range(num_layers)], self.num_neurons)
                else:
                    acts = torch.stack([self._cap_acts[layer_idx] for layer_idx in range(num_layers)]).half().numpy()
                resid = torch.stack([self._cap_resid[layer_idx] for layer_idx in range(num_layers)]).half().cpu()
            finally:
                self._cap_acts = None
                self._cap_resid = None
                self._cap_read = None
                self._push_state = None
                self._abl_mask = None
                self._abl_any = False

            impact_scale = (self.dnorm / self.resid_p50[:, None]).cpu().numpy().astype(np.float32)
            self.trace = {
                "ids": ids,
                "n_prompt": len(prompt_ids),
                "tokens": self.token_strs(ids),
                "acts": acts,
                "impact_scale": impact_scale,
                "lens_ids": np.stack(lens_ids),
                "lens_p": np.stack(lens_probs),
                "resid": resid,
            }
            return self.trace

    def trace_jlens(self, t: int, k: int = 5) -> tuple[np.ndarray, np.ndarray] | None:
        """Return J-lens top-``k`` ``(ids, probs)`` at token ``t`` of the current trace, or ``None``."""
        if self.jlens is None or self.trace is None:
            return None
        from . import jlens

        with self.lock:
            ids, probs = jlens.read(self, self.jlens, self.trace["resid"][:, t : t + 1].to(self.dev), k)
        return ids[:, 0], probs[:, 0]

    def trace_top_neurons(
        self,
        t: int | None = None,
        k: int = 40,
        response_only: bool = True,
    ) -> list[tuple[int, int, float, float, int]]:
        """Return top-``k`` neurons by residual impact at token ``t`` (or across the response when ``t is None``)."""
        tr = self.trace
        acts = tr["acts"]
        _, seq_len, num_neurons = acts.shape
        scale = tr["impact_scale"]

        if isinstance(acts, SparseActs):
            return acts.top(scale, k, t=t, start=tr["n_prompt"] if response_only and tr["n_prompt"] < seq_len else 1)

        if t is not None:
            act_slice = acts[:, t, :].astype(np.float32)
            impact = act_slice * scale
            flat = np.abs(impact).ravel()
            top_k = min(k, flat.size)
            idx = np.argpartition(-flat, top_k - 1)[:top_k] if top_k < flat.size else np.arange(flat.size)
            idx = idx[np.argsort(-flat[idx])]
            return [
                (int(i // num_neurons), int(i % num_neurons), float(impact.ravel()[i]), float(act_slice.ravel()[i]), t)
                for i in idx
            ]

        start = tr["n_prompt"] if response_only and tr["n_prompt"] < seq_len else 1
        act_slice = acts[:, start:, :].astype(np.float32)  # [L, T', N]
        impact = act_slice * scale[:, None, :]
        argmax_pos = np.abs(impact).argmax(axis=1)  # [L, N]
        best = np.take_along_axis(impact, argmax_pos[:, None, :], axis=1)[:, 0, :]  # [L, N]

        flat = np.abs(best).ravel()
        top_k = min(k, flat.size)
        idx = np.argpartition(-flat, top_k - 1)[:top_k] if top_k < flat.size else np.arange(flat.size)
        idx = idx[np.argsort(-flat[idx])]

        out: list[tuple[int, int, float, float, int]] = []
        for i in idx:
            layer_idx, neuron_idx = int(i // num_neurons), int(i % num_neurons)
            tok_pos = int(argmax_pos[layer_idx, neuron_idx]) + start
            out.append((layer_idx, neuron_idx, float(best[layer_idx, neuron_idx]), float(acts[layer_idx, tok_pos, neuron_idx]), tok_pos))
        return out

    # ------------------------------------------------------------------ neuron views
    def lens(self, layer: int, neuron: int, k: int = 12) -> dict[str, list[str]]:
        """Decode top-``k`` tokens promoted (+) and suppressed (-) by ``(layer, neuron)``."""
        with self.lock, torch.inference_mode():
            down_col = self._down_col(layer, neuron)
            vecs = torch.stack([down_col, -down_col])[:, None] * (
                self.resid_p50[layer] / down_col.float().norm().clamp(min=1e-8)
            ).to(down_col.dtype)
            if self.jlens is not None:
                vecs = vecs.to(self.jlens.dtype) @ self.jlens[layer].T
            logits = self._head(self.final_norm(vecs))[:, 0]
            up = logits[0].topk(k).indices.tolist()
            down = logits[1].topk(k).indices.tolist()

        return {"up": [self.tok.decode([t]) for t in up], "down": [self.tok.decode([t]) for t in down]}

    # ------------------------------------------------------------------ scoring & attribution
    def _batches(self, rows: Sequence[tuple[Any, ...]], budget: int = 6144) -> Iterator[list[int]]:
        """Group sequence rows into length-sorted batches within a padded-token ``budget``."""
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        batch: list[int] = []
        max_len = 0

        for idx in order:
            cand_max = max(max_len, len(rows[idx][0]))
            if batch and cand_max * (len(batch) + 1) > budget:
                yield batch
                batch, cand_max = [], len(rows[idx][0])
            batch.append(idx)
            max_len = cand_max

        if batch:
            yield batch

    def score(
        self,
        rows: Sequence[tuple[Any, ...]],
        cancel: threading.Event | None = None,
        budget: int = 6144,
        abl: bool | Sequence[bool] | None = None,
    ) -> np.ndarray:
        """Compute teacher-forced (weighted) mean target log-probability for each row.

        Each row is ``(ids, target_start, pushes)`` or ``(ids, target_start, pushes, token_weights)``.
        """
        out = np.zeros(len(rows), np.float64)

        with self.lock:
            flags, _ = self._abl_flags(abl, len(rows))

            for batch_idx in self._batches(rows, budget):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()

                seqs = [rows[i][0] for i in batch_idx]
                max_len = max(len(seq) for seq in seqs)
                pad = self.tok.pad_token_id
                ids = torch.tensor([list(q) + [pad] * (max_len - len(q)) for q in seqs], device=self.dev)
                mask = torch.tensor([[1] * len(q) + [0] * (max_len - len(q)) for q in seqs], device=self.dev)
                self._push_state = self._make_push_state([rows[i][2] for i in batch_idx], len(batch_idx))
                if flags is not None:
                    self._abl_mask = torch.tensor([flags[i] for i in batch_idx], device=self.dev)
                    self._abl_any = True

                spans = [(rows[i][1], max(rows[i][1], len(rows[i][0]))) for i in batch_idx]

                try:
                    with torch.inference_mode():
                        logprobs = self._target_logprobs(ids, mask, spans)
                finally:
                    self._push_state = None
                    self._abl_mask = None
                    self._abl_any = False

                cursor = 0
                for span_idx, row_idx in enumerate(batch_idx):
                    start, end = spans[span_idx]
                    span = logprobs[cursor : cursor + end - start]
                    cursor += end - start

                    if end <= start:
                        continue

                    weights = rows[row_idx][3] if len(rows[row_idx]) > 3 else None
                    if weights is None:
                        out[row_idx] = float(span.mean())
                    else:
                        weight_t = torch.tensor(weights, device=span.device, dtype=span.dtype)
                        out[row_idx] = float((span * weight_t).sum() / weight_t.sum())

        return out

    def _target_logprobs(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        spans: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """Compute ``log p(next token)`` only over target spans to avoid materializing full-vocab logits."""
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

        row_indices: list[int] = []
        tok_indices: list[int] = []
        for row_idx, (start, end) in enumerate(spans):
            if end <= start:
                continue
            row_indices.extend([row_idx] * (end - start))
            tok_indices.extend(range(start - 1, end - 1))

        if not row_indices:
            return torch.zeros(0, device=ids.device)

        rb = torch.tensor(row_indices, device=ids.device)
        rt = torch.tensor(tok_indices, device=ids.device)
        targets = ids[rb, rt + 1]

        chunks: list[torch.Tensor] = []
        for offset in range(0, len(rb), 256):
            logits = self._head(hidden[rb[offset : offset + 256], rt[offset : offset + 256]])
            chunks.append(logits.gather(-1, targets[offset : offset + 256, None])[:, 0] - torch.logsumexp(logits, dim=-1))

        return torch.cat(chunks)

    def attribute(
        self,
        items: Sequence[tuple[Any, ...]],
        cancel: threading.Event | None = None,
        pushes: Sequence[Push] | None = None,
        abl: bool = False,
    ) -> np.ndarray:
        """Compute exact first-order steering gradient ``dJ/d(dose)`` for every neuron: ``[L, N]`` numpy."""
        num_layers, num_neurons = self.dnorm.shape
        # For MoE models, place deltas on MLP output ([H]) and project through down columns afterwards.
        width = self.hidden if self.ink else num_neurons

        with self.lock:
            self._deltas = [
                torch.zeros(width, device=self.dev, dtype=self.model.dtype, requires_grad=True)
                for _ in range(num_layers)
            ]
            self._push_state = self._make_push_state([pushes], 1) if pushes else None
            if abl:
                self._abl_mask = torch.tensor([True], device=self.dev)
                self._abl_any = True

            try:
                for ids, start, weight, *tok_weights in items:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()

                    input_ids = torch.tensor([ids], device=self.dev)
                    with torch.enable_grad():
                        logprobs = self._target_logprobs(input_ids, None, [(start, len(ids))])
                        if tok_weights and tok_weights[0] is not None:
                            tw = torch.tensor(tok_weights[0], device=logprobs.device, dtype=logprobs.dtype)
                            loss = (logprobs * tw).sum() / tw.sum()
                        else:
                            loss = logprobs.mean()
                        (weight * loss).backward()

                grads = torch.stack(
                    [
                        delta.grad.float() if delta.grad is not None else torch.zeros(width, device=self.dev)
                        for delta in self._deltas
                    ]
                )
                if self.ink:
                    scores = self._down_scores(range(num_layers), grads)
                    grads = torch.stack([scores[layer_idx] for layer_idx in range(num_layers)])
            finally:
                self._deltas = None
                self._push_state = None
                self._abl_mask = None
                self._abl_any = False

            return (grads * self.unit).cpu().numpy()  # chain rule: add value = dose * unit

    def neurons_for_word(self, word: str, k: int = 24, skip_last: int = 0) -> list[tuple[int, int, float]]:
        """Rank neurons whose down-projection most aligns with ``word``'s unembedding (via J-lens if available)."""
        if not word.strip():
            self.last_search_partial = False
            return []
        with self.lock, torch.inference_mode():
            ids: set[int] = set()
            partial = False
            variants = (
                " " + word,
                word,
                " " + word.capitalize(),
                word.capitalize(),
                " " + word.lower(),
                word.lower(),
            )
            for variant in variants:
                toks = self.tok(variant, add_special_tokens=False).input_ids
                if len(toks) == 1:
                    ids.add(toks[0])

            if not ids:
                partial = True
                fallback = self.tok(" " + word, add_special_tokens=False).input_ids
                if not fallback:
                    self.last_search_partial = True
                    return []
                ids.add(fallback[0])

            self.last_search_partial = partial

            unembed = self.lm_head.weight
            gain = getattr(self.final_norm, "weight", None)
            gain_f: torch.Tensor | float = gain.float() if gain is not None else 1.0
            target_dir = (unembed[sorted(ids)].float().mean(0) - unembed.mean(0, dtype=torch.float32)) * gain_f
            ref_norm = float(self.resid_p50[-1])

            end_layer = max(0, self.num_layers - skip_last)
            layer_range = range(end_layer)
            layer_dirs = [
                target_dir if self.jlens is None else (target_dir.to(self.jlens.dtype) @ self.jlens[layer_idx]).float()
                for layer_idx in layer_range
            ]
            raw_scores = self._down_scores(layer_range, layer_dirs)

            best: list[tuple[int, int, float]] = []
            for layer_idx in layer_range:
                scores = raw_scores[layer_idx] / self.dnorm[layer_idx] * (self.resid_p50[layer_idx] / ref_norm)
                top = scores.abs().topk(min(k, scores.numel()))
                best.extend((layer_idx, int(n), float(scores[n])) for n in top.indices.tolist())

            best.sort(key=lambda item: -abs(item[2]))
            return best[:k]

    def mean_acts(self, ids: Sequence[int], start: int) -> np.ndarray:
        """Return every neuron's mean activation over positions ``start:`` as ``[L, N]`` numpy."""
        with self.lock:
            self._cap_acts = {}
            try:
                with torch.inference_mode():
                    self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
                if self.ink:
                    return SparseActs(
                        [self._cap_acts[layer_idx] for layer_idx in range(self.num_layers)],
                        self.num_neurons,
                    ).mean(start)
                return torch.stack(
                    [self._cap_acts[layer_idx][start:].mean(0) for layer_idx in range(self.num_layers)]
                ).numpy()
            finally:
                self._cap_acts = None

    def reference_nll(
        self,
        row_pushes: Sequence[Sequence[Push]],
        cancel: threading.Event | None = None,
        abl: bool | Sequence[bool] | None = None,
    ) -> np.ndarray:
        """Measure mean NLL on ``REFERENCE_TEXT`` for each push set to quantify fluency cost."""
        ids = self.tok(REFERENCE_TEXT).input_ids
        rows = [(ids, 1, pushes) for pushes in row_pushes]
        return -self.score(rows, cancel, abl=abl)
