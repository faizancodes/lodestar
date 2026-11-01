"""Live model for Lodestar: tracing, neuron reading, steering, teacher-forced scoring and gradient attribution.

A "neuron" is one unit at the input of a layer's MLP down_proj, where silu(gate) * up lives, so reading, adding and
multiplying are all exact. Works with any Llama-style decoder in transformers (Qwen, Llama, Mistral, Gemma, ...) whose
layers expose `mlp.down_proj`.

Doses for `add` pushes are in units of the layer's median residual norm, divided by the neuron's down-column norm, so
dose 1.0 on any neuron moves the residual stream by about one typical residual's length. `mul` pushes multiply the
activation (0 = ablate).

A "direction" is a vector in the residual stream (a concept's difference in means, see interp/directions.py). A `dir`
push adds dose / len(layers) times the direction's contrast size to the output of each of its layers, so dose 1 moves
the stream by one average difference between the concept's two poles; `erase` clamps the stream's component along it
to where ordinary prompts sit, at every layer.
"""
import os
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# transformers 5 moves weights onto the device from a pool of threads, and on Apple GPUs concurrent copies race in
# torch's Metal kernel cache and abort the process (malloc corruption in MetalShaderLibrary). One thread is safe.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

LAYER_RE = re.compile(r"\blayers\.(\d+)\.")  # a decoder layer's weights, by name ("model.layers.13.mlp.down_proj.weight")

REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


@dataclass
class Push:
    layer: int | None  # None for a direction push, which carries its own layers
    neuron: int | None  # None = every neuron in the layer's MLP (mul only)
    dose: float
    mode: str = "add"  # "add" | "mul" | "dir" | "erase"
    dir_id: str | None = None
    layers: tuple = ()  # dir: the layers the dose is spread over; erase: the layer the direction is taken from

    @staticmethod
    def of(d):
        if d.get("kind") == "dir":
            op = "erase" if d.get("op") == "erase" else "dir"
            return Push(None, None, float(d["dose"]), op, str(d["dir"]), tuple(int(l) for l in d.get("layers") or ()))
        return Push(int(d["layer"]), None if d.get("neuron") is None else int(d["neuron"]), float(d["dose"]),
                    d.get("mode", "add"))


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = self.tok = None
        self.model_id = None
        self.status = "idle"
        self.progress = None  # 0..1 while a model loads, else None
        # while a model loads: {"stage": "open" | "weights" | "measure", "layers": L | None, "lit": [L] 0..1 | None},
        # how far along it is and how much of each layer's weights has arrived, for the loading scene
        self.load_state = None
        self.trace = None
        self.jlens = None  # [L, H, H] on the device once computed or loaded (interp/jlens.py)

        self._push_state = None  # {layer: (add [B, N] | None, mul [B, N] | None), "dir": {layer: [B, H]}, "erase": [...]}
        self.dirs = {}  # direction id -> {"unit": [L, H] float32 on device, "mag": [L] list, "band": [layers],
        #                                "neutral": [L, H] float32 on device | None}

        self.abl_vec = None  # [H] refusal direction to erase from every layer's residual stream, unit norm
        self.abl_coef = 1.0  # how much of the projection to remove (1.0 = fully)
        self.abl_layer = None  # the layer the direction was extracted from, for the UI
        self._abl_mask = None  # [B] bool, per row: whether this pass erases the direction
        self._abl_any = False

        self._deltas = None  # per-layer [N] tensors requiring grad, for attribution
        self._cap_acts = None  # {layer: [T, N] cpu} during a trace
        self._cap_resid = None  # {layer: [T, H]} during a trace or norm measurement
        self._read = None  # while generating with gauges: {layer: (units [D_l, H], ref [D_l], scale [D_l], cols [D_l])}
        self._read_buf = None  # [B, D]: this pass's reading of each gauged direction at the newest position
        self._handles = []

    # ---------------------------------------------------------------- loading
    def load(self, model_id):
        with self.lock:
            if self.model_id == model_id and self.model is not None:
                return self.info()
            self.status = f"loading {model_id}"
            self.load_state = {"stage": "open", "layers": self._layer_count(model_id), "lit": None}
            self.progress = 0.0
            t0 = time.perf_counter()
            took = {}
            self.unload()
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

            # the tokenizer (~0.7 s) loads alongside the weights rather than before them
            tok_box = {}
            tok_thread = threading.Thread(target=lambda: tok_box.update(tok=AutoTokenizer.from_pretrained(model_id)))
            tok_thread.start()
            self.progress = 0.04

            # straight onto the device: loading on the CPU first would hold two copies of the weights at once
            prev = hf_logging.set_tqdm_hook(self._weights_progress_hook)
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=dev)
            finally:
                hf_logging.set_tqdm_hook(prev)

            tok_thread.join()
            if "tok" not in tok_box:
                raise RuntimeError(f"Couldn't load the tokenizer for {model_id}")
            tok = tok_box["tok"]

            took["weights"] = time.perf_counter() - t0
            self.progress = 0.9
            self.load_state = {**self.load_state, "stage": "measure"}

            n_params = sum(p.numel() for p in model.parameters())
            if n_params < 2.5e9:  # small models fit comfortably in fp32, which keeps gradients and lenses precise
                model = model.float()
            model = model.eval()
            model.requires_grad_(False)

            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            self.tok, self.model, self.dev, self.model_id = tok, model, dev, model_id
            self.n_params = n_params
            self.layers = self._find_layers(model)
            self.final_norm = self._find_norm(model)
            self.lm_head = model.get_output_embeddings()
            self.backbone = getattr(model, model.base_model_prefix)  # the transformer without the vocabulary head

            for i, layer in enumerate(self.layers):
                self._handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._mlp_hook(i)))
                self._handles.append(layer.register_forward_hook(self._resid_hook(i)))

            self._use_builtin_steering(model_id)

            t1 = time.perf_counter()
            with torch.no_grad():
                self.dnorm = torch.stack([l.mlp.down_proj.weight.float().norm(dim=0) for l in self.layers])  # [L, N]
            took["norms"] = time.perf_counter() - t1

            t1 = time.perf_counter()
            self.resid_p50 = self._measure_resid_norms()
            took["reference pass"] = time.perf_counter() - t1

            self.unit = (self.resid_p50[:, None] / self.dnorm.clamp(min=1e-6))  # [L, N] activation per unit dose
            self.end_ids = self._end_ids()

            took["total"] = time.perf_counter() - t0
            took["setup"] = took["total"] - took["weights"]
            print(f"loaded {model_id}: " + ", ".join(f"{k} {v:.1f}s" for k, v in took.items()), flush=True)

            self.progress = None
            self.load_state = None
            self.status = "ready"
            return self.info()

    @staticmethod
    def _layer_count(model_id):
        """The model's number of layers from its config, before any weights load (None when it can't be read)."""
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            return getattr(getattr(cfg, "text_config", None) or cfg, "num_hidden_layers", None)
        except Exception:
            return None

    def _weights_progress_hook(self, factory, args, kwargs):
        """Route transformers' "Loading weights" bar into self.progress (0.04 to 0.9 of the whole load), and into
        load_state: how much of each layer's weights has arrived, counted against the tensors each layer has."""
        bar = factory(*args, **kwargs)
        if kwargs.get("desc") != "Loading weights":
            return bar

        eng = self
        names = [x[0] if isinstance(x, tuple) else x for x in list(args[0])] if args else []
        layer_of = lambda name: int(m.group(1)) if isinstance(name, str) and (m := LAYER_RE.search(name)) else None
        per = {}
        for nm in names:
            if (l := layer_of(nm)) is not None:
                per[l] = per.get(l, 0) + 1

        L = max(per) + 1 if per else eng.load_state.get("layers")
        got = [0] * (L or 0)
        eng.load_state = {**eng.load_state, "stage": "weights", "layers": L, "lit": [0.0] * (L or 0)}

        class Watched:
            def __iter__(self):
                total = getattr(bar, "total", None) or len(names)
                for i, x in enumerate(bar, 1):
                    if total:
                        eng.progress = 0.04 + 0.86 * i / total

                    l = layer_of(x[0] if isinstance(x, tuple) else x)
                    if l is not None and l < len(got) and per.get(l):
                        got[l] += 1
                        eng.load_state["lit"][l] = round(got[l] / per[l], 2)
                    yield x

            def __getattr__(self, k):
                return getattr(bar, k)

            def __enter__(self):
                bar.__enter__()
                return self

            def __exit__(self, *a):
                return bar.__exit__(*a)

        return Watched()

    def _use_builtin_steering(self, model_id):
        """A copy saved from Steer carries its recipe twice: as biases fitted to run in any app, and exactly, in
        lodestar_steering.safetensors. Here the exact one replaces the fitted one, so the copy steers in this app
        just as the recipe did live."""
        spec = getattr(self.model.config, "lodestar_steering", None)
        if not spec:
            return

        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        local = Path(model_id) / spec["file"]
        exact = load_file(str(local if local.exists() else hf_hub_download(model_id, spec["file"])))

        with torch.no_grad():
            for l in spec["layers"]:
                self.layers[l + 1].self_attn.o_proj.bias.zero_()
                b = exact[f"layer.{l}"].to(self.dev, self.model.dtype)
                self._handles.append(self.layers[l].mlp.down_proj.register_forward_hook(lambda m, i, o, b=b: o + b))

    def unload(self):
        for h in self._handles:
            h.remove()
        self._handles = []

        self.model = self.tok = None
        self.model_id = None
        self.trace = None
        self.jlens = None
        self.dirs = {}
        self.abl_vec = self._abl_mask = None
        self._abl_any = False

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def _find_layers(model):
        for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
            obj = model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
            except AttributeError:
                continue
            if hasattr(obj[0], "mlp") and hasattr(obj[0].mlp, "down_proj"):
                return obj

        raise RuntimeError("unsupported architecture: no layers with mlp.down_proj found")

    @staticmethod
    def _find_norm(model):
        for path in ("model.norm", "model.language_model.norm", "transformer.ln_f"):
            obj = model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue

        raise RuntimeError("no final norm found")

    def _end_ids(self):
        ids = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id:
                ids.add(i)

        return {i for i in ids if i is not None}

    def info(self):
        if self.model is None or self.status != "ready":  # still measuring layers right after the weights load
            return {"loaded": False, "status": self.status}

        return {"loaded": True, "status": self.status, "model": self.model_id, "device": self.dev,
                "dtype": str(self.model.dtype).replace("torch.", ""), "params": self.n_params,
                "layers": len(self.layers), "neurons": int(self.dnorm.shape[1]),
                "hidden": int(self.layers[0].mlp.down_proj.out_features)}

    def abl_of(self, vec, coef, layer=None):
        """Store the unit refusal direction to erase (raw residual coords), as a tensor on the model's device."""
        if self.model is None:
            raise RuntimeError("no model loaded")

        v = torch.tensor([float(x) for x in vec], device=self.dev).float()
        v = v / v.norm().clamp(min=1e-8)
        self.abl_vec = v.to(self.model.dtype)
        self.abl_vec_f = v  # keep the float32 copy for weight edits
        self.abl_coef = float(coef)
        self.abl_layer = layer

    def clear_abl(self):
        self.abl_vec = self._abl_mask = None
        self.abl_layer = None
        self._abl_any = False

    @staticmethod
    def _abl_flags(abl, n):
        """Normalize a per-row ablation spec (True for all rows, a bool list, or None)."""
        if abl is None:
            return None, False
        flags = [bool(abl)] * n if isinstance(abl, bool) else [bool(x) for x in abl]
        return flags, any(flags)

    def prompt_end_len(self):
        """How many tokens the chat template puts after the user's words (e.g. Qwen3's
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" is 9): the common tail of two templated prompts."""
        a, b = self.chat_ids("What is the capital of France?"), self.chat_ids("tell me a joke")
        n = 0
        while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
            n += 1
        return max(1, min(n, 12))

    def prompt_end_states(self, prompts, n_pos=1, cancel=None, budget=6144):
        """Residual stream at the last n_pos tokens of each prompt (chat templated, left padded), after every layer:
        [n_prompts, L+1, n_pos, H] float16 cpu. Index 0 is the embedding output, i>0 is the post-layer-i residual."""
        with self.lock:
            ids_list = [self.chat_ids(p) for p in prompts]
            out = []

            for bi in self._batches([(x,) for x in ids_list], budget):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()

                seqs = [ids_list[i] for i in bi]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)

                with torch.inference_mode():
                    hs = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                                    output_hidden_states=True).hidden_states
                    last = torch.stack([h[:, -n_pos:, :] for h in hs], 1)  # [B, L+1, n_pos, H]
                    out.append(last.half().cpu())
                    del hs, last

            return torch.cat(out).numpy() if out else np.zeros(
                (0, len(self.layers) + 1, n_pos, int(self.layers[0].mlp.down_proj.out_features)), np.float16)

    # ---------------------------------------------------------------- hooks
    def _mlp_hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._cap_acts is not None:
                self._cap_acts[i] = h[0].detach().float().cpu()

            changed = False
            st = self._push_state.get(i) if self._push_state else None
            if st is not None:
                add, mul = st
                if mul is not None:
                    h = h * mul[:, None, :].to(h.dtype)
                if add is not None:
                    h = h + add[:, None, :].to(h.dtype)
                changed = True

            if self._deltas is not None:
                h = h + self._deltas[i].to(h.dtype)
                changed = True

            return (h,) if changed else None

        return hook

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            changed = False
            st = self._push_state
            off = st["dir"].get(i) if st and "dir" in st else None

            if off is not None:
                h = h + off[:, None, :]
                changed = True

            if self._abl_any and self.abl_vec is not None:
                mask = self._abl_mask.to(h.dtype)[:, None, None]
                v = self.abl_vec
                proj = (h @ v)[:, :, None] * v  # the component of each position along the refusal direction
                h = h - mask * self.abl_coef * proj
                changed = True

            for coef, v, ref in (st.get("erase") or ()) if st else ():
                # clamp to where ordinary prompts sit along v, not to 0: zeroing a direction that ordinary text sits
                # far from would push every reply toward its other pole
                p = h @ v - ref[i]
                h = h - coef[:, None, None] * p[:, :, None] * v
                changed = True

            if self._cap_resid is not None:
                self._cap_resid[i] = h.detach()[0]

            rd = self._read.get(i) if self._read else None
            if rd is not None:
                units, ref, scale, cols = rd
                self._read_buf[:, cols] += (h[:, -1].float() @ units.T - ref) * scale

            if changed:
                return (h,) if isinstance(out, tuple) else h

        return hook

    def _measure_resid_norms(self):
        ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
        self._cap_resid = {}

        try:
            with torch.inference_mode():
                self.model(input_ids=ids, use_cache=False)
            return torch.tensor([float(self._cap_resid[l][1:].float().norm(dim=-1).median())
                                 for l in range(len(self.layers))], device=self.dev)
        finally:
            self._cap_resid = None

    def _make_push_state(self, row_pushes, B):
        """row_pushes: list (len B) of lists of Push. Returns {layer: (add [B,N], mul [B,N])}, plus "dir":
        {layer: [B, H]} for direction pushes and "erase": [(coef [B], unit [H], ref [L])] for erased directions."""
        N = self.dnorm.shape[1]
        st = {}
        dirs, erase = {}, {}

        for r, pushes in enumerate(row_pushes):
            for p in pushes or []:
                if p.mode in ("dir", "erase"):
                    d = self.dirs.get(p.dir_id)
                    if d is None:
                        raise ValueError(f"No direction called {p.dir_id!r} for this model.")

                    if p.mode == "dir":
                        for l in p.layers:
                            off = dirs.get(l)
                            if off is None:
                                off = dirs[l] = torch.zeros(B, d["unit"].shape[1], device=self.dev, dtype=torch.float32)
                            off[r] += (p.dose / len(p.layers) * d["mag"][l]) * d["unit"][l]
                    else:
                        src = p.layers[0] if p.layers else d["band"][len(d["band"]) // 2]
                        key = (p.dir_id, src)

                        if key not in erase:
                            v = d["unit"][src]

                            # where ordinary prompts sit along v at each layer (0 when the direction has no neutral means)
                            ref = d["neutral"] @ v if d.get("neutral") is not None \
                                else torch.zeros(len(self.layers), device=self.dev)
                            dt = self.model.dtype
                            erase[key] = [torch.zeros(B, device=self.dev), v.to(dt), ref.to(dt)]

                        erase[key][0][r] += p.dose
                    continue

                add, mul = st.get(p.layer, (None, None))
                if p.mode == "mul":
                    if mul is None:
                        mul = torch.ones(B, N, device=self.dev)
                    if p.neuron is None:
                        mul[r] *= p.dose
                    else:
                        mul[r, p.neuron] *= p.dose
                else:
                    if p.neuron is None:
                        raise ValueError("whole-layer pushes must use mode 'mul'")
                    if add is None:
                        add = torch.zeros(B, N, device=self.dev)
                    add[r, p.neuron] += p.dose * self.unit[p.layer, p.neuron]

                st[p.layer] = (add, mul)

        if dirs:
            st["dir"] = {l: off.to(self.model.dtype) for l, off in dirs.items()}
        if erase:
            st["erase"] = [(c.to(self.model.dtype), v, ref) for c, v, ref in erase.values()]

        return st or None

    # ---------------------------------------------------------------- tokens
    def chat_ids(self, prompt, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.messages_ids(msgs)

    def messages_ids(self, messages):
        if self.tok.chat_template is None:
            text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
            return self.tok(text).input_ids

        try:
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                               tokenize=True)
        except Exception:  # templates without system-role support: fold the system prompt into the first user turn
            if messages and messages[0]["role"] == "system":
                sys_, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
                rest[0]["content"] = f"{sys_}\n\n{rest[0]['content']}"
                return self.messages_ids(rest)
            raise

        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def token_strs(self, ids):
        """Display string for each token, keeping multi-byte characters intact on the token that completes them."""
        out, prev = [], ""

        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            if full.endswith("\ufffd") and i < len(ids) - 1:
                out.append("")
                continue
            out.append(full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[i]]))
            prev = full

        return out

    # ---------------------------------------------------------------- generation
    def _read_plan(self, dir_ids):
        """How to read directions as gauges: at each layer of a direction's band, the stream's position along it,
        from where ordinary prompts sit, in units of the gap between its two poles, averaged over the band."""
        by_layer = {}
        for c, did in enumerate(dir_ids):
            d = self.dirs.get(did)
            if d is None:
                raise ValueError(f"No direction called {did!r} for this model.")

            band = d["band"] or list(range(len(self.layers) // 3, 2 * len(self.layers) // 3))
            for l in band:
                v = d["unit"][l]
                ref = float(d["neutral"][l] @ v) if d.get("neutral") is not None else 0.0
                by_layer.setdefault(l, []).append((v, ref, 1 / (d["mag"][l] * len(band)), c))

        return {l: (torch.stack([x[0] for x in xs]), torch.tensor([x[1] for x in xs], device=self.dev),
                    torch.tensor([x[2] for x in xs], device=self.dev), torch.tensor([x[3] for x in xs], device=self.dev))
                for l, xs in by_layer.items()}

    def generate(self, rows, max_new=160, temperature=0.0, cancel=None, seed=0, abl=None, read=None):
        """rows: list of (prompt_ids, [Push]). Decodes all rows in one batch. `abl` erases the refusal direction
        from the residual stream of the rows it marks (True/False/bool list). Yields (row, text_delta) and finally
        (None, [generated ids per row]). With `read` (direction ids), each delta comes as (row, (text, readings)):
        for every token in it, each direction's gauge at the position that chose the token."""
        with self.lock:
            B = len(rows)
            flags, any_abl = self._abl_flags(abl, B)

            seqs = [r[0] for r in rows]
            T = max(map(len, seqs))
            pad = self.tok.pad_token_id
            ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
            mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)

            done, out_ids, sent = [False] * B, [[] for _ in range(B)], [""] * B
            held = [[] for _ in range(B)]  # readings of the tokens not yet sent as text
            gen = torch.Generator(device="cpu").manual_seed(seed)

            self._push_state = self._make_push_state([r[1] for r in rows], B)
            if any_abl:
                self._abl_mask = torch.tensor(flags, device=self.dev)
                self._abl_any = True
            if read:
                self._read = self._read_plan(read)

            def step(**kw):
                if read:
                    self._read_buf = torch.zeros(B, len(read), device=self.dev)
                return self.model(**kw)

            try:
                with torch.inference_mode():
                    out = step(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)

                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()

                        logits = out.logits[:, -1].float()
                        if temperature > 0:
                            probs = torch.softmax(logits / temperature, -1).cpu()
                            nxt = torch.multinomial(probs, 1, generator=gen)[:, 0].to(self.dev)
                        else:
                            nxt = logits.argmax(-1)

                        vals = self._read_buf.cpu().tolist() if read else None
                        for r in range(B):
                            if done[r]:
                                continue

                            t = int(nxt[r])
                            if t in self.end_ids:
                                done[r] = True
                                continue

                            out_ids[r].append(t)
                            if read:
                                held[r].append([round(x, 3) for x in vals[r]])

                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, (text[len(sent[r]):], held[r]) if read else text[len(sent[r]):]
                                sent[r], held[r] = text, []

                        if all(done):
                            break

                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = step(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                   past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                self._read = self._read_buf = None
                self._push_state = None
                self._abl_mask = None
                self._abl_any = False

            yield None, out_ids

    def generate_text(self, rows, max_new=160, temperature=0.0, cancel=None, abl=None):
        final = None
        for r, x in self.generate(rows, max_new, temperature, cancel, abl=abl):
            if r is None:
                final = x
        return [self.tok.decode(ids, skip_special_tokens=True) for ids in final], final

    # ---------------------------------------------------------------- tracing
    def run_trace(self, prompt_ids, response_ids, pushes=None, topk_lens=5, abl=False):
        """One forward pass over prompt + response, capturing every neuron at every token and the logit lens at every
        layer. Stored on self.trace for later queries."""
        with self.lock:
            ids = list(prompt_ids) + list(response_ids)
            L = len(self.layers)
            self._cap_acts, self._cap_resid = {}, {}

            self._push_state = self._make_push_state([pushes or []], 1)
            if abl:
                self._abl_mask = torch.ones(1, dtype=torch.bool, device=self.dev)
                self._abl_any = True

            try:
                with torch.inference_mode():
                    # the hooks capture what's needed; the model's own logits aren't
                    self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)

                    lens_ids, lens_p = [], []
                    for l in range(L):
                        r = self._cap_resid[l]
                        lg = self.lm_head(self.final_norm(r)).float()
                        p = torch.softmax(lg, -1)
                        top = p.topk(topk_lens, dim=-1)
                        lens_ids.append(top.indices.cpu().numpy())
                        lens_p.append(top.values.cpu().numpy())
                        del lg, p

                acts = torch.stack([self._cap_acts[l] for l in range(L)]).half().numpy()  # [L, T, N]
                resid = torch.stack([self._cap_resid[l] for l in range(L)]).half().cpu()  # [L, T, H], for the J-lens
            finally:
                self._cap_acts = self._cap_resid = None
                self._push_state = None
                self._abl_mask = None
                self._abl_any = False

            impact_scale = (self.dnorm / self.resid_p50[:, None]).cpu().numpy().astype(np.float32)  # [L, N]
            self.trace = {
                "ids": ids, "n_prompt": len(prompt_ids), "tokens": self.token_strs(ids), "acts": acts,
                "impact_scale": impact_scale, "lens_ids": np.stack(lens_ids), "lens_p": np.stack(lens_p), "resid": resid,
            }

            return self.trace

    def trace_jlens(self, t, k=5):
        """The J-lens at token t of the trace, every layer: (ids [L, k], p [L, k]), or None without a lens."""
        if self.jlens is None or self.trace is None:
            return None
        from . import jlens
        with self.lock:
            ids, p = jlens.read(self, self.jlens, self.trace["resid"][:, t : t + 1].to(self.dev), k)
        return ids[:, 0], p[:, 0]

    def trace_top_neurons(self, t=None, k=40, response_only=True):
        """Neurons with the largest impact (|activation| x down-column norm / layer residual median) at token t, or
        over the whole response when t is None. Returns [(layer, neuron, signed impact, activation, token)]."""
        tr = self.trace
        acts = tr["acts"]
        L, T, N = acts.shape
        sc = tr["impact_scale"]

        if t is not None:
            a = acts[:, t, :].astype(np.float32)
            imp = a * sc
            flat = np.abs(imp).ravel()
            idx = np.argpartition(-flat, k)[:k]
            idx = idx[np.argsort(-flat[idx])]

            return [(int(i // N), int(i % N), float(imp.ravel()[i]), float(a.ravel()[i]), t) for i in idx]

        s = tr["n_prompt"] if response_only and tr["n_prompt"] < T else 1
        a = acts[:, s:, :].astype(np.float32)  # [L, T', N]
        imp = a * sc[:, None, :]
        am = np.abs(imp).argmax(1)  # [L, N]
        best = np.take_along_axis(imp, am[:, None, :], 1)[:, 0, :]  # [L, N]

        flat = np.abs(best).ravel()
        idx = np.argpartition(-flat, k)[:k]
        idx = idx[np.argsort(-flat[idx])]

        out = []
        for i in idx:
            l, n = int(i // N), int(i % N)
            tt = int(am[l, n]) + s
            out.append((l, n, float(best[l, n]), float(acts[l, tt, n]), tt))

        return out

    # ---------------------------------------------------------------- neuron views
    def lens(self, layer, n, k=12):
        """Tokens the neuron's down column pushes up (+) and down (-): through the layer's J-lens when the model has one
        (what the later layers make of the push), else straight through the final norm and unembedding."""
        with self.lock, torch.inference_mode():
            d = self.layers[layer].mlp.down_proj.weight[:, n]
            vecs = torch.stack([d, -d])[:, None] * (self.resid_p50[layer] / d.float().norm()).to(d.dtype)
            if self.jlens is not None:
                vecs = vecs.to(self.jlens.dtype) @ self.jlens[layer].T
            logits = self.lm_head(self.final_norm(vecs))[:, 0].float()
            up = logits[0].topk(k).indices.tolist()
            down = logits[1].topk(k).indices.tolist()

        return {"up": [self.tok.decode([t]) for t in up], "down": [self.tok.decode([t]) for t in down]}

    # ---------------------------------------------------------------- scoring
    def _batches(self, rows, budget=6144):
        """Split rows (each with a token list as first element) into batches under a padded-token budget."""
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        batch, mx = [], 0

        for i in order:
            m = max(mx, len(rows[i][0]))
            if batch and m * (len(batch) + 1) > budget:
                yield batch
                batch, m = [], len(rows[i][0])
            batch.append(i)
            mx = m

        if batch:
            yield batch

    def score(self, rows, cancel=None, budget=6144, abl=None):
        """Teacher-forced mean log-probability of each row's target span. rows: list of (ids, target_start, [Push]) or
        (ids, target_start, [Push], weights), where weights (one per target token) make it a weighted mean.
        `abl` erases the refusal direction from the rows it marks. Returns np.array [len(rows)]."""
        out = np.zeros(len(rows), np.float64)

        with self.lock:
            flags, _ = self._abl_flags(abl, len(rows))

            for bi in self._batches(rows, budget):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()

                seqs = [rows[i][0] for i in bi]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id
                ids = torch.tensor([q + [pad] * (T - len(q)) for q in seqs], device=self.dev)
                mask = torch.tensor([[1] * len(q) + [0] * (T - len(q)) for q in seqs], device=self.dev)
                self._push_state = self._make_push_state([rows[i][2] for i in bi], len(bi))
                if flags is not None:
                    self._abl_mask = torch.tensor([flags[i] for i in bi], device=self.dev)
                    self._abl_any = True

                spans = [(rows[i][1], max(rows[i][1], len(rows[i][0]))) for i in bi]

                try:
                    with torch.inference_mode():
                        lp = self._target_logprobs(ids, mask, spans)
                finally:
                    self._push_state = None
                    self._abl_mask = None
                    self._abl_any = False

                k = 0
                for j, i in enumerate(bi):
                    s, n = spans[j]
                    span = lp[k:k + n - s]
                    k += n - s

                    if n <= s:
                        continue

                    w = rows[i][3] if len(rows[i]) > 3 else None
                    if w is None:
                        out[i] = float(span.mean())
                    else:
                        w = torch.tensor(w, device=span.device, dtype=span.dtype)
                        out[i] = float((span * w).sum() / w.sum())

        return out

    def _target_logprobs(self, ids, mask, spans):
        """log p(next token) at the target positions only. spans: per row (start, end) of the target tokens.
        Runs the backbone once, then the vocabulary projection in chunks over just those positions: building
        float log-softmaxes over a 150k vocabulary at every position (prompts included) took gigabytes per batch
        on an 8B model and drove the machine into swap."""
        h = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

        rb, rt = [], []
        for j, (s, n) in enumerate(spans):
            rb += [j] * (n - s)
            rt += range(s - 1, n - 1)

        rb = torch.tensor(rb, device=ids.device)
        rt = torch.tensor(rt, device=ids.device)
        tgt = ids[rb, rt + 1]

        out = []
        for k in range(0, len(rb), 256):
            lg = self.lm_head(h[rb[k:k + 256], rt[k:k + 256]]).float()
            out.append(lg.gather(-1, tgt[k:k + 256, None])[:, 0] - torch.logsumexp(lg, -1))

        return torch.cat(out) if out else torch.zeros(0, device=ids.device)

    def attribute(self, items, cancel=None, pushes=None, abl=False):
        """First-order steering effect of every neuron. items: list of (ids, target_start, weight) or
        (ids, target_start, weight, token_weights). Objective J = sum(weight * (weighted) mean target logprob). Returns dJ/d(dose) for an `add` push on each neuron: [L, N] numpy,
        i.e. J(dose) ~= J(0) + dose * grad. A constant added at every position is exactly what steering does.
        `pushes` (and `abl`) take the gradient with a recipe already on, for refining it."""
        L, N = self.dnorm.shape

        with self.lock:
            self._deltas = [torch.zeros(N, device=self.dev, dtype=self.model.dtype, requires_grad=True)
                            for _ in range(L)]

            self._push_state = self._make_push_state([pushes], 1) if pushes else None
            if abl:
                self._abl_mask, self._abl_any = torch.tensor([True], device=self.dev), True

            try:
                for ids, s, w, *tw in items:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()

                    x = torch.tensor([ids], device=self.dev)

                    with torch.enable_grad():
                        lp = self._target_logprobs(x, None, [(s, len(ids))])
                        if tw and tw[0] is not None:
                            tw_ = torch.tensor(tw[0], device=lp.device, dtype=lp.dtype)
                            lp = (lp * tw_).sum() / tw_.sum()
                        else:
                            lp = lp.mean()
                        (w * lp).backward()

                g = torch.stack([d.grad.float() if d.grad is not None else torch.zeros(N, device=self.dev)
                                 for d in self._deltas])
            finally:
                self._deltas = None
                self._push_state = None
                self._abl_mask, self._abl_any = None, False

            return (g * self.unit).cpu().numpy()  # chain rule: add value = dose * unit

    def neurons_for_word(self, word, k=24, skip_last=0):
        """Neurons whose push most directly raises (or, pushed negative, raises) the logit of `word`: the word's
        unembedding row (centred on the vocabulary mean, through the final norm's gain) against each neuron's down
        column, per unit dose. Returns [(layer, neuron, signed score)], strongest first; a negative score means the
        neuron must be turned down to favour the word. With a J-lens the word is first carried back to each layer
        (J_l^T u), so a neuron scores by what the later layers make of its push, not only its direct effect; without
        one, later layers dominate."""
        with self.lock, torch.inference_mode():
            # only spellings that are one whole token; a fragment like "P" from " pirate" would match noise
            ids, partial = set(), False
            variants = (" " + word, word, " " + word.capitalize(), word.capitalize(), " " + word.lower(), word.lower())
            for v in variants:
                t = self.tok(v, add_special_tokens=False).input_ids
                if len(t) == 1:
                    ids.add(t[0])

            if not ids:  # the word is always split: fall back to its first piece, and say so
                partial = True
                ids.add(self.tok(" " + word, add_special_tokens=False).input_ids[0])

            self.last_search_partial = partial

            W = self.lm_head.weight
            g = getattr(self.final_norm, "weight", None)
            g = g.float() if g is not None else 1.0
            u = (W[sorted(ids)].float().mean(0) - W.mean(0, dtype=torch.float32)) * g  # [H]
            ref = float(self.resid_p50[-1])

            best = []
            for l, layer in enumerate(self.layers[: len(self.layers) - skip_last]):
                D = layer.mlp.down_proj.weight
                ul = u if self.jlens is None else (u.to(self.jlens.dtype) @ self.jlens[l]).float()
                sc = (ul.to(D.dtype) @ D).float() / self.dnorm[l] * (self.resid_p50[l] / ref)  # [N]
                top = sc.abs().topk(k)
                best += [(l, int(n), float(sc[n])) for n in top.indices.tolist()]

            best.sort(key=lambda x: -abs(x[2]))
            return best[:k]

    def mean_acts(self, ids, start):
        """Every neuron's mean activation over positions start..end of ids: [L, N] numpy."""
        with self.lock:
            self._cap_acts = {}

            try:
                with torch.inference_mode():
                    self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
                return torch.stack([self._cap_acts[l][start:].mean(0) for l in range(len(self.layers))]).numpy()
            finally:
                self._cap_acts = None

    def reference_nll(self, row_pushes, cancel=None, abl=None):
        """Mean NLL of a fixed neutral passage under each push set: how much a push damages ordinary language."""
        ids = self.tok(REFERENCE_TEXT).input_ids
        rows = [(ids, 1, p) for p in row_pushes]
        return -self.score(rows, cancel, abl=abl)
