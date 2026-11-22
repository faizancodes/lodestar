"""Inkling 2 Mini (Thinking Machines, model_type "inkling2_loop") for Lodestar.

Three things set it apart from the Llama-style decoders the rest of the engine was written for:

- Its MLPs are mixtures of 128 experts, and the checkpoint stores the experts in NVFP4 (layers 0-19) and MXFP8
  (layers 20-23). transformers unpacks them to bf16 on load, about 53 GB, which no 32 GB Mac holds. Here they stay
  packed on the GPU (about 16 GB) and each forward unpacks only the experts its tokens are routed to. Where even
  that doesn't fit (a 32 GB Mac), the experts stay in the checkpoint file and each forward reads the ones it's routed
  to onto the GPU, keeping the most recently used there (ExpertCache) up to a size the memory check sets. Replies are
  slower.
- It is looped: layers 6-17 run three times per token with shared weights, so 24 layers of weights make 48 layer
  applications. Lodestar treats each application as a layer of its own, numbered in the order they run (0-5, the
  three passes over 6-17, then 18-23). A forward hook on a looped layer fires three times per token, so hooks go
  through VirtualLayer, which passes on only the call of its pass.
- Its layers don't add into one residual stream. Each layer reads an attention-weighted mix of "block states" (the
  partial sums at every sixth layer) and returns the next partial sum, and the output head reads one more such mix.
  A layer's output is still where pushes, erasing and directions act; the logit lens reads it through the head's mix
  (`readout`), which is what the model would predict if it stopped there.

A neuron is one unit of one expert's down projection input: neuron n of a layer is unit n % 1152 of expert
n // 1152, so a layer has 147,456. Only the 8 experts a token is routed to are active, so a neuron is 0 at the
tokens that skip its expert. Adding to a neuron adds its down column to the layer's MLP output at every position,
which is what raising its activation does, whichever experts the router picks.
"""
import contextlib
import fcntl
import functools
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_TYPES = ("inkling2_loop",)

# the reply starts as a text message, which skips the thinking block the model would otherwise open
REPLY_PREFIX = ("<|message_model|>", "<|content_text|>")
END_TOKENS = ("<|end_message|>", "<|content_model_end_sampling|>")

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# One pass over an expert's bytes on Apple GPUs. The same unpacking in torch ops (gather, scale, cast) took 2.8 ms
# per expert matrix against 0.5 here, and a forward unpacks up to 128 experts in each of 48 layer applications.
METAL = r"""
#include <metal_stdlib>
using namespace metal;
constant float E2M1[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

// 4 bytes = 8 values (the even one in each byte's low nibble); every 16 values share a float8 e4m3 scale
kernel void nvfp4(device bfloat* out, device const uchar4* codes, device const uchar* scales,
                  device const float* fp8, constant float& factor, uint i [[thread_position_in_grid]]) {
    uchar4 b = codes[i];
    float s = fp8[scales[i >> 1]] * factor;
    uint o = 8 * i;
    out[o]     = bfloat(E2M1[b.x & 15] * s); out[o + 1] = bfloat(E2M1[b.x >> 4] * s);
    out[o + 2] = bfloat(E2M1[b.y & 15] * s); out[o + 3] = bfloat(E2M1[b.y >> 4] * s);
    out[o + 4] = bfloat(E2M1[b.z & 15] * s); out[o + 5] = bfloat(E2M1[b.z >> 4] * s);
    out[o + 6] = bfloat(E2M1[b.w & 15] * s); out[o + 7] = bfloat(E2M1[b.w >> 4] * s);
}

// 4 float8 e4m3 values; every 32 share a power-of-two scale 2^(s - 127)
kernel void mxfp8(device bfloat* out, device const uchar4* codes, device const uchar* scales,
                  device const float* fp8, uint i [[thread_position_in_grid]]) {
    uchar4 b = codes[i];
    float s = exp2(float(scales[i >> 3]) - 127.0);
    uint o = 4 * i;
    out[o] = bfloat(fp8[b.x] * s); out[o + 1] = bfloat(fp8[b.y] * s);
    out[o + 2] = bfloat(fp8[b.z] * s); out[o + 3] = bfloat(fp8[b.w] * s);
}

// y[t, r] = sum_c W[r, c] x[t, c] straight from W's packed bytes, for up to MV_ROWS rows of x: one simdgroup per row
// of W, each lane taking every 32nd scale group. Decoding a token needs W once, so unpacking all of it (above) would
// write 8 times the bytes this reads.
#define MV_ROWS 8
kernel void nvfp4_mv(device float* y, device const uchar* codes, device const uchar* scales, device const float* fp8,
                     device const bfloat* x, constant float& factor, constant uint& cols, constant uint& n,
                     constant uint& rows, uint gid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint r = gid / 32, G = cols / 16;
    device const uint2* c = (device const uint2*)(codes + r * (cols / 2));
    device const uchar* s = scales + r * G;
    float acc[MV_ROWS] = {0};
    for (uint g = lane; g < G; g += 32) {
        uint2 b = c[g];
        float v[16];
        for (uint k = 0; k < 4; k++) {
            uint lo = (b.x >> (8 * k)) & 255, hi = (b.y >> (8 * k)) & 255;
            v[2 * k] = E2M1[lo & 15]; v[2 * k + 1] = E2M1[lo >> 4];
            v[8 + 2 * k] = E2M1[hi & 15]; v[8 + 2 * k + 1] = E2M1[hi >> 4];
        }
        float sc = fp8[s[g]] * factor;
        for (uint t = 0; t < n; t++) {
            device const bfloat* xt = x + t * cols + g * 16;
            float d = 0;
            for (uint k = 0; k < 16; k++) d += v[k] * float(xt[k]);
            acc[t] += d * sc;
        }
    }
    for (uint t = 0; t < n; t++) {
        float a = simd_sum(acc[t]);
        if (lane == 0) y[t * rows + r] = a;
    }
}

kernel void mxfp8_mv(device float* y, device const uchar* codes, device const uchar* scales, device const float* fp8,
                     device const bfloat* x, constant uint& cols, constant uint& n, constant uint& rows,
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = gid / 32, G = cols / 32;
    device const uchar4* c = (device const uchar4*)(codes + r * cols);
    device const uchar* s = scales + r * G;
    float acc[MV_ROWS] = {0};
    for (uint g = lane; g < G; g += 32) {
        float v[32];
        for (uint k = 0; k < 8; k++) {
            uchar4 b = c[g * 8 + k];
            v[4 * k] = fp8[b.x]; v[4 * k + 1] = fp8[b.y]; v[4 * k + 2] = fp8[b.z]; v[4 * k + 3] = fp8[b.w];
        }
        float sc = exp2(float(s[g]) - 127.0);
        for (uint t = 0; t < n; t++) {
            device const bfloat* xt = x + t * cols + g * 32;
            float d = 0;
            for (uint k = 0; k < 32; k++) d += v[k] * float(xt[k]);
            acc[t] += d * sc;
        }
    }
    for (uint t = 0; t < n; t++) {
        float a = simd_sum(acc[t]);
        if (lane == 0) y[t * rows + r] = a;
    }
}
"""
MV_ROWS = 8  # up to this many rows of input, an expert multiplies from its packed bytes (see nvfp4_mv)
_metal = None


def _kernels():
    global _metal
    if _metal is None:
        _metal = torch.mps.compile_shader(METAL)
    return _metal


def _config(model_id):
    p = Path(model_id) / "config.json"
    if p.exists():
        return json.loads(p.read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    except Exception:
        return {}


def is_inkling(model_id):
    return _config(model_id).get("model_type") in MODEL_TYPES


# ------------------------------------------------------------------------------------------------ packed experts
class OnDisk:
    """A [n, ...] tensor left in a safetensors file, read one leading index at a time. The reads skip macOS's file
    cache (F_NOCACHE): streaming 16 GB of experts through it would crowd out everything else on the Mac, and
    ExpertCache keeps the ones that get reused."""

    def __init__(self, fd, offset, shape, dtype):
        self.fd, self.offset, self.shape, self.dtype = fd, offset, tuple(shape), dtype
        self.row = int(np.prod(self.shape[1:])) * np.dtype(dtype).itemsize

    def __getitem__(self, i):
        buf = np.empty(self.row, dtype=np.uint8)
        os.preadv(self.fd, [buf], self.offset + i * self.row)
        return torch.from_numpy(buf.view(self.dtype).reshape(self.shape[1:]))


@functools.cache
def _open_uncached(path):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    return fd


def on_disk(path, prefix):
    """The tensors in a safetensors file whose names start with `prefix` (with it cut off): the experts' codes and
    scales as OnDisk, the small per-expert factors read now."""
    dtypes = {"U8": np.uint8, "F8_E4M3": np.uint8, "I8": np.int8, "F32": np.float32}
    fd = _open_uncached(str(path))
    n = int.from_bytes(os.pread(fd, 8, 0), "little")
    header = json.loads(os.pread(fd, n, 8))
    out = {}
    for k, v in header.items():
        if k.startswith(prefix):
            t = OnDisk(fd, 8 + n + v["data_offsets"][0], v["shape"], dtypes[v["dtype"]])
            small = v["data_offsets"][1] - v["data_offsets"][0] < 1 << 20
            out[k[len(prefix):]] = torch.from_numpy(np.stack([t[i] for i in range(t.shape[0])])) if small else t
    return out


class ExpertCache:
    """Streamed experts copied onto the GPU, the most recently used kept up to `budget` bytes across all layers."""

    def __init__(self, budget, dev):
        self.budget, self.dev, self.used = budget, dev, 0
        self.held = OrderedDict()  # key -> (tensors, bytes)
        self.hits = self.misses = 0

    def get(self, key, read):
        """The tensors stored under key, or read() copied to the GPU and stored."""
        hit = self.held.get(key)
        if hit is not None:
            self.held.move_to_end(key)
            self.hits += 1
            return hit[0]
        self.misses += 1
        ts = tuple(t.to(self.dev) for t in read())
        n = sum(t.numel() * t.element_size() for t in ts)
        while self.held and self.used + n > self.budget:
            self.used -= self.held.popitem(last=False)[1][1]
        self.held[key] = (ts, n)
        self.used += n
        return ts

    def fill(self, items):
        """Read every missing (key, read) at once: a layer knows which experts its tokens go to before it runs any,
        and parallel reads go about twice as fast as one at a time (preadv lets go of the GIL)."""
        missing = [(k, r) for k, r in items if k not in self.held]
        if missing:
            for (k, _), ts in zip(missing, _readers.map(lambda kr: kr[1](), missing)):
                self.get(k, lambda ts=ts: ts)


_readers = ThreadPoolExecutor(8, thread_name_prefix="inkling-read")


class PackedExperts:
    """One layer's 128 experts as stored in the checkpoint, unpacked one expert at a time.

    w13 rows alternate gate and up (row 2k is unit k's gate, 2k+1 its up); w2 is [hidden, units]. With a cache, the
    codes and scales stay in the checkpoint file (see `on_disk`) and each expert is read onto the GPU when it's used."""

    def __init__(self, tensors, dev, cache=None, key=None):
        self.nvfp4 = "w13_weight_amax" in tensors
        self.cache, self.key = cache, key
        u8 = (lambda t: t) if cache is not None else (lambda t: (t.view(torch.uint8) if t.dtype != torch.uint8 else t).to(dev))
        self.w = {"w13": u8(tensors["w13_weight"]), "w2": u8(tensors["w2_weight"])}
        self.s = {"w13": u8(tensors["w13_weight_scale"]), "w2": u8(tensors["w2_weight_scale"])}
        if self.nvfp4:
            self.f = {k: (tensors[f"{k}_weight_amax"].float() / tensors[f"{k}_weight_outer_scale_denominator"].float()).to(dev)
                      for k in ("w13", "w2")}
            self.f_list = {k: v.tolist() for k, v in self.f.items()}  # read per call without waiting on the GPU
        self.metal = str(dev) == "mps"
        self.n_experts = self.w["w13"].shape[0]
        self.units = self.w["w2"].shape[2] * (2 if self.nvfp4 else 1)
        self.hidden = self.w["w2"].shape[1]

        # byte -> value tables: two E2M1 codes per byte (the even element in the low nibble), float8 e4m3 scales
        fp8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        fp8[fp8.isnan()] = 0.0
        e2m1 = torch.tensor(E2M1)
        b = torch.arange(256)
        self.pair = torch.stack((e2m1[b & 0xF], e2m1[b >> 4]), -1).to(dev)  # [256, 2]
        self.fp8 = fp8.to(dev)

    def packed(self, name, e):
        """Expert e's codes and scales for matrix `name`, on the GPU."""
        if self.cache is None:
            return self.w[name][e], self.s[name][e]
        return self.cache.get((self.key, name, e), lambda: (self.w[name][e], self.s[name][e]))

    def prefetch(self, experts):
        """Bring these experts onto the GPU together, ahead of using them one by one."""
        if self.cache is not None:
            self.cache.fill([((self.key, name, e), lambda name=name, e=e: (self.w[name][e], self.s[name][e]))
                             for e in experts for name in ("w13", "w2")])

    def weight(self, name, e, dtype=torch.bfloat16):
        """Expert e's matrix `name` ("w13" [2*units, hidden] or "w2" [hidden, units]), unpacked."""
        codes, scale = self.packed(name, e)
        rows = codes.shape[0]
        if self.metal:
            out = torch.empty(rows, scale.shape[1] * (16 if self.nvfp4 else 32), dtype=torch.bfloat16, device=codes.device)
            if self.nvfp4:
                _kernels().nvfp4(out, codes, scale, self.fp8, self.f_list[name][e], threads=out.numel() // 8)
            else:
                _kernels().mxfp8(out, codes, scale, self.fp8, threads=out.numel() // 4)
            return out if dtype == torch.bfloat16 else out.to(dtype)
        if self.nvfp4:
            vals = self.pair[codes.int()].view(rows, scale.shape[1], 16)
            vals = vals * (self.fp8[scale.int()] * self.f[name][e])[..., None]
        else:
            vals = self.fp8[codes.int()].view(rows, scale.shape[1], 32)
            vals = vals * torch.exp2(scale.float() - 127)[..., None]
        return vals.view(rows, -1).to(dtype)

    def mv(self, name, e, x):
        """x @ W.T in float32 for expert e's matrix `name`, read straight from its packed bytes: for up to MV_ROWS rows
        of x on Apple GPUs, where it's ~10x faster than unpacking W."""
        codes, scale = self.packed(name, e)
        rows, cols, n = codes.shape[0], scale.shape[1] * (16 if self.nvfp4 else 32), x.shape[0]
        x = x.to(torch.bfloat16).contiguous()
        y = torch.empty(n, rows, dtype=torch.float32, device=x.device)
        if self.nvfp4:
            _kernels().nvfp4_mv(y, codes, scale, self.fp8, x, self.f_list[name][e], cols, n, rows, threads=rows * 32,
                                group_size=256)
        else:
            _kernels().mxfp8_mv(y, codes, scale, self.fp8, x, cols, n, rows, threads=rows * 32, group_size=256)
        return y

    def down_columns(self, idx, dtype=torch.float32):
        """Down columns of neurons idx (expert * units + unit): [hidden, len(idx)]."""
        idx = torch.as_tensor(idx).reshape(-1).tolist()
        out = torch.empty(self.hidden, len(idx), dtype=dtype, device=self.pair.device)
        by_e = {}
        for j, n in enumerate(idx):
            by_e.setdefault(n // self.units, []).append((j, n % self.units))
        for e, js in by_e.items():
            W = self.weight("w2", e, dtype)
            out[:, [j for j, _ in js]] = W[:, [u for _, u in js]]
        return out

    def down_scores(self, U, dtype=torch.float32):
        """U [k, hidden] against every neuron's down column: [k, n_experts * units]."""
        U = U.to(dtype)
        return torch.cat([U @ self.weight("w2", e, dtype) for e in range(self.n_experts)], -1)

    def down_norms(self):
        return torch.cat([self.weight("w2", e, torch.float32).norm(dim=0) for e in range(self.n_experts)])


class _UnpackedLinear(torch.autograd.Function):
    """x @ W.T for an expert's packed W, unpacking it again in the backward pass instead of keeping it: autograd would
    otherwise hold every routed expert's unpacked weights, gigabytes per layer, until the backward ran."""

    @staticmethod
    def forward(ctx, x, store, name, e):
        ctx.store, ctx.name, ctx.e = store, name, e
        return x @ store.weight(name, e, x.dtype).T

    @staticmethod
    def backward(ctx, gy):
        return gy @ ctx.store.weight(ctx.name, ctx.e, gy.dtype), None, None, None


def _linear(x, store, name, e):
    if torch.is_grad_enabled() and x.requires_grad:
        return _UnpackedLinear.apply(x, store, name, e)
    if store.metal and x.shape[0] <= MV_ROWS:
        return store.mv(name, e, x).to(x.dtype)
    return x @ store.weight(name, e, x.dtype).T


class MoeControl:
    """What the engine asks of one layer application: per-row multipliers on its neurons, a vector added to its output
    per row, a tensor requiring grad added to its output, and where to send its activations."""

    __slots__ = ("add", "capture", "delta", "mul")

    def __init__(self, mul=None, add=None, delta=None, capture=None):
        self.mul, self.add, self.delta, self.capture = mul, add, delta, capture


class PackedMoe(nn.Module):
    """Stands in for Inkling2LoopSparseMoeBlock: the same router and gated activation, with packed experts."""

    def __init__(self, block, store, phys, tracker):
        super().__init__()
        self.gate = block.gate
        self.act_fn = block.experts.act_fn
        self.store = store
        self.phys = phys
        self.tracker = tracker
        self.control = None  # set by the engine: virtual layer -> MoeControl | None

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        _, weights, idx = self.gate(x)  # [B*T, k] each
        v = self.tracker.current(self.phys)
        ctl = self.control(v) if self.control is not None else None

        K, units, st = idx.shape[1], self.store.units, self.store
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=st.n_experts).tolist()
        st.prefetch([e for e, n in enumerate(counts) if n])
        tok_of = order // K
        w_flat = weights.reshape(-1)

        out = torch.zeros(B * T, H, device=x.device, dtype=torch.float32)
        cap = torch.zeros(B * T * K, units, device=x.device, dtype=torch.float16) \
            if ctl is not None and ctl.capture is not None else None
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            pos = order[start:start + n]
            tok = tok_of[start:start + n]
            start += n

            gu = _linear(x[tok], st, "w13", e)
            a = self.act_fn(gu[:, 0::2], gu[:, 1::2])  # [n, units]
            if ctl is not None and ctl.mul is not None:
                a = a * ctl.mul[tok // T, e * units:(e + 1) * units].to(a.dtype)
            if cap is not None:
                cap[pos] = a.detach().half()
            y = _linear(a, st, "w2", e).float() * w_flat[pos, None]
            out = out.index_add(0, tok, y)

        out = out.to(hidden_states.dtype).view(B, T, H)
        if ctl is not None:
            if ctl.add is not None:
                out = out + ctl.add[:, None, :].to(out.dtype)
            if ctl.delta is not None:
                out = out + ctl.delta.to(out.dtype)
            if cap is not None:
                ctl.capture(v, idx.view(B, T, K), cap.view(B, T, K, units))
        return out


# ------------------------------------------------------------------------------------------------ loop bookkeeping
class LoopTracker:
    """Which pass each looped layer is on during the current forward, so hooks know which application they see."""

    def __init__(self, cfg):
        self.start, self.end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        self.passes, self.n_phys = cfg.loop_num_passes, cfg.num_hidden_layers
        self.n_loop = self.end - self.start
        self.calls = [0] * self.n_phys

    @property
    def n_virtual(self):
        return self.n_phys + (self.passes - 1) * self.n_loop

    def virtual(self, phys, pass_):
        if phys < self.start:
            return phys
        if phys < self.end:
            return self.start + pass_ * self.n_loop + (phys - self.start)
        return phys + (self.passes - 1) * self.n_loop

    def physical(self, v):
        if v < self.start:
            return v
        if v < self.start + self.passes * self.n_loop:
            return self.start + (v - self.start) % self.n_loop
        return v - (self.passes - 1) * self.n_loop

    def current(self, phys):
        return self.virtual(phys, max(self.calls[phys] - 1, 0))

    def attach(self, text_model):
        def reset(*_):
            self.calls = [0] * self.n_phys
        handles = [text_model.register_forward_pre_hook(reset)]
        for i, layer in enumerate(text_model.layers):
            def count(*_, i=i):
                self.calls[i] += 1
            handles.append(layer.register_forward_pre_hook(count))
        return handles


class VirtualLayer:
    """One application of a decoder layer. Hooks registered here fire only for this application's pass."""

    def __init__(self, module, phys, v, tracker):
        self.module, self.phys, self.v, self.tracker = module, phys, v, tracker

    def __getattr__(self, name):  # anything else (self_attn, mlp, named_children, ...) is the shared module's
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def register_forward_hook(self, fn):
        return self.hook(self.module, fn)

    def register_forward_pre_hook(self, fn):
        return self.hook(self.module, fn, pre=True)

    def hook(self, sub, fn, pre=False):
        """A hook on this layer or one of its submodules that fires only during this application's pass. (The
        layer's pre-hooks run after the tracker's counting hook, registered first, so the count is this call's.)"""
        if pre:
            return sub.register_forward_pre_hook(
                lambda mod, inp: fn(mod, inp) if self.tracker.current(self.phys) == self.v else None)
        return sub.register_forward_hook(
            lambda mod, inp, out: fn(mod, inp, out) if self.tracker.current(self.phys) == self.v else None)


# ------------------------------------------------------------------------------------------------ loading
@contextlib.contextmanager
def _experts_left_out():
    """Build the model without expert weights (they load packed afterwards), and keep transformers from unpacking
    them: the checkpoint's expert tensors then go unread instead of becoming 53 GB of bf16."""
    from transformers import conversion_mapping as cm
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    class NoExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(config)
            # empty stand-ins for the weight init, which looks experts up by this class's name
            self.register_buffer("gate_up_proj", torch.empty(0), persistent=False)
            self.register_buffer("down_proj", torch.empty(0), persistent=False)

    # replace the text model's own mapping (registering one under the class name would apply both, renaming twice)
    key = "inkling2_loop_text"
    full = cm.get_checkpoint_conversion_mapping(key)
    mapping = [t for t in full if not any(str(p).startswith(r"mlp\.experts\.w") for p in t._original_source_patterns)]
    was_user = key in cm.USER_REGISTERED_MAPPINGS
    cm.register_checkpoint_conversion_mapping(key, mapping, overwrite=True)
    orig = M.Inkling2LoopExperts
    M.Inkling2LoopExperts = NoExperts
    try:
        yield M
    finally:
        M.Inkling2LoopExperts = orig
        cm._checkpoint_conversion_mapping_cache[key] = full
        if not was_user:
            cm.USER_REGISTERED_MAPPINGS.discard(key)


def _weights_files(model_id):
    root = Path(model_id)
    if not root.exists():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(model_id, ignore_patterns=["mtp.safetensors"]))
    index = root / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = sorted(f.name for f in root.glob("*.safetensors") if f.name != "mtp.safetensors")
    return root, [root / f for f in files]


def load(model_id, dev, on_layer=None, cache_bytes=None):
    """The text model with packed experts on `dev`, plus what the engine needs to drive it:
    (model, info) where info = {"tracker", "layers" (a VirtualLayer per application), "text", "n_params", "handles"}.
    on_layer(i, n) is called as each layer's experts arrive. With cache_bytes, the experts stream from the checkpoint
    file through a GPU cache of that size instead of loading."""
    from safetensors import safe_open
    from transformers import AutoConfig

    root, files = _weights_files(model_id)
    cfg = AutoConfig.from_pretrained(root).text_config

    with _experts_left_out() as M:
        model = M.Inkling2LoopForCausalLM.from_pretrained(root, config=cfg, dtype=torch.bfloat16, device_map=dev)

    text = model.model
    tracker = LoopTracker(cfg)
    handles = tracker.attach(text)

    n_params = sum(p.numel() for p in model.parameters())
    prefix = "model.llm.layers.{}.mlp.experts."
    t0 = time.perf_counter()
    cache = ExpertCache(cache_bytes, dev) if cache_bytes else None
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for i, layer in enumerate(text.layers):
                p = prefix.format(i)
                names = [k for k in keys if k.startswith(p)]
                if not names:
                    continue
                tensors = on_disk(f, p) if cache else {k[len(p):]: sf.get_tensor(k) for k in names}
                store = PackedExperts(tensors, dev, cache, i)
                n_params += store.n_experts * store.units * store.hidden * 3
                layer.mlp = PackedMoe(layer.mlp, store, i, tracker)
                if on_layer:
                    on_layer(i, len(text.layers))
    print(f"inkling: {'streamed' if cache else 'packed'} experts in {time.perf_counter() - t0:.1f}s", flush=True)

    model.eval()
    model.requires_grad_(False)
    layers = [VirtualLayer(text.layers[tracker.physical(v)], tracker.physical(v), v, tracker)
              for v in range(tracker.n_virtual)]
    return model, {"tracker": tracker, "layers": layers, "text": text, "n_params": n_params, "handles": handles,
                   "cfg": cfg, "cache": cache}


def readout(text, mod, inp, h):
    """What the output head would read if the model stopped after this layer: its mix of the block states so far and
    the layer's output h (the head's own attention over blocks, before the final norm)."""
    blocks = inp[1]
    if getattr(mod, "is_block_boundary", False):
        blocks = torch.cat([blocks, inp[0].unsqueeze(-2)], dim=-2)
    return text.final_input(blocks, h)[0].to(h.dtype)
