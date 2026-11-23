"""The J-lens: a logit lens corrected for downstream layer Jacobians.

Implements the J-lens from Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models"
(2026). Rather than projecting layer ``l``'s residual ``h`` directly through the final norm and unembedding, the
J-lens first maps ``h`` to the final layer via ``J_l``, the Jacobian of the final-layer residual with respect to
layer ``l``'s residual averaged over prompts, source positions ``t``, and all later positions ``t' >= t``::

    J_l = E[ d h_final,t' / d h_l,t ],    lens(h) = softmax(W_U norm(J_l h))

Each model's lens is persisted to ``DATA/jlens/<model>.safetensors`` as ``J`` with shape ``[L, H, H]`` in float16.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .engine import Cancelled
from .utils import slug, unwrap_tensor


if TYPE_CHECKING:
    from .engine import Engine


# Varied ordinary prose and code used to estimate the average layer-to-output Jacobian.
PASSAGES = [
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, and "
    "by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. The bridge "
    "burned twice. Each time the town rebuilt it in stone a little wider than before, and each time the neighbours "
    "raised their tolls to match, until a bishop was sent to settle the matter and stayed for thirty years.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\n\n\ndef sort(xs):\n    if len(xs) < 2:\n        return xs\n    mid = len(xs) // 2\n"
    "    return merge(sort(xs[:mid]), sort(xs[mid:]))\n\n\nif __name__ == '__main__':\n    print(sort([5, 3, 9, 1, 4]))\n",
    "She asked me what I thought of the plan, and I told her honestly that it seemed rushed. She didn't say anything "
    "for a while. Then she picked up her coat, looked out at the rain, and said that rushed was the only speed she had "
    "left. I offered to drive her to the station. She said no, she'd rather walk, and that I should finish my tea.",
    "Photosynthesis takes place in the chloroplasts. Light is absorbed by chlorophyll, and its energy splits water "
    "into oxygen, protons and electrons. The electrons pass along a chain of proteins in the thylakoid membrane, "
    "pumping protons as they go, and the gradient this builds drives ATP synthase. In the Calvin cycle the ATP and "
    "NADPH are spent fixing carbon dioxide into sugar.",
    "Preheat the oven to 200 degrees. Toss the potatoes in olive oil with salt, pepper and two crushed cloves of "
    "garlic, then spread them on a tray in a single layer so they roast rather than steam. After twenty minutes, turn "
    "them with a spatula and add the rosemary. They are done when the edges are brown and crisp and a knife slides "
    "into the middle without resistance.",
    "The tenant shall keep the premises in good repair and shall not make any alteration to the structure without the "
    "prior written consent of the landlord, such consent not to be unreasonably withheld. Rent is payable monthly in "
    "advance on the first day of each month. If any payment is more than fourteen days late, interest shall accrue at "
    "four percent above the base rate until it is paid in full.",
    "Let x be a real number with x^2 - 5x + 6 = 0. Factoring, (x - 2)(x - 3) = 0, so x = 2 or x = 3. More generally, "
    "a quadratic ax^2 + bx + c has roots (-b ± sqrt(b^2 - 4ac)) / 2a, and the discriminant b^2 - 4ac tells us how many "
    "there are: two if it is positive, one if it is zero, and none among the reals if it is negative.",
    "Shares fell sharply on Tuesday after the central bank signalled that interest rates would stay higher for longer "
    "than markets had expected. Technology stocks led the decline, while energy companies edged up on the back of "
    "rising oil prices. Analysts said investors were now pricing in only one cut before the end of the year, down "
    "from three a month ago.",
    "The old lighthouse keeper climbed the stairs every evening at dusk, one hundred and twelve of them, counting under "
    "his breath. At the top he trimmed the wick, polished the great lens with a soft cloth, and lit the lamp. Then he "
    "sat by the window with his pipe and watched the beam sweep out across the water, over the rocks, and back again.",
    "To reset your password, open Settings and choose Account. Under Security, select Change password. You will be "
    "asked for your current password, then for the new one twice. It must be at least twelve characters long. If you "
    "have forgotten your current password, sign out and use the Forgot password link on the sign-in screen instead.",
    "Dear Maria, thank you so much for having us last weekend. The children haven't stopped talking about the lake and "
    "the boat and the enormous dog. I've enclosed the photographs I promised, including the one of Tom falling off the "
    "jetty, which I think you'll agree is the best of them. Do let us know when you're next in town. With love, Anna.",
    "Volcanoes form where magma reaches the surface. Most lie along the boundaries of tectonic plates: where an oceanic "
    "plate dives beneath another, water released from it lowers the melting point of the mantle above, and the magma "
    "that results rises to build chains of explosive volcanoes such as those around the Pacific. Others, like Hawaii, "
    "sit over hot spots far from any boundary.",
]

QA = [
    (
        "How do I keep basil alive indoors?",
        "Basil wants light more than anything: put it in the sunniest window you have, ideally six hours of direct sun a "
        "day. Water when the top of the soil feels dry, and pinch off the flower buds as soon as they appear, because "
        "once it flowers the leaves turn bitter and the plant stops growing. Harvest from the top, just above a pair of "
        "leaves, and it will branch out and grow bushier.",
    ),
    (
        "Why is the sky blue?",
        "Sunlight contains every colour, and as it passes through the atmosphere it scatters off the gas molecules in the "
        "air. Shorter wavelengths scatter much more strongly than longer ones, so blue light is scattered across the "
        "whole sky while red and yellow mostly pass straight through. When you look anywhere other than at the sun, you "
        "see that scattered blue light. At sunset the light travels through more air, and the blues are scattered away.",
    ),
    (
        "Can you explain what a hash map is?",
        "A hash map stores pairs of keys and values so that you can look a value up by its key quickly. It keeps an array "
        "of buckets, runs each key through a hash function to get a number, and uses that number to pick a bucket. Two "
        "keys can land in the same bucket, so each bucket holds a short list, or the map probes for the next free slot. "
        "As long as the buckets stay sparse, lookups take constant time on average.",
    ),
    (
        "Write a short poem about autumn.",
        "The maples burn along the lane,\nthe light goes amber, low and thin,\nthe geese write lines across the rain\nand "
        "every door lets woodsmoke in.\nThe garden lets its colour go,\nthe apples soften where they fall,\nand somewhere "
        "underneath, below,\nthe roots are listening for it all.",
    ),
]


def corpus(eng: Engine, seq_len: int) -> list[list[int]]:
    """Build tokenized rows of exact length ``seq_len`` from passages and chat QA turns."""
    rows = [eng.encode(passage) for passage in PASSAGES]
    for question, answer in QA:
        rows.append(eng.chat_ids(question) + eng.encode(answer))
    rows = [row[:seq_len] for row in rows if len(row) >= seq_len]
    if not rows:
        raise RuntimeError("No passage is long enough for the lens's context length.")
    return rows


def path_for(root: str | Path, model_id: str) -> Path:
    """Return the ``.safetensors`` file path storing ``model_id``'s J-lens under ``root``."""
    return Path(root) / f"{slug(model_id)}.safetensors"


@torch.no_grad()
def _scale(jacobians: torch.Tensor) -> torch.Tensor:
    """Rescale each layer's matrix to unit RMS singular value (Frobenius norm / sqrt(H)).

    Averaging over position pairs shrinks the norm, and while the final norm rescales activations anyway,
    a very small ``J h`` would sit near the normalization epsilon.
    """
    hidden = jacobians.shape[-1]
    scale = (jacobians.flatten(1).norm(dim=1) / (hidden**0.5)).clamp(min=1e-12)
    return jacobians / scale[:, None, None]


def compute(
    eng: Engine,
    cancel: threading.Event | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    T: int = 64,
    batch: int = 8,
    skip: int = 1,
) -> torch.Tensor:
    """Compute per-layer J-lens Jacobians ``[L, H, H]`` as float32 on CPU.

    Args:
        eng: Loaded :class:`~interp.engine.Engine`.
        cancel: Optional event signalling cancellation.
        progress: Optional callback ``(done, total, elapsed_seconds)`` invoked after each batch.
        T: Sequence length per corpus row.
        batch: Number of basis coordinates processed per backward pass.
        skip: Number of leading positions (attention sinks) excluded from the average.

    Returns:
        Normalized Jacobian tensor of shape ``[L, H, H]`` on CPU.
    """
    num_layers = eng.num_layers
    hidden_size = eng.hidden_size
    rows = corpus(eng, T)
    jacobians = torch.zeros(num_layers, hidden_size, hidden_size)
    kept_positions = torch.arange(skip, T)
    num_pairs = len(kept_positions) * (len(kept_positions) + 1) / 2  # (t, t') with t <= t', both kept
    embed = eng.model.get_input_embeddings()

    caps: dict[int | str, torch.Tensor] = {}

    def _make_capture(key: int | str):
        def _hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            caps[key] = unwrap_tensor(out)

        return _hook

    hooks = [layer.register_forward_hook(_make_capture(layer_idx)) for layer_idx, layer in enumerate(eng.layers)]
    target: int | str = num_layers - 1
    if eng.ink:
        target = "head"
        hooks.append(eng.ink["text"].final_input.register_forward_hook(_make_capture("head")))

    done = 0
    t0 = time.perf_counter()
    model = eng.model

    try:
        for k0 in range(0, hidden_size, batch):
            if (cancel is not None and cancel.is_set()) or eng.model is not model:
                raise Cancelled()

            ks = list(range(k0, min(hidden_size, k0 + batch)))
            batch_size = len(ks)
            ids = torch.tensor([rows[(k0 + offset) % len(rows)] for offset in range(batch_size)])
            caps.clear()
            # Keep every GPU allocation and free under the engine lock (MPS is single-threaded).
            with eng.lock, torch.enable_grad():
                x = embed(ids.to(eng.dev)).detach().requires_grad_(True)
                eng.backbone(inputs_embeds=x, use_cache=False)
                hs = [caps[layer_idx] for layer_idx in range(num_layers)]
                out = caps[target]
                # Probe b reads coordinate ks[b] of the head's stream summed over kept positions.
                probe = torch.zeros(batch_size, T, hidden_size, device=eng.dev, dtype=out.dtype)
                probe[torch.arange(batch_size)[:, None], kept_positions[None, :], torch.tensor(ks)[:, None]] = 1
                grads = torch.autograd.grad((out * probe).sum(), hs if eng.ink else hs[:-1])
                for layer_idx, grad in enumerate(grads):
                    jacobians[layer_idx, ks] = (grad[:, skip:].float().sum(1) / num_pairs).cpu()
                del grads, hs, out, x, probe
                caps.clear()

            if not eng.ink:
                # Final layer's Jacobian with respect to itself is scaled identity (diagonal entries only).
                jacobians[num_layers - 1, ks, ks] = len(kept_positions) / num_pairs

            done += batch_size
            if progress is not None:
                progress(done, hidden_size, time.perf_counter() - t0)
    finally:
        for handle in hooks:
            handle.remove()

    return _scale(jacobians)


def save(jacobians: torch.Tensor, path: Path) -> None:
    """Save a computed J-lens tensor as float16 ``.safetensors``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"J": jacobians.half().contiguous()}, str(path))


def load(path: Path, eng: Engine) -> torch.Tensor | None:
    """Load a saved J-lens onto ``eng.dev`` in the model's dtype, or ``None`` if absent or shape-mismatched."""
    if not path.exists():
        return None
    jacobians = load_file(str(path))["J"]
    if tuple(jacobians.shape) != (eng.num_layers, eng.hidden_size, eng.hidden_size):
        return None
    return jacobians.to(eng.dev, eng.model.dtype)


def read(
    eng: Engine,
    jacobians: torch.Tensor,
    resid: torch.Tensor | Sequence[torch.Tensor] | dict[int, torch.Tensor],
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode top-``k`` J-lens token IDs and probabilities at every layer and position.

    Args:
        eng: Loaded :class:`~interp.engine.Engine`.
        jacobians: J-lens tensor of shape ``[L, H, H]`` on the model device.
        resid: Per-layer residual stream of shape ``[L, T, H]`` (or mapping ``layer -> [T, H]``).
        k: Number of top tokens to return per layer and position.

    Returns:
        Tuple ``(ids, probs)`` of numpy arrays, each of shape ``[L, T, k]``.
    """
    ids: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for layer_idx in range(eng.num_layers):
            layer_resid = resid[layer_idx].to(jacobians.dtype)
            logits = eng._head(eng.final_norm(layer_resid @ jacobians[layer_idx].T))
            top = torch.softmax(logits, dim=-1).topk(k, dim=-1)
            ids.append(top.indices.cpu().numpy())
            probs.append(top.values.cpu().numpy())
            del logits
    return np.stack(ids), np.stack(probs)
