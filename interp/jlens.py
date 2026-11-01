"""The J-lens (Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models", 2026): a logit
lens corrected for what the layers after it do to the stream.

The logit lens reads layer l's residual h straight through the final norm and unembedding, as if nothing came after
it, so middle layers mostly read as noise. The J-lens first maps h to the last layer with J_l, the Jacobian of the
last layer's residual with respect to layer l's, averaged over prompts, source positions t and every later position
t' >= t:

    J_l = E[ d h_final,t' / d h_l,t ],    lens(h) = softmax(W_U norm(J_l h))

What it reads out is what the paper calls the J-space: the few concepts a layer holds ready for the model to put
into words, which is often not the next token.

Computing it: a backward pass from <e_k, sum_t' h_final,t'> gives, at every layer and every source position at
once, row k of the sum over t' >= t of those Jacobians. One pass per basis vector e_k fills every layer's matrix;
the rows are batched (one prompt per row of the batch, a different one each time, so the average over prompts comes
from the rows too). That is hidden-size backward passes in all: a few minutes on a 1-4B model, longer on an 8B.

Each model's lens is saved in DATA/jlens/<model>.safetensors as J [L, H, H] float16.
"""
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .engine import Cancelled

# Text to average over: varied, ordinary writing (the paper uses a pretraining-like sample), a few as chat turns.
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
    ("How do I keep basil alive indoors?",
     "Basil wants light more than anything: put it in the sunniest window you have, ideally six hours of direct sun a "
     "day. Water when the top of the soil feels dry, and pinch off the flower buds as soon as they appear, because "
     "once it flowers the leaves turn bitter and the plant stops growing. Harvest from the top, just above a pair of "
     "leaves, and it will branch out and grow bushier."),
    ("Why is the sky blue?",
     "Sunlight contains every colour, and as it passes through the atmosphere it scatters off the gas molecules in the "
     "air. Shorter wavelengths scatter much more strongly than longer ones, so blue light is scattered across the "
     "whole sky while red and yellow mostly pass straight through. When you look anywhere other than at the sun, you "
     "see that scattered blue light. At sunset the light travels through more air, and the blues are scattered away."),
    ("Can you explain what a hash map is?",
     "A hash map stores pairs of keys and values so that you can look a value up by its key quickly. It keeps an array "
     "of buckets, runs each key through a hash function to get a number, and uses that number to pick a bucket. Two "
     "keys can land in the same bucket, so each bucket holds a short list, or the map probes for the next free slot. "
     "As long as the buckets stay sparse, lookups take constant time on average."),
    ("Write a short poem about autumn.",
     "The maples burn along the lane,\nthe light goes amber, low and thin,\nthe geese write lines across the rain\nand "
     "every door lets woodsmoke in.\nThe garden lets its colour go,\nthe apples soften where they fall,\nand somewhere "
     "underneath, below,\nthe roots are listening for it all."),
]


def corpus(eng, T):
    """Token rows of exactly T tokens (the passages and the chat turns, cut to length; short ones are dropped)."""
    rows = [eng.encode(p) for p in PASSAGES]
    for q, a in QA:
        rows.append(eng.chat_ids(q) + eng.encode(a))
    rows = [r[:T] for r in rows if len(r) >= T]
    if not rows:
        raise RuntimeError("No passage is long enough for the lens's context length.")
    return rows


def path_for(root, model_id):
    from .directions import slug
    return Path(root) / f"{slug(model_id)}.safetensors"


@torch.no_grad()
def _scale(J):
    """Rescale each layer's matrix to unit mean singular value squared (Frobenius / sqrt(H)): averaging over pairs
    shrinks it by the number of later positions, and the final norm then rescales anyway, but a tiny J h would sit
    under the norm's epsilon."""
    H = J.shape[-1]
    return J / (J.flatten(1).norm(dim=1) / H ** 0.5).clamp(min=1e-12)[:, None, None]


def compute(eng, cancel=None, progress=None, T=64, batch=8, skip=1):
    """J [L, H, H] float32 cpu. progress(done, total) is called after every backward pass. `skip` leading positions
    (the attention sink, whose norm is enormous) are left out of both ends of the average."""
    L, H = len(eng.layers), int(eng.layers[0].mlp.down_proj.out_features)
    rows = corpus(eng, T)
    J = torch.zeros(L, H, H)
    S = torch.arange(skip, T, device=eng.dev)
    pairs = len(S) * (len(S) + 1) / 2  # (t, t') with t <= t', both kept
    emb = eng.model.get_input_embeddings()

    caps = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o[0] if isinstance(o, tuple) else o))
             for l, layer in enumerate(eng.layers)]
    done, t0, model = 0, time.perf_counter(), eng.model
    try:
        for k0 in range(0, H, batch):
            if (cancel is not None and cancel.is_set()) or eng.model is not model:  # stopped, or another model opened
                raise Cancelled()

            ks = list(range(k0, min(H, k0 + batch)))
            B = len(ks)
            ids = torch.tensor([rows[(k0 // batch * batch + b) % len(rows)] for b in range(B)], device=eng.dev)
            caps.clear()
            with eng.lock, torch.enable_grad():
                x = emb(ids).detach().requires_grad_(True)
                eng.backbone(inputs_embeds=x, use_cache=False)
                hs = [caps[l] for l in range(L)]
                # probe b reads coordinate ks[b] of the last layer's stream, summed over the kept positions
                probe = torch.zeros(B, T, H, device=eng.dev, dtype=hs[-1].dtype)
                probe[torch.arange(B, device=eng.dev)[:, None], S[None, :], torch.tensor(ks, device=eng.dev)[:, None]] = 1
                grads = torch.autograd.grad((hs[-1] * probe).sum(), hs[:-1])

            for l, g in enumerate(grads):
                J[l, ks] = (g[:, skip:].float().sum(1) / pairs).cpu()  # row k of J_l
            J[L - 1, ks] = torch.eye(H)[ks] * (len(S) / pairs)
            del grads, hs, x, probe
            caps.clear()

            done += B
            if progress:
                progress(done, H, time.perf_counter() - t0)
    finally:
        for h in hooks:
            h.remove()

    return _scale(J)


def save(J, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"J": J.half().contiguous()}, str(path))


def load(path, eng):
    """The saved lens on the device in the model's dtype, or None if there isn't one that fits this model."""
    if not path.exists():
        return None
    J = load_file(str(path))["J"]
    L, H = len(eng.layers), int(eng.layers[0].mlp.down_proj.out_features)
    if tuple(J.shape) != (L, H, H):
        return None
    return J.to(eng.dev, eng.model.dtype)


def read(eng, J, resid, k=5):
    """Top-k J-lens tokens and probabilities at every layer and position: (ids [L, T, k], p [L, T, k]) numpy.
    resid: [L, T, H] (or {layer: [T, H]}), each layer's residual after it."""
    ids, ps = [], []
    with torch.inference_mode():
        for l in range(len(eng.layers)):
            r = resid[l].to(J.dtype)
            lg = eng.lm_head(eng.final_norm(r @ J[l].T)).float()
            top = torch.softmax(lg, -1).topk(k, dim=-1)
            ids.append(top.indices.cpu().numpy())
            ps.append(top.values.cpu().numpy())
            del lg
    return np.stack(ids), np.stack(ps)
