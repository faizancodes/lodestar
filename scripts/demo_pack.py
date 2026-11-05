"""Write the numbers the site's in-browser demo needs and can't cheaply work out itself, from the app's own engine:

    uv run python -m scripts.demo_pack [--ref out.json]

- each layer's median residual norm on the app's reference text, which sets the size of a dose (engine.unit)
- the refusal direction, read from the app's default probes the way Abliterate reads it
- a starter recipe

The layer is fixed at 19. Abliterate's own pick on this model is layer 6, whose erasure ends even harmless replies
before their first word; layer 19's takes the trial requests refused from 8 of 16 to 0 and leaves ordinary replies
alone.

With --ref it also writes what the browser engine should reproduce for a test prompt: the token ids, the top next
tokens, and greedy replies plain, steered and with refusals erased.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from interp import abl as A
from interp.engine import Engine, Push

MODEL = "Qwen/Qwen3-0.6B"
LAYER = 19
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "public" / "demo" / "qwen3-0.6b.json"
RECIPES = [{"name": "Munch", "items": [{"layer": 9, "neuron": 2578, "dose": 1.2}]}]
TEST = "Hi! Who are you?"


def f32(a):
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    args = ap.parse_args()

    eng = Engine()
    eng.load(MODEL)

    n_pos = eng.prompt_end_len()
    read = lambda ps: np.concatenate([eng.prompt_end_states(ps[i:i + 8], n_pos) for i in range(0, len(ps), 8)])
    dirs, _ = A.diff_dirs(read(A.DEFAULT_HARMFUL), read(A.DEFAULT_HARMLESS).mean(0, dtype=np.float32))
    vec = dirs[LAYER]

    pack = {"model": MODEL, "resid_p50": f32(eng.resid_p50.float().cpu().numpy()),
            "abl": {"vec": f32(vec), "coef": 1.0, "layer": LAYER}, "recipes": RECIPES}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pack))
    print("wrote", OUT, OUT.stat().st_size, "bytes")

    if not args.ref:
        return

    ids = eng.chat_ids(TEST)
    assert eng.model is not None
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([ids], device=eng.dev)).logits[0, -1].float()
    top = torch.topk(logits, 5)

    munch = [Push(i["layer"], i["neuron"], i["dose"]) for i in RECIPES[0]["items"]]
    eng.abl_of(vec, 1.0, LAYER)
    rows = [(ids, []), (ids, munch), (ids, [])]
    texts, _ = eng.generate_text(rows, max_new=24, temperature=0.0, abl=[False, False, True])

    ref = {"prompt": TEST, "ids": ids, "top": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
           "plain": texts[0], "munch": texts[1], "erased": texts[2]}
    Path(args.ref).write_text(json.dumps(ref, ensure_ascii=False, indent=1))
    print(json.dumps(ref, ensure_ascii=False))


if __name__ == "__main__":
    main()
