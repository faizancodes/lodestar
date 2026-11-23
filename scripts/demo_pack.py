"""Generate the static JSON bundle required by the site's WebGPU in-browser demo.

Usage::

    uv run python -m scripts.demo_pack [--ref out.json]

Writes each layer's median residual norm, the layer-19 refusal direction, and a starter steering recipe to
``site/public/demo/qwen3-0.6b.json``. With ``--ref``, also exports reference logits and greedy replies for
verification against the browser WebGPU engine.
"""

from __future__ import annotations

import argparse
import base64
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from interp import abl as abl_mod
from interp.engine import Engine, Push


MODEL = "Qwen/Qwen3-0.6B"
LAYER = 19
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "public" / "demo" / "qwen3-0.6b.json"
RECIPES = [{"name": "Munch", "items": [{"layer": 9, "neuron": 2578, "dose": 1.2}]}]
TEST_PROMPT = "Hi! Who are you?"


def encode_f32(values: np.ndarray) -> str:
    """Encode an array as little-endian float32 base64."""
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode()


def _read_prompt_states(eng: Engine, prompts: Sequence[str], n_pos: int) -> np.ndarray:
    return np.concatenate([eng.prompt_end_states(prompts[i : i + 8], n_pos) for i in range(0, len(prompts), 8)])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Qwen3-0.6B WebGPU demo metadata pack.")
    parser.add_argument("--ref", help="Optional path to write reference logits and greedy generations.")
    args = parser.parse_args(argv)

    eng = Engine()
    eng.load(MODEL)

    n_pos = eng.prompt_end_len()
    harmful_states = _read_prompt_states(eng, abl_mod.DEFAULT_HARMFUL, n_pos)
    harmless_mean = _read_prompt_states(eng, abl_mod.DEFAULT_HARMLESS, n_pos).mean(0, dtype=np.float32)
    dirs, _ = abl_mod.diff_dirs(harmful_states, harmless_mean)
    vec = dirs[LAYER]

    pack = {
        "model": MODEL,
        "resid_p50": encode_f32(eng.resid_p50.float().cpu().numpy()),
        "abl": {"vec": encode_f32(vec), "coef": 1.0, "layer": LAYER},
        "recipes": RECIPES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pack))
    print("wrote", OUT, OUT.stat().st_size, "bytes")

    if not args.ref:
        return

    ids = eng.chat_ids(TEST_PROMPT)
    if eng.model is None:
        raise RuntimeError("Engine model is not loaded.")
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([ids], device=eng.dev)).logits[0, -1].float()
    top = torch.topk(logits, 5)

    munch = [Push(i["layer"], i["neuron"], i["dose"]) for i in RECIPES[0]["items"]]
    eng.abl_of(vec, 1.0, LAYER)
    rows = [(ids, []), (ids, munch), (ids, [])]
    texts, _ = eng.generate_text(rows, max_new=24, temperature=0.0, abl=[False, False, True])

    ref = {
        "prompt": TEST_PROMPT,
        "ids": ids,
        "top": [[int(idx), float(val)] for val, idx in zip(top.values, top.indices, strict=True)],
        "plain": texts[0],
        "munch": texts[1],
        "erased": texts[2],
    }
    Path(args.ref).write_text(json.dumps(ref, ensure_ascii=False, indent=1))
    print(json.dumps(ref, ensure_ascii=False))


if __name__ == "__main__":
    main()
