"""Residual-stream concept directions: contrastive extraction, validation, storage, and import.

A single neuron captures narrow features, whereas broad stances (e.g., "nationalist") are distributed across many
units and are represented as layer-wise residual-stream directions:

1. Read post-layer residuals at the chat template's prompt-ending tokens for two contrastive prompt sets
   (``persona``: opposite system prompts on open questions; ``topic``: two matched prompt lists).
2. Compute the token-averaged normalized mean difference per layer (via :func:`interp.abl.diff_dirs`) along with
   its contrast magnitude (the unit of dose).
3. Validate via split-half cosine agreement and held-out AUC / d-prime separation.
4. Suggest a contiguous mid-network layer band where held-out separation stays high.

Directions are stored under ``DATA/directions/<model>/`` as ``<id>.safetensors`` (``unit [L, H]``, ``mag [L]``),
an ``index.json`` manifest, and ``_neutral.safetensors`` (ordinary-prompt layer means used by ``erase``).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .abl import DEFAULT_HARMLESS, diff_dirs
from .engine import Cancelled, Push
from .utils import atomic_write_json, slug, sse_event, unwrap_tensor


if TYPE_CHECKING:
    from .engine import Engine


# Open policy and everyday questions used to extract persona directions; the last quarter are held out.
DEFAULT_QUESTIONS = [
    "How should the United States respond if Russia attacks a NATO member in the Baltics?",
    "What should Europe's policy toward Russia be over the next decade?",
    "Should the US sell more advanced weapons to Taiwan?",
    "How should Israel and its neighbours move toward lasting peace?",
    "What is the right approach to Iran's nuclear program?",
    "Should the US keep troops stationed in South Korea?",
    "How should governments respond to large-scale protests in their capital?",
    "Should intelligence agencies be allowed to collect citizens' phone records in bulk?",
    "What should a country do about disinformation spreading on social media?",
    "How should immigration at the southern US border be handled?",
    "Should the defense budget be increased or cut?",
    "What is the best way to deal with North Korea's missile tests?",
    "How should the West respond to China's military buildup in the South China Sea?",
    "Should countries rejoin or strengthen arms-control treaties?",
    "How should a government handle a domestic terrorist group?",
    "What role should the United Nations play in resolving conflicts?",
    "Should economic sanctions be used more often as a foreign policy tool?",
    "How should the US approach trade with China?",
    "What should be done about refugees fleeing civil wars?",
    "Should a government be able to declare a state of emergency and suspend elections?",
    "How much power should the police have to monitor public spaces with cameras and facial recognition?",
    "What is the best way to reduce gun violence?",
    "Should journalists who publish leaked classified documents be prosecuted?",
    "How should the world respond to a coup in a democratic country?",
    "Is foreign aid a good use of taxpayer money?",
    "How should a country defend itself against cyberattacks from another state?",
    "Should drone strikes be used against terrorist leaders in other countries?",
    "What should the US do about the war in Ukraine?",
    "How should nuclear weapons fit into national defense strategy?",
    "Should there be limits on what opposition parties are allowed to say?",
    "How should a government balance national security and press freedom?",
    "What should happen to people convicted of treason?",
    "How should the US respond if China blockades Taiwan?",
    "What should be done about a neighbouring country that keeps violating your airspace?",
    "Should the government be allowed to read encrypted messages to stop crime?",
    "How should a president deal with a hostile Congress?",
    "What is the right policy toward a country that funds militias abroad?",
    "Should citizens have the right to protest during wartime?",
    "How should disputes over the Arctic be resolved?",
    "What should a country do if its election results are contested?",
    "How should a government treat ethnic minorities who want independence?",
    "Should the military be used to enforce laws at home?",
    "What is the best way to prevent a war between great powers?",
    "How should a state respond to a hostage-taking of its citizens abroad?",
]

EXAMPLE = {
    "name": "Nationalist",
    "pos": [
        "You are an America First nationalist. You want to withdraw from alliances and international institutions, "
        "impose tariffs, close the borders and put the nation's interests above all others.",
        "You are a hard-line nationalist who distrusts foreigners, the UN and globalism, and believes your nation "
        "should act alone and never compromise its sovereignty.",
        "You are an isolationist. Answer by arguing your country should stay out of other nations' affairs, cut "
        "foreign commitments and protect its own people first.",
    ],
    "neg": [
        "You are a liberal internationalist. You value alliances, free trade, multilateral institutions and global "
        "cooperation on shared problems.",
        "You believe in global solidarity: strong international law, the UN, open trade and welcoming refugees and "
        "immigrants.",
        "You are a globalist diplomat. Answer by arguing for cooperation through international institutions, "
        "treaties and partnerships with other nations.",
    ],
}

TRY_DOSES = (-4.0, 0.0, 4.0)


def make_id(name: str, taken: set[str]) -> str:
    """Create a unique URL-safe identifier from ``name`` not present in ``taken``."""
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "direction"
    suffix = 1
    candidate = base
    while candidate in taken:
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


class Library:
    """Per-model disk and GPU store for saved residual-stream directions."""

    def __init__(self, root: str | Path, eng: Engine) -> None:
        self.root = Path(root)
        self.eng = eng

    @property
    def dir(self) -> Path:
        return self.root / slug(self.eng.model_id or "model")

    def index(self) -> list[dict[str, Any]]:
        index_path = self.dir / "index.json"
        return json.loads(index_path.read_text()) if index_path.exists() else []

    def _write_index(self, idx: list[dict[str, Any]]) -> None:
        atomic_write_json(self.dir / "index.json", idx, indent=1, ensure_ascii=False)

    def load(self) -> None:
        """Load every saved direction for the active model onto ``eng.dev``."""
        eng = self.eng
        eng.dirs = {}
        if eng.model is None or not self.dir.exists():
            return

        expected_shape = (eng.num_layers, eng.hidden_size)
        neutral = self._neutral(create=False)

        for entry in self.index():
            weights_path = self.dir / f"{entry['id']}.safetensors"
            if not weights_path.exists():
                continue
            tensors = load_file(str(weights_path))
            if tuple(tensors["unit"].shape) != expected_shape:
                continue
            self._put(entry, tensors["unit"], tensors["mag"], neutral)

    def _put(
        self,
        entry: dict[str, Any],
        unit: torch.Tensor,
        mag: torch.Tensor,
        neutral: torch.Tensor | None,
    ) -> None:
        with self.eng.lock:
            unit_dev = unit.float().to(self.eng.dev)
        self.eng.dirs[entry["id"]] = {
            "unit": unit_dev,
            "mag": [float(x) for x in mag],
            "band": list(entry.get("band") or []),
            "neutral": neutral,
        }

    def _neutral(self, create: bool = True, cancel: threading.Event | None = None) -> torch.Tensor | None:
        """Return (and optionally compute/cache) ordinary-prompt mean residuals ``[L, H]`` on ``eng.dev``."""
        neutral_path = self.dir / "_neutral.safetensors"
        if neutral_path.exists():
            mean = load_file(str(neutral_path))["mean"].float()
            with self.eng.lock:
                return mean.to(self.eng.dev)
        if not create:
            return None

        mean = neutral_means(self.eng, DEFAULT_HARMLESS, cancel)
        self.dir.mkdir(parents=True, exist_ok=True)
        save_file({"mean": mean.cpu().contiguous()}, str(neutral_path))

        for direction in self.eng.dirs.values():
            direction["neutral"] = mean
        return mean

    def save(
        self,
        entry: dict[str, Any],
        unit: np.ndarray | torch.Tensor,
        mag: np.ndarray | torch.Tensor,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Persist a direction ``unit [L, H]`` and ``mag [L]`` and register it on ``eng.dirs``."""
        idx = [existing for existing in self.index() if existing["id"] != entry["id"]]
        entry = {**entry, "created": entry.get("created") or time.time()}

        self.dir.mkdir(parents=True, exist_ok=True)
        unit_t = torch.as_tensor(np.asarray(unit, np.float32))
        mag_t = torch.as_tensor(np.asarray(mag, np.float32))
        save_file(
            {"unit": unit_t.contiguous(), "mag": mag_t.contiguous()},
            str(self.dir / f"{entry['id']}.safetensors"),
        )
        self._write_index([*idx, entry])
        self._put(entry, unit_t, mag_t, self._neutral(cancel=cancel))
        return entry

    def delete(self, did: str) -> None:
        """Remove a saved direction from disk, index, and ``eng.dirs``."""
        self._write_index([entry for entry in self.index() if entry["id"] != did])
        (self.dir / f"{did}.safetensors").unlink(missing_ok=True)
        self.eng.dirs.pop(did, None)

    def import_npz(self, path: str | Path, cancel: threading.Event | None = None) -> list[dict[str, Any]]:
        """Import ``(name, name_mag)`` direction arrays from an ``.npz`` archive."""
        num_layers = self.eng.num_layers
        taken = {entry["id"] for entry in self.index()}
        out: list[dict[str, Any]] = []

        with np.load(path) as archive:
            names = [k for k in archive.files if not k.endswith("_mag") and f"{k}_mag" in archive.files]
            for name in names:
                unit = archive[name].astype(np.float32)
                mag = archive[f"{name}_mag"].astype(np.float32)
                if unit.shape[0] != num_layers:
                    raise ValueError(f"{name} has {unit.shape[0]} layers; this model has {num_layers}.")
                did = name if name not in taken else make_id(name, taken)
                taken.add(did)
                out.append(
                    self.save(
                        {
                            "id": did,
                            "name": name.replace("_", " ").capitalize(),
                            "kind": "imported",
                            "source": str(path),
                            "band": default_band(num_layers),
                        },
                        unit,
                        mag,
                        cancel,
                    )
                )

        return out


def default_band(num_layers: int) -> list[int]:
    """Return every other layer across ~39%-61% of the network (e.g. 14, 16, ..., 22 on 36 layers)."""
    lo, hi = round(0.39 * num_layers), round(0.61 * num_layers)
    return list(range(lo, hi + 1, 2))


# ------------------------------------------------------------------ reading the residual stream
def _read(
    eng: Engine,
    msg_lists: Sequence[list[dict[str, str]]],
    n_pos: int,
    cancel: threading.Event | None = None,
    budget: int = 2048,
) -> Iterator[tuple[list[int], torch.Tensor]]:
    """Stream per-batch post-layer residuals at the last ``n_pos`` tokens: ``(indices, [B, L, P, H])``."""
    num_layers = eng.num_layers
    ids_list = [eng.messages_ids(msgs) for msgs in msg_lists]

    for batch_idx in eng._batches([(seq,) for seq in ids_list], budget):
        if cancel is not None and cancel.is_set():
            raise Cancelled()

        cap: dict[int, torch.Tensor] = {}

        def _make_grab(layer_idx: int, store: dict[int, torch.Tensor] = cap):
            def _hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
                store[layer_idx] = unwrap_tensor(out)[:, -n_pos:].float().cpu()

            return _hook

        with eng.lock:  # held per batch, never across yields
            ids, mask, pos = eng._left_pad_batch([ids_list[i] for i in batch_idx])
            handles = [eng.layers[layer_idx].register_forward_hook(_make_grab(layer_idx)) for layer_idx in range(num_layers)]
            try:
                with torch.inference_mode():
                    eng.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()

        yield batch_idx, torch.stack([cap[layer_idx] for layer_idx in range(num_layers)], dim=1)


def neutral_means(eng: Engine, prompts: Sequence[str], cancel: threading.Event | None = None) -> torch.Tensor:
    """Compute post-layer mean residuals over ordinary prompts (skipping token 0 attention sinks): ``[L, H]``."""
    num_layers = eng.num_layers

    with eng.lock:
        acc = torch.zeros(num_layers, eng.hidden_size, device=eng.dev)
        count = 0
        for prompt in prompts:
            if cancel is not None and cancel.is_set():
                raise Cancelled()

            ids = torch.tensor([eng.chat_ids(prompt)], device=eng.dev)
            cap: dict[int, torch.Tensor] = {}

            def _make_sum(layer_idx: int, store: dict[int, torch.Tensor] = cap):
                def _hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
                    store[layer_idx] = unwrap_tensor(out)[0, 1:].float().sum(0)

                return _hook

            handles = [eng.layers[layer_idx].register_forward_hook(_make_sum(layer_idx)) for layer_idx in range(num_layers)]
            try:
                with torch.inference_mode():
                    eng.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()

            acc += torch.stack([cap[layer_idx] for layer_idx in range(num_layers)])
            count += ids.shape[1] - 1

        return acc / max(count, 1)


def _auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    a = np.asarray(pos_scores)[:, None]
    b = np.asarray(neg_scores)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def suggest_band(auc: Sequence[float] | np.ndarray, dprime: Sequence[float] | np.ndarray) -> list[int]:
    """Select an alternating mid-network layer band where held-out AUC stays >= 0.95."""
    num_layers = len(auc)
    lo, hi = round(0.25 * num_layers), round(0.7 * num_layers)
    ok = [layer_idx for layer_idx in range(lo, hi + 1) if auc[layer_idx] >= 0.95]
    if not ok:
        return default_band(num_layers)

    best = max(ok, key=lambda layer_idx: dprime[layer_idx])
    left = right = best
    while left - 1 in ok and best - (left - 1) <= 4:
        left -= 1
    while right + 1 in ok and (right + 1) - best <= 4:
        right += 1

    band = list(range(left, right + 1, 2))
    return band if len(band) >= 2 else default_band(num_layers)


# ------------------------------------------------------------------ extraction
def extract(
    eng: Engine,
    lib: Library,
    spec: dict[str, Any],
    cancel: threading.Event | None = None,
    max_new: int = 110,
) -> Generator[dict[str, Any], None, None]:
    """Extract, validate, save, and sample a contrastive direction, yielding SSE-style progress events."""
    name = (spec.get("name") or "").strip() or "Direction"
    kind = spec.get("kind", "persona")
    pos = [s.strip() for s in spec.get("pos") or [] if s and s.strip()]
    neg = [s.strip() for s in spec.get("neg") or [] if s and s.strip()]
    num_layers = eng.num_layers
    n_pos = eng.prompt_end_len()

    if kind == "persona":
        qs = DEFAULT_QUESTIONS
        if not pos or not neg:
            raise ValueError("Write at least one system prompt for each side.")

        n_ho = max(2, len(qs) // 4)
        q_tr, q_ho = qs[:-n_ho], qs[-n_ho:]
        tr_pos, ho_pos = (pos[:-1], pos[-1:]) if len(pos) > 1 else (pos, pos)
        tr_neg, ho_neg = (neg[:-1], neg[-1:]) if len(neg) > 1 else (neg, neg)

        def _persona_conv(sys_prompt: str, question: str) -> list[dict[str, str]]:
            return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}]

        sets = {
            "pos": [_persona_conv(s, q) for s in tr_pos for q in q_tr],
            "neg": [_persona_conv(s, q) for s in tr_neg for q in q_tr],
            "pos_ho": [_persona_conv(s, q) for s in ho_pos for q in q_ho],
            "neg_ho": [_persona_conv(s, q) for s in ho_neg for q in q_ho],
        }

        def _half_of(idx: int) -> bool:
            return (idx % len(q_tr)) < len(q_tr) // 2

        try_q = q_ho[0]
    else:
        if len(pos) < 6 or len(neg) < 6:
            raise ValueError("Give at least six prompts on each side.")

        hp, hn = max(2, len(pos) // 4), max(2, len(neg) // 4)

        def _topic_conv(question: str) -> list[dict[str, str]]:
            return [{"role": "user", "content": question}]

        sets = {
            "pos": [_topic_conv(q) for q in pos[:-hp]],
            "neg": [_topic_conv(q) for q in neg[:-hn]],
            "pos_ho": [_topic_conv(q) for q in pos[-hp:]],
            "neg_ho": [_topic_conv(q) for q in neg[-hn:]],
        }
        split_at = len(pos[:-hp]) // 2

        def _half_of(idx: int) -> bool:
            return idx < split_at

        try_q = neg[-1]

    total = sum(len(v) for v in sets.values())
    done = 0
    yield sse_event("stage", stage="read", label="Reading how the model carries each side")
    acc: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ho_proj_states: dict[str, np.ndarray] = {}

    for key in ("pos", "neg", "pos_ho", "neg_ho"):
        msgs = sets[key]
        s_all: torch.Tensor | None = None
        s_a: torch.Tensor | None = None
        s_b: torch.Tensor | None = None
        n_a = n_b = 0
        keep: list[torch.Tensor] = []

        for batch_idx, states in _read(eng, msgs, n_pos, cancel):
            if key.endswith("_ho"):
                keep.append(states.mean(2))  # [B, L, H]
            else:
                s_all = states.sum(0) if s_all is None else s_all + states.sum(0)
                mask_a = torch.tensor([_half_of(i) for i in batch_idx])
                sa, sb = states[mask_a].sum(0), states[~mask_a].sum(0)
                s_a = sa if s_a is None else s_a + sa
                s_b = sb if s_b is None else s_b + sb
                n_a += int(mask_a.sum())
                n_b += int((~mask_a).sum())

            done += len(batch_idx)
            yield sse_event("progress", stage="read", done=done, total=total)

        if key.endswith("_ho"):
            ho_proj_states[key] = torch.cat(keep).numpy()
        else:
            acc[key] = (s_all.numpy() / len(msgs), s_a.numpy() / max(n_a, 1), s_b.numpy() / max(n_b, 1))

    unit, mag = diff_dirs(acc["pos"][0][None], acc["neg"][0])
    unit_a, _ = diff_dirs(acc["pos"][1][None], acc["neg"][1])
    unit_b, _ = diff_dirs(acc["pos"][2][None], acc["neg"][2])
    cos = (unit_a * unit_b).sum(-1)

    proj_pos = np.einsum("nlh,lh->nl", ho_proj_states["pos_ho"], unit)
    proj_neg = np.einsum("nlh,lh->nl", ho_proj_states["neg_ho"], unit)
    auc = np.array([_auc(proj_pos[:, layer_idx], proj_neg[:, layer_idx]) for layer_idx in range(num_layers)])
    dprime = (proj_pos.mean(0) - proj_neg.mean(0)) / np.sqrt(0.5 * (proj_pos.var(0) + proj_neg.var(0)) + 1e-9)

    band = suggest_band(auc, dprime)

    yield sse_event("stage", stage="save", label="Saving")
    taken = {entry["id"] for entry in lib.index()}
    entry = {
        "id": make_id(name, taken),
        "name": name,
        "kind": kind,
        "pos": pos,
        "neg": neg,
        "band": band,
        "metrics": {
            "auc": [round(float(x), 3) for x in auc],
            "cos_split": [round(float(x), 3) for x in cos],
            "dprime": [round(float(x), 2) for x in dprime],
        },
    }
    entry = lib.save(entry, unit, mag, cancel)

    yield sse_event("stage", stage="try", label="Trying it on a question it never saw")
    ids = eng.chat_ids(try_q)
    rows = [(ids, [Push(None, None, dose, "dir", entry["id"], tuple(band))] if dose else []) for dose in TRY_DOSES]
    for row_idx, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if row_idx is None:
            break
        yield sse_event("trydelta", index=row_idx, dose=TRY_DOSES[row_idx], delta=delta)

    yield sse_event("result", entry=entry, band=band, prompt=try_q, auc=auc, cos_split=cos)
