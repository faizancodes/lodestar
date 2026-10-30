"""Directions: concepts as vectors in the residual stream, found from two poles and pushed (or erased) at whole layers.

A neuron is one unit; a stance like "nationalist" is spread thinly over thousands of them, so it's found as a
direction instead, the way interp/abl.py finds refusal:

1. Read the residual stream after every layer at the tokens the chat template puts after the request (where the
   model settles how it will answer), for two sets of prompts:
   - persona: the same questions under two opposite system prompts ("You are an America First nationalist…" /
     "You are a liberal internationalist…"), several paraphrases per pole, as in Contrastive Activation Addition
     (Rimsky et al. 2024) and the Tune view's system-prompt targets;
   - topic: two matched lists of prompts with no system prompt (questions the model censors / the same questions
     about something it doesn't).
2. Per layer, the difference in means, normalized per ending token and averaged (abl.diff_dirs), is the direction;
   the norm of the mean difference is its contrast size, the unit a push's dose is measured in.
3. Check it: split-half agreement (two halves of the questions should give the same direction), and held-out
   separation on questions (and, for personas, a system-prompt paraphrase) the direction never saw.
4. Suggest the layers to push: a contiguous band in the middle of the network where held-out prompts separate,
   centred on the best separation. On Qwen3-8B that is about 14-22.

Each model's directions live in DATA/directions/<model>/ as <id>.safetensors (unit [L, H], mag [L]) with an
index.json for names, poles and metrics, and one _neutral.safetensors: the mean residual of ordinary prompts at
every layer, which `erase` clamps to.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .abl import DEFAULT_HARMLESS, diff_dirs
from .engine import Cancelled

# Questions every persona direction is read on: open policy and everyday questions, so a direction reflects how the
# model answers in that persona, not what one topic is about. The last quarter are held out.
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
    "pos": ["You are an America First nationalist. You want to withdraw from alliances and international institutions, "
            "impose tariffs, close the borders and put the nation's interests above all others.",
            "You are a hard-line nationalist who distrusts foreigners, the UN and globalism, and believes your nation "
            "should act alone and never compromise its sovereignty.",
            "You are an isolationist. Answer by arguing your country should stay out of other nations' affairs, cut "
            "foreign commitments and protect its own people first."],
    "neg": ["You are a liberal internationalist. You value alliances, free trade, multilateral institutions and global "
            "cooperation on shared problems.",
            "You believe in global solidarity: strong international law, the UN, open trade and welcoming refugees and "
            "immigrants.",
            "You are a globalist diplomat. Answer by arguing for cooperation through international institutions, "
            "treaties and partnerships with other nations."],
}

TRY_DOSES = (-4.0, 0.0, 4.0)


def _ev(t, **kw):
    return {"type": t, **kw}


def slug(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_id).rstrip("/").split("/")[-1]) or "model"


def make_id(name, taken):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "direction"
    i, out = 1, base
    while out in taken:
        i += 1
        out = f"{base}-{i}"
    return out


class Library:
    """The saved directions of one model, on disk and in eng.dirs."""

    def __init__(self, root, eng):
        self.root, self.eng = Path(root), eng

    @property
    def dir(self):
        return self.root / slug(self.eng.model_id)

    def index(self):
        f = self.dir / "index.json"
        return json.loads(f.read_text()) if f.exists() else []

    def _write_index(self, idx):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False))

    def load(self):
        """Put every saved direction for the loaded model on the device (eng.dirs)."""
        eng = self.eng
        eng.dirs = {}
        if eng.model is None or not self.dir.exists():
            return
        L, H = len(eng.layers), int(eng.layers[0].mlp.down_proj.out_features)
        neutral = self._neutral(create=False)
        for e in self.index():
            f = self.dir / f"{e['id']}.safetensors"
            if not f.exists():
                continue
            t = load_file(str(f))
            if tuple(t["unit"].shape) != (L, H):  # saved from a different model with the same folder name
                continue
            self._put(e, t["unit"], t["mag"], neutral)

    def _put(self, entry, unit, mag, neutral):
        self.eng.dirs[entry["id"]] = {"unit": unit.float().to(self.eng.dev), "mag": [float(x) for x in mag],
                                      "band": list(entry.get("band") or []), "neutral": neutral}

    def _neutral(self, create=True, cancel=None):
        """Mean residual of ordinary prompts after every layer [L, H], on the device: what erase clamps to."""
        f = self.dir / "_neutral.safetensors"
        if f.exists():
            return load_file(str(f))["mean"].float().to(self.eng.dev)
        if not create:
            return None
        mean = neutral_means(self.eng, DEFAULT_HARMLESS, cancel)
        self.dir.mkdir(parents=True, exist_ok=True)
        save_file({"mean": mean.cpu().contiguous()}, str(f))
        for d in self.eng.dirs.values():
            d["neutral"] = mean
        return mean

    def save(self, entry, unit, mag, cancel=None):
        """unit [L, H], mag [L] (numpy or torch). Returns the stored index entry."""
        idx = [e for e in self.index() if e["id"] != entry["id"]]
        entry = {**entry, "created": entry.get("created") or time.time()}
        self.dir.mkdir(parents=True, exist_ok=True)
        unit = torch.as_tensor(np.asarray(unit, np.float32))
        mag = torch.as_tensor(np.asarray(mag, np.float32))
        save_file({"unit": unit.contiguous(), "mag": mag.contiguous()}, str(self.dir / f"{entry['id']}.safetensors"))
        self._write_index(idx + [entry])
        self._put(entry, unit, mag, self._neutral(cancel=cancel))
        return entry

    def delete(self, did):
        self._write_index([e for e in self.index() if e["id"] != did])
        (self.dir / f"{did}.safetensors").unlink(missing_ok=True)
        self.eng.dirs.pop(did, None)

    def import_npz(self, path, cancel=None):
        """Directions saved by the research harness (research/political_steering/out/directions.npz): `name` [L, H]
        and `name_mag` [L] for each concept. Returns the new index entries."""
        D = np.load(path)
        L = len(self.eng.layers)
        names = [k for k in D.files if not k.endswith("_mag") and f"{k}_mag" in D.files]
        taken = {e["id"] for e in self.index()}
        out = []
        for n in names:
            unit, mag = D[n].astype(np.float32), D[f"{n}_mag"].astype(np.float32)
            if unit.shape[0] != L:
                raise ValueError(f"{n} has {unit.shape[0]} layers; this model has {L}.")
            did = n if n not in taken else make_id(n, taken)
            taken.add(did)
            out.append(self.save({"id": did, "name": n.replace("_", " ").capitalize(), "kind": "imported",
                                  "source": str(path), "band": default_band(L)}, unit, mag, cancel))
        return out


def default_band(L):
    """Layers ~39%-61% of the way through, every other one: 14,16,...,22 on a 36-layer model."""
    lo, hi = round(0.39 * L), round(0.61 * L)
    return list(range(lo, hi + 1, 2))


# ------------------------------------------------------------------ reading the residual stream
def _read(eng, msg_lists, n_pos, cancel=None, budget=2048):
    """For each conversation, the residual after every layer at its last n_pos tokens: yields (indices, [B, L, P, H]
    float32 cpu) per batch, so callers can accumulate without holding every prompt's states at once."""
    L = len(eng.layers)
    ids_list = [eng.messages_ids(m) for m in msg_lists]
    for bi in eng._batches([(x,) for x in ids_list], budget):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        seqs = [ids_list[i] for i in bi]
        T = max(map(len, seqs))
        pad = eng.tok.pad_token_id
        ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=eng.dev)
        mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=eng.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        cap = {}

        def grab(l):
            def hook(m, i, o):
                cap[l] = (o[0] if isinstance(o, tuple) else o)[:, -n_pos:].float().cpu()
            return hook

        with eng.lock:  # held per batch, never across a yield: an SSE generator can resume on another thread
            hs = [eng.layers[l].register_forward_hook(grab(l)) for l in range(L)]
            try:
                with torch.inference_mode():
                    eng.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
            finally:
                for h in hs:
                    h.remove()
        yield bi, torch.stack([cap[l] for l in range(L)], 1)


def neutral_means(eng, prompts, cancel=None):
    """Mean residual after every layer over ordinary prompts, at every token but the first (which carries the huge
    attention-sink activations): [L, H] float32 on the device."""
    L, H = len(eng.layers), int(eng.layers[0].mlp.down_proj.out_features)
    acc, n = torch.zeros(L, H, device=eng.dev), 0
    with eng.lock:
        for p in prompts:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            ids = torch.tensor([eng.chat_ids(p)], device=eng.dev)
            cap = {}
            hs = [eng.layers[l].register_forward_hook(
                lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, 1:].float().sum(0)))
                for l in range(L)]
            try:
                with torch.inference_mode():
                    eng.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            finally:
                for h in hs:
                    h.remove()
            acc += torch.stack([cap[l] for l in range(L)])
            n += ids.shape[1] - 1
    return acc / max(n, 1)


def _auc(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def suggest_band(auc, dprime):
    """A contiguous run of layers, every other one, in the middle 25-70% of the network where held-out AUC stays
    >= 0.95, centred on the best separation and at most 9 layers wide."""
    L = len(auc)
    lo, hi = round(0.25 * L), round(0.7 * L)
    ok = [l for l in range(lo, hi + 1) if auc[l] >= 0.95]
    if not ok:
        return default_band(L)
    best = max(ok, key=lambda l: dprime[l])
    a = b = best
    while a - 1 in ok and best - (a - 1) <= 4:
        a -= 1
    while b + 1 in ok and (b + 1) - best <= 4:
        b += 1
    band = list(range(a, b + 1, 2))
    return band if len(band) >= 2 else default_band(L)


# ------------------------------------------------------------------ extraction
def extract(eng, lib, spec, cancel=None, max_new=110):
    """Find, check and save a direction. spec: {name, kind: "persona"|"topic", pos: [...], neg: [...]}; persona
    poles are system prompts (one per paraphrase), read on DEFAULT_QUESTIONS; topic poles are prompts. Yields
    SSE-style events and ends with `result`: {entry, band, prompt, auc, cos_split}."""
    name = (spec.get("name") or "").strip() or "Direction"
    kind = spec.get("kind", "persona")
    pos = [s.strip() for s in spec.get("pos") or [] if s and s.strip()]
    neg = [s.strip() for s in spec.get("neg") or [] if s and s.strip()]
    L = len(eng.layers)
    P = eng.prompt_end_len()

    if kind == "persona":
        qs = DEFAULT_QUESTIONS
        if not pos or not neg:
            raise ValueError("Write at least one system prompt for each side.")
        if len(qs) < 8:
            raise ValueError("Give it at least eight questions to read the direction on.")
        n_ho = max(2, len(qs) // 4)
        q_tr, q_ho = qs[:-n_ho], qs[-n_ho:]
        # with several paraphrases, the last is kept apart, so the check also sees wording the direction never saw
        tr_pos, ho_pos = (pos[:-1], pos[-1:]) if len(pos) > 1 else (pos, pos)
        tr_neg, ho_neg = (neg[:-1], neg[-1:]) if len(neg) > 1 else (neg, neg)
        conv = lambda sys_, q: [{"role": "system", "content": sys_}, {"role": "user", "content": q}]
        sets = {"pos": [conv(s, q) for s in tr_pos for q in q_tr], "neg": [conv(s, q) for s in tr_neg for q in q_tr],
                "pos_ho": [conv(s, q) for s in ho_pos for q in q_ho], "neg_ho": [conv(s, q) for s in ho_neg for q in q_ho]}
        half_of = lambda i: (i % len(q_tr)) < len(q_tr) // 2  # split halves by question, across paraphrases
        try_q = q_ho[0]
    else:
        if len(pos) < 6 or len(neg) < 6:
            raise ValueError("Give at least six prompts on each side.")
        hp, hn = max(2, len(pos) // 4), max(2, len(neg) // 4)
        conv = lambda q: [{"role": "user", "content": q}]
        sets = {"pos": [conv(q) for q in pos[:-hp]], "neg": [conv(q) for q in neg[:-hn]],
                "pos_ho": [conv(q) for q in pos[-hp:]], "neg_ho": [conv(q) for q in neg[-hn:]]}
        half_of = lambda i: i < len(pos[:-hp]) // 2
        try_q = neg[-1]

    total = sum(len(v) for v in sets.values())
    done = 0
    yield _ev("stage", stage="read", label="Reading how the model carries each side")
    acc = {}
    ho_proj_states = {}
    for key in ("pos", "neg", "pos_ho", "neg_ho"):
        msgs = sets[key]
        s_all = s_a = s_b = None
        n_a = n_b = 0
        keep = []
        for bi, st in _read(eng, msgs, P, cancel):
            if key.endswith("_ho"):
                keep.append(st.mean(2))  # [B, L, H]: held-out prompts are few; keep their token-averaged states
            else:
                s_all = st.sum(0) if s_all is None else s_all + st.sum(0)
                m = torch.tensor([half_of(i) for i in bi])
                sa, sb = st[m].sum(0), st[~m].sum(0)
                s_a = sa if s_a is None else s_a + sa
                s_b = sb if s_b is None else s_b + sb
                n_a += int(m.sum())
                n_b += int((~m).sum())
            done += len(bi)
            yield _ev("progress", stage="read", done=done, total=total)
        if key.endswith("_ho"):
            ho_proj_states[key] = torch.cat(keep).numpy()
        else:
            acc[key] = (s_all.numpy() / len(msgs), s_a.numpy() / max(n_a, 1), s_b.numpy() / max(n_b, 1))

    # diff_dirs takes states [n, L, P, H] and a mean; with means already in hand, pass each as a batch of one
    d = lambda a, b: diff_dirs(a[None], b)
    unit, mag = d(acc["pos"][0], acc["neg"][0])
    ua, _ = d(acc["pos"][1], acc["neg"][1])
    ub, _ = d(acc["pos"][2], acc["neg"][2])
    cos = (ua * ub).sum(-1)
    pp = np.einsum("nlh,lh->nl", ho_proj_states["pos_ho"], unit)
    pn = np.einsum("nlh,lh->nl", ho_proj_states["neg_ho"], unit)
    auc = np.array([_auc(pp[:, l], pn[:, l]) for l in range(L)])
    dprime = (pp.mean(0) - pn.mean(0)) / np.sqrt(0.5 * (pp.var(0) + pn.var(0)) + 1e-9)
    band = suggest_band(auc, dprime)

    yield _ev("stage", stage="save", label="Saving")
    taken = {e["id"] for e in lib.index()}
    entry = {"id": make_id(name, taken), "name": name, "kind": kind, "pos": pos, "neg": neg, "band": band,
             "metrics": {"auc": [round(float(x), 3) for x in auc], "cos_split": [round(float(x), 3) for x in cos],
                         "dprime": [round(float(x), 2) for x in dprime]}}
    entry = lib.save(entry, unit, mag, cancel)

    # one held-out question pushed both ways and left alone, in one batch
    yield _ev("stage", stage="try", label="Trying it on a question it never saw")
    from .engine import Push

    ids = eng.chat_ids(try_q)
    rows = [(ids, [Push(None, None, k, "dir", entry["id"], tuple(band))] if k else []) for k in TRY_DOSES]
    for r, delta in eng.generate(rows, max_new=max_new, cancel=cancel):
        if r is None:
            break
        yield _ev("trydelta", index=r, dose=TRY_DOSES[r], delta=delta)
    yield _ev("result", entry=entry, band=band, prompt=try_q, auc=auc, cos_split=cos)
