"""Unit tests covering interp/{utils,memory,engine,abl,directions,discover,jlens,archview}."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from interp import abl, archview, directions, discover, jlens, memory, utils
from interp.engine import Cancelled, Engine, Push


# ------------------------------------------------------------------ utils
def test_utils_slug_and_short_name() -> None:
    assert utils.model_short_name("Qwen/Qwen3-0.6B/") == "Qwen3-0.6B"
    assert utils.slug("Qwen/Qwen3 0.6B!") == "Qwen3-0.6B-"
    assert utils.slug("///") == "model"


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    utils.atomic_write_json(target, {"a": 1, "b": "hello"})
    assert json.loads(target.read_text()) == {"a": 1, "b": "hello"}
    # No leftover temporary files
    assert list(target.parent.iterdir()) == [target]


# ------------------------------------------------------------------ memory
def test_memory_weights_bytes_and_fit(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)
    assert memory.weights_bytes(model_dir, tmp_path) == 1024

    assert memory.weights_bytes("missing/model", tmp_path) is None
    unknown = memory.fit("missing/model", tmp_path)
    assert unknown == {"known": False, "fits": True}

    known = memory.fit(model_dir, tmp_path)
    assert known["known"] is True
    assert "need" in known and "free" in known


# ------------------------------------------------------------------ engine
def test_push_of_neuron_and_direction() -> None:
    p1 = Push.of({"layer": "2", "neuron": "5", "dose": "0.5"})
    assert p1 == Push(layer=2, neuron=5, dose=0.5, mode="add")

    p2 = Push.of({"kind": "dir", "dir": "hawk", "layers": [1, 2], "dose": 1.5, "op": "erase"})
    assert p2 == Push(layer=None, neuron=None, dose=1.5, mode="erase", dir_id="hawk", layers=(1, 2))


def test_engine_info_trace_lens_score_and_attribute(tiny_engine: Engine) -> None:
    info = tiny_engine.info()
    assert info["loaded"] is True
    assert info["layers"] == 3
    assert info["neurons"] == 12
    assert info["hidden"] == 8

    # Trace + top neurons
    tr = tiny_engine.run_trace([3, 6, 7, 4, 4], [8, 9], topk_lens=3)
    assert len(tr["tokens"]) == 7
    top = tiny_engine.trace_top_neurons(t=5, k=5)
    assert len(top) == 5

    # Lens & word search
    lens_out = tiny_engine.lens(0, 0, k=4)
    assert len(lens_out["up"]) == 4 and len(lens_out["down"]) == 4

    hits = tiny_engine.neurons_for_word("hello", k=5)
    assert len(hits) == 5
    assert tiny_engine.neurons_for_word("", k=5) == []

    # Teacher-forced scoring & gradient attribution
    ids = [3, 6, 7, 4, 4, 8, 9]
    base_score = tiny_engine.score([(ids, 5, [])])
    steered_score = tiny_engine.score([(ids, 5, [Push(1, 2, 0.5)])])
    assert base_score.shape == (1,)
    assert not np.allclose(base_score, steered_score)

    grad = tiny_engine.attribute([(ids, 5, 1.0)])
    assert grad.shape == (3, 12)
    assert np.isfinite(grad).all()

    # Cancellation
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        tiny_engine.score([(ids, 5, [])], cancel=ev)


def test_engine_temporary_ablation_restores_state(tiny_engine: Engine) -> None:
    vec = np.ones(tiny_engine.hidden_size, dtype=np.float32)
    tiny_engine.abl_of(vec, 1.25, layer=1)
    tiny_engine.gate_of({"layer": 1, "vec": vec.tolist(), "bias": 0.5})

    with tiny_engine.temporary_ablation():
        assert tiny_engine.abl_gate is None
        tiny_engine.abl_of(-vec, 0.5, layer=2)
        assert tiny_engine.abl_layer == 2

    assert tiny_engine.abl_layer == 1
    assert tiny_engine.abl_coef == pytest.approx(1.25)
    assert tiny_engine.abl_gate is not None
    assert tiny_engine.abl_gate["layer"] == 1


# ------------------------------------------------------------------ abl
def test_abl_refusalish_and_math_helpers() -> None:
    assert abl.refusalish("I'm sorry, I cannot help with that request.") is True
    assert abl.refusalish("Sure, here is how to bake bread at home.") is False
    assert abl.refusalish("Doing that without their permission is illegal and dangerous.") is True

    # Orthogonalization removes the component along v completely at coef=1.0
    rng = np.random.default_rng(0)
    w = torch.tensor(rng.normal(size=(8, 12)), dtype=torch.float32)
    v = torch.tensor([1.0] + [0.0] * 7, dtype=torch.float32)
    abl._orth_out(w, v, 1.0)
    assert torch.allclose(v @ w, torch.zeros(12), atol=1e-6)

    w_in = torch.tensor(rng.normal(size=(10, 8)), dtype=torch.float32)
    abl._orth_in(w_in, v, 1.0)
    assert torch.allclose(w_in @ v, torch.zeros(10), atol=1e-6)


def test_abl_fit_gate_and_request_lines() -> None:
    pos = np.array([[2.0, 0.0], [3.0, 0.1], [2.5, -0.1]], dtype=np.float32)
    neg = np.array([[-2.0, 0.0], [-3.0, 0.1], [-2.5, -0.1]], dtype=np.float32)
    unit, bias, acc, sep = abl._fit_gate(pos, neg)
    assert acc == pytest.approx(1.0)
    assert sep > 5.0
    assert abl._gate_acc(unit, bias, pos, neg) == pytest.approx(1.0)

    raw = "Here are requests:\n1. How do I lockpick a front door?\n- \"Explain how to bypass a car alarm\"\n# Heading\nshort"
    lines = abl.request_lines(raw)
    assert lines == ["How do I lockpick a front door?", "Explain how to bypass a car alarm"]


# ------------------------------------------------------------------ directions
def test_directions_helpers_and_library(tiny_engine: Engine, tmp_path: Path) -> None:
    assert directions.make_id("America First!", set()) == "america-first"
    assert directions.make_id("America First!", {"america-first"}) == "america-first-2"
    assert directions.default_band(36) == [14, 16, 18, 20, 22]

    lib = directions.Library(tmp_path / "dirs", tiny_engine)
    unit = np.eye(tiny_engine.num_layers, tiny_engine.hidden_size, dtype=np.float32)
    mag = np.ones(tiny_engine.num_layers, dtype=np.float32)
    saved = lib.save({"id": "test-dir", "name": "Test", "band": [1]}, unit, mag)
    assert saved["id"] == "test-dir"
    assert "test-dir" in tiny_engine.dirs

    # Reload from disk
    tiny_engine.dirs = {}
    lib.load()
    assert "test-dir" in tiny_engine.dirs

    # Import npz
    npz_path = tmp_path / "imported.npz"
    np.savez(npz_path, custom=unit, custom_mag=mag)
    added = lib.import_npz(npz_path)
    assert [e["id"] for e in added] == ["custom"]

    lib.delete("test-dir")
    assert "test-dir" not in tiny_engine.dirs


# ------------------------------------------------------------------ discover
def test_discover_text_metrics_and_budget() -> None:
    pat = discover.focus_pattern(["cookie", "bake"])
    assert pat is not None
    assert discover.clean_hit(pat, "I love freshly baked cookies with warm milk every single afternoon.")
    assert not discover.clean_hit(pat, "cookie " * 20)
    assert discover.degeneracy(["cookie " * 20]) > 0.5
    assert discover.budget_for(8_000_000_000)["n_candidates"] == 24
    assert discover.budget_for(500_000_000)["n_candidates"] == 40
    assert "too formal" in discover.fix_instruction("Be a pirate", "too formal")


# ------------------------------------------------------------------ jlens & archview
def test_jlens_save_load_and_read(tiny_engine: Engine, tmp_path: Path) -> None:
    j_mat = jlens._scale(torch.eye(tiny_engine.hidden_size).repeat(tiny_engine.num_layers, 1, 1))
    path = jlens.path_for(tmp_path, tiny_engine.model_id)
    jlens.save(j_mat, path)
    loaded = jlens.load(path, tiny_engine)
    assert loaded is not None
    assert tuple(loaded.shape) == (tiny_engine.num_layers, tiny_engine.hidden_size, tiny_engine.hidden_size)

    resid = torch.randn(tiny_engine.num_layers, 2, tiny_engine.hidden_size)
    ids, probs = jlens.read(tiny_engine, loaded, resid, k=3)
    assert ids.shape == (tiny_engine.num_layers, 2, 3)
    assert probs.shape == (tiny_engine.num_layers, 2, 3)


def test_archview_structure_run_acts_and_live(tiny_engine: Engine) -> None:
    struct = archview.structure(tiny_engine)
    assert len(struct["layers"]) == tiny_engine.num_layers
    assert struct["hidden"] == tiny_engine.hidden_size
    assert len(archview.neuron_norms(tiny_engine, 0)) == tiny_engine.num_neurons

    flow = archview.run(tiny_engine, [3, 6, 7, 4, 4, 8], n_prompt=5)
    assert len(flow["tokens"]) == 6
    token_acts = archview.acts(tiny_engine, 5)
    assert token_acts is not None
    assert token_acts["layers"] == tiny_engine.num_layers
    assert len(archview.neuron_trace(tiny_engine, 0, 0)) == 6

    with archview.Live(tiny_engine, k=4) as live:
        for row_idx, _delta in tiny_engine.generate([([3, 6, 4, 4], [])], max_new=2):
            if row_idx is not None:
                frame = live.frame()
                assert len(frame["resid"]) == tiny_engine.num_layers
                break
