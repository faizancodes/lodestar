"""FastAPI endpoint unit tests for interp/server.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from interp import directions, server
from interp.engine import Engine, Push


@pytest.fixture
def client(tiny_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Bind interp.server to an isolated temp data dir and the tiny CPU test engine."""
    monkeypatch.setattr(server, "eng", tiny_engine)
    monkeypatch.setattr(server, "dirs", directions.Library(tmp_path / "directions", tiny_engine))
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setattr(server, "RECIPES", tmp_path / "recipes.json")
    monkeypatch.setattr(server, "ABL_FILE", tmp_path / "abl.json")
    monkeypatch.setattr(server, "LAST_MODEL", tmp_path / "last_model.txt")
    monkeypatch.setattr(server, "PROBES_FILE", tmp_path / "abl-probes.json")
    monkeypatch.setattr(server, "CHATS", tmp_path / "chats")
    monkeypatch.setattr(server, "JLENS", tmp_path / "jlens")
    monkeypatch.setattr(server, "COPIES", tmp_path / "models")
    monkeypatch.setattr(server, "arch_cache", {})
    server.serving["pushes"] = []
    server.serving["log"].clear()
    server.serving["active"] = 0
    return TestClient(server.app)


def test_finite_and_reverse_helpers() -> None:
    assert server._finite({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
    assert server._reverse(Push(1, 2, 0.5, "add")) == Push(1, 2, -0.5, "add")
    assert server._reverse(Push(1, 2, 0.0, "mul")) == Push(1, 2, 2.0, "mul")
    assert server._reverse(Push(None, None, 1.0, "erase", "d", (1,))) is None


def test_status_arch_neuron_and_search_endpoints(client: TestClient) -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    assert res.json()["loaded"] is True

    arch_res = client.get("/api/arch")
    assert arch_res.status_code == 200
    assert arch_res.json()["hidden"] == 8

    assert client.get("/api/arch/neurons", params={"layer": 99}).status_code == 400
    assert client.get("/api/neuron", params={"layer": 0, "n": 0}).status_code == 200
    assert client.get("/api/search", params={"word": "hello", "k": 4}).status_code == 200


def test_recipes_and_chats_crud(client: TestClient) -> None:
    saved = client.post("/api/recipes", json={"name": "Test", "items": []})
    assert saved.status_code == 200
    rid = saved.json()["id"]
    assert len(client.get("/api/recipes").json()) == 1
    assert client.delete(f"/api/recipes/{rid}").status_code == 200
    assert client.get("/api/recipes").json() == []

    # Invalid chat id rejected
    assert client.post("/api/chats", json={"id": "../bad", "title": "x", "messages": []}).status_code == 400

    chat_doc = {"id": "chat-1", "title": "Hello", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/api/chats", json=chat_doc).status_code == 200
    assert client.get("/api/chats").json()[0]["turns"] == 1
    assert client.get("/api/chats/chat-1").json()["title"] == "Hello"
    assert client.delete("/api/chats/chat-1").status_code == 200
    assert client.get("/api/chats/chat-1").status_code == 404


def test_copy_and_remove_reject_traversal(client: TestClient, tmp_path: Path) -> None:
    # Path traversal outside /Volumes rejected
    res = client.post("/api/models/copy", json={"path": str(tmp_path)})
    assert res.status_code == 400

    # Removing a path outside COPIES rejected
    res = client.post("/api/models/remove", json={"path": str(tmp_path)})
    assert res.status_code == 400


def test_openai_v1_models_and_completions(client: TestClient) -> None:
    models_res = client.get("/v1/models")
    assert models_res.status_code == 200
    ids = [m["id"] for m in models_res.json()["data"]]
    assert ids == ["tiny-model", "tiny-model-plain"]

    # Validation errors return OpenAI error envelope
    bad = client.post("/v1/chat/completions", json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]})
    assert bad.status_code == 404
    assert bad.json()["error"]["type"] == "not_found_error"

    # Non-streaming completion succeeds
    ok = client.post(
        "/v1/chat/completions",
        json={"model": "tiny-model-plain", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert math.isfinite(payload["usage"]["total_tokens"])

    # Streaming completion returns valid SSE frames ending in [DONE]
    stream_res = client.post(
        "/v1/chat/completions",
        json={"model": "tiny-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stream": True},
    )
    assert stream_res.status_code == 200
    lines = [ln for ln in stream_res.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    first_chunk = json.loads(lines[0][6:])
    assert first_chunk["object"] == "chat.completion.chunk"
