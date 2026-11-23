"""Replay recorded ``/api`` exchanges from ``e2e/fixtures/<scenario>.json`` and serve a built UI folder.

Usage::

    uv run python e2e/harness/mock.py 8796 [ui-dir, default web/dist]

Each ``METHOD /path?query`` key maintains a FIFO queue of recorded responses; once drained, the final response
repeats for polling loops. Request body mismatches and unrecorded endpoints are logged and surfaced via
``GET /__fixture/log``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "e2e" / "fixtures"
UI = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "web" / "dist"
GAP = float(os.environ.get("MOCK_GAP", "0.012"))  # seconds between replayed SSE events
POLL_KEEP = 6  # max replayed steps for long GET polling runs

STATE: dict[str, Any] = {
    "name": None,
    "queues": {},
    "requests": [],
    "mismatches": [],
    "unrecorded": [],
    "chats": {},
}

GB = 1024**3
MEMORY = {
    "total": 32 * GB,
    "used": 14 * GB,
    "app": 4 * GB,
    "others": [{"pid": 1, "name": "Safari", "bytes": 2 * GB}],
    "swap": 0,
    "pressure": "normal",
    "swapping": False,
}
_SERVE_LOG = [
    {"at": 1793409424, "model": "Qwen3-0.6B-plain", "prompt": 18, "tokens": 27, "seconds": 2.2, "finish": "stop"},
    {"at": 1793409422, "model": "Qwen3-0.6B", "prompt": 18, "tokens": 40, "seconds": 3.0, "finish": "length"},
]


def _key(method: str, path: str, query: str) -> str:
    return f"{method} {path}" + (f"?{query}" if query else "")


def _thin(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subsample long GET polling runs evenly while preserving first and last responses."""
    if len(entries) <= POLL_KEEP or entries[0]["method"] != "GET":
        return entries
    idx = sorted({round(i * (len(entries) - 1) / (POLL_KEEP - 1)) for i in range(POLL_KEEP)})
    return [entries[i] for i in idx]


def _chat_summary(chat_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chat_doc["id"],
        "title": chat_doc["title"],
        "model": chat_doc["model"],
        "updated": chat_doc["updated"],
        "turns": sum(m["role"] == "user" for m in chat_doc["messages"]),
    }


async def load(req: Request) -> JSONResponse:
    """Load a named recorded or assembled fixture scenario and reset replay queues."""
    name = req.query_params.get("name")
    if not name:
        return JSONResponse({"error": "Missing fixture name."}, status_code=400)

    fixture_path = FIXTURES / f"{name}.json"
    if not fixture_path.exists():
        fixture_path = FIXTURES / "assembled" / f"{name}.json"
    if not fixture_path.exists():
        return JSONResponse({"error": f"Fixture not found: {name}"}, status_code=404)

    log = json.loads(fixture_path.read_text())
    queues: dict[str, list[dict[str, Any]]] = {}
    for entry in log:
        queues.setdefault(_key(entry["method"], entry["path"], entry["query"]), []).append(entry)

    STATE.update(
        name=name,
        queues={k: _thin(v) for k, v in queues.items()},
        requests=[],
        mismatches=[],
        unrecorded=[],
        chats={},
    )
    return JSONResponse({"ok": True, "keys": len(queues)})


async def report(_req: Request) -> JSONResponse:
    """Return replay diagnostics (recorded requests, body mismatches, and unrecorded calls)."""
    return JSONResponse({k: STATE[k] for k in ("name", "requests", "mismatches", "unrecorded")})


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


async def api(req: Request) -> Response:
    """Replay a recorded ``/api/*`` exchange or serve deterministic stand-ins for ambient endpoints."""
    raw = await req.body()
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = raw.decode("utf-8", "replace")

    req_key = _key(req.method, req.url.path, req.url.query)
    STATE["requests"].append({"key": req_key, "body": body})

    queue = None if req.method == "POST" and req.url.path == "/api/serve" else STATE["queues"].get(req_key)
    fallback_entry: dict[str, Any] | None = None
    if not queue and req.url.path == "/api/arch/neurons":
        # Background per-layer neuron fetches vary with timing; serve a recorded layer without mutating its queue.
        donor = next((v for kk, v in STATE["queues"].items() if kk.startswith("GET /api/arch/neurons")), None)
        if donor:
            fallback_entry = donor[0]

    if not queue and fallback_entry is None:
        standin = _stand_in(req.method, req.url.path, body)
        if standin is not None:
            return standin

        STATE["unrecorded"].append({"key": req_key, "body": body})
        return JSONResponse({"error": f"Not recorded: {req_key}"}, status_code=599)

    entry = fallback_entry if fallback_entry is not None else (queue.pop(0) if len(queue) > 1 else queue[0])
    if req.method != "GET" and not _same_json(body, entry["body"]):
        STATE["mismatches"].append({"key": req_key, "sent": body, "recorded": entry["body"]})

    if req.url.path == "/api/serve":
        return JSONResponse({**entry["response"], "log": _SERVE_LOG})
    if entry["type"] != "sse":
        return JSONResponse(entry["response"], status_code=entry["status"] or 200)

    async def _gen():
        for ev in entry["events"]:
            await asyncio.sleep(GAP)
            yield f"data: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), status_code=entry["status"] or 200, media_type="text/event-stream")


def _stand_in(method: str, path: str, body: Any) -> JSONResponse | None:
    """Return a deterministic stand-in response for non-model endpoints, or ``None``."""
    if path == "/api/memory":
        return JSONResponse(MEMORY)
    if path == "/api/jlens" and method == "GET":
        return JSONResponse({"ready": True, "status": "done", "done": 1024, "total": 1024, "elapsed": 284.2})
    if path == "/api/memory/fit":
        return JSONResponse({"known": True, "fits": True})
    if path == "/api/serve":
        if method == "POST":
            return JSONResponse({"ok": True})
        return JSONResponse(
            {
                "models": [
                    {"id": "Qwen3-0.6B", "about": "As set in Lodestar"},
                    {"id": "Qwen3-0.6B-plain", "about": "Plain"},
                ],
                "log": _SERVE_LOG,
                "active": 0,
            }
        )

    chats = STATE["chats"]
    if path == "/api/chats" and method == "GET":
        return JSONResponse(sorted((_chat_summary(c) for c in chats.values()), key=lambda c: -c["updated"]))
    if path == "/api/chats" and method == "POST":
        chats[body["id"]] = {**body, "updated": len(STATE["requests"])}
        return JSONResponse(_chat_summary(chats[body["id"]]))

    if path.startswith("/api/chats/"):
        cid = path.rsplit("/", 1)[1]
        if method == "DELETE":
            chats.pop(cid, None)
            return JSONResponse({"ok": True})
        if cid not in chats:
            return JSONResponse({"error": "That chat is gone."}, status_code=404)
        return JSONResponse(chats[cid])

    return None


async def index(_req: Request) -> FileResponse:
    return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})


app = Starlette(
    routes=[
        Route("/__fixture/load", load, methods=["POST"]),
        Route("/__fixture/log", report),
        Route("/api/{rest:path}", api, methods=["GET", "POST", "PATCH", "DELETE"]),
        Route("/", index),
        Mount("/", StaticFiles(directory=UI, check_dir=False)),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8796, log_level="warning")
