"""Replays recorded /api exchanges (e2e/fixtures/<scenario>.json) and serves a UI folder, so the parity suite runs
without a model and the UI can be checked against them.

    uv run python e2e/harness/mock.py 8796 [ui-dir, default web/dist]

Each method + path + query has a queue of recorded responses, served in order; the last one repeats once the queue
runs out (a poll that runs longer). A request whose body differs from the recorded one is logged as a mismatch, and
one with no recording at all as unrecorded: GET /__fixture/log returns both, and the tests fail on either.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "e2e" / "fixtures"
UI = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "web" / "dist"
GAP = float(os.environ.get("MOCK_GAP", "0.012"))  # seconds between streamed events, so text still streams in
POLL_KEEP = 6  # a long poll (loading, exporting) replays as at most this many steps

S = {"name": None, "queues": {}, "requests": [], "mismatches": [], "unrecorded": []}


def key(method, path, query):
    return f"{method} {path}" + (f"?{query}" if query else "")


def thin(entries):
    """Keep a long run of GET polls short: evenly spaced steps, always the first and the last."""
    if len(entries) <= POLL_KEEP or entries[0]["method"] != "GET":
        return entries
    idx = sorted({round(i * (len(entries) - 1) / (POLL_KEEP - 1)) for i in range(POLL_KEEP)})
    return [entries[i] for i in idx]


async def load(req: Request):
    name = req.query_params["name"]
    log = json.loads((FIXTURES / f"{name}.json").read_text())

    queues = {}
    for e in log:
        queues.setdefault(key(e["method"], e["path"], e["query"]), []).append(e)

    S.update(name=name, queues={k: thin(v) for k, v in queues.items()}, requests=[], mismatches=[], unrecorded=[])
    return JSONResponse({"ok": True, "keys": len(queues)})


async def report(req: Request):
    return JSONResponse({k: S[k] for k in ("name", "requests", "mismatches", "unrecorded")})


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


async def api(req: Request):
    raw = await req.body()
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = raw.decode("utf-8", "replace")

    k = key(req.method, req.url.path, req.url.query)
    S["requests"].append({"key": k, "body": body})

    q = S["queues"].get(k)
    if not q:
        S["unrecorded"].append({"key": k, "body": body})
        return JSONResponse({"error": f"Not recorded: {k}"}, status_code=599)

    e = q.pop(0) if len(q) > 1 else q[0]
    if req.method != "GET" and not same(body, e["body"]):
        S["mismatches"].append({"key": k, "sent": body, "recorded": e["body"]})

    if e["type"] != "sse":
        return JSONResponse(e["response"], status_code=e["status"] or 200)

    async def gen():
        for ev in e["events"]:
            await asyncio.sleep(GAP)
            yield f"data: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), status_code=e["status"] or 200, media_type="text/event-stream")


async def index(req: Request):
    return FileResponse(UI / "index.html", headers={"Cache-Control": "no-store"})


app = Starlette(routes=[
    Route("/__fixture/load", load, methods=["POST"]),
    Route("/__fixture/log", report),
    Route("/api/{rest:path}", api, methods=["GET", "POST", "PATCH", "DELETE"]),
    Route("/", index),
    Mount("/", StaticFiles(directory=UI)),
])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8796, log_level="warning")
