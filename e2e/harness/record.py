"""ASGI recording wrapper around the real Lodestar server for E2E parity fixtures.

Writes every ``/api`` exchange to ``e2e/fixtures/<scenario>.json`` so ``e2e/harness/mock.py`` can replay it
deterministically without a loaded model. Scenarios are started via ``POST /__fixture/start?name=<scenario>``.

Usage::

    AUTO_INTERP_DATA=e2e/.data uv run python e2e/harness/record.py 8795
"""

from __future__ import annotations

import codecs
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import uvicorn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "e2e" / "fixtures"
# Ambient endpoints answered by mock stand-ins (including /api/memory/fit and /api/chats/*).
STANDS_IN = ("/api/memory", "/api/chats")

from interp.server import app as real_app  # noqa: E402


def _json_or_text(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


class Recorder:
    """ASGI middleware that captures JSON and SSE ``/api`` exchanges per test scenario."""

    def __init__(self, asgi_app: Any) -> None:
        self.app = asgi_app
        self.name: str | None = None
        self.log: list[dict[str, Any]] = []

    def _save(self, name: str | None = None, log: list[dict[str, Any]] | None = None) -> None:
        target_name = self.name if name is None else name
        target_log = self.log if log is None else log
        if target_name:
            FIXTURES.mkdir(parents=True, exist_ok=True)
            (FIXTURES / f"{target_name}.json").write_text(json.dumps(target_log, indent=1, ensure_ascii=False))

    async def _control(self, scope: dict[str, Any], send: Any) -> None:
        query = parse_qs(scope["query_string"].decode())
        if scope["path"] == "/__fixture/start":
            names = query.get("name")
            if not names or not names[0]:
                body = json.dumps({"error": "Missing fixture name."}).encode()
                await send({"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": body})
                return
            self.name = names[0]
            self.log = []
            self._save()
            status = 200
            body = json.dumps({"ok": True, "name": self.name}).encode()
        else:
            status = 404
            body = json.dumps({"error": f"Unknown control route: {scope['path']}"}).encode()

        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"].startswith("/__fixture/"):
            return await self._control(scope, send)
        if not scope["path"].startswith("/api/") or not self.name or scope["path"].startswith(STANDS_IN):
            return await self.app(scope, receive, send)

        # Capture the active scenario at request start so a late-finishing request cannot pollute the next scenario.
        scenario_name = self.name
        scenario_log = self.log

        chunks: list[bytes] = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break

        raw = b"".join(chunks)
        sent_body = False

        async def _replay_receive() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        t0 = time.monotonic()
        entry: dict[str, Any] = {
            "method": scope["method"],
            "path": scope["path"],
            "query": scope["query_string"].decode(),
            "body": _json_or_text(raw),
            "status": None,
            "type": None,
        }
        out: list[bytes] = []
        timed: list[tuple[float, bytes]] = []

        async def _tap(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                entry["status"] = msg["status"]
                ctype = dict(msg.get("headers") or []).get(b"content-type", b"").decode()
                entry["type"] = "sse" if "event-stream" in ctype else "json"
            elif msg["type"] == "http.response.body" and msg.get("body"):
                out.append(msg["body"])
                timed.append((round(time.monotonic() - t0, 3), msg["body"]))
            await send(msg)

        try:
            await self.app(scope, _replay_receive, _tap)
        finally:
            if entry["type"] == "sse":
                events: list[dict[str, Any]] = []
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                buf = ""
                for dt, chunk in timed:
                    buf += decoder.decode(chunk)
                    while "\n\n" in buf:
                        line, buf = buf.split("\n\n", 1)
                        if line.startswith("data: "):
                            try:
                                events.append({"dt": dt, "data": json.loads(line[6:])})
                            except ValueError:
                                continue
                buf += decoder.decode(b"", final=True)
                if buf.startswith("data: "):
                    with contextlib.suppress(ValueError):
                        events.append({"dt": round(time.monotonic() - t0, 3), "data": json.loads(buf[6:])})
                entry["events"] = events
            else:
                entry["response"] = _json_or_text(b"".join(out))

            entry["ms"] = round((time.monotonic() - t0) * 1000)
            scenario_log.append(entry)
            self._save(scenario_name, scenario_log)


app = Recorder(real_app)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8795, log_level="warning")
