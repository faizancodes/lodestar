"""The real Lodestar server with every /api exchange written to e2e/fixtures/<scenario>.json, so the mock server
can replay it later without a model. The scenario is set by the tests: POST /__fixture/start?name=<scenario>.

    AUTO_INTERP_DATA=e2e/.data uv run python e2e/harness/record.py 8795

Run it against a throwaway data folder: the tests save recipes, find directions and write exported models.
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "e2e" / "fixtures"

from interp.server import app as real_app  # noqa: E402  (after sys.path)


def _json_or_text(raw: bytes):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


class Recorder:
    def __init__(self, app):
        self.app = app
        self.name = None
        self.log = []

    def _save(self):
        if self.name:
            FIXTURES.mkdir(parents=True, exist_ok=True)
            (FIXTURES / f"{self.name}.json").write_text(json.dumps(self.log, indent=1, ensure_ascii=False))

    async def _control(self, scope, send):
        q = parse_qs(scope["query_string"].decode())
        if scope["path"] == "/__fixture/start":
            self.name = q["name"][0]
            self.log = []
            self._save()
        elif scope["path"] == "/__fixture/stop":
            self.name = None
        body = json.dumps({"ok": True, "name": self.name}).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"].startswith("/__fixture/"):
            return await self._control(scope, send)
        if not scope["path"].startswith("/api/") or not self.name:
            return await self.app(scope, receive, send)

        # read the whole request body, then hand it on in one message; after that, the real receive, so a
        # streaming response still hears about a client that disconnects
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        raw = b"".join(chunks)
        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        t0 = time.monotonic()
        entry = {"method": scope["method"], "path": scope["path"], "query": scope["query_string"].decode(),
                 "body": _json_or_text(raw), "status": None, "type": None}
        out, timed = [], []

        async def tap(msg):
            if msg["type"] == "http.response.start":
                entry["status"] = msg["status"]
                ctype = dict(msg.get("headers") or []).get(b"content-type", b"").decode()
                entry["type"] = "sse" if "event-stream" in ctype else "json"
            elif msg["type"] == "http.response.body" and msg.get("body"):
                out.append(msg["body"])
                timed.append((round(time.monotonic() - t0, 3), msg["body"]))
            await send(msg)

        try:
            await self.app(scope, replay_receive, tap)
        finally:
            if entry["type"] == "sse":
                events, buf = [], ""
                for dt, b in timed:
                    buf += b.decode("utf-8", "replace")
                    while "\n\n" in buf:
                        line, buf = buf.split("\n\n", 1)
                        if line.startswith("data: "):
                            events.append({"dt": dt, "data": json.loads(line[6:])})
                entry["events"] = events
            else:
                entry["response"] = _json_or_text(b"".join(out))
            entry["ms"] = round((time.monotonic() - t0) * 1000)
            self.log.append(entry)
            self._save()


app = Recorder(real_app)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8795, log_level="warning")
