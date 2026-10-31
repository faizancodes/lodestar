"""Start the server headless, load a small model, and check that a reply and a trace come back. CI runs it on Windows,
where nothing else exercises the engine: uv run python scripts/smoke.py [model]."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
PORT = 8797
URL = f"http://127.0.0.1:{PORT}"


def call(path, body=None, timeout=60):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(path, body):
    """The events of a text/event-stream, ending at its end event; an error event fails the run."""
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "error":
                sys.exit(f"{path}: {ev['message']}")
            if ev["type"] == "end":
                return events
            events.append(ev)
    sys.exit(f"{path}: the stream stopped before its end event")


def main():
    env = {**os.environ, "AUTO_INTERP_PORT": str(PORT), "AUTO_INTERP_DATA": tempfile.mkdtemp(prefix="lodestar-")}
    server = subprocess.Popen([sys.executable, "-m", "interp.app", "--server"], env=env)

    try:
        for _ in range(120):
            try:
                call("/api/status", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            sys.exit("the server didn't start")

        call("/api/load", {"model": MODEL})
        start = time.time()
        while (s := call("/api/status"))["status"] != "ready":
            if s["status"].startswith("error"):
                sys.exit(s["status"])
            if time.time() - start > 1800:
                sys.exit("the model took more than 30 minutes to load")
            time.sleep(2)
        print(f"loaded {s['model']} on {s['device']} ({s['dtype']}, {s['layers']} layers) in {time.time() - start:.0f}s",
              flush=True)

        prompt = "Name a colour."
        reply = "".join(ev["delta"] for ev in stream("/api/chat", {
            "messages": [{"role": "user", "content": prompt}], "max_new": 12, "temperature": 0}) if ev["type"] == "delta")
        if not reply.strip():
            sys.exit("the reply was empty")
        print("reply:", repr(reply), flush=True)

        tr = [ev for ev in stream("/api/trace", {"prompt": prompt, "response": reply}) if ev["type"] == "trace"]
        if not tr or not tr[0]["tokens"]:
            sys.exit("the trace had no tokens")
        print(f"trace: {len(tr[0]['tokens'])} tokens", flush=True)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
