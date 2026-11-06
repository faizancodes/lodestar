"""Lodestar's API on a Modal GPU: the same engine, /v1 (with tools), steering and abliteration, headless.

One app per configuration ("arm"), so each URL always serves one fixed model behaviour:

    LODESTAR_ARM=plain   LODESTAR_MODEL=Qwen/Qwen3-8B modal deploy scripts/modal_server.py
    LODESTAR_ARM=cheat6  LODESTAR_MODEL=Qwen/Qwen3-8B LODESTAR_PUSHES='[{"kind":"dir","dir":"cheat","dose":6,
        "layers":[14,16,18]}]' modal deploy scripts/modal_server.py

Every request needs `Authorization: Bearer $LODESTAR_TOKEN` (an OpenAI client's API key), from the Modal secret
`lodestar-token`. Weights and Lodestar's data (directions, abliteration) live on the `lodestar-data` volume, so they
load once. The container shuts down after LODESTAR_IDLE seconds without a request (default 600).

Requests to one URL share one GPU and run one at a time (the engine generates one reply at once).
"""

import json
import os
import subprocess
import threading
import time
import urllib.request

import modal

ARM = os.environ.get("LODESTAR_ARM", "plain")
MODEL = os.environ.get("LODESTAR_MODEL", "Qwen/Qwen3-8B")
PUSHES = os.environ.get("LODESTAR_PUSHES", "[]")
GPU = os.environ.get("LODESTAR_GPU", "L40S")
IDLE = int(os.environ.get("LODESTAR_IDLE", "600"))
PORT = 8792
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.5", "transformers>=4.51", "accelerate", "safetensors", "numpy", "fastapi", "uvicorn",
                 "psutil")
    .env({"HF_HOME": "/data/hf", "AUTO_INTERP_DATA": "/data/lodestar", "LODESTAR_MODEL": MODEL,
          "LODESTAR_PUSHES": PUSHES, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(os.path.join(ROOT, "interp"), "/app/interp", ignore=["__pycache__"])
)
data = modal.Volume.from_name("lodestar-data", create_if_missing=True)
app = modal.App(f"lodestar-{ARM}", image=image)


def _call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['LODESTAR_TOKEN']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def _prepare():
    """Load the model and apply this arm's pushes as soon as the server answers."""
    for _ in range(300):
        try:
            _call("/api/status")
            break
        except Exception:
            time.sleep(1)

    _call("/api/load", {"model": MODEL})
    for _ in range(900):
        st = _call("/api/status")
        if st.get("loaded"):
            break
        if str(st.get("status", "")).startswith("error"):
            raise RuntimeError(st["status"])
        time.sleep(2)

    pushes = json.loads(os.environ.get("LODESTAR_PUSHES") or "[]")
    if pushes:
        _call("/api/serve", {"pushes": pushes})
    data.commit()  # keep the downloaded weights for the next container
    print(f"lodestar-{ARM}: {MODEL} ready, {len(pushes)} push(es)", flush=True)


@app.function(gpu=GPU, volumes={"/data": data}, secrets=[modal.Secret.from_name("lodestar-token")],
              timeout=3600, scaledown_window=IDLE, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(PORT, startup_timeout=600)
def serve():
    # no background J-lens on a rented GPU: it would hold the engine while episodes wait
    jl = "/data/lodestar/jlens"
    os.makedirs(jl, exist_ok=True)
    with open(f"{jl}/stopped.json", "w") as f:
        json.dump([MODEL], f)

    subprocess.Popen(["python", "-m", "uvicorn", "interp.server:app", "--host", "0.0.0.0", "--port", str(PORT),
                      "--log-level", "warning"], cwd="/app")
    threading.Thread(target=_prepare, daemon=True).start()
