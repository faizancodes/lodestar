"""A persistent Python namespace with the model loaded, for interactive exploration: POST code to
127.0.0.1:8799/run and get stdout back. Local research tool only."""
import contextlib, io, sys, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from lib import *  # noqa
from concepts import *  # noqa
eng = load(); steer = Steer(eng); D = dict(np.load(OUT / "directions.npz"))
NS = globals()

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        code = self.rfile.read(int(self.headers["Content-Length"])).decode()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(code, NS)
            except Exception:
                traceback.print_exc()
        out = buf.getvalue().encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass

print("ready", flush=True)
HTTPServer(("127.0.0.1", 8799), H).serve_forever()
