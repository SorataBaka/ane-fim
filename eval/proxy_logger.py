"""Logging proxy: sits between llama.vscode and llama-server and saves real /infill traffic.

    python3 eval/proxy_logger.py [--listen 8012] [--upstream http://127.0.0.1:8013]

Every POST /infill is saved to eval/llama_vscode_requests/NNNN.json as
{"path", "request", "response", "status", "elapsed_ms", "time"}.
All other requests are forwarded untouched.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

OUT_DIR = Path(__file__).parent / "llama_vscode_requests"
HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}

counter_lock = Lock()


def next_path():
    with counter_lock:
        n = len(list(OUT_DIR.glob("*.json")))
        return OUT_DIR / f"{n:04d}.json"


class Proxy(BaseHTTPRequestHandler):
    upstream = ""

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        req = urllib.request.Request(self.upstream + self.path, data=body, headers=headers, method=self.command)

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status, resp_headers, resp_body = resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            status, resp_headers, resp_body = e.code, e.headers, e.read()
        except urllib.error.URLError as e:
            self.send_error(502, f"upstream unreachable: {e.reason}")
            return
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

        if self.command == "POST" and self.path.startswith("/infill"):
            self._save(body, resp_body, status, elapsed_ms)

    def _save(self, body, resp_body, status, elapsed_ms):
        def parse(raw):
            try:
                return json.loads(raw)
            except (ValueError, TypeError):
                return (raw or b"").decode("utf-8", "replace")

        record = {
            "path": self.path,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "elapsed_ms": round(elapsed_ms, 1),
            "request": parse(body),
            "response": parse(resp_body),
        }
        out = next_path()
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        # With n_cmpl > 1, llama-server returns a list of completions instead of one object.
        resp = record["response"]
        first = resp[0] if isinstance(resp, list) and resp else resp
        evaluated = first.get("tokens_evaluated") if isinstance(first, dict) else None
        n = len(resp) if isinstance(resp, list) else 1
        print(f"{out.name}  {status}  {elapsed_ms:7.1f} ms  n_cmpl={n}  tokens_evaluated={evaluated}", flush=True)

    do_GET = do_POST = do_OPTIONS = _forward

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8012)
    ap.add_argument("--upstream", default="http://127.0.0.1:8013")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    Proxy.upstream = args.upstream.rstrip("/")
    print(f"Proxy 127.0.0.1:{args.listen} -> {Proxy.upstream}, saving /infill to {OUT_DIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.listen), Proxy).serve_forever()


if __name__ == "__main__":
    main()
