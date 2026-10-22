"""ane-fim prototype server: llama.cpp-compatible /infill backed by the ANE model.

    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/server-py/server.py [--model lut6|fp16] [--port 8012]

One model, one request at a time. A newer request cancels the one in flight
(llama.vscode fires a request per keystroke; only the latest matters).
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine import Engine
from prompt import PromptBuilder

PROJECT_DIR = Path(__file__).resolve().parent.parent

engine = None
builder = None
model_lock = threading.Lock()
ticket_lock = threading.Lock()
latest_ticket = 0
cfg = None


def trim_suffix_overlap(text, suffix):
    """Cut generated lines once they start repeating the code after the cursor."""
    after_cursor_line = suffix.split("\n", 1)[1] if "\n" in suffix else ""
    first = next((s.strip() for s in after_cursor_line.splitlines() if s.strip()), None)
    lines = text.split("\n")
    if first:
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == first:
                lines = lines[:i]
                break
    while len(lines) > 1 and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def indent_of(line):
    return len(line) - len(line.lstrip())


def make_stop_rule(prefix, suffix):
    """Where a suggestion should end, based on the cursor's line.

    Returns cut(text) -> (kept_text, done).
    - Cursor line has code before or after it: finish this line only.
    - Cursor on an empty/whitespace line: allow a block, but stop before a blank line
      or a line indented less than the cursor (the block has ended).
    """
    line_before = prefix.rsplit("\n", 1)[-1]
    line_after = suffix.split("\n", 1)[0]

    if line_before.strip() or line_after.strip():
        def cut_line(text):
            i = text.find("\n")
            return (text, False) if i < 0 else (text[:i], True)
        return cut_line

    base = len(line_before)  # whitespace already typed = the block's indent

    def cut_block(text):
        lines = text.split("\n")
        for i, line in enumerate(lines[1:], 1):
            complete = i < len(lines) - 1
            if (complete and not line.strip()) or (line.strip() and indent_of(line) < base):
                return "\n".join(lines[:i]), True
        return text, False
    return cut_block


def empty_response(stop_type="limit"):
    return {"content": "", "stop": True, "stop_type": stop_type, "tokens_cached": 0,
            "timings": {"prompt_n": 0, "prompt_ms": 0, "predicted_n": 0, "predicted_ms": 0, "cache_n": 0}}


def infill(body):
    global latest_ticket
    n_predict = body.get("n_predict", 64)
    if n_predict == 0:  # llama.vscode cache warm-up for input_extra; ignored until Phase 6
        return empty_response()

    with ticket_lock:
        latest_ticket += 1
        ticket = latest_ticket
    is_stale = lambda: ticket != latest_ticket

    with model_lock:
        if is_stale():
            return empty_response("cancel")
        max_tokens = min(n_predict, cfg.max_tokens)
        prefix = body.get("input_prefix", "") + body.get("prompt", "")
        suffix = body.get("input_suffix", "")
        ids = builder.build(prefix, suffix, max_tokens)
        rule = make_stop_rule(prefix, suffix)
        out, info = engine.complete(ids, max_tokens, body.get("t_max_predict_ms"), cfg.hard_ms, is_stale,
                                    stop_text=lambda t: rule(t)[1],
                                    eog_rule=cfg.eog_rule and "infill" in body.get("samplers", ["infill"]))

    text, _ = rule(engine.tokenizer.decode(out, skip_special_tokens=True))
    text = trim_suffix_overlap(text, suffix)
    if info["stop_type"] in ("limit", "cancel") and "\n" in text.rstrip("\n"):
        text = text[:text.rstrip("\n").rfind("\n")]  # cut off before an unfinished last line
    print(f"prompt {len(ids):5d}  cached {info['tokens_cached']:5d}  new {info['prompt_n']:4d} "
          f"({info['prompt_ms']:5.0f} ms)  gen {info['predicted_n']:3d} ({info['predicted_ms']:5.0f} ms)  "
          f"{info['stop_type']:6s} {text[:40]!r}", flush=True)
    return {
        "content": text,
        "stop": True,
        "stop_type": info["stop_type"],
        "tokens_cached": info["tokens_cached"],
        "timings": {
            "prompt_n": info["prompt_n"],
            "prompt_ms": round(info["prompt_ms"], 1),
            "predicted_n": info["predicted_n"],
            "predicted_ms": round(info["predicted_ms"], 1),
            "cache_n": info["tokens_cached"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            if engine is None:
                self._send(503, {"status": "loading"})
            else:
                self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/infill"):
            self._send(404, {"error": "not found"})
            return
        if engine is None:
            self._send(503, {"error": "loading"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            self._send(200, infill(body))
        except Exception as e:  # keep serving; report the error to the client
            print(f"error: {e!r}", flush=True)
            self._send(500, {"error": str(e)})

    def log_message(self, *args):
        pass


def load(meta):
    global engine, builder
    t0 = time.perf_counter()
    e = Engine(meta)
    b = PromptBuilder(e.tokenizer, e.ctx, cfg.suffix_tokens)
    e.complete(b.build("def hello():\n    ", "\n", 8), 8)  # warm up the ANE graphs
    b.anchor, e.cached = None, []
    builder, engine = b, e
    print(f"model ready: {meta} (ctx {e.ctx}) in {time.perf_counter() - t0:.1f}s", flush=True)


def main():
    global cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lut6", "fp16", "lut6-logits"], default="lut6",
                    help="lut6/fp16 are argmax builds (greedy only); lut6-logits is needed for --eog-rule")
    ap.add_argument("--eog-rule", action="store_true",
                    help="llama.cpp infill-style early end-of-text (needs lut6-logits). Off by default: "
                         "it mostly cuts suggestions short (see CLAUDE.md Phase 3a)")
    ap.add_argument("--size", default="1.5b", help="model size folder: 1.5b (default) or 0.5b")
    ap.add_argument("--meta", help="explicit meta.yaml (overrides --model/--size)")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--max-tokens", type=int, default=48, help="server cap on n_predict")
    ap.add_argument("--hard-ms", type=float, default=500, help="stop generating after this, newline or not")
    ap.add_argument("--suffix-tokens", type=int, default=128, help="max suffix tokens (re-prefilled every keystroke)")
    cfg = ap.parse_args()

    meta = cfg.meta or PROJECT_DIR / f"models/qwen25coder-{cfg.size.lower()}-ctx2048-{cfg.model}/meta.yaml"
    threading.Thread(target=load, args=(meta,), daemon=True).start()
    print(f"listening on http://127.0.0.1:{cfg.port} (loading {cfg.model})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
