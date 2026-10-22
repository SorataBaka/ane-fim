"""Replay captured llama.vscode /infill requests against a server, in order, and report latency.

    python3 eval/bench_latency.py --url http://127.0.0.1:8014 [--n 200] [--start 0] [--show 5]

Requests are sent one at a time (the editor's real pattern is one live request at a time).
Warm-up requests (n_predict 0) are sent too, since a server has to handle them.
"""

import argparse
import glob
import json
import time
import urllib.request
from pathlib import Path

REQ_DIR = Path(__file__).resolve().parent / "llama_vscode_requests"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8012")
    ap.add_argument("--n", type=int, default=0, help="max requests (0 = all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--show", type=int, default=5, help="print this many sample completions")
    ap.add_argument("--duration", type=float, default=0, help="loop over the requests for this many seconds")
    a = ap.parse_args()

    files = sorted(glob.glob(str(REQ_DIR / "*.json")))[a.start:]
    if a.n:
        files = files[:a.n]

    def request_stream():
        if not a.duration:
            yield from files
            return
        end = time.perf_counter() + a.duration
        while True:
            for f in files:
                if time.perf_counter() >= end:
                    return
                yield f

    ms, cached, new, gen, shown = [], [], [], [], 0
    wall0 = time.perf_counter()
    for f in request_stream():
        req = json.load(open(f))["request"]
        data = json.dumps(req).encode()
        t0 = time.perf_counter()
        r = urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/infill", data=data,
                                                          headers={"Content-Type": "application/json"}), timeout=60)
        resp = json.loads(r.read())
        if req.get("n_predict") == 0:
            continue
        ms.append((time.perf_counter() - t0) * 1000)
        first = resp[0] if isinstance(resp, list) else resp
        t = first.get("timings", {})
        cached.append(t.get("cache_n", first.get("tokens_cached", 0)) or 0)
        new.append(t.get("prompt_n", 0))
        gen.append(t.get("predicted_n", 0))
        if shown < a.show and first.get("content", "").strip():
            print(f"  {Path(f).name}: {(req['input_prefix'][-30:] + req.get('prompt', ''))!r:45} -> {first['content'][:50]!r}")
            shown += 1

    wall = time.perf_counter() - wall0
    print(f"\n{a.url}: {len(ms)} completions in {wall:.1f} s ({len(ms) / wall:.2f}/s)")
    print(f"  latency ms      p50 {pct(ms, 50):6.0f}   p95 {pct(ms, 95):6.0f}   max {max(ms):6.0f}")
    print(f"  new prompt tok  p50 {pct(new, 50):6.0f}   p95 {pct(new, 95):6.0f}")
    print(f"  cached tok      p50 {pct(cached, 50):6.0f}")
    print(f"  generated tok   p50 {pct(gen, 50):6.0f}   p95 {pct(gen, 95):6.0f}")


if __name__ == "__main__":
    main()
