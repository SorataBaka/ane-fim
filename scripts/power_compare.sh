#!/bin/bash
# Phase 5: power while replaying real llama.vscode requests, ANE server vs llama-server (GPU).
# Both servers must already be running. Run as your normal user (calls sudo for powermetrics):
#   scripts/power_compare.sh [seconds_per_phase]
# Don't type in VS Code while it runs (the editor would send requests too).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SECS="${1:-30}"
ANE_URL="${ANE_URL:-http://127.0.0.1:8012}"
GPU_URL="${GPU_URL:-http://127.0.0.1:8013}"
INTERVAL_MS=500
SAMPLES=$(( SECS * 1000 / INTERVAL_MS ))
OUT_DIR="$PROJECT_DIR/eval/power/compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

for url in "$ANE_URL" "$GPU_URL"; do
    curl -sf "$url/health" > /dev/null || { echo "Server not healthy: $url"; exit 1; }
done

sudo -v   # ask for the password once, up front

sample() {
    sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i "$INTERVAL_MS" -n "$SAMPLES" > "$1" 2>/dev/null
}

replay() {  # url name
    echo "Replaying against $2 for ${SECS}s..."
    sample "$OUT_DIR/$2.power.txt" &
    local pm=$!
    python3 "$PROJECT_DIR/eval/bench_latency.py" --url "$1" --duration "$SECS" --show 0 > "$OUT_DIR/$2.bench.txt"
    wait "$pm"
}

echo "Idle (both servers loaded) for ${SECS}s..."
sample "$OUT_DIR/idle.power.txt"
replay "$ANE_URL" ane
sleep 10
replay "$GPU_URL" gpu

python3 - "$OUT_DIR" <<'EOF' | tee "$OUT_DIR/summary.txt"
import re, sys
from pathlib import Path
d = Path(sys.argv[1])

def power(name):
    vals = {"CPU": [], "GPU": [], "ANE": [], "Total": []}
    for line in (d / f"{name}.power.txt").read_text().splitlines():
        m = re.match(r"(CPU|GPU|ANE) Power: (\d+) mW", line)
        if m:
            vals[m[1]].append(int(m[2]))
        m = re.match(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", line)
        if m:
            vals["Total"].append(int(m[1]))
    return {k: sum(v) / len(v) / 1000 for k, v in vals.items() if v}

def bench(name):
    text = (d / f"{name}.bench.txt").read_text()
    n, secs = re.search(r"(\d+) completions in ([\d.]+) s", text).groups()
    p50, p95 = re.search(r"latency ms\s+p50\s+(\d+)\s+p95\s+(\d+)", text).groups()
    return int(n), float(secs), int(p50), int(p95)

idle = power("idle")
print(f"{'':10s} {'CPU W':>6s} {'GPU W':>6s} {'ANE W':>6s} {'Total W':>8s} {'req/s':>6s} {'p50 ms':>7s} {'p95 ms':>7s} {'J/req':>6s} {'extra J/req':>11s}")
print(f"{'idle':10s} {idle['CPU']:6.2f} {idle['GPU']:6.2f} {idle['ANE']:6.2f} {idle['Total']:8.2f}")
for name, label in (("ane", "ANE"), ("gpu", "llama GPU")):
    p = power(name)
    n, secs, p50, p95 = bench(name)
    rate = n / secs
    print(f"{label:10s} {p['CPU']:6.2f} {p['GPU']:6.2f} {p['ANE']:6.2f} {p['Total']:8.2f} {rate:6.1f} {p50:7d} {p95:7d} "
          f"{p['Total'] / rate:6.3f} {(p['Total'] - idle['Total']) / rate:11.3f}")
print("\nJ/req = total CPU+GPU+ANE energy per completion; extra J/req = same minus idle power.")
EOF
