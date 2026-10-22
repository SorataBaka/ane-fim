#!/bin/bash
# Measure average CPU/GPU/ANE power: idle first, then while the ANE model generates.
# Run as your normal user (it calls sudo for powermetrics itself):
#   scripts/measure_power.sh [model_dir] [seconds]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANEMLL_DIR="${ANEMLL_DIR:-$HOME/src/Anemll}"
MODEL_DIR="${1:-$PROJECT_DIR/models/qwen25coder-0.5b-ctx1024-fp16}"
SECONDS_TO_SAMPLE="${2:-20}"
INTERVAL_MS=500
SAMPLES=$(( SECONDS_TO_SAMPLE * 1000 / INTERVAL_MS ))
WARMUP=15   # model load + chat.py warmup before sampling starts

OUT_DIR="$PROJECT_DIR/eval/power/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

sudo -v   # ask for the password once, up front

# Average "<Name> Power: N mW" lines from a powermetrics log.
summarize() {
    awk '
        /^CPU Power:/ { cpu += $3; n_cpu++ }
        /^GPU Power:/ { gpu += $3; n_gpu++ }
        /^ANE Power:/ { ane += $3; n_ane++ }
        /^Combined Power/ { tot += $(NF-1); n_tot++ }
        END {
            printf "  CPU %6.0f mW | GPU %6.0f mW | ANE %6.0f mW | Total %6.0f mW  (%d samples)\n",
                cpu/n_cpu, gpu/n_gpu, ane/n_ane, tot/n_tot, n_tot
        }' "$1"
}

sample() {
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i "$INTERVAL_MS" -n "$SAMPLES" > "$1" 2>/dev/null
}

# Single-line FIM prompts, fed to chat.py one per line, forever.
feed_prompts() {
    sleep 1
    while true; do
        # Stop as soon as chat.py has exited (write to closed pipe fails).
        echo '<|fim_prefix|>def quicksort(arr):<|fim_suffix|><|fim_middle|>' || return 0
        echo '<|fim_prefix|>function debounce(fn, ms) {<|fim_suffix|>}<|fim_middle|>' || return 0
        echo '<|fim_prefix|>func parseConfig(path string) (*Config, error) {<|fim_suffix|>}<|fim_middle|>' || return 0
        echo '<|fim_prefix|>class LRUCache:<|fim_suffix|><|fim_middle|>' || return 0
    done 2>/dev/null
}

echo "Idle: sampling ${SECONDS_TO_SAMPLE}s (don't touch the Mac)..."
sample "$OUT_DIR/idle.txt"

echo "Loading model and generating in a loop..."
cd "$ANEMLL_DIR"
source env-anemll/bin/activate
feed_prompts | python tests/chat.py --meta "$MODEL_DIR/meta.yaml" \
    --no-template --max-tokens 64 > "$OUT_DIR/workload.log" 2>&1 &
WORKLOAD_PID=$!
disown   # we kill it on purpose later; skip bash's "Terminated"/"Broken pipe" notices
sleep "$WARMUP"

echo "Generating: sampling ${SECONDS_TO_SAMPLE}s..."
sample "$OUT_DIR/generate.txt"

kill "$WORKLOAD_PID" 2>/dev/null || true
pkill -f "tests/chat.py --meta $MODEL_DIR" 2>/dev/null || true

{
    echo "Model: $MODEL_DIR"
    echo "Idle:"
    summarize "$OUT_DIR/idle.txt"
    echo "Generating (ANE):"
    summarize "$OUT_DIR/generate.txt"
    echo "Decode speed during run:"
    grep -E 'Inference:' "$OUT_DIR/workload.log" | awk '{s+=$2; n++} END {if (n) printf "  %.1f tok/s avg over %d completions\n", s/n, n}'
} | tee "$OUT_DIR/summary.txt"
