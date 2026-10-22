#!/bin/bash
# Convert Qwen2.5-Coder-0.5B (base) to Core ML for the ANE using ANEMLL.
# Usage: [SIZE=1.5B] [CHUNK=2] [CONTEXT=2048] [ARGMAX=0] scripts/convert.sh [fp16|lut6]
#   defaults: SIZE=0.5B, CHUNK=1, CONTEXT=1024, argmax on, fp16
# ARGMAX=0 keeps full logits (needed for the infill EOG stop rule); output dir gets a "-logits" suffix.
# CHUNK: see `python anemll/utils/calc_chunk_split.py Qwen/Qwen2.5-Coder-<SIZE> --lut 6` (1.5B LUT6 → 2).
set -euo pipefail

ANEMLL_DIR="${ANEMLL_DIR:-$HOME/src/Anemll}"   # pinned at tag 0.3.5_beta
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIZE="${SIZE:-0.5B}"
CHUNK="${CHUNK:-1}"
MODEL="Qwen/Qwen2.5-Coder-${SIZE}"
CONTEXT="${CONTEXT:-1024}"
BATCH=64
MODE="${1:-fp16}"

case "$MODE" in
    # --no-lut is listed in convert_model.sh --help but not implemented (0.3.5_beta);
    # empty LUT values are what actually disable quantization.
    fp16) QUANT=(--lut1 "" --lut2 "" --lut3 "") ;;
    lut6) QUANT=(--lut1 6 --lut2 6 --lut3 6) ;;
    *) echo "Unknown mode: $MODE (use fp16 or lut6)"; exit 1 ;;
esac

ARGMAX_FLAG=(--argmax)
SUFFIX=""
if [ "${ARGMAX:-1}" = "0" ]; then
    ARGMAX_FLAG=()
    SUFFIX="-logits"
fi

SIZE_LC="$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')"
OUT="$PROJECT_DIR/models/qwen25coder-${SIZE_LC}-ctx${CONTEXT}-${MODE}${SUFFIX}"
mkdir -p "$OUT"

cd "$ANEMLL_DIR"
source env-anemll/bin/activate
./anemll/utils/convert_model.sh \
    --model "$MODEL" \
    --output "$OUT" \
    --context "$CONTEXT" \
    --batch "$BATCH" \
    --chunk "$CHUNK" \
    ${ARGMAX_FLAG[@]+"${ARGMAX_FLAG[@]}"} \
    "${QUANT[@]}" \
    2>&1 | tee "$OUT/convert.log"
