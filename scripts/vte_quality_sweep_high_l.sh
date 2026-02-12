#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Focused high-layer sweep for flat-horizon quality convergence.
# Override any variable at callsite if needed.
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
LAYERS_LIST="${LAYERS_LIST:-64 96 128 160 192 256}"
TRACE_STEPS_LIST="${TRACE_STEPS_LIST:-320}"
OUT_DIR="${OUT_DIR:-frames/vte_quality_sweep_high_l}"
SCENE_FILTER="${SCENE_FILTER:-^flat$}"
SHOT_FILTER="${SHOT_FILTER:-^flat_horizon_[ab]$}"

echo "Running high-L sweep:"
echo "  WIDTH=$WIDTH HEIGHT=$HEIGHT"
echo "  LAYERS_LIST=$LAYERS_LIST"
echo "  TRACE_STEPS_LIST=$TRACE_STEPS_LIST"
echo "  OUT_DIR=$OUT_DIR"
echo "  SCENE_FILTER=$SCENE_FILTER"
echo "  SHOT_FILTER=$SHOT_FILTER"

WIDTH="$WIDTH" \
HEIGHT="$HEIGHT" \
LAYERS_LIST="$LAYERS_LIST" \
TRACE_STEPS_LIST="$TRACE_STEPS_LIST" \
OUT_DIR="$OUT_DIR" \
SCENE_FILTER="$SCENE_FILTER" \
SHOT_FILTER="$SHOT_FILTER" \
"$ROOT_DIR/scripts/vte_quality_sweep.sh"
