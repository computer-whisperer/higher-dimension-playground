#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
LAYERS_LIST="${LAYERS_LIST:-8 16 32 64}"
TRACE_STEPS_LIST="${TRACE_STEPS_LIST:-160 320 640}"
OUT_DIR="${OUT_DIR:-frames/vte_quality_sweep}"
SCENE_FILTER="${SCENE_FILTER:-}"
SHOT_FILTER="${SHOT_FILTER:-}"
VTE_DISPLAY_MODE="${VTE_DISPLAY_MODE:-}"
VTE_SLICE_LAYER="${VTE_SLICE_LAYER:-}"
VTE_THICK_HALF_WIDTH="${VTE_THICK_HALF_WIDTH:-}"
GPU_SCREENSHOT_SOURCE="${GPU_SCREENSHOT_SOURCE:-render-buffer}"

mkdir -p "$OUT_DIR"

# scene shot_name pos(x y z w) angles_deg(yaw pitch xw zw) yw
SHOTS=(
  "flat flat_horizon_a 16.69 4.32 -15.22 -4.00 22 -15 22 0 0"
  "flat flat_horizon_b 17.18 4.32 -14.84 -4.00 22 -15 22 0 0"
  "demo-cubes cubes_overview 0.00 1.80 -9.50 -5.00 22 -10 22 0 0"
  "demo-cubes cubes_close 0.50 0.80 -5.80 -3.60 28 -8 22 0 0"
)

echo "Output directory: $OUT_DIR"
echo "Layers: $LAYERS_LIST"
echo "Trace steps: $TRACE_STEPS_LIST"
if [[ -n "$SCENE_FILTER" ]]; then
  echo "Scene filter (regex): $SCENE_FILTER"
fi
if [[ -n "$SHOT_FILTER" ]]; then
  echo "Shot filter (regex): $SHOT_FILTER"
fi
if [[ -n "$VTE_DISPLAY_MODE" ]]; then
  echo "VTE display mode: $VTE_DISPLAY_MODE"
fi
if [[ -n "$VTE_SLICE_LAYER" ]]; then
  echo "VTE slice layer: $VTE_SLICE_LAYER"
fi
if [[ -n "$VTE_THICK_HALF_WIDTH" ]]; then
  echo "VTE thick half width: $VTE_THICK_HALF_WIDTH"
fi
echo "GPU screenshot source: $GPU_SCREENSHOT_SOURCE"

EXTRA_ARGS=()
if [[ -n "$VTE_DISPLAY_MODE" ]]; then
  EXTRA_ARGS+=(--vte-display-mode "$VTE_DISPLAY_MODE")
fi
if [[ -n "$VTE_SLICE_LAYER" ]]; then
  EXTRA_ARGS+=(--vte-slice-layer "$VTE_SLICE_LAYER")
fi
if [[ -n "$VTE_THICK_HALF_WIDTH" ]]; then
  EXTRA_ARGS+=(--vte-thick-half-width "$VTE_THICK_HALF_WIDTH")
fi

for shot in "${SHOTS[@]}"; do
  read -r scene shot_name px py pz pw yaw pitch xw zw yw <<<"$shot"
  if [[ -n "$SCENE_FILTER" ]] && ! [[ "$scene" =~ $SCENE_FILTER ]]; then
    continue
  fi
  if [[ -n "$SHOT_FILTER" ]] && ! [[ "$shot_name" =~ $SHOT_FILTER ]]; then
    continue
  fi
  for layers in $LAYERS_LIST; do
    for steps in $TRACE_STEPS_LIST; do
      echo "Rendering scene=$scene shot=$shot_name layers=$layers steps=$steps"
      cargo run -p polychora --release -- \
        --backend voxel-traversal \
        --scene "$scene" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --layers "$layers" \
        --vte-max-trace-steps "$steps" \
        "${EXTRA_ARGS[@]}" \
        --gpu-screenshot \
        --gpu-screenshot-source "$GPU_SCREENSHOT_SOURCE" \
        --screenshot-pos "$px" "$py" "$pz" "$pw" \
        --screenshot-angles-deg "$yaw" "$pitch" "$xw" "$zw" \
        --screenshot-yw "$yw"

      out_file="${OUT_DIR}/${scene}_${shot_name}_L${layers}_S${steps}.png"
      cp frames/gpu_render.png "$out_file"
      echo "  -> $out_file"
    done
  done
done

echo "Sweep complete."
