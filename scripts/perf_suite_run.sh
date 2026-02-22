#!/usr/bin/env bash
set -euo pipefail

# Multi-seed perf suite runner.
#
# Environment variables:
#   PERF_SEEDS          Space-separated list of world seeds (default: "1337 42 9001")
#   PERF_WORLD_TYPE     World generator (default: "massive-platforms")
#   PERF_WARMUP         Warmup frames per scenario (default: 180)
#   PERF_SAMPLE         Sample frames per scenario (default: 600)
#   PERF_TAG            Tag prefix for report filenames (default: "run")
#   PERF_EXTRA_ARGS     Extra arguments passed to polychora (default: "")
#   PERF_BINARY         Path to polychora binary (default: "target/release/polychora")
#   PERF_REPORT_DIR     Output directory for reports (default: "profiles")
#   PERF_SPAWN_ENTITIES Entity count for BVH testing (default: 0)

SEEDS="${PERF_SEEDS:-1337 42 9001}"
WORLD_TYPE="${PERF_WORLD_TYPE:-massive-platforms}"
WARMUP="${PERF_WARMUP:-180}"
SAMPLE="${PERF_SAMPLE:-600}"
TAG="${PERF_TAG:-run}"
EXTRA_ARGS="${PERF_EXTRA_ARGS:-}"
BINARY="${PERF_BINARY:-target/release/polychora}"
REPORT_DIR="${PERF_REPORT_DIR:-profiles}"
SPAWN_ENTITIES="${PERF_SPAWN_ENTITIES:-0}"

if [[ ! -x "$BINARY" ]]; then
  echo "error: binary not found or not executable: $BINARY" >&2
  echo "hint: run 'cargo build -p polychora --release' first" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

echo "perf-suite-runner"
echo "  seeds: $SEEDS"
echo "  world_type: $WORLD_TYPE"
echo "  warmup: $WARMUP  sample: $SAMPLE"
echo "  tag: $TAG"
echo "  spawn_entities: $SPAWN_ENTITIES"
echo "  binary: $BINARY"
echo ""

for seed in $SEEDS; do
  unix_ts=$(date +%s)
  report_file="${REPORT_DIR}/perf-suite-${TAG}-${WORLD_TYPE}-seed${seed}-${unix_ts}.json"
  echo "=== Seed $seed ==="
  echo "  report: $report_file"

  # shellcheck disable=SC2086
  "$BINARY" \
    --perf-suite \
    --perf-suite-warmup-frames "$WARMUP" \
    --perf-suite-sample-frames "$SAMPLE" \
    --perf-suite-exit-on-complete true \
    --perf-suite-report "$report_file" \
    --perf-suite-spawn-entities "$SPAWN_ENTITIES" \
    --singleplayer-world-type "$WORLD_TYPE" \
    --singleplayer-world-seed "$seed" \
    $EXTRA_ARGS

  if [[ -f "$report_file" ]]; then
    echo "  done: $report_file"
  else
    echo "  warning: report file not created: $report_file" >&2
  fi
  echo ""
done

echo "All seeds complete. Reports in $REPORT_DIR/"
