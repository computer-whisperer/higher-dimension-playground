#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/perf_compare.sh
  scripts/perf_compare.sh <base_report.json> <head_report.json>

Compares two polychora perf-suite JSON reports and prints per-scenario deltas.

Defaults:
  With no arguments, compares the two newest reports under profiles/.
  Newer report is treated as head; older as base.
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

base_report=""
head_report=""

if [[ $# -eq 0 ]]; then
  mapfile -t latest_reports < <(ls -1t profiles/perf-suite-*.json 2>/dev/null | head -n 2)
  if [[ ${#latest_reports[@]} -lt 2 ]]; then
    echo "error: expected at least 2 reports under profiles/ (found ${#latest_reports[@]})" >&2
    echo "hint: run perf suite twice or pass explicit report paths" >&2
    exit 1
  fi
  head_report="${latest_reports[0]}"
  base_report="${latest_reports[1]}"
elif [[ $# -eq 2 ]]; then
  base_report="$1"
  head_report="$2"
else
  usage >&2
  exit 1
fi

if [[ ! -f "$base_report" ]]; then
  echo "error: base report not found: $base_report" >&2
  exit 1
fi
if [[ ! -f "$head_report" ]]; then
  echo "error: head report not found: $head_report" >&2
  exit 1
fi

for report in "$base_report" "$head_report"; do
  schema=$(jq -r '.schema // empty' "$report")
  if [[ "$schema" != "polychora.perf_suite.v1" ]]; then
    echo "error: unsupported schema in $report: '$schema'" >&2
    exit 1
  fi
done

echo "perf-report-compare"
echo "  base: $base_report"
echo "  head: $head_report"

echo ""
base_meta=$(jq -r '"warmup=\(.warmup_frames) sample=\(.sample_frames) scenarios=\(.scenario_count) elapsed_s=\((.elapsed_seconds|tonumber)) backend=\(.render_backend)"' "$base_report")
head_meta=$(jq -r '"warmup=\(.warmup_frames) sample=\(.sample_frames) scenarios=\(.scenario_count) elapsed_s=\((.elapsed_seconds|tonumber)) backend=\(.render_backend)"' "$head_report")
echo "base-meta: $base_meta"
echo "head-meta: $head_meta"
echo ""

jq -n -r --slurpfile base "$base_report" --slurpfile head "$head_report" '
  ($base[0].scenarios | map({key: .label, value: .}) | from_entries) as $bmap
  | $head[0].scenarios[]
  | .label as $label
  | $bmap[$label] as $b
  | [
      $label,
      ($b.client_cpu.avg_ms // null),
      (.client_cpu.avg_ms // null),
      ($b.render_gpu.avg_ms // null),
      (.render_gpu.avg_ms // null)
    ]
  | @tsv
' | awk -F '\t' '
function fmt(v) {
  if (v == "" || v == "null") return "na";
  return sprintf("%.3f", v + 0.0);
}
function delta(a, b) {
  if (a == "" || b == "" || a == "null" || b == "null") return "na";
  return sprintf("%+.3f", (b + 0.0) - (a + 0.0));
}
function pct(a, b) {
  if (a == "" || b == "" || a == "null" || b == "null") return "na";
  a = a + 0.0;
  b = b + 0.0;
  if (a == 0.0) return "na";
  return sprintf("%+.1f%%", ((b - a) / a) * 100.0);
}
BEGIN {
  cpu_sum = 0.0;
  gpu_sum = 0.0;
  cpu_n = 0;
  gpu_n = 0;
  print "scenario                  cpu_base  cpu_head   cpu_d   cpu_%   gpu_base  gpu_head   gpu_d   gpu_%";
  print "-----------------------------------------------------------------------------------------------";
}
{
  label = $1;
  cb = $2;
  ch = $3;
  gb = $4;
  gh = $5;

  cd = delta(cb, ch);
  cp = pct(cb, ch);
  gd = delta(gb, gh);
  gp = pct(gb, gh);

  printf "%-24s %8s %8s %7s %7s %9s %9s %7s %7s\n", label, fmt(cb), fmt(ch), cd, cp, fmt(gb), fmt(gh), gd, gp;

  if (cd != "na") {
    cpu_sum += cd + 0.0;
    cpu_n += 1;
  }
  if (gd != "na") {
    gpu_sum += gd + 0.0;
    gpu_n += 1;
  }
}
END {
  print "-----------------------------------------------------------------------------------------------";
  if (cpu_n > 0 || gpu_n > 0) {
    cpu_avg = (cpu_n > 0) ? sprintf("%+.3f", cpu_sum / cpu_n) : "na";
    gpu_avg = (gpu_n > 0) ? sprintf("%+.3f", gpu_sum / gpu_n) : "na";
    printf "%-24s %8s %8s %7s %7s %9s %9s %7s %7s\n", "avg delta", "-", "-", cpu_avg, "-", "-", "-", gpu_avg, "-";
  }
}
'
