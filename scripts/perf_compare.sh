#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/perf_compare.sh [--phases] [base_report.json] [head_report.json]

Compares two polychora perf-suite JSON reports and prints per-scenario deltas.

Options:
  --phases   When both reports are v2, print per-phase GPU breakdown under each scenario.

Defaults:
  With no arguments, compares the two newest reports under profiles/.
  Newer report is treated as head; older as base.
USAGE
}

show_phases=false
positional=()

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --phases) show_phases=true ;;
    *) positional+=("$arg") ;;
  esac
done

base_report=""
head_report=""

if [[ ${#positional[@]} -eq 0 ]]; then
  mapfile -t latest_reports < <(ls -1t profiles/perf-suite-*.json 2>/dev/null | head -n 2)
  if [[ ${#latest_reports[@]} -lt 2 ]]; then
    echo "error: expected at least 2 reports under profiles/ (found ${#latest_reports[@]})" >&2
    echo "hint: run perf suite twice or pass explicit report paths" >&2
    exit 1
  fi
  head_report="${latest_reports[0]}"
  base_report="${latest_reports[1]}"
elif [[ ${#positional[@]} -eq 2 ]]; then
  base_report="${positional[0]}"
  head_report="${positional[1]}"
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

# Accept both v1 and v2 schemas.
for report in "$base_report" "$head_report"; do
  schema=$(jq -r '.schema // empty' "$report")
  if [[ "$schema" != "polychora.perf_suite.v1" && "$schema" != "polychora.perf_suite.v2" ]]; then
    echo "error: unsupported schema in $report: '$schema'" >&2
    exit 1
  fi
done

base_schema=$(jq -r '.schema' "$base_report")
head_schema=$(jq -r '.schema' "$head_report")

echo "perf-report-compare"
echo "  base: $base_report ($base_schema)"
echo "  head: $head_report ($head_schema)"

echo ""
base_meta=$(jq -r '"warmup=\(.warmup_frames) sample=\(.sample_frames) scenarios=\(.scenario_count) elapsed_s=\((.elapsed_seconds|tonumber)) backend=\(.render_backend)"' "$base_report")
head_meta=$(jq -r '"warmup=\(.warmup_frames) sample=\(.sample_frames) scenarios=\(.scenario_count) elapsed_s=\((.elapsed_seconds|tonumber)) backend=\(.render_backend)"' "$head_report")
echo "base-meta: $base_meta"
echo "head-meta: $head_meta"

# Show world type/seed if v2.
base_world=$(jq -r 'if .singleplayer_world_type then "world_type=\(.singleplayer_world_type) seed=\(.singleplayer_world_seed)" else "" end' "$base_report")
head_world=$(jq -r 'if .singleplayer_world_type then "world_type=\(.singleplayer_world_type) seed=\(.singleplayer_world_seed)" else "" end' "$head_report")
if [[ -n "$base_world" ]]; then echo "base-world: $base_world"; fi
if [[ -n "$head_world" ]]; then echo "head-world: $head_world"; fi
echo ""

# Determine if both are v2 (for render_config and phases display).
both_v2=false
if [[ "$base_schema" == "polychora.perf_suite.v2" && "$head_schema" == "polychora.perf_suite.v2" ]]; then
  both_v2=true
fi

# Build TSV with scenario data. For v2, include render_config info.
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
      (.render_gpu.avg_ms // null),
      (if $b.render_config then "\($b.render_config.vte_max_trace_steps)/\($b.render_config.vte_max_trace_distance)" else "" end),
      (if .render_config then "\(.render_config.vte_max_trace_steps)/\(.render_config.vte_max_trace_distance)" else "" end)
    ]
  | @tsv
' | awk -F '\t' -v show_rc="$both_v2" '
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
  if (show_rc == "true") {
    print "scenario                  cpu_base  cpu_head   cpu_d   cpu_%   gpu_base  gpu_head   gpu_d   gpu_%  render_config";
    print "----------------------------------------------------------------------------------------------------------------";
  } else {
    print "scenario                  cpu_base  cpu_head   cpu_d   cpu_%   gpu_base  gpu_head   gpu_d   gpu_%";
    print "-----------------------------------------------------------------------------------------------";
  }
}
{
  label = $1;
  cb = $2;
  ch = $3;
  gb = $4;
  gh = $5;
  rc_base = $6;
  rc_head = $7;

  cd = delta(cb, ch);
  cp = pct(cb, ch);
  gd = delta(gb, gh);
  gp = pct(gb, gh);

  if (show_rc == "true" && rc_head != "") {
    printf "%-24s %8s %8s %7s %7s %9s %9s %7s %7s  %s\n", label, fmt(cb), fmt(ch), cd, cp, fmt(gb), fmt(gh), gd, gp, rc_head;
  } else {
    printf "%-24s %8s %8s %7s %7s %9s %9s %7s %7s\n", label, fmt(cb), fmt(ch), cd, cp, fmt(gb), fmt(gh), gd, gp;
  }

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
  if (show_rc == "true") {
    print "----------------------------------------------------------------------------------------------------------------";
  } else {
    print "-----------------------------------------------------------------------------------------------";
  }
  if (cpu_n > 0 || gpu_n > 0) {
    cpu_avg = (cpu_n > 0) ? sprintf("%+.3f", cpu_sum / cpu_n) : "na";
    gpu_avg = (gpu_n > 0) ? sprintf("%+.3f", gpu_sum / gpu_n) : "na";
    printf "%-24s %8s %8s %7s %7s %9s %9s %7s %7s\n", "avg delta", "-", "-", cpu_avg, "-", "-", "-", gpu_avg, "-";
  }
}
'

# Per-phase GPU breakdown (only when --phases and both reports are v2).
if [[ "$show_phases" == "true" && "$both_v2" == "true" ]]; then
  echo ""
  echo "=== Per-Phase GPU Breakdown ==="
  echo ""

  jq -n -r --slurpfile base "$base_report" --slurpfile head "$head_report" '
    ($base[0].scenarios | map({key: .label, value: .}) | from_entries) as $bmap
    | $head[0].scenarios[]
    | .label as $label
    | $bmap[$label] as $b
    | (($b["render_gpu.phases"] // []) | map({key: .name, value: .}) | from_entries) as $bp
    | ((.["render_gpu.phases"] // []) | map({key: .name, value: .}) | from_entries) as $hp
    | ([($bp | keys[]), ($hp | keys[])] | unique) as $all_phases
    | $label,
      ($all_phases[] | . as $pn |
        "  " + $pn + "\t" +
        (if $bp[$pn] then ($bp[$pn].avg_ms | tostring) else "na" end) + "\t" +
        (if $hp[$pn] then ($hp[$pn].avg_ms | tostring) else "na" end)
      )
  ' | awk -F '\t' '
  function fmt(v) {
    if (v == "na" || v == "") return "na";
    return sprintf("%.3f", v + 0.0);
  }
  function delta(a, b) {
    if (a == "na" || b == "na" || a == "" || b == "") return "na";
    return sprintf("%+.3f", (b + 0.0) - (a + 0.0));
  }
  function pct(a, b) {
    if (a == "na" || b == "na" || a == "" || b == "") return "na";
    a = a + 0.0; b = b + 0.0;
    if (a == 0.0) return "na";
    return sprintf("%+.1f%%", ((b - a) / a) * 100.0);
  }
  {
    if (NF == 1) {
      # Scenario label
      print $0;
    } else {
      # Phase row
      name = $1;
      b = $2;
      h = $3;
      printf "  %-28s %8s -> %8s  %8s %7s\n", name, fmt(b), fmt(h), delta(b, h), pct(b, h);
    }
  }
  '
fi
