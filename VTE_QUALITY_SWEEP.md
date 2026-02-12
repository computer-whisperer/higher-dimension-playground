# VTE Quality Sweep

This document defines reproducible screenshot sweeps for evaluating VTE quality
vs. performance, with fixed scenes/cameras so artifacts are comparable.

## Included Scenes

- `flat`: the existing flat-world baseline.
- `demo-cubes`: voxelized version of the demo cube layout:
  - 2x2x2x2 outer lattice of colored blocks (materials 1..5 cycling)
  - bright center block (material 13)

Use `--scene flat` or `--scene demo-cubes` with `game`.

## Sweep Script

Run:

```bash
scripts/vte_quality_sweep.sh
```

Default sweep:

- `layers`: `8 16 32 64`
- `vte-max-trace-steps`: `160 320 640`
- resolution: `960x540`

Outputs are written to `frames/vte_quality_sweep/` with filenames:

`<scene>_<shot>_L<layers>_S<trace_steps>.png`

## Useful Overrides

```bash
LAYERS_LIST="16 32 64 96" \
TRACE_STEPS_LIST="320 640 960" \
OUT_DIR="frames/vte_quality_sweep_run2" \
scripts/vte_quality_sweep.sh
```

```bash
WIDTH=1280 HEIGHT=720 scripts/vte_quality_sweep.sh
```

## Notes

- `--layers` controls hidden-dimension sample count for VTE directly.
- `--vte-max-trace-steps` controls traversal budget (quality/perf tradeoff).
- `--vte-max-trace-distance` caps traversal distance (raise for long-range debug checks).
- `--vte-display-mode` selects resolve/diagnostic output (`integral`, `slice`, `thick-slice`, `debug-compare`, `debug-integral`).
- The script uses fixed camera presets to make artifact comparison stable.

## Debug Compare Flags

Long-lived VTE compare toggles:

- `R4D_VTE_REFERENCE_COMPARE=1`: run reference tracer alongside fast DDA in debug compare/integral modes.
- `R4D_VTE_REFERENCE_MISMATCH_ONLY=1`: render only mismatch classes (suppresses matching output).
- `R4D_VTE_COMPARE_SLICE_ONLY=1`: compare only selected layer (`--vte-slice-layer`) for faster iteration.

## VTE Experiment Ledger

Track every VTE correctness/perf experiment here so we can revert changes that
hurt performance without improving quality.

### Logging Rule

For each attempted fix, record:

- `id`: short unique identifier (`EXP-YYYYMMDD-...`)
- `scope`: what path changed (`chunk DDA`, `in-chunk DDA`, `debug tooling`, etc.)
- `mode`: where evaluated (`integral`, `slice`, `debug-compare`, etc.)
- `quality_result`: improved / unchanged / regressed
- `perf_result`: measured impact (or `TBD`)
- `decision`: `keep`, `revert`, or `pending`

If `quality_result=unchanged` and `perf_result` is a regression in non-debug
mode, default decision should be `revert`.

### Current Entries

| id | scope | mode | quality_result | perf_result | decision | notes |
|---|---|---|---|---|---|---|
| `EXP-20260211-REFCOMPARE-MISMATCH-ONLY` | Debug compare visualization and env toggles (`R4D_VTE_REFERENCE_COMPARE`, `R4D_VTE_REFERENCE_MISMATCH_ONLY`) | `debug-compare` | Improved diagnostics clarity | Debug-only overhead | keep | Separates real mismatches from normal chunk colors. |
| `EXP-20260211-CHUNK-TIE-RESEED` | Tie-crossing reseed in chunk DDA | `debug-compare` | Unresolved (purple mismatch remained) | Likely negative (extra arithmetic, no quality gain) | revert | Replaced by deterministic single-axis stepping. |
| `EXP-20260211-CHUNK-ABSOLUTE-T` | Recompute `chunkCoord` from absolute `t` each chunk step | `debug-compare` | Improved stability under boundary cases | `TBD` in normal mode | keep (provisional) | Still present in current fix path. |
| `EXP-20260211-CHUNK-FORWARD-PROBE` | Forward probe (`t+eps`) for chunk ownership | `debug-compare` | Improved boundary ownership stability | `TBD` in normal mode | keep (provisional) | Needed with chunk-entry ownership semantics. |
| `EXP-20260211-INCHUNK-ENTRY-BIAS` | Scale-aware `tEnter`/interval bias in in-chunk DDA | `debug-compare` | Improved robustness at chunk boundaries | `TBD` in normal mode | keep (provisional) | Keeps interval math scale-aware instead of fixed epsilon. |
| `EXP-20260211-COMPARE-SLICE-ONLY` | Compare only selected layer in debug mode (`R4D_VTE_COMPARE_SLICE_ONLY`) | `debug-compare` | Improved diagnostic iteration speed | Large debug perf win, no normal-mode impact | keep | Keeps mismatch semantics while avoiding full `W*H*L` reference trace cost. |
| `EXP-20260211-HIGH-BUDGET-COMPARE` | Raised compare tracing budgets (`--vte-max-trace-steps 4096`, `--vte-max-trace-distance 4096`) | `debug-compare` | Necessary for stress diagnostics, not a fix by itself | Severe debug perf regression (~1 FPS) | keep (debug-only workflow) | Use only for targeted captures; not required for routine sweeps. |
| `EXP-20260211-FIRST-MISMATCH-TRACE-META` | Added first-mismatch telemetry (`chunk_steps`, `final_t`, `remaining_voxels`, `last_chunk`) | `debug-compare` | Improved root-cause visibility | Debug-only overhead | keep | Supports diagnosing chunk-budget misses without full per-step trace dumps. |
| `FIX-20260211-CHUNK-ENTRY-OWNERSHIP` | Direction-aware chunk boundary ownership (`entryChunkCoord`) | all VTE modes | Fixed chunk pop-in and compare mismatches (`mismatch=0` at repro) | `TBD` in normal mode | keep | Resolves boundary pinning that caused chunk-budget stalls near `t≈0`. |
| `EXP-20260212-CHUNK-SLAB-HOTPATH-GATE` | Skip per-chunk slab interval recomputation on normal VTE path (keep for debug/compare) | all VTE modes | `TBD` (expected unchanged) | `TBD` (expected stage_a improvement) | pending | DDA interval already provides `[currentT, chunkExitT]`; slab kept for diagnostic strictness. |
| `EXP-20260212-INTEGRAL-STAGEA-FUSION` | In integral mode, perform layer accumulation in Stage A and skip Stage B dispatch | integral | `TBD` (expected unchanged) | `TBD` (expected total frame reduction, especially Stage B) | pending | Avoids full-frame layer readback pass in Stage B for integral mode. |
| `CLEANUP-20260211-COMPARE-READBACK-GATING` | Gate compare buffer reset/readback to debug-compare modes only | all VTE modes | No visual change outside debug compare | Reduces CPU-side overhead in normal rendering | keep | Prevents unconditional per-frame debug buffer map/reset work. |
| `CLEANUP-20260211-TIE-FLAG-GATING` | Compute tie/zero-interval debug flags only in debug modes | all VTE modes | No visual change outside debug compare | Reduces per-chunk arithmetic in normal rendering | keep | Keeps diagnostics fidelity while trimming non-debug hot path. |
| `CLEANUP-20260212-VISIBLE-HASH-GATING` | Compute/print `vh` visible-set hash only when reference compare is enabled | all VTE modes | No visual change | Reduces per-frame CPU hash work and HUD noise in normal mode | keep | Hash remains available for mismatch/debug runs. |
