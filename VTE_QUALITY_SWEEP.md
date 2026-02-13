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
- `decision`: `keep`, `revert`, `pending`, or `on_ice`

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
| `EXP-20260212-CHUNK-SLAB-HOTPATH-GATE` | Skip per-chunk slab interval recomputation on normal VTE path (keep for debug/compare) | all VTE modes | Unchanged in reference compare (`mismatch=0`) | Part of measured stage-a reduction in aggregate | keep | DDA interval `[currentT, chunkExitT]` is sufficient on normal path; slab retained for strict debug/compare. |
| `EXP-20260212-INTEGRAL-STAGEA-FUSION` | In integral mode, perform layer accumulation in Stage A and skip Stage B dispatch | integral | Unchanged in reference compare (`mismatch=0`) | `vte_stage_b` removed in integral mode (`0.000 ms`) | keep | Eliminates full-frame Stage B resolve dispatch for integral path. |
| `EXP-20260212-INTEGRAL-DIR-BASIS-RECURRENCE` | Integral fast path uses transformed direction basis + angle recurrence instead of per-layer matrix/trig recompute | integral | Unchanged in reference compare (`mismatch=0`) | Contributes to lower Stage A ALU cost at high `layers` | keep | Preserves layer-angle sequence while reducing per-sample math overhead. |
| `EXP-20260212-VISIBLE-BOUNDS-LOOKUP-CULL` | Upload visible chunk min/max bounds; skip hash lookup when chunk DDA coordinate is outside that bounds AABB | all VTE modes | Unchanged in reference compare (`mismatch=0`) | Reduces useless hash probes outside visible chunk region | keep | Effective empty-space pruning before lookup. |
| `EXP-20260212-REANCHOR-RATE-GATE` | Keep frequent chunk-state re-anchor for debug/reference paths, reduce frequency on normal path | all VTE modes | Unchanged in reference compare (`mismatch=0`) | Reduces normal-path re-anchor overhead | keep | Keeps robust diagnostics while trimming hot-path work. |
| `EXP-20260212-VISIBLE-BOUNDS-INTERVAL-CLIP` | Intersect ray against visible-chunk world AABB; start traversal at entry and cap by exit | all VTE modes | Unchanged in reference compare (`mismatch=0`) | Improves Stage A stability by clipping traversal interval | keep | Prevents stepping outside guaranteed-empty ray ranges. |
| `FIX-20260212-INTEGRAL-REDUCED-STORAGE-LAYERS` | Separate logical sample layers from pixel-buffer storage layers; use 1 storage layer for explicit VTE integral startup path | voxel integral | Unchanged output in integral mode | Fixes 4K startup crash (`max_buffer_size`) and keeps high FPS | keep | Logical `--layers` sampling preserved; storage reduced to avoid `W*H*L` buffer blowup. |
| `EXP-20260212-STAGEA-MACRO-OCCUPANCY-SKIP` | Add per-chunk `2x2x2x2` macro occupancy masks and skip in-chunk voxel material/occupancy fetches when macro cell is empty | all VTE modes | Expected unchanged (conservative fallback on invalid offsets) | `TBD` (targets `vte_stage_a`) | keep (provisional) | Implements deferred L3 in-chunk hierarchy culling path. |
| `EXP-20260213-STAGEA-MACRO-CELL-CACHE` | Cache current macro-cell occupancy during in-chunk DDA; refresh only when voxel traversal enters a new `2x2x2x2` macro cell | all VTE modes | Expected unchanged first-hit behavior | `TBD` (targets `vte_stage_a` in sparse edited chunks) | keep (provisional) | Avoids repeated macro-word loads for multiple voxel steps inside the same macro cell. |
| `EXP-20260213-STAGEA-SPARSE-CHUNK-SOLID-AABB-CLIP` | Upload per-chunk local solid AABB (`min/max` voxel coords) and clip in-chunk DDA interval against that AABB before voxel stepping | all VTE modes | Expected unchanged first-hit behavior (conservative chunk-local envelope) | `TBD` (targets `vte_stage_a` spikes after sparse block edits) | keep (provisional) | Reduces empty-voxel stepping in chunks with a small edited solid region (e.g. new `y=0` chunks over flat slab). |
| `TOOLING-20260213-VTE-SWEEP-CHUNK-SOLID-CLIP-GATE` | Runtime flag/toggle for chunk solid-AABB clip and sweep matrix expansion to include clip on/off profiles | all VTE modes | No rendering change when clip state unchanged | Enabled direct same-scene A/B to prove significance | superseded | Superseded by `CLEANUP-20260213-VTE-FLAG-PRUNE` once clip was proven high-value in repeated sweeps. |
| `EXP-20260213-STAGEA-YSLICE-REEVAL-WITH-CHUNK-CLIP` | Re-evaluate y-slice fastpath after introducing sparse chunk solid-AABB clip | non-debug VTE modes | Improved in current live sweeps (with/without entities) | Stage-A reduction in latest A/B sweeps (~2 ms in sampled scene) | keep (provisional) | Y-slice fastpath is now always enabled on normal VTE path; old CLI toggle retained as deprecated no-op for compatibility. |
| `CLEANUP-20260213-VTE-FLAG-PRUNE` | Remove runtime y-slice-fastpath / chunk-solid-clip toggles from sweep axis; keep both always enabled in normal path | normal VTE modes | Unchanged relative to proven baseline configuration | Reduces runtime/sweep complexity; allows shader/compiler simplification | keep | Runtime sweep axis now focuses on `entities` and new lookup-cache toggle. Deprecated CLI toggles are accepted as no-ops for compatibility. |
| `EXP-20260213-STAGEA-YSLICE-LOOKUP-CACHE-PRECHECK` | Add optional y-slice sparse chunk-presence precheck (bitmask) before chunk-hash lookup in y-slice DDA path | non-debug VTE modes | Unchanged in repeated sweeps | Flat/noise-level impact in sampled scenes | superseded | Replaced by direct y-slice chunk-index lookup table experiment (`EXP-20260213-STAGEA-YSLICE-DIRECT-LOOKUP-TABLE`). |
| `EXP-20260213-STAGEA-YSLICE-DIRECT-LOOKUP-TABLE` | Build per-y-slice direct `x/z/w -> chunkIndex` table and use it to bypass hash lookup on lookup-cache path | non-debug VTE modes | Expected unchanged first-hit behavior (exact chunk index mapping) | `TBD` (targets `vte_stage_a` by removing hash probes on covered slices) | keep (provisional) | Still sweep-gated via runtime y-slice lookup-cache toggle (`F7`) for same-scene A/B. |
| `TOOLING-20260213-VTE-SWEEP-ENTITIES-DEFAULT` | Make runtime `F8` sweep default to entity-preserving profiles (lookup-cache on/off with entities enabled); keep no-entities profiles opt-in via `R4D_VTE_SWEEP_INCLUDE_NO_ENTITIES=1` | runtime sweep tooling | N/A (measurement workflow only) | Improves relevance of routine sweeps to shipping behavior | keep | Keeps entity-disable measurements available for occasional attribution without treating removal as a target optimization path. |
| `EXP-20260212-STAGEA-YSLICE-INTERVAL-FASTPATH` | Build per-`chunk_y` x/z/w bounds and trace non-debug rays via y-slice AABB intervals + 3D chunk DDA inside each slice | non-debug VTE modes | Mixed in earlier ad-hoc block-edit runs (pre chunk-solid-clip tuning) | Previously regressed in that earlier configuration | superseded | Superseded by `EXP-20260213-STAGEA-YSLICE-REEVAL-WITH-CHUNK-CLIP`; keep this row for historical context only. |
| `EXP-20260212-STAGEA-DYNAMIC-LOOKUP-CAPACITY` | Size chunk hash lookup table per frame (`~2x visible`, power-of-two) instead of always using max-capacity mask | all VTE modes | Expected unchanged hit/miss behavior | `TBD` (targets `vte_stage_a` hash lookup cache efficiency) | keep (provisional) | Reduces random lookup working-set size for small visible chunk sets. |
| `EXP-20260212-STAGEA-PER-RAY-CHUNK-BUDGET-CAP` | Cap per-ray chunk DDA loop by conservative crossing estimate from clipped traversal interval + `tDeltaChunk` | normal VTE modes | Expected unchanged first-hit results | `TBD` (targets Stage-A tail latency from long miss-ray loops) | keep (provisional) | Leaves debug/reference compare behavior unchanged. |
| `EXP-20260212-VTE-ENTITY-BVH-ALWAYS` | Use BVH traversal for any non-zero entity tetra set (`linear_threshold=0`) | all VTE modes with entities | Unchanged entity intersection result | `TBD` (targets `vte_stage_a` variance from per-ray linear loops) | keep (provisional) | Keeps CPU/shader threshold policy aligned to avoid mixed traversal paths. |
| `EXP-20260212-VTE-OVERLAY-FAST-ZW` | VTE overlay raster path skips O(n^2) occlusion prepass and uses reduced hidden-dimension sample count | VTE post-raster overlay path | Slightly reduced preview fidelity (acceptable for block/item preview) | `TBD` (targets `tet_raster`) | keep (provisional) | Non-VTE tetra path keeps full-quality occlusion + 256-sample integration. |
| `CLEANUP-20260211-COMPARE-READBACK-GATING` | Gate compare buffer reset/readback to debug-compare modes only | all VTE modes | No visual change outside debug compare | Reduces CPU-side overhead in normal rendering | keep | Prevents unconditional per-frame debug buffer map/reset work. |
| `CLEANUP-20260211-TIE-FLAG-GATING` | Compute tie/zero-interval debug flags only in debug modes | all VTE modes | No visual change outside debug compare | Reduces per-chunk arithmetic in normal rendering | keep | Keeps diagnostics fidelity while trimming non-debug hot path. |
| `CLEANUP-20260212-VISIBLE-HASH-GATING` | Compute/print `vh` visible-set hash only when reference compare is enabled | all VTE modes | No visual change | Reduces per-frame CPU hash work and HUD noise in normal mode | keep | Hash remains available for mismatch/debug runs. |

### On-Ice Future Optimizations

These are intentionally deferred while game features advance:

- Investigate `frame_end` tail cost (~2.8 ms at 1080p/4K) in present/copy/sync path.
- Add tiled/layer-chunked storage path for non-integral VTE display modes at very high resolutions.
- Explore adaptive hidden-dimension sampling to preserve quality at lower effective layer cost.
