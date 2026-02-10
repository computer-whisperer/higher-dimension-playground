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
- The script uses fixed camera presets to make artifact comparison stable.
