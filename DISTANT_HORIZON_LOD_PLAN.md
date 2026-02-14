# Distant Horizon / LOD Plan (4D Voxel World)

## Goal

Render a much farther 4D horizon without exploding GPU traversal cost, RAM, or save size, while keeping:

- deterministic seed-driven procgen
- server-authoritative world state
- copy-on-write edits only (persist edits, not generated base world)
- compatibility with current VTE near-field visuals

## Current Constraints (from code)

- `CHUNK_SIZE = 8`, `CHUNK_VOLUME = 4096` (`crates/polychora/src/shared/voxel/mod.rs`).
- Current active voxel set is a single near window around camera (`RENDER_DISTANCE = 64`) (`crates/polychora/src/scene.rs`, `crates/polychora/src/scene/voxel_runtime.rs`).
- VTE uses one flat visible chunk set and one lookup table per frame (`src/render/vte.rs`, `slang-shaders/src/voxel.slang`).
- Ray cost is linear in chunk traversal depth (`maxTraceSteps`, `maxTraceDistance`) (`slang-shaders/src/voxel.slang`).
- Server already has deterministic virtual chunk generation (base + structures + keepout) and only materializes chunk overrides for edits (`crates/polychora/src/server/mod.rs`, `crates/polychora/src/server/procgen.rs`, `crates/polychora/src/shared/voxel/world.rs`).

This means we already have the right ownership split (server generation + sparse edited overrides), but no distant representation.

## Efficient GPU Strategy for Distant 4D Objects

### Recommendation: Ringed Multi-Resolution Voxel LOD (Clipmap-style)

Use multiple voxel levels instead of only L0 chunks:

- L0 (near): exact current chunks (1x voxel scale).
- L1 (mid): downsampled chunks (2x voxel scale).
- L2 (far): downsampled chunks (4x voxel scale).
- Optional L3+: 8x, etc.

Each LOD chunk still stores `8^4` occupancy/material cells, but each cell covers larger world-space volume.

Why this is efficient in this engine:

- Reuses current VTE chunk/payload model and interning pipeline.
- Avoids introducing a fully new far-geometry renderer.
- Keeps shader math in DDA form (already optimized/debugged).
- Greatly reduces chunk-step count at long distance.

### Why not far meshes/impostors first

For this engine today, voxel clipmaps are lower risk than introducing a second distant renderer because:

- distance shading and hidden-dimension semantics are already in VTE,
- edit/procgen determinism is naturally voxel-domain,
- server streaming protocol already chunk-oriented.

## Rendering Architecture

### 1) Separate Visible Sets Per LOD

Build independent visible chunk sets for L0/L1/L2 with non-overlapping distance bands:

- `R0`: near exact
- `R1`: mid (coarse)
- `R2`: far (coarser)

Do not render the same space in multiple levels at once; ring partitioning avoids double hits and fighting.

### 2) Per-LOD Trace Passes in One Shader Invocation

In shader, trace rings near-to-far:

1. Trace L0 in `[0, R0]`.
2. If miss, trace L1 in `(R0, R1]`.
3. If miss, trace L2 in `(R1, R2]`.

This keeps near quality exact and far cost bounded.

### 3) LOD-Aware Chunk Header

Extend chunk metadata with (or equivalent derived constants):

- `lod_level`
- `cell_world_size = 1 << lod_level`
- `chunk_world_span = CHUNK_SIZE * cell_world_size`

Then generalize in-chunk traversal from voxel size `1` to cell size `cell_world_size`.

### 4) Material Resolve at Coarse Levels

For coarse cells, use dominant non-air material in covered fine region.

MVP rule:

- occupancy: cell is solid if any child solid (conservative)
- material: most frequent non-air id (fallback: first non-air)

This is stable and deterministic.

## Server + Worldgen Accommodations

### 1) Add Pure Virtual Sampling APIs

Server needs a reusable API to generate both L0 and LODn without forcing L0 realization:

- `sample_virgin_voxel(world_pos)` (deterministic base + structures + keepout)
- `build_virgin_chunk(chunk_pos, lod_level)` (aggregates virgin samples)

Edits are overlaid from sparse override chunks.

Key point: LOD generation must never require inserting all intermediate L0 chunks.

### 2) LOD Chunk Caches on Server (Ephemeral)

Keep server-side runtime caches:

- `lod_cache[level][lod_chunk_pos] -> interned payload id`
- invalidation map from edited L0 chunk -> impacted LOD chunks

On edit:

- update/trim L0 override as today
- invalidate impacted L1/L2 entries
- regenerate lazily on next client demand

### 3) Streaming Protocol Extension

Add LOD-aware chunk messages:

- load batch: `(lod_level, chunks...)`
- unload batch: `(lod_level, chunk positions...)`

Server decides per-client ring radii and sends each level independently.

### 4) Save-File Policy

Keep save format focused on authoritative edits:

- persist only L0 overrides (current approach)
- do not persist LOD caches

On load, LOD data is rebuilt lazily from deterministic seed + overrides.

This preserves small saves even after large exploration.

## Chunk Instancing / Dedupe Alignment

The current renderer already payload-interns identical chunk voxel words. Reuse the same interning idea across all LOD levels:

- hash key should include `lod_level` + packed occupancy/material words
- payload pools can be per-level or unified with `lod_level` in key

This is especially valuable for far levels where many chunks collapse to identical patterns.

## Migration and Compatibility

- Existing worlds continue to load unchanged.
- If no LOD data is present (initial rollout), behavior stays current.
- LOD is runtime-generated from current world seed/keepout/overrides, so no data migration is required for saved worlds.

## Implementation Phases

### Phase 1 (MVP, lowest risk)

- Add L1 only (2x cell size), one extra ring.
- Server computes/streams L1 chunk batches.
- Client keeps separate L0 and L1 frame inputs.
- Shader adds two-pass near->mid traversal.

Success criteria:

- near visuals unchanged
- distant horizon appears beyond current range
- no save size growth from exploration

### Phase 2

- Add L2 and configurable ring radii.
- Add edit-driven invalidation fanout for all levels.
- Add debug counters per level (steps, hit %, misses by budget).

### Phase 3

- Optimize coarse material resolve and temporal stability.
- Optional sky/fog blend policy for far coarse levels.
- Optional per-level trace budgets.

## Immediate Code Touchpoints

- Server generation/streaming:
  - `crates/polychora/src/server/mod.rs`
  - `crates/polychora/src/server/procgen.rs`
  - `crates/polychora/src/shared/protocol.rs`
- Client ingest/world application:
  - `crates/polychora/src/main.rs`
- VTE runtime frame assembly:
  - `crates/polychora/src/scene.rs`
  - `crates/polychora/src/scene/voxel_runtime.rs`
- GPU metadata and traversal:
  - `src/render/vte.rs`
  - `src/render.rs`
  - `slang-shaders/src/voxel.slang`

## Open Decisions

- Ring defaults (`R0`, `R1`, `R2`) and max level count.
- Coarse occupancy policy (`any-solid` vs threshold) for each level.
- Whether to hard-cut rings or blend around boundaries.

## Practical Defaults Proposal

- L0: `0..96`
- L1: `96..384`
- L2: `384..1536`
- Levels: start with `L0 + L1` first, then add L2.

These should be tunable runtime flags, but fixed defaults are good enough for first implementation.
