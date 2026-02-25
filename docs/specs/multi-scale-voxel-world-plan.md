# Multi-Scale Voxel World Plan (Mixed Scale Chunk Populations)

Status: Proposed  
Updated: 2026-02-25  
Owners: Codex + Christian

## 1) Goal

Support power-of-two voxel scales in the same world space, including mixed populations (for example, scale=1 and scale=1/2 content colocated in one area), without breaking deterministic world edits, render correctness, or save/network compatibility.

## 2) Core Decision

Use a single world tree model with scale-aware `ChunkArray` leaves, while preserving the fixed local chunk payload shape (`8x8x8x8` voxels per payload).

This is paired with one hard invariant:

1. World-space ownership is unique. No world-space region may be represented by multiple non-empty leaves.

If that invariant is enforced, runtime "finer wins" precedence rules are not required.

## 3) Current Constraints

1. Chunk payload geometry is fixed at `CHUNK_SIZE=8`, `CHUNK_VOLUME=4096`.
2. Region/render tree query paths assume a single chunk lattice and first-match ownership.
3. GPU traversal and leaf lookup assume one chunk coordinate space and uniform voxel scale.
4. Editing and collision currently sample only unit voxel coordinates.
5. Save/load chunk-array blob format currently has no scale field.

## 4) Representation Rules

### 4.1 Scale encoding

1. Add `scale_exp: i8` to `ChunkArrayData`.
2. Effective voxel cell size is `2^scale_exp` in world units.
3. Initial implementation scope: non-positive exponents only (`0`, `-1`, `-2`, ...) to enable finer-than-unit editing first.

### 4.2 Mixed populations inside finer chunks

Allow coarse semantic blocks to be materialized inside finer chunk payloads by stamping repeated entries.  
For one step finer (`scale=1/2`), one coarse cell maps to `2x2x2x2 = 16` fine voxels.

This is an intentional storage/texturing trick for early mixed-scale authoring and can be validated with warn-only diagnostics when coherence is violated.

### 4.3 World-space ownership invariant

Two leaves may overlap in chunk-coordinate terms only if their world-space occupied volumes are disjoint.  
Canonicalization must resolve conflicts at write/splice time so query and traversal never see ambiguous ownership.

## 5) Project Phases

### Phase A (Milestone 1): Data Model + Invariant Foundation

Build scale-aware storage and validation first, without changing runtime GPU traversal semantics.

### Phase B: Scale-Aware Query/Edit API Surface

Introduce explicit scaled chunk/voxel query and mutation APIs while preserving current unit-voxel wrappers.

### Phase C: Render Runtime and Shader Support

Extend leaf headers, lookup, and traversal math to support scale-aware chunk addressing.

### Phase D: Gameplay Sampling (Edit/Collision) Integration

Make edit raycast and collision resolve scale-aware occupancy.

### Phase E: Optimization and Compression

Tune memory pressure, merge rules, and fine/coarse stamping behavior.

## 6) Milestone 1 (First Work Chunk)

### 6.1 Scope

1. Add `scale_exp` field to `ChunkArrayData` with serde default `0`.
2. Thread `scale_exp` through chunk-array construction, cloning, slicing, remap, and encode/decode helpers.
3. Add world-space overlap validator utilities for region/render trees.
4. Enforce validation in splice/normalize/consolidation paths.
5. Extend save blob schema to persist `scale_exp` with backward-compatible load defaults.
6. Add tests for:
   1. round-trip serialization with and without scale field,
   2. mixed-scale non-overlapping acceptance,
   3. mixed-scale overlapping rejection,
   4. consolidation not crossing scale boundaries.
7. Keep nonzero-scale leaves blocked from VTE upload for now with explicit diagnostics, to avoid silent misrender while Phase C is pending.

### 6.2 Non-goals

1. No shader traversal changes yet.
2. No gameplay collision/edit behavior changes yet.
3. No network protocol redesign beyond serde compatibility of existing subtree payloads.

### 6.3 Primary touchpoints

1. `crates/polychora/src/shared/chunk_payload.rs`
2. `crates/polychora/src/shared/region_tree/tree.rs`
3. `crates/polychora/src/shared/render_tree/mod.rs`
4. `crates/polychora/src/save_v4.rs`
5. `crates/polychora/src/scene/voxel_runtime.rs` (guard rails only in this milestone)

### 6.4 Acceptance criteria

1. Data model supports `scale_exp` end-to-end in memory and save/load.
2. Tree operations cannot produce world-space overlap across mixed scales.
3. Existing scale=0 behavior remains unchanged.
4. Nonzero-scale content is either rejected or explicitly skipped in voxel-frame build with clear logs (never silently interpreted as scale=0).

## 7) Risks and Mitigations

1. Risk: invariant checks add overhead in hot mutation paths.  
   Mitigation: debug/assert-heavy checks plus cheap fast-path checks in release.
2. Risk: consolidation logic accidentally merges across scales.  
   Mitigation: enforce equal-scale precondition for chunk-array merges.
3. Risk: save compatibility regressions.  
   Mitigation: additive blob field with defaulting; add fixture tests for legacy blobs.

## 8) Open Decisions

1. Exact canonical world-space interval convention (recommended: half-open `[min, max)` in fixed-point units).
2. Whether to cap minimum allowed `scale_exp` in Milestone 1.
3. Whether stamp-coherence validation is warn-only or hard-fail in debug.

## 9) Immediate Next Step

Implement Milestone 1 in this order:

1. `ChunkArrayData.scale_exp` + serde/save plumbing.
2. World-space bounds conversion utilities.
3. Region/render tree overlap validation + merge guardrails.
4. VTE upload guard for nonzero-scale leaves.
5. Unit/integration tests.
