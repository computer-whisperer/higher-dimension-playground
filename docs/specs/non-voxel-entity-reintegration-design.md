# Non-Voxel Entity Reintegration Design (VTE + Dynamic Accel)

Status: Proposed (GPU-build direction accepted)
Updated: 2026-02-21
Owners: Codex + Christian

## 1) Goal

Reintroduce non-voxel entities into the rewritten VTE pipeline with:

1. depth-correct composition against voxel world hits,
2. stable behavior under rapid movement,
3. no architectural regression in the world-BVH mutation refactor.

## 2) Problem Summary

Current code already contains entity tetra traversal and a dedicated BVH path, but integration is currently disconnected:

1. VTE app path does not pass non-voxel instances to `render_voxel_frame`.
2. VTE shader entity evaluation is hard-disabled.
3. World-BVH work is active and must not absorb high-frequency entity motion churn.

This means non-voxel entities currently do not participate in VTE first-hit shading.

## 3) Design Decision

## 3.1 Chosen direction

Use a dedicated non-voxel acceleration path sampled alongside voxel traversal.

1. Keep world voxel BVH ownership/mutation unchanged.
2. Maintain separate entity accel resources and update cadence.
3. Resolve nearest-hit competition in VTE Stage A (entity hit vs voxel hit).

## 3.2 Why this is the right choice now

1. Rapid movers are a high-frequency stream and should not trigger world-BVH edits.
2. It aligns with prior working architecture and current code scaffolding.
3. It preserves a clean migration path toward unified mixed leaf types later.

## 3.3 Explicitly deferred

Immediate merge of entities into the world render BVH is deferred.

Reason: it couples unrelated mutation frequencies and increases risk while the world-BVH mutation path is still being stabilized.

## 3.4 Build-side decision (accepted)

Entity BVH construction is GPU-resident for the current architecture phase.

1. Use existing GPU LBVH passes for entity accel construction/update.
2. CPU remains responsible for instance simulation, culling inputs, and scheduling/budget policy.
3. Avoid full CPU-side per-frame entity tetra BVH builds plus node uploads.

Rationale:

1. Entity transforms can change every frame; GPU construction avoids CPU-side BVH rebuild/upload churn.
2. The renderer already has a mature GPU BVH construction path and diagnostics for entity BVH correctness.

## 4) Target Runtime Model

Two independent scene domains are sampled per VTE ray:

1. Voxel domain:
   - world chunk/leaf/BVH metadata,
   - mutation-batch-driven GPU updates.
2. Entity domain:
   - dynamic non-voxel instance list,
   - entity acceleration structure.

VTE Stage A computes:

1. nearest voxel hit (if any),
2. nearest entity hit up to voxel-hit distance,
3. final hit = nearest of the two.

## 5) Entity Acceleration Architecture

## 5.1 Near-term reintegration (Phase 0/1)

Re-enable the existing single-structure entity tetra BVH path:

1. preprocess active entity instances into tetra stream,
2. rebuild entity BVH on GPU when entity scene hash changes,
3. trace entity BVH from Stage A.

This restores correctness quickly with low architectural risk.

## 5.2 Rapid-movement optimization target (Phase 2)

Move from monolithic per-change rebuild toward two-level accel:

1. static model accel (BLAS-like):
   - model-space tetra data and optional model-local BVH,
   - built once per model asset.
2. dynamic instance accel (TLAS-like):
   - per-instance world AABB + transform + model reference,
   - GPU refit-first each update window,
   - GPU rebuild only when quality/degeneracy thresholds are exceeded.

This isolates high-frequency movement updates from full tetra rebuild cost.

## 5.3 Refit vs rebuild policy

1. Refit path:
   - default for position/orientation/scale changes with stable instance set.
2. Rebuild path:
   - instance add/remove bursts,
   - significant tree-quality degradation,
   - explicit safety fallback.

Use per-frame budgets and carry-over queues when over budget.

## 6) Data Contracts

## 6.1 App -> renderer (VTE mode)

Renderer VTE entry accepts:

1. voxel frame input (existing),
2. entity instance slice for Stage A entity traversal,
3. optional overlay-only instance slice (legacy preview path compatibility).

## 6.2 Renderer -> shader

Shader-visible entity data includes:

1. entity tetra payload and/or instance refs,
2. entity accel nodes,
3. counts/roots needed for traversal,
4. debug flags for compare modes.

## 6.3 Correctness invariant

If entity count is non-zero, Stage A must evaluate entity intersections (not compile-time or runtime-disabled).

## 7) Frame and Update Scheduling

## 7.1 Frame path invariant

Camera movement does not trigger world scene rebuilds.

Entity update work is bounded and independent:

1. consume queued entity transform edits,
2. run refit/rebuild under budgets,
3. reuse previous accel if budget exhausted.

## 7.2 Budget knobs

Planned controls:

1. max entity update CPU ms/frame,
2. max entity update GPU dispatch/upload bytes/frame,
3. max entity accel ops/frame.

Overflow is deferred, not forced into blocking full rebuilds.

## 8) Diagnostics and Validation

## 8.1 Required runtime diagnostics

1. entity instance count,
2. entity tetra count,
3. entity accel mode this frame (`none|refit|rebuild|deferred`),
4. entity accel GPU ms,
5. entity compare mismatch counters (linear vs BVH),
6. reason for rebuild fallback.

## 8.2 Correctness gates

1. Entity visible in VTE when present.
2. Entity/voxel occlusion ordering is nearest-hit correct.
3. Compare diagnostics remain stable (no systematic BVH mismatch growth).

## 8.3 Performance gates

1. Rapid mover scenarios do not induce world-BVH rebuild spikes.
2. Entity-heavy movement maintains bounded frame-time variance under configured budgets.

## 9) Implementation Phasing

## Phase 0: Reintegration parity

1. Restore VTE API wiring for entity instance input.
2. Re-enable shader-side entity evaluation gate.
3. Keep existing entity BVH compare diagnostics.

## Phase 1: Hardening and telemetry

1. Add explicit per-frame entity accel telemetry.
2. Confirm no regression to voxel mutation path or world-BVH ownership.

## Phase 2: Rapid-movement optimization

1. Introduce refit-first dynamic entity accel updates.
2. Add rebuild threshold and budgeted fallback behavior.

## Phase 3: Unified render-scene preparation

1. Define bridge contract to future `LeafTetSegmentRef` integration.
2. Keep side-channel removable without behavior changes.

## 10) Risks and Mitigations

1. Risk: entity accel and world accel interfere in buffers/layout.
   - Mitigation: keep allocator/buffer domains separate.
2. Risk: frequent movement still triggers full GPU entity rebuild.
   - Mitigation: refit-first policy + budgets + deferred queue.
3. Risk: GPU BVH build cost spikes under bursty topology change.
   - Mitigation: quality thresholds + rebuild budget + optional multi-frame staging.
4. Risk: stale or partial entity updates causing temporal artifacts.
   - Mitigation: generation tracking and explicit fallback reason logging.

## 11) Out of Scope (This Design Cycle)

1. Network protocol redesign for entity render payloads.
2. Immediate replacement of the world-BVH mutation architecture.
3. Asset-system-wide multi-model authoring overhaul.

## 12) Open Decisions

1. Target initial entity scale for performance gate (for example 16, 64, 256 active movers).
2. Refit quality threshold policy (fixed interval rebuild vs SAH degradation metric).
3. Whether overlay-preview compatibility remains default-off after reintegration.
4. Whether Phase 2 introduces dedicated GPU TLAS buffers now, or first reuses current flattened-tetra LBVH path with stricter rebuild budgets.
