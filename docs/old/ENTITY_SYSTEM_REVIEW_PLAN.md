# Entity System Review And Improvement Plan

Date: 2026-02-14
Branch: `agent/entity-system-review`

## Current Setup

1. Protocol
- Entities are represented by `EntitySnapshot` and streamed through `ServerMessage::EntityPositions`.
- `EntityKind` currently has a single variant: `TestCube`.
- Source: `crates/polychora/src/shared/protocol.rs`.

2. Server state model
- Server stores entities in a single `HashMap<u64, EntityState>` inside `ServerState`.
- There is no separate entity simulation module/system layer.
- Source: `crates/polychora/src/server/mod.rs`.

3. Spawn lifecycle
- Entities are spawned at server startup via `spawn_default_test_entities`.
- Three hardcoded test cubes are created every boot.
- Source: `crates/polychora/src/server/mod.rs`.

4. Replication model
- Broadcast thread sends full `Vec<EntitySnapshot>` to all clients each server tick.
- On client join, server also emits per-entity `EntitySpawned` snapshots.
- Source: `crates/polychora/src/server/mod.rs`.

5. Client state and render path
- Client stores remote entities in `HashMap<u64, RemoteEntityState>`.
- Client smooths only `position`; orientation is received but not used for rendering.
- Rendered entity rotation is client-local animation from `(entity_id, time_s)`, not authoritative from server state.
- Source: `crates/polychora/src/main.rs`.

## Main Gaps

1. Hardcoded test entities
- No gameplay-facing API for spawn/despawn/update.
- `EntityDestroyed` exists in protocol but has no active server producer path.

2. No simulation authority beyond static transforms
- No entity simulation loop (AI, movement, state transitions, physics integration).
- `last_update_ms` is sent but not used for client-side time-aware interpolation/extrapolation.

3. Scalability limits
- Full entity snapshots are broadcast to all clients each tick.
- No entity interest management by streamed chunk window or distance.
- Client/handler cleanup uses `Vec::contains` retain patterns (O(n^2) in entity count).

4. Data model mismatch
- Protocol includes orientation, but rendering ignores it.
- Entity behavior and composition are implicit in code paths instead of explicit components/archetypes.

5. Persistence and tooling
- Entities are not persisted with world saves; world restart resets to default test entities.
- No dedicated tests for entity replication, interpolation, or spawn/despawn correctness.

## Recommended Roadmap

### Phase 0: Foundation Cleanup (small, high leverage)

1. Extract entity logic into a dedicated module:
- `server/entity/mod.rs` for storage + update entry points.
- `client/entity/mod.rs` for interpolation/presentation state.

2. Fix immediate correctness and scaling issues:
- Use orientation in render transform instead of local spin.
- Replace O(n^2) `seen.contains` cleanup with `HashSet<u64>`.

3. Add minimal observability:
- Counters for entities simulated, replicated, bytes per tick, and dropped/stale snapshots.

### Phase 1: Authoritative Simulation + Better Replication

1. Add a server entity simulation step:
- Fixed simulation tick (`sim_hz`) separate from broadcast tick (`net_hz`).
- Keep current `tick_hz` for net transport until split config is introduced.

2. Move to delta-oriented messages:
- Keep initial full snapshot for join.
- Send per-tick changed transforms only (position/orientation/scale/material).
- Retain full snapshot fallback for resync/debug mode.

3. Add entity interest management:
- Reuse player chunk stream window.
- Send entity updates only for entities inside client interest bounds.

### Phase 2: Lifecycle + Persistence

1. Explicit runtime lifecycle API:
- Authoritative spawn, despawn, and mutate paths with reason/cause fields.
- Ensure `EntitySpawned` and `EntityDestroyed` are emitted from actual runtime transitions.

2. Persist entities:
- Add an entity section or sidecar payload in world save/load path.
- Preserve IDs or remap deterministically on load.

### Phase 3: Gameplay Hooks

1. Interaction messages:
- Client intent messages for entity interactions (use, damage, trigger).
- Server validates and emits resulting authoritative changes.

2. Entity-voxel integration:
- Query voxel collisions, stepping, grounding, and material effects.
- Keep this deterministic on server; clients only predict visuals.

3. Audio/VFX hooks:
- Drive remote entity sounds/events from authoritative state changes, not local heuristics.

### Phase 4: 4D-Specific Features

1. 4D orientation and motion model:
- Represent orientation with stable 4D basis/rotor form.
- Support compound plane rotations as first-class state.

2. 4D-native behaviors:
- W-phase entities (visible/interactive across W slices).
- Hyperplane patrol/ambush logic and 4D occlusion-aware sensing.

## Suggested First Implementation Slice

1. Keep `TestCube`, but make it fully authoritative:
- Server updates orientation/position each sim tick.
- Client consumes and renders server orientation directly.

2. Add a lightweight delta message:
- `EntityTransforms { server_time_ms, updates: Vec<...> }`.
- Keep `EntityPositions` temporarily for compatibility during migration.

3. Add interest filtering:
- Reuse existing streamed chunk center/radius data for per-client filtering.

This slice gives immediate wins in correctness, bandwidth, and architecture without requiring full ECS adoption in one step.

