# Architecture Review — July 2026

Snapshot of the codebase as of branch `aetna-ui-integration` (commit `bc92bf7` + uncommitted
hotbar scale-badge work). Written to re-establish visibility into decisions made during the
previous development stretch, with focus on the entity replication path and its performance
characteristics.

## Repo reality vs. old documentation

The crate layout drifted from what CLAUDE.md described:

- There is **no `polychora-server` or `polychora-common` crate**. The server lives at
  `crates/polychora/src/server/`, the protocol at `crates/polychora/src/shared/protocol.rs`,
  both inside the game crate. The server binary is `crates/polychora/src/bin/polychora-server.rs`.
- Two crates exist that the docs didn't mention:
  - `crates/polychora-plugin-api` — the WASM plugin ABI (entity declarations, sim config,
    tick/model/ability opcodes).
  - `crates/polychora-content` — excluded from the workspace, compiled separately to WASM and
    embedded in the client/server binary via `include_bytes!(env!("POLYCHORA_CONTENT_WASM_PATH"))`
    (`plugin_loader.rs`).
- **egui is completely gone** (zero references, including Cargo.lock). The UI is Damascene
  (formerly Aetna), an external retained-mode GUI framework. All UI surfaces (main menu, HUD,
  inventory, pause/settings, dev console, block GUI) live in
  `crates/polychora/src/app_damascene_ui.rs` (~3600 lines) as `El` trees, composited by the
  render context via `damascene_vulkano::Runner` each frame.

## High-level architecture

**One process model, two transports.** The server is embeddable: singleplayer runs an
integrated server in-process over mpsc channels (`server/mod.rs` `connect_local_client`);
multiplayer uses TCP with postcard serialization and 4-byte LE length framing. Thread-per-client
on both ends: a blocking reader thread and a writer thread fed by an **unbounded**
`std::sync::mpsc` channel (`server/runtime_net.rs` `spawn_client_thread`). All server state —
world, entities, mobs, players, per-client interest tracking — is behind a single
`Arc<Mutex<ServerState>>` (`server/core_state.rs`).

**Tick structure.** One broadcast thread at 10 Hz (`tick_hz`, `server/config.rs`); entity
simulation targets 30 Hz via up to 3 catch-up substeps per broadcast tick
(`ENTITY_SIM_STEP_MAX_PER_BROADCAST`, `server/mod.rs`). Each tick holds the global lock across
the entire sim + replication-prep phase — WASM entity ticks, mob A* pathfinding, collision,
block ticks, item pickup, and building every client's replication batch
(`runtime_net.rs` broadcast loop). Sends happen after the lock drops.

**Entity system — fully data-driven.** Entities are `(namespace, entity_type)` pairs with a 4D
`EntityPose` (position/orientation/velocity `[f32;4]` + scale) and an opaque `data: Vec<u8>`
blob. Declarations (category `Player|Accent|Mob`, sim mode `Parametric|PhysicsDriven`,
locomotion, textures, ability params) come from the WASM content plugin manifest. Behavior is
three WASM entry points:

- `OP_ENTITY_TICK` — Parametric entities return `SetPose`; Mobs return `Steer`, integrated by
  native physics/nav (`server/mob_sim/`).
- `OP_ENTITY_ABILITY` — creeper detonation, phase-spider blink.
- `OP_ENTITY_MODEL` — procedural animated tesseract-part models. **No voxel models, no mesh
  assets, nothing on the wire** — model geometry is computed client-side from
  (type, id, elapsed time, speed, scale).

Content entities: cube/rotor/drifter (Accents), seeker/creeper/phase_spider (Mobs); the engine
natively registers player avatars and item stacks (`builtin_content.rs`). Entities persist in
the save-v4 format as postcard blobs, region-keyed (`save_v4/entity_io.rs`).

**Client structure.** Net reader/writer threads feed an mpsc queue drained at the top of each
frame, capped at 64 messages/frame (`app_multiplayer.rs` `poll_multiplayer_events`). Remote
entities live in two HashMaps on `App` (`remote_players`, `remote_entities`) holding pose +
smoothing state only. Interpolation is dead-reckoning with exponential smoothing and
teleport-snap.

**Rendering (VTE).** Voxel world and entities are fully separate paths:

- *World*: region BVH over dense 8⁴ chunks in flat GPU arrays, generation-gated so steady-state
  frames upload nothing; edits apply bounded mutation batches; residency changes trigger a
  background-thread full rebuild with a one-time buffer swap (`scene/frame_builder.rs`,
  `render.rs` voxel upload path).
- *Entities*: rebuilt every frame as tesseract `ModelInstance`s, uploaded as a flat instance
  buffer, with a GPU-side BVH that is hash-gated — refit-only when topology is unchanged, full
  rebuild on spawn/despawn or every 120 frames (`render.rs` non-voxel BVH pass). Spawning an
  entity never touches chunks, meshing, or world buffers.

## The entity spawn round trip

Spawn triggers: spawn-egg item (client sends `ClientMessage::SpawnEntity`), console `/spawn`,
block-tick side effects (spawner block), item drops. Server-side
(`runtime_net.rs` `spawn_entity_from_request`): validate against registry, insert into
`entity_store` — **no message is sent at spawn time**. Entities reach clients only via the next
tick's interest diff: `build_entity_replication_batches` (`server/mod.rs`) computes each
client's visible set by chunk distance, diffs against last tick, and emits `EntitySpawned`
(full snapshot) for newly visible, `EntityDestroyed` for departed, and one batched
`EntityTransforms` for **all** visible entities — every tick, moved or not. Client-side,
`EntitySpawned` is an O(1) HashMap insert; the entity renders the same frame.

The wire design is a reasonable delta-replication scheme. The problems are mechanical:

### Server, per tick, under the global lock

1. `collect_live_replication_frame` iterates **all** live entity records and **clones every
   entity in full** (including the `data` blob); each client's visibility pass then clones
   every visible snapshot **again**. O(entities) + O(clients × visible) deep clones per tick,
   whether anything changed or not.
2. Despawned entities leave tombstone `EntityRecord`s that are never GC'd, so the all-records
   scan grows monotonically over a session.
3. Mob pathfinding, WASM ticks, and collision run under the same lock that client reader
   threads contend on for `UpdatePlayer`/`WorldInterestUpdate`/world queries. More entities →
   longer lock hold → input processing stalls behind the tick.

### Wire/transport

4. `EntityTransforms` includes every visible entity every tick with no moved-since-last-tick
   gating.
5. Writer threads serialize per-client (no serialize-once broadcast buffer), flush after every
   message on a `BufWriter` with `TCP_NODELAY` (many tiny writes), and the feeding channel is
   unbounded — a stalled client accumulates memory without bound.

### Client, per frame — likely the observed symptom

6. `remote_entity_instances` (`app_multiplayer.rs`) makes a **WASM VM call per entity per
   frame** — postcard-encode `EntityModelInput`, invoke `OP_ENTITY_MODEL`, postcard-decode —
   plus up to 10 texture-ref resolutions per entity, with **no caching**
   (`shared/wasm/manager.rs` `call_slot` goes straight to the VM). Every spawned entity
   permanently adds per-frame cost: spawning doesn't hitch, it *ratchets* — frame time scales
   with entity count. Models are animated via `elapsed_s`, so naive memoization won't work,
   but the call doesn't need to run at render rate, and per-type results could be shared
   across same-type entities with per-id phase offsets applied engine-side.

There is also a fixed ≤100 ms spawn latency (10 Hz tick, no immediate broadcast), plus the
client's 64-messages-per-frame drain cap adding frames of delay under load — sluggish spawn
response, distinct from throughput.

## Overall assessment

Structure is better than feared: clean entity/world separation in the renderer, sane delta
replication semantics, a well-factored plugin ABI (the mob-centric → entity-first refactor
landed well). The poor decisions are concentrated in mechanical choices: clone-everything-
per-tick under one global mutex, no change-gating on transforms, no tombstone GC, and —
most visibly — evaluating entity models through a WASM boundary once per entity per frame
with zero reuse. None of these require re-architecting the protocol; they are targeted fixes
in `server/mod.rs`'s replication pass and the client's instance builder.

## Status update (2026-07-01)

Fixes landed after this review was written:

- **Client WASM path (finding 6)**: model evaluations cached and rate-limited to 30 Hz with a
  64-evals-per-frame cap; material tables cached per entity lifetime; the animation clock bug
  (`elapsed_s` resetting every network tick) fixed via a per-entity `spawned_at`.
- **Server clones (finding 1)**: the replication frame now carries pose-sized entries; full
  `EntitySnapshot`s are cloned only for entities newly entering a client's visible set.
- **Tombstones (finding 2)**: removed entirely — despawn deletes the `EntityRecord`
  (`remove_entity_record`); nothing consumed tombstones.
- **Transform gating (finding 4)**: `EntityTransforms` include only entities whose pose changed
  bitwise since the last broadcast, with a keepalive resend every 10th tick.

Still open: transport-layer items (finding 5: per-client re-serialization, flush-per-message,
unbounded send queues), the global-mutex lock scope (finding 3), and a gap discovered during
this work: **the runtime tick-loop save persists chunks and players only** —
`resolve_entities_for_save` is reached only from migration/worldgen full saves, so live
sessions do not persist non-player entities (persistent item drops/mobs are lost on restart).

## Dependency note: Aetna → Damascene

The UI framework was originally consumed as *path* dependencies on a sibling "Aetna" repo,
which was renamed upstream to **Damascene** — leaving the paths dangling and the tree
unbuildable. Resolved 2026-07-01: the dependency now comes from crates.io
(`damascene-core` / `damascene-vulkano` 0.4.5), all identifiers and the UI module were
renamed (`app_aetna_ui.rs` → `app_damascene_ui.rs`), and the 0.3→0.4 API breaks were
migrated (`Pointer` struct for pointer events, `Color::srgb_u8`/`with_alpha_u8`,
`text_input_with`/`apply_event` argument order, `slider(key, value)`).
