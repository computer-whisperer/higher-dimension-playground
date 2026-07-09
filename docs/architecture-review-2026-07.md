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

- **Entity persistence gap** (discovered during this work): the runtime tick-loop save
  persisted chunks and players only, and startup never loaded entities — spawned mobs/accents
  were lost on restart. Fixed: startup loads and respawns all persisted entities (unknown
  types are retained un-respawned so a plugin-less session can't erase them);
  `persistent_entities_dirty` (spawn/despawn/pose change) triggers an entity-subtree rewrite
  in the interval save; a best-effort final save runs when the broadcast loop exits.
  Known limits: quit-to-desktop can exit the process before the detached final save runs
  (loss bounded by `save_interval`), and the dedicated server has no signal handler; reliable
  quit-time saving needs the client to join server shutdown. Item stacks remain intentionally
  transient. Parametric accents whose pose animates server-side (rotor/drifter) keep the
  entity subtree perpetually dirty — one entity-blob rewrite per save interval; the semantic
  fix would be persisting `home_position` instead of the animated pose.

- **Transport (finding 5)**: TCP clients now receive pre-encoded frames (`ClientSink::Tcp`)
  so broadcasts and world-patch envelope groups encode once and share one `Arc<[u8]>` across
  recipients; local (singleplayer) clients keep their zero-serialization channel
  (`ClientSink::Local`). The writer thread flushes once per burst instead of per message.
  Send queues are byte-capped at 256 MiB per client (exact reserve-then-refund accounting);
  a client exceeding it is disconnected and can rejoin fresh. Note: a *single* frame larger
  than the cap would disconnect every interested client on each sync attempt — frames that
  size were already pathological (previously unbounded memory instead).

Still open: the global-mutex lock scope (finding 3) — before redesigning, gather evidence
from the server's `profile server-cpu` output (`msg_avg`/`tick_max`) in a real session with
mobs active. Separate pre-existing bug found during transport smoke-testing: the content
plugin's `MazeGenerator::generate` traps in WASM (`memcmp` backtrace) during procgen on a
flat-worldgen dedicated server; procgen structures near spawn silently fail to place.

- **Maze WASM trap (2026-07-09)**: root cause was fuel exhaustion ("all fuel consumed"),
  reproduced via a WASM-level test — 23/24 maze seeds trapped; the sole survivor used 436M
  of the 500M fuel budget and 1.72 MB of the 2 MB output cap. Fixed on both sides: the
  content plugin's chunk dedup went from an O(chunks²) 8 KB-memcmp linear scan to
  hash-bucketed dedup (shared by maze + blueprint structures), maze rasterization became
  chunk-major (no per-voxel div/rem), and `PROCGEN_EXECUTION_LIMITS` rose to 4B fuel /
  64 MB output. Largest maze now: ~1.8B fuel, 9.4 MB output, ~90 ms. A generate-output LRU
  (8 entries, keyed by structure/seed/orientation/origin) was added to `ProcgenWasmCaller`
  since world queries re-generate overlapping structures each time; WASM guest errors now
  keep their anyhow cause chain. Regression tests: `maze_prepare_and_generate_many_seeds`
  (procgen_wasm.rs), `procgen_structures_place_near_spawn` (flat_world_generator.rs).

## Regression found in play-testing (2026-07-05)

Blank hotbar/inventory thumbnails after the Damascene swap: Damascene 0.4's `Runner::draw()`
host contract requires `record_uploads(&mut builder)` between `prepare()` and
`begin_render_pass` (Aetna 0.3 uploaded image textures eagerly inside prepare). Without it,
image widgets sample uninitialized GPU textures and render blank. Fixed in `src/render.rs`;
verified via automated screenshot. A `console:<server command>` auto-command was added to
`--commands` for scripted repros (spawn entities etc. without manual input).

## Dependency note: Aetna → Damascene

The UI framework was originally consumed as *path* dependencies on a sibling "Aetna" repo,
which was renamed upstream to **Damascene** — leaving the paths dangling and the tree
unbuildable. Resolved 2026-07-01: the dependency now comes from crates.io
(`damascene-core` / `damascene-vulkano` 0.4.5), all identifiers and the UI module were
renamed (`app_aetna_ui.rs` → `app_damascene_ui.rs`), and the 0.3→0.4 API breaks were
migrated (`Pointer` struct for pointer events, `Color::srgb_u8`/`with_alpha_u8`,
`text_input_with`/`apply_event` argument order, `slider(key, value)`).
