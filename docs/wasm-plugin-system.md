# WASM Plugin System Design

## Status: Brainstorming

## Goal

Migrate blocks, items, entities, and mobs to be defined and operated by WebAssembly plugins. The first-party Polychora content becomes a WASM module itself, eating our own dog food from day one.

---

## 1. ID Philosophy: Random u32 Everywhere

### The problem with sequential IDs

Sequential IDs (0, 1, 2, 3...) silently grow unintended semantics:
- You can't deprecate `id=1` stone and replace it with `id=67` smooth stone without migration headaches
- Cross-domain confusion: if `block_type=1` and `entity_type=1` are both just "1", bugs where you accidentally use a block ID as an entity ID produce wrong-but-plausible results (you spawn a creeper when you meant to reference stone)
- Sequential IDs invite hardcoding and implicit ordering assumptions

### The approach: one-time random u32 for every ID

Every block type, entity type, item type, and namespace gets a **randomly chosen u32** at creation time. This is a permanent, stable identifier.

- `polychora-content` namespace: one random u32, chosen once (e.g. `0xa37f_1c4b`)
- Each block gets its own random u32 (stone might be `0x7e2d_3f91`, dirt might be `0xb4c8_02ae`)
- Each entity type gets its own random u32
- Each item type gets its own random u32
- Namespace 0 remains the special "internal engine" namespace — not random, just reserved

**Benefits:**
- Accidental cross-domain use produces obviously wrong results (lookup miss) rather than subtly wrong results
- Deprecation is trivial: stop recognizing an old ID, add a new one
- No ordering assumptions — IDs have no inherent meaning beyond identity
- Collision risk is negligible in 32-bit space for realistic content counts

**Convention:** IDs are assigned once when a content entry is first created and never change. They can live as constants in the plugin source.

### What about namespace 0?

Namespace 0 = **internal engine constructs**. These are the minimal set of things with hard-coded engine semantics that cannot be defined by any plugin:

- **Air** block (0, 0) — empty space, used everywhere in chunk storage
- **Player avatar** entity (0, 0) — hardwired into camera, input, collision
- **Block stack** item (0, 0) — fundamental inventory wrapper for any block

The type IDs within namespace 0 can stay simple (0) since there are so few of them and they're truly special. Everything else — all 68 current blocks, all mobs, all future content — migrates to the `polychora-content` plugin with random IDs.

### Migration from current IDs

The v4 format is new enough that nothing important depends on the current ID assignments. This migration is a clean break:

1. All 68 blocks get new random u32 type IDs under the `polychora-content` namespace
2. All entity types get new random u32 type IDs
3. The v4 world spec updates to use the new IDs
4. A one-off migration tool handles the existing v3 world (separate task)

No backward compatibility shim needed for the v4 format — we update the spec.

---

## 2. Namespace Architecture

### Proposed model

**Namespace 0 — "internal engine"**
- Reserved, not a plugin
- Contains only constructs the engine must understand intrinsically
- Minimal: air, player avatar, block_stack item
- No visible/rendered content — purely structural

**Namespace `0x????_????` — `polychora-content`**
- The first-party WASM module
- Gets a one-time-chosen random u32 as its namespace ID
- Contains all gameplay content: stone, dirt, mobs, tools, etc.
- Eats our own dog food — uses the exact same plugin interface as third-party

**Other random u32 namespaces — third-party plugins**
- Each plugin declares a stable random namespace ID in its manifest
- Server validates no collisions at load time
- World files store the namespace ID directly (no remapping needed since IDs are globally unique)

### Namespace ID assignment: stable random, not server-assigned

Since every plugin picks a random u32 at creation time, there's no need for server-assigned IDs. The namespace ID is part of the plugin's identity, baked into its source code. Benefits:

- World files are self-describing: `(namespace, type_id)` pairs are globally meaningful
- No namespace table needed in world file headers
- No server→client namespace mapping negotiation
- Collision probability for N plugins: ~N²/2³² (negligible for realistic N)

Server still validates at load time that no two loaded plugins share a namespace ID.

---

## 3. Plugin ABI Architecture

### Current ABI (`shared/wasm/abi.rs`)

```
polychora_abi_version() -> i32          // must return 1
polychora_alloc(len) -> ptr             // guest allocator
polychora_free(ptr, len)                // guest deallocator
polychora_call(opcode, in_ptr, in_len, out_ptr, out_cap) -> i32   // dispatch
```

The current model: single dispatch entry point, opcode selects the operation, input/output are opaque byte buffers. The host manages allocation, fuel limits, and memory bounds. **No host-imported functions** — the linker is empty, so the guest cannot call back into the host.

Opcodes today: `EntitySimulation = 1`, `ModelLogic = 2`.

### The key insight: where to draw the host/guest boundary

Looking at `mob_sim.rs` reveals the natural split. The mob simulation loop does:

1. **Navigation** — A* pathfinding over 4D grid cells, LOS checks, walkability queries (~400 lines)
2. **Collision** — voxel-level collision detection, binary search resolution, step-up climbing (~250 lines)
3. **Steering** — given a waypoint + mob params + time, compute direction + speed (~300 lines per archetype)
4. **Integration** — apply velocity, resolve final position (~50 lines)
5. **Special abilities** — phase spider blink, creeper detonation triggers (~100 lines)

Steps 1, 2, and 4 need deep `ServerState` access (world voxels, chunk caches, entity store). Step 3 is **pure computation** — it receives pre-resolved data and returns a steering command:

```rust
// What entity tick functions receive today (via EntityTickInput):
//   entity_ns, entity_type, entity_id, position, home_position, scale,
//   target_position, path_following, simple_steer, now_ms, phase_offset,
//   preferred_distance, tangent_weight, locomotion
// And return EntityTickOutput:
//   Steer { desired_direction, speed_factor }    — for PhysicsDriven mobs
//   SetPose { position, orientation, scale }     — for Parametric accents
```

This separation already exists in the code. **The plugin boundary goes between steps 2 and 3** — the host does the heavy lifting (pathfinding, collision, physics), the plugin makes behavioral decisions.

### Three candidate architectures

#### Architecture A: "Typed opcode dispatch" (extend current model)

Keep the existing `polychora_call(opcode, in, out)` primitive. Define typed input/output schemas per opcode in `polychora-plugin-api`. Each hook is an opcode with a known serialization format.

```
Host: serialize MobSteeringInput → bytes
Host: call polychora_call(MOB_STEERING, in_ptr, in_len, out_ptr, out_cap)
Guest: deserialize input, compute, serialize MobSteeringOutput → out_ptr
Host: deserialize MobSteeringOutput from out_ptr
```

**Pros:**
- Minimal ABI surface — one dispatch function, everything else is data
- No host imports means perfect sandboxing — guest can't do anything the host doesn't explicitly feed it
- Easy to add new hooks (just new opcodes + schemas)
- Type safety comes from the shared `polychora-plugin-api` crate, not the ABI boundary itself
- Already built and tested

**Cons:**
- Every call pays serialization cost (postcard is fast, but it's not zero)
- Guest cannot query the host during execution (no "what block is at position X?")
- Complex hooks need the host to pre-gather all possibly-needed data into the input

#### Architecture B: "Host callbacks" (add imported functions)

Add host-imported functions to the wasmtime linker so the guest can call back into the host:

```wasm
(import "polychora" "query_block" (func (param i32 i32 i32 i32) (result i64)))
(import "polychora" "query_entities_near" (func (param f32 f32 f32 f32 f32 i32 i32) (result i32)))
```

**Pros:**
- Plugins can query world state on demand — much more powerful
- No need to pre-gather data; host provides it lazily
- Enables complex plugin logic (e.g., "scan for lava below me", "find nearest entity of type X")

**Cons:**
- Much more complex ABI — each callback is a new function signature to maintain forever
- Harder to sandbox — callbacks give the guest a read channel into host state
- Harder to reason about fuel — host callbacks execute outside the fuel budget
- More complex wasmtime setup (linker needs host function registrations)
- Version compatibility: adding/changing callbacks is an ABI break

#### Architecture C: "Separate exports per hook"

Each hook is a separate WASM exported function instead of routing through a single dispatcher:

```wasm
(export "polychora_mob_steering" (func ...))
(export "polychora_block_interact" (func ...))
```

**Pros:**
- Host can check which hooks a plugin implements by checking exports
- Slightly less dispatch overhead (no opcode switch)

**Cons:**
- ABI surface grows with every hook type
- Parameters still need to go through memory (WASM functions only pass i32/i64/f32/f64)
- The "check which hooks exist" benefit can be achieved with a manifest flag instead
- More rigid — hard to add optional hooks without the module failing to link

### Recommendation: Architecture A, with Architecture B as a future extension

**Start with Architecture A (typed opcode dispatch).** It's already built, it's simple, and it covers the immediate needs perfectly. The mob steering boundary is clean — the host resolves navigation and collision, passes a small typed struct to the guest, gets back direction + speed.

The limitation of Architecture A (guest can't query the host) is not a problem for the hooks we need first:
- **Mob steering** — host provides waypoint, mob params, time. Pure computation.
- **Block interactions** — host provides block data, player context. Guest returns an action.
- **Content declarations** — manifest data, no host queries needed.

**Architecture B (host callbacks) can be added later** when we need it, and it's additive — adding host imports doesn't break existing modules that don't use them. The trigger for adding it would be plugin logic that genuinely needs to query arbitrary world state at runtime (e.g., a mob that scans for specific block types, or a block that checks its neighbors).

### Content declaration: via `GetManifest` opcode

Plugins declare their content via the `GetManifest` opcode (0x10). The host calls it once at load time with an empty input. The guest returns a postcard-serialized `PluginManifest`.

This is better than a WASM data section export because:
- The manifest can be computed (e.g., generate block variants programmatically)
- Uses the same call machinery as everything else
- No need to parse WASM data sections separately

### Hook taxonomy

Hooks are grouped by domain. Each hook has a typed input struct and a typed output struct, defined in `polychora-plugin-api`. The opcode is a `u32` (expanded from the current `i32` — gives room and avoids signed confusion).

#### Lifecycle hooks

| Opcode | Name | Input | Output | When |
|--------|------|-------|--------|------|
| `0x0010` | `GetManifest` | `()` | `PluginManifest` | Once at load time |
| `0x0011` | `GetTextures` | `()` | `TextureBatch` | Once at load time, after GetManifest |

#### Entity hooks

| Opcode | Name | Input | Output | When |
|--------|------|-------|--------|------|
| `0x0100` | `EntityTick` | `EntityTickInput` | `EntityTickOutput` | Per entity with sim_config, per sim tick |
| `0x0101` | `EntityAbility` | `EntityAbilityCheck` | `EntityAbilityResult` | Per entity with abilities, per sim tick |
| `0x0102` | `MobSpawn` | `MobSpawnInput` | `MobSpawnOutput` | When a mob is spawned (future) |

#### Block hooks

| Opcode | Name | Input | Output | When |
|--------|------|-------|--------|------|
| `0x0200` | `BlockInteract` | `BlockInteractInput` | `BlockInteractOutput` | Player right-clicks a block |
| `0x0201` | `BlockBreak` | `BlockBreakInput` | `BlockBreakOutput` | Block is broken |
| `0x0202` | `BlockPlace` | `BlockPlaceInput` | `BlockPlaceOutput` | Block is about to be placed |

#### Item hooks

| Opcode | Name | Input | Output | When |
|--------|------|-------|--------|------|
| `0x0300` | `ItemUse` | `ItemUseInput` | `ItemUseOutput` | Player uses an item |

### Concrete data schemas (implemented)

> **Note:** The early brainstorm schemas below evolved during implementation.
> See `polychora-plugin-api/src/` for the actual current types:
> - `manifest.rs`: `PluginManifest`, `EntityDeclaration` (with `sim_config: Option<EntitySimConfig>`)
> - `entity.rs`: `EntitySimConfig`, `SimulationMode` (PhysicsDriven | Parametric)
> - `entity_tick_abi.rs`: `EntityTickInput`, `EntityTickOutput` (Steer | SetPose), `EntityAbilityCheck`, `EntityAbilityResult`
> - `opcodes.rs`: `OP_ENTITY_TICK` (0x0100), `OP_ENTITY_ABILITY` (0x0101)

### Why not batch entity ticks?

One `polychora_call` per entity per tick seems expensive. But:
- postcard serialization of `EntityTickInput` is ~100 bytes. Deserialization is essentially a memcpy.
- The current fuel budget (200k) is generous for the per-entity math.
- wasmtime function call overhead is ~microseconds.
- At 20 entities × 20 ticks/sec = 400 calls/sec. This is trivially cheap.

If it ever matters, batching is an additive change — define a `BatchEntityTick` opcode that takes `Vec<EntityTickInput>` and returns `Vec<EntityTickOutput>`. No ABI break needed.

### What about per-entity persistent state?

The current `MobState` (in `server/types.rs`) includes `MobNavigationState` which the host manages, plus per-mob params (phase_offset, speeds, etc.) set at spawn time.

Approach: **host owns all persistent state, plugin is stateless per call.** The host passes relevant state in the input and uses the output to update it. If a plugin needs custom per-entity state, the `data: Vec<u8>` field on `Entity` carries it — the host passes it through and the plugin can update it in the output.

```rust
// Future extension for stateful entities
pub struct EntityTickInput {
    pub entity_type_id: u32,
    pub entity_data: Vec<u8>,   // plugin's custom state blob, opaque to host
    pub position: [f32; 4],
    pub dt_ms: u64,
    // ... context
}

pub struct EntityTickOutput {
    pub updated_data: Option<Vec<u8>>,  // updated state blob, or None if unchanged
    pub actions: Vec<EntityAction>,
}
```

This keeps WASM instances stateless (no accumulated heap state between calls) while still allowing per-entity custom data. The host snapshot/replication system already handles `Vec<u8>` entity data.

---

## 4. Plugin Manager Evolution

### Current: slot-based

`WasmPluginManager` has named slots (`EntitySimulation`, `ModelLogic`). One module per slot. This doesn't map to the namespace model where each plugin is a module with multiple capabilities.

### Proposed: namespace-keyed plugin registry

```rust
struct PluginRegistry {
    plugins: HashMap<u32, LoadedPlugin>,  // namespace_id → plugin
}

struct LoadedPlugin {
    namespace_id: u32,
    identity: PluginIdentity,        // name, version, hash
    manifest: PluginManifest,        // declared blocks, entities, items
    instance: WasmRuntimeInstance,   // live WASM instance for behavior calls
}

struct PluginManifest {
    blocks: Vec<BlockDeclaration>,
    entities: Vec<EntityDeclaration>,
    items: Vec<ItemDeclaration>,
}

struct BlockDeclaration {
    type_id: u32,              // random u32 within this namespace
    name: String,
    category: String,
    color_hint: [u8; 3],       // fallback color for minimaps, particles
    texture: TextureRef,       // (namespace, texture_id) — ns=0 for procedural
    transparent: bool,
    light_emission: u8,
    // ... more properties as needed
}
```

The load sequence:
1. Load WASM bytes → compile module
2. Instantiate → call `polychora_abi_version()` to validate
3. Call `GetManifest` opcode → read manifest, validate namespace ID uniqueness
4. Call `GetTextures` opcode → receive pixel data for all plugin textures
5. Upload textures to GPU pool → build `(namespace, texture_id) → gpu_slot` mapping
6. Resolve block material tokens: for each block, look up `block.texture` → material token
7. Build lookup tables: `(namespace, type_id) → plugin + declaration + resolved material token`
8. Plugin is now live for behavior calls

---

## 5. Texture System: `(namespace, texture_id)` as Universal Handle

### The idea

Textures are a first-class namespaced resource, just like blocks, entities, and items. Every texture is identified by `(namespace: u32, texture_id: u32)` where both IDs are randomly chosen. Plugins reference textures by this handle — they never see GPU internals.

### Texture handles vs. GPU material tokens

These are separate concerns with an indirection layer between them:

```
Plugin world                          Host/GPU world
─────────────                         ──────────────
(namespace, texture_id)   ──map──►    GPU material token (u16)
   random u32 pairs                      bit 15 = procedural vs texture-pool
   stable, permanent                     bits [14:0] = shader case or pool slot
   cross-referenceable                   ephemeral, host-assigned, can change
```

The `(namespace, texture_id)` handle is the stable, permanent identifier that lives in world files, block declarations, and cross-plugin references. The GPU material token is an internal detail the host assigns at load time and can reassign freely (e.g., on texture eviction/reload).

### All namespaces can have any kind of texture

**Namespace 0** can have both procedural textures (generated by shader code) and normal uploaded textures. Some of the current procedural materials are simple enough that they should just become pixel data.

**Plugin namespaces** similarly — `polychora-content` will provide uploaded textures, and might also reference namespace 0 textures (procedural or otherwise).

Every texture gets a random u32 ID regardless of whether it's procedural or uploaded. The current procedural shader cases (1-68 in `sampleMaterial()`) are a GPU implementation detail that the host manages. A procedural texture `(0, 0x3f8a_b2c1)` and an uploaded texture `(0, 0x7e2d_01ff)` both live in namespace 0 — the host knows which ones are procedural and which need GPU upload.

### Texture pipeline at plugin load time

```
1. Host calls GetManifest → plugin returns manifest
   (blocks reference textures by (namespace, texture_id) handles)

2. Host calls GetTextures → plugin returns Vec<TextureDeclaration>
   (texture_id + dimensions + format + pixel data)

3. Host registers each texture in the texture registry:
   (plugin_namespace, texture_id) → TextureEntry

4. Host uploads non-procedural textures to GPU pool → assigns slots

5. Host builds the material token map:
   (namespace, texture_id) → gpu_material_token
   - Procedural textures: token = shader_case_id (bit 15 = 0)
   - Uploaded textures: token = 0x8000 | pool_slot (bit 15 = 1)

6. Host resolves block material tokens:
   For each block: look up block.texture handle → material token
```

### Data structures

```rust
/// Universal texture handle — references a texture from any namespace.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureRef {
    pub namespace: u32,
    pub texture_id: u32,
}

/// Texture declaration returned by GetTextures opcode.
#[derive(Serialize, Deserialize)]
pub struct TextureDeclaration {
    pub texture_id: u32,         // random u32, scoped to this plugin's namespace
    pub width: u32,
    pub height: u32,
    pub depth: u32,              // 1 for flat, >1 for 3D volumetric
    pub format: TextureFormat,   // R8G8B8A8_SRGB, etc.
    pub pixels: Vec<u8>,         // raw pixel data
}
```

### Cross-namespace texture references

A mod can reference any texture from any loaded namespace:
- A "mossy stone" mod references polychora-content's stone texture by its `TextureRef`
- `polychora-content` can reference namespace 0 procedural textures for blocks that still use them
- The host resolves all handles to GPU tokens — cross-references are free, no re-uploading

### Why this works

- **Plugins stay stateless.** Declare textures with random IDs, reference them by `TextureRef`. Never touch GPU concepts.
- **Host owns GPU lifecycle.** The `(namespace, texture_id) → material_token` mapping is host-internal. Upload, evict, reassign slots — all transparent to plugins.
- **Procedural and uploaded textures coexist in the same handle space.** The host knows which is which. Plugins don't need to care. A block can be migrated from procedural to uploaded (or vice versa) by changing the host mapping, not the block declaration.
- **Gradual migration.** Current procedural materials get random texture IDs in namespace 0. Simple ones can be converted to uploaded textures at any pace. Complex procedural ones (rainbow, crystal lattice, tidal glass) stay as shader code. The block declarations don't change either way.

### GPU-side details

The existing `TexturePool` (`src/render/texture_pool.rs`) handles the uploaded side:
- 256 slots, 3D textures, `PARTIALLY_BOUND` descriptors
- `upload_texture_3d()` returns a slot index
- Shader: `texturePool[poolIndex].SampleLevel(sampler, pos.xyz, 0.0)`

The host maintains the mapping layer:
- `HashMap<(u32, u32), u16>` — `(namespace, texture_id) → gpu_material_token`
- Procedural entries map to shader case IDs (bit 15 = 0)
- Uploaded entries map to `0x8000 | pool_slot` (bit 15 = 1)
- Resolved at world-load and block-place time, not per-frame

### Migration path

Since the v4 format is new enough to break:
1. Assign random u32 texture IDs to all 68 current procedural materials in namespace 0
2. Update the v4 spec: all block IDs become random u32 under `polychora-content` namespace
3. `block_to_material_token()` becomes a registry lookup through the texture handle system
4. Blocks in `polychora-content` manifest reference `TextureRef { namespace: 0, texture_id: 0x.... }` for procedural materials
5. Over time, migrate simple procedural materials to uploaded textures (either in ns 0 or in polychora-content) — the block declarations don't change, only the host's routing
6. v3 world migration: one-off tool that remaps old sequential IDs → new random IDs

---

## 6. The First WASM Module: `polychora-content`

### What it should contain (MVP)

The first module to build. It proves the entire pipeline works end to end.

**MVP scope:**
- Plugin manifest with a random namespace ID
- Declares all 68 block types with new random u32 type IDs, names, categories, colors
- Declares the 3 mob types with archetype + defaults
- Maps first-party blocks to procedural material shader IDs
- EntityTick opcode implementation for all 6 entity types (mobs + accents)

**What this proves:**
- Plugin loading and manifest reading works
- Block registry is driven by plugin data, not compile-time constants
- Random IDs work end to end through storage, protocol, and rendering
- Entity behavior runs in WASM with the same behavior as today's native code

### Language

Rust targeting `wasm32-unknown-unknown`. Natural fit — can share types via a `no_std` API crate. Third-party plugin SDKs for other languages can come later.

### Crate structure

```
crates/
  polychora-content/        # First-party WASM plugin source (target: wasm32-unknown-unknown)
    Cargo.toml
    src/
      lib.rs                # ABI exports, dispatch
      blocks.rs             # Block declarations + properties (random IDs defined here)
      entities.rs           # Entity declarations
      items.rs              # Item declarations
      entity_tick.rs        # Entity tick logic (mob steering + accent animation)
  polychora-plugin-api/     # Shared contract types between host and guest
    Cargo.toml              # no_std, serde, postcard
    src/
      lib.rs
      manifest.rs           # PluginManifest, BlockDeclaration, EntityDeclaration, etc.
      entity_tick_abi.rs    # EntityTickInput/Output, EntityAbilityCheck/Result types
```

`polychora-plugin-api` is the contract crate — used by both the host (to deserialize) and the guest (to serialize). Must be `no_std` compatible.

---

## 7. Build Integration

### Approach: `build.rs` + subprocess with separate `--target-dir`

A `build.rs` in the host crate shells out to `cargo build --target wasm32-unknown-unknown` for the `polychora-content` crate. The key trick: use a **separate `--target-dir`** to avoid deadlocking on Cargo's lock file (the outer cargo holds `target/.cargo-lock`; a subprocess targeting the same dir deadlocks).

```rust
// build.rs (in the host crate that embeds the .wasm)
let wasm_target_dir = out_dir.join("wasm-target");
Command::new("cargo")
    .args(["build", "--release", "--target", "wasm32-unknown-unknown",
           "--manifest-path", &content_crate_manifest,
           "--target-dir", &wasm_target_dir])
    .status()?;

// Then the host crate embeds:
const POLYCHORA_CONTENT_WASM: &[u8] = include_bytes!(env!("POLYCHORA_CONTENT_WASM_PATH"));
```

This is the approach Substrate/Polkadot uses at scale. Works on stable Rust today.

**Trade-offs:**
- Separate target dir means the WASM deps build in isolation (no cache sharing with the workspace's native target)
- `cargo:rerun-if-changed` needs to cover the content crate's source files
- `cargo check` will trigger a full WASM build (unavoidable with build.rs)

**Future option:** `-Z bindeps` (RFC 3028, nightly-only) would let Cargo handle the cross-compilation natively in the dep graph. Still unstable as of early 2026 but is the right long-term answer. Migration from `build.rs` to `-Z bindeps` would be straightforward when it stabilizes.

---

## 8. Client-Server Plugin Flow

### Server side
1. Server loads plugin WASM modules from a `plugins/` directory (or embedded for first-party)
2. Reads manifests, validates no namespace collisions, builds registries
3. On client connect: sends plugin list (namespace IDs, names, versions, hashes)
4. During gameplay: calls plugin opcodes for mob steering, block interactions, etc.

### Client side
1. Receives plugin list from server
2. For `polychora-content`: uses embedded copy (no download needed, validated by hash match)
3. For third-party: downloads WASM bytes from server (or refuses)
4. Builds client-side registries for rendering (material tokens, entity visuals)
5. Optionally runs `ClientSpeculative` instances for prediction

### New protocol messages

```rust
// Server → Client
PluginList {
    plugins: Vec<PluginInfo>,  // namespace_id, name, version, wasm_hash
}

PluginModuleData {
    namespace_id: u32,
    wasm_bytes: Vec<u8>,       // or chunked transfer
}

// Client → Server
RequestPluginModule {
    namespace_id: u32,
}
```

---

## 9. Execution Model

### One WASM instance per plugin, persistent across calls

Each loaded plugin gets one `WasmRuntimeInstance` that lives for the lifetime of the server (or client session). The instance persists its WASM linear memory between calls, but **we treat it as stateless by convention** — all meaningful state flows through the input/output buffers, and the `Entity::data` field carries per-entity state.

The instance's heap may accumulate allocator state between calls, but the plugin should not rely on global mutable state for correctness. This makes it safe to snapshot, replicate, and predict.

### Fuel budget per call, not per tick

Each `polychora_call` gets a fresh fuel budget (currently 200k). For mob steering this is generous — the math is simple trig and vector ops. If a plugin exhausts fuel, the call fails and the host falls back to a safe default (e.g., no movement).

### Determinism: server-authoritative, client predicts

Server result is canonical. Client can run the same WASM module for prediction (using `WasmExecutionRole::ClientSpeculative`), but we do **not** require bitwise-identical results. WASM float semantics are mostly deterministic (IEEE 754), but NaN bit patterns and optimization differences can vary. The server corrects any client misprediction.

### World state access: host provides, guest computes

No host callbacks for now (Architecture A). The host pre-resolves everything the guest needs:
- For mob steering: position, waypoint (from host-side A* pathfinding), mob params, time
- For block interactions: block data, player position, face
- For entity ticks: entity data blob, position, dt

If a future hook genuinely needs on-demand world queries, we add host imports (Architecture B) as an additive extension. The trigger: a plugin that needs to scan arbitrary world positions at runtime.

### Open question: client-only plugins

For now, all plugins load server-side and are distributed to clients at connection time. Eventually there may be a role for client-only plugins (custom rendering, HUD overlays, input macros). These wouldn't need the server at all — the client loads them from a local `plugins/` directory. Deferred to Phase 6+.

---

## 10. Incremental Implementation Plan

### Phase 1: Plugin API crate + namespace registry
- Create `polychora-plugin-api` crate with manifest types (`no_std`, postcard)
- Build `PluginRegistry` that replaces static registries in the host
- Namespace 0 internal content stays hardcoded; all lookups route through registry
- No WASM yet — just the data structures and lookup paths
- Assign random u32 IDs to all existing content

### Phase 2: First WASM module (declarations only)
- Create `polychora-content` crate targeting `wasm32-unknown-unknown`
- Implement ABI exports + manifest with all block/entity/item declarations
- Host loads module, reads manifest, populates registry
- Rendering uses registry-based `block_to_material_token()` lookup
- Game runs identically but content is now plugin-driven

### Phase 3: Entity simulation in WASM ✅
- Entity tick logic (mob steering + accent animation) runs in `polychora-content` WASM plugin
- `OP_ENTITY_TICK` opcode covers both `PhysicsDriven` mobs (returns `Steer`) and `Parametric` accents (returns `SetPose`)
- `OP_ENTITY_ABILITY` opcode handles detonation + blink triggers
- Hardcoded accent animation removed from engine; all 6 entity types simulated via WASM

### Phase 4: Block interactions in WASM
- Block break/place/interact opcodes
- Drop tables defined in plugin
- Block properties (hardness, transparency) from manifest

### Phase 5: Client integration + protocol
- `PluginList` / `PluginModuleData` protocol messages
- Client-side registry population from server data
- Client-speculative execution for prediction

### Phase 6: Third-party plugin support
- Plugin loading from files
- Permission/sandboxing model
- Modding API documentation

---

## 11. Design Philosophy

**Start minimal, iterate aggressively.** The ABI is not a public API yet — we control both sides (host and the only plugin). We should expect to mutate opcode schemas, add/remove hooks, and reshape data structures as we discover the right abstractions through actual use. The opcode dispatch model supports this well: adding a new hook is a new opcode + new structs, changing a hook's schema is a recompile of both sides, and removing a hook just stops calling it.

The restricted ABI surface (no host callbacks, stateless calls, host-resolved navigation) is a feature, not a limitation. It constrains the design space to something we can reason about and evolve confidently. We can always add complexity later; removing it is much harder.

## 12. Resolved Decisions

- **IDs:** Random u32 everywhere — namespaces, block types, entity types, item types
- **Namespace 0:** Internal engine only (air, player, block_stack). No visible content.
- **Build:** `build.rs` + subprocess with separate `--target-dir`. Migrate to `-Z bindeps` when stable.
- **ABI architecture:** Typed opcode dispatch (Architecture A). No host callbacks initially.
- **State model:** WASM instances are persistent but logically stateless. Host owns all persistent state. Per-entity custom data flows through `Entity::data` field.
- **Determinism:** Server-authoritative. Client predicts, server corrects.
- **Plugin distribution:** Server loads all plugins. Client receives at connection time. Client-only plugins deferred.

## 12. Open Discussion Points

1. **Namespace 0 scope** — is air + player avatar + block_stack the complete list? Or do engine-level blocks like grid floor / avatar marker also stay internal?

2. **Manifest format** — postcard (already a dep, compact, serde) seems like the obvious choice. Any reason to prefer CBOR (self-describing, already used for item metadata)?

3. **Material system for plugins** — start with "plugins declare a color + optional procedural_material_id, procedural path is first-party only"? Or do we need the texture pool path working from day one?

4. **Testing strategy** — how to test plugins in CI without a full game instance? Likely: unit tests on manifest serialization round-trips + opcode contract tests using the test WAT module pattern already in `runtime.rs`.

5. **Opcode numbering** — the current ABI uses `i32` for opcodes. Should we widen to `u32`? The proposed opcode ranges (0x0010, 0x0100, 0x0200, 0x0300) fit in either, but `u32` is cleaner for a registry of positive IDs.

6. **How does `MobAbilityAction::Blink` work?** — the host still needs to do collision resolution for the blink destination. Should the plugin return the desired blink parameters and the host resolves a valid destination? (This matches the "host does physics, plugin does decisions" pattern.)
