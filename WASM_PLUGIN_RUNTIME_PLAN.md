# WASM Plugin Runtime + Cache Plan

## Objective
Build a shared WebAssembly runtime and module cache that can power mod/plugin execution across systems (including mobs/models), while preserving strict server authority.

Initial target: runtime engine + cache first, then one gameplay hook.

## Scope
In scope:
- Deterministic wasm runtime used by both server and client code paths.
- Runtime limits and safety controls for untrusted modules.
- Hash-keyed module cache (memory + optional disk persistence).
- First gameplay hook where plugin output affects behavior.

Out of scope for first pass:
- Full migration of all gameplay systems to wasm.
- Plugin network/distribution protocol details.
- Procgen/worldgen or block-edit plugin integration.
- Dynamic linking between wasm modules.

## Existing Integration Points
Current behavior is concentrated in these paths:
- Server simulation loop: `crates/polychora/src/server/mod.rs` (`tick_entity_simulation_window`, `simulate_mobs`, `simulate_*_step`).
- Multiplayer protocol definitions: `crates/polychora/src/shared/protocol.rs`.
- Client message handling: `crates/polychora/src/app_multiplayer.rs`.

The immediate invocation seam is server mob steering; client overlap is speculative/presentation-only.

## Proposed Runtime Architecture
Add a shared module tree under `crates/polychora/src/shared/wasm/`:
- `abi.rs`: request/response payload structs + opcode constants.
- `runtime.rs`: `wasmtime`-based loader, instance lifecycle, and invocation.
- `cache.rs`: hash-keyed memory cache + optional disk cache for module bytes.
- `manager.rs`: host-facing registry, cache lookup, and activation logic.

Use `wasmtime` (Cranelift backend) as the initial runtime engine:
- Better throughput for high-frequency calls.
- Explicit fuel metering and configurable limits.
- Mature compilation/caching ecosystem.

## Plugin Unit and Identity
A module is keyed by a content hash:
- `module_hash`: SHA-256 of wasm bytes.
- `module_id`: logical name (human/config-facing).
- `kind`: behavior class (`mob_steering`, `model_logic`, etc).
- `abi_version`: runtime ABI version.
- `limits`: memory pages, max input bytes, max output bytes, fuel budget.

Distribution (network/file/modpack) is an adapter concern and should only provide `wasm_bytes`.

## ABI Contract (v1)
Required guest exports:
- `polychora_abi_version() -> i32`
- `polychora_alloc(len: i32) -> i32`
- `polychora_free(ptr: i32, len: i32)`
- `polychora_call(opcode: i32, in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32`

Call flow:
1. Host allocates guest input/output buffers with `polychora_alloc`.
2. Host writes postcard-encoded request bytes into guest memory.
3. Host invokes `polychora_call`.
4. Return value is output byte length (`>= 0`) or negative error code.
5. Host reads output bytes and decodes postcard response.

Rules:
- No WASI.
- No mutable globals imported from host.
- Only deterministic host imports, and keep initial v1 imports empty unless required.

## Determinism Rules
To support "evaluate exactly" semantics:
- Fixed invocation schedule (based on server tick index, not wall-clock).
- Identical input payload bytes on server and client for the same tick.
- No host time/random imports.
- Fuel-limited execution with deterministic trap behavior.
- Sanitize NaN/inf in host boundaries before/after calls.

Verification mechanism:
- Optional per-plugin rolling digest over `(tick, request_bytes, response_bytes)`.
- Server may send periodic digest checkpoints; client compares and logs mismatch.

Authority rule:
- Server output is authoritative.
- Client plugin output for overlapping systems is speculative/lag-hiding only and must reconcile to server state.

## Safety and Resource Controls
Per-invocation limits:
- `max_input_bytes`
- `max_output_bytes`
- `max_memory_pages`
- `max_fuel`

Failure handling:
- Trap, decode failure, limit breach, or bad ABI marks plugin instance unhealthy.
- Host falls back to built-in Rust behavior for that system.
- Client logs plugin fault but keeps session alive.

## Transport/Distribution Boundary
Not part of runtime core.

Runtime + cache API should accept raw module bytes and hashes from any source:
- local files
- modpack bundles
- server push/pull protocol

A future network adapter can be layered without changing core execution APIs.

## First Gameplay Hook
Use wasm first for mob steering, not full physics.

Why this first:
- High value for behavior experimentation.
- Minimal host capability requirements.
- Low risk: host still owns collision, nav, and authoritative entity updates.

Hook location:
- Replace Rust `simulate_seeker_step`, `simulate_creeper_step`, `simulate_phase_spider_step` decisions with plugin call output.
- Keep `update_mob_navigation_state`, collision resolution, explosions, and replication in host Rust.
- Server remains canonical for movement outcomes.
- Client path may optionally run the same steering plugin for prediction/smoothing only.

Initial request/response payloads:
- Request includes archetype, locomotion mode, tick time, dt, current transform, nearest target, and path-following flag.
- Response includes desired direction `[f32; 4]` and `speed_factor`.

Fallback behavior:
- If plugin not active or fails call, use existing Rust steering logic.

## Next Hook Candidates
- Model behavior/animation driving for non-authoritative visuals.
- Additional mob behavior logic beyond steering.

## Incremental Rollout Plan
Phase 1: Runtime + cache skeleton
- Add wasm runtime module and ABI validation.
- Add hash-keyed memory cache with optional disk backing.
- Add unit tests with tiny fixture wasm modules.

Phase 2: Manager + slot activation
- Add plugin slot registry (`mob_steering`, `model_logic`, etc).
- Add activation and fallback policies.

Phase 3: Mob steering integration (server authoritative)
- Add steering plugin slot and runtime call path.
- Keep current Rust steering as fallback.
- Add optional client-side speculative invocation for smoothing only.
- Add deterministic digest logging/checkpoints.

Phase 4: Transport adapters (optional)
- Add server-distribution adapter and/or modpack loader using runtime/cache APIs.

## Testing Strategy
- Runtime unit tests: ABI mismatch, bad exports, traps, memory/fuel limits, hash checks.
- Gameplay tests: compare plugin steering output against known vectors.
- Determinism tests: identical request bytes produce identical response bytes on server/client harness.

## Open Design Decisions
- Persistent plugin state format (save file integration through `custom_global_payload` or dedicated plugin blob).
- Signature/trust model for module provenance (hash-only vs signed manifests).
- Whether to allow floating-point-heavy plugins in strict deterministic mode or require fixed-point ABI in later phase.
