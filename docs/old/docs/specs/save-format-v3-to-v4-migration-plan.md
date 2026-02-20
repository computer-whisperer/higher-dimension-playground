# Save Format v3 -> v4 Migration Plan

Status: Draft
Updated: 2026-02-17

## Goal
Deliver a clean versioned transition from deployed v3 saves (flat region-leaf index) to true v4 saves (tree-structured index), with zero player/world/entity data loss.

## Problem Statement
1. Existing worlds are labeled `manifest.version = 3` and use the implemented `save_v3.rs` schema.
2. The prior "v3 spec" document described the intended tree index design, which was not actually implemented in v3.
3. We must preserve on-disk truth, freeze v3 semantics, and ship tree index as v4 with explicit migration.

## Non-Negotiable Invariants
1. No silent format upgrades.
2. No in-place destructive migration without backup.
3. No loss of:
   - block overrides
   - players/inventory
   - persistent entities
   - opaque global bytes (`custom_global_payload`)
4. Old binaries must fail cleanly on unknown future versions.

## Version Contract
1. v3 = deployed flat index format (`SAVE_FORMAT_V3_SPEC.md`).
2. v4 = tree-index format target (`SAVE_FORMAT_V4_SPEC.md`).
3. Runtime loader behavior:
   - v3 world: load via `save_v3` path only.
   - v4 world: load via new `save_v4` path only.
   - unknown version: hard error with actionable message.

## Implementation Phases

## Phase 0: Freeze and Validate v3
1. Add/expand conformance tests for v3 read/write against `SAVE_FORMAT_V3_SPEC.md`.
2. Add a regression corpus of real migrated worlds (small/medium/large) for roundtrip verification.
3. Ensure tooling preserves unknown `custom_global_payload` bytes.

## Phase 1: Implement v4 Core (New Module)
1. Add `save_v4` module with explicit structs and serializers.
2. Implement tree index encoding/decoding per `SAVE_FORMAT_V4_SPEC.md`.
3. Keep append-only data blob model and checksum rules.
4. Add deterministic serialization ordering tests.
5. Add corruption detection tests (bad header, bad checksum, bad refs).

## Phase 2: Build Migration Tooling
1. Add `worldgen-cli migrate-v3-to-v4` command.
2. Migration flow:
   - read v3
   - materialize canonical world + entities + players + global bytes
   - write v4 to new output root
   - verify roundtrip equivalence
   - atomically swap only if requested
3. Default behavior writes to a new directory; optional `--in-place` requires backup.
4. Write migration report (counts, bounds, payload bytes, checksum summary).

## Phase 3: Runtime Cutover
1. Server/runtime can read v3 and v4.
2. Server writes only one format selected by explicit config.
3. Default remains v3 until migration tool is validated on real worlds.
4. After acceptance, flip default write format to v4.

## Verification Matrix
1. Roundtrip equivalence:
   - block materialization equality over sampled bounds
   - entity id/set equality
   - player record equality
   - global payload byte equality
2. Stress:
   - sparse far-apart overrides
   - dense local edits
   - many entities crossing regions
3. Failure injection:
   - interrupted write before manifest swap
   - corrupt blob record checksum
   - dangling blob ref in index

## Rollback Strategy
1. Migration keeps source v3 world untouched by default.
2. In-place migration always creates timestamped backup first.
3. Runtime keeps v3 loader indefinitely for rollback and forensic recovery.

## Immediate Next Steps
1. Add v3 conformance tests that lock current schema behavior.
2. Scaffold `save_v4` structs and loader/writer API surface.
3. Implement migration command skeleton with dry-run and report output.
