# Design: Shapes-as-Magic

Status: vision captured 2026-07-10 (from project owner); both prototypes landed
and runtime-verified 2026-07-10 (commits `d5de01c`, `e58a0a3` + infra fix
`8052905`). Working today: build a sigil around a Resonator (or stamp one from
a dispenser blueprint), interact to cast; Gate teleports along +w, Beacon
erects a light pillar, Summon calls seekers; the same sigil at scale −1 casts
at tier ×2. Everything below the Vision section now describes shipped
prototype behavior plus the still-open design questions.
This document records a core piece of the original project vision that predates the
repo's documentation, plus the concept-proving work derived from it. Treat the
"Vision" section as authoritative user intent; everything after it is
interpretation and prototype scoping that the owner has not yet reviewed.

## Vision (owner's words, 2026-07-10)

> The core idea was shapes-as-magic, where a given 4d structure can activate and
> perform world effects. The blueprint system was intended to enable a
> densification growth curve, where higher technology would allow for denser
> "hexes" to be cast on one placement. It's an idea that needs a game loop and
> overall balance system to fit into.

## Interpretation

- **The spellbook is geometry.** A specific arrangement of blocks — a *sigil* —
  is latent magic. Activating it performs a world effect. The shape *is* the
  spell: knowing a spell means knowing a 4D structure, and casting well means
  building well. This makes the game's central skill (thinking and building in
  4D) also its progression currency.
- **4D-native by construction.** Sigils can require genuinely four-dimensional
  geometry (a tesseract frame, a duoprism ring, an axis pair through w). A
  player who only thinks in 3D slices can copy a sigil; a player who understands
  4D can design one. Effects can also be 4D-native (translation along w,
  cross-slice reach).
- **Blueprints are the casting mechanism.** A blueprint item stamps an entire
  structure in one placement action. "Casting a hex" = placing a sigil-carrying
  blueprint. Hand-building a sigil block-by-block is the zero-tech floor of the
  same curve.
- **Densification is the growth curve.** The voxel engine is multi-scale
  (`BlockData.scale_exp`; the region tree stores chunks at any power-of-two
  scale). The same sigil built at scale −1 packs 16× the voxels into the same
  4D footprint (2⁴ per halving). Higher technology → blueprints at finer
  scales → denser sigils per placement → stronger effects. Density, not just
  size, is the vertical axis of progression — which also means power growth
  doesn't consume more world space.
- **Needs a game loop to live in.** Acquisition (where do you learn sigils /
  get blueprints?), cost (what does casting consume?), and balance are
  deliberately unresolved. The prototypes below prove mechanics, not economy.

## Engine mapping (verified against code, 2026-07-10)

The engine already contains most of the machinery:

| Vision element | Existing mechanism |
|---|---|
| Stamp a structure in one placement | Blueprint item path: client `place_blueprint` (`app_gameplay_loop.rs`) → `send_set_tree_core(origin, tree)` → server splices a `RegionTreeCore` into the world. Works at any `scale_exp`; the Blueprint Dispenser already emits a scale −2 structure. |
| Density tiers | `BlockData.scale_exp` + multi-scale region tree; VTE renders mixed scales natively. |
| Interactive trigger blocks | `OP_BLOCK_INTERACT` runs client-side WASM (`block_gui.rs::try_block_interact`); side effects processed in `app_gameplay_loop.rs::process_block_interact_side_effects`. |
| Content-defined behavior | The WASM plugin declares blocks and implements interact/tick logic; sigil patterns and effects belong there, keeping magic data-driven. |
| World-effect vocabulary | `SideEffect` enum (plugin API): today `SpawnEntity`, `UpdateBlockMetadata`, `ConsumeHeldItem`, `GiveItem`. |

What's missing, and what the prototype adds:

1. **Structure sensing.** No path exposes the world's blocks to the plugin.
   Prototype: the client snapshots a bounded neighborhood around an interacted
   block (`scene.get_block_data`) into `BlockInteractInput`, gated by a new
   `BlockDeclaration` field so ordinary blocks pay nothing.
2. **World-effect side effects.** Prototype adds `EditWorldTree { offset, tree }`
   (reuses the blueprint `send_set_tree_core` path) and `TeleportPlayer
   { offset }`; `SpawnEntity` becomes allowed from interact (routed through the
   existing `send_multiplayer_spawn_entity`).
3. **Sigil recognition.** Pure logic in the content crate: canonicalize the
   snapshot (offsets relative to the catalyst, per block type, per scale) and
   match against declared patterns.

Note on authority: like block placement and blueprints, activation is
client-authoritative today (the whole edit path is). Server-side validation is
a later hardening concern, not a prototype concern.

## Prototype 1 — sigil activation (vertical slice)

A **Resonator** block is the catalyst. The player builds a sigil around it (or
stamps one via blueprint), then interacts with the resonator:

1. Client snapshots non-air blocks within a fixed radius of the resonator.
2. Content plugin matches the snapshot against its sigil registry.
3. On a match, the plugin returns world-effect side effects; on no match,
   nothing happens (dev-console feedback for debuggability).

Initial sigil set (chosen to span effect categories, one per category):

- **Gate sigil** → mobility: teleport the caster along +w — the 4D-native
  effect; a shortcut through the axis 3D-thinkers forget.
- **Beacon sigil** → world edit: erect a light pillar above the sigil.
- **Summon sigil** → entities: spawn mobs/fauna near the sigil.

v1 limitations (documented, deliberate): sigil must be centered on the
resonator (translation-fixed), no rotation invariance (4D has 192 orientation
symmetries — later), cooldown via block metadata instead of a casting cost.

## Prototype 2 — densification tiers

The same sigil pattern recognized at `scale_exp` 0 (tier 1) and −1 (tier 2),
with effect magnitude scaling by tier (teleport distance, pillar height, spawn
count). The Blueprint Dispenser grows a selection GUI offering each sigil as a
blueprint at each tier, so one placement stamps resonator + sigil — the full
"denser hex per placement" loop in miniature:

    hand-build tier 1  →  cast from blueprint  →  tier 2 blueprint, same
    footprint, 16× density, stronger effect

## Open design questions (for the owner)

1. **Activation trigger.** Catalyst-block interact (prototyped) vs.
   self-activation when the shape completes vs. a wand/focus item?
2. **Consumption.** Are sigil blocks consumed on cast (spell = ammunition), do
   they persist as enchanted monuments (spell = infrastructure), or does the
   cast consume a *separate* reagent while the shape persists? Prototype uses
   cooldown-only; this is the biggest open balance lever.
3. **Materials as ingredients.** Same shape in different block types — different
   spell, or different potency, or irrelevant? (Prototype: patterns are
   type-sensitive, so materials-as-ingredients is the default trajectory.)
4. **Knowledge acquisition.** Where do players learn sigils? Loot-chest
   blueprints in procgen structures would tie magic to exploration (the
   `BlockData.extra_data` plumbing for pre-seeded chests mostly exists).
5. **What do effects target?** Caster-centric (buffs, teleports), sigil-centric
   (area effects, standing enchantments), or remote (projected at a target)?
6. **"Hex" scope.** Is every blueprint placement a cast, or are only
   sigil-bearing blueprints magical? I.e., is building itself magic, or is
   magic a subset of building?

## Non-goals of the prototypes

- No economy/balance numbers worth defending — magnitudes are placeholders.
- No survival-mode integration (costs, damage) — there is no damage system yet.
- No rotation-invariant recognition, no multi-sigil composition, no remote
  targeting. All are plausible next steps once the core loop feels right.
