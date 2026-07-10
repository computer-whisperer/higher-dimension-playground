//! Sigil recognition and casting — the shapes-as-magic prototype.
//!
//! A Resonator block is the catalyst. When the player interacts with it, the
//! host sends a snapshot of the exact-fit blocks surrounding the resonator
//! (offsets in cells of the resonator's own scale). This module matches the
//! snapshot against a registry of sigil shapes and, on a match, returns world
//! effects as side effects.
//!
//! Density is the growth curve: the same sigil built at a finer voxel scale
//! (resonator and pattern blocks all at `scale_exp = -1`) is a higher *tier*
//! and casts a stronger effect. Tier multiplier = 2^(-scale_exp).
//!
//! v1 limitations (see docs/design-shapes-as-magic.md): patterns are anchored
//! to the resonator cell (no translation search), orientation-sensitive (no
//! rotation invariance), and superset-tolerant (extra nearby blocks are
//! ignored). Spatially disjoint sigils can therefore coexist around one
//! resonator (e.g. Beacon + Summon); when more than one matches, the cast is
//! rejected as interference rather than silently picking one.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::content_ids::*;
use polychora_plugin_api::gui_abi::{BlockInteractInput, BlockInteractOutput, SnapshotBlock};
use polychora_plugin_api::region_tree::{
    Aabb4, BlockData, ChunkCoord, RegionNodeKind, RegionTreeCore,
};
use polychora_plugin_api::side_effects::{SideEffect, WasmCallResult};

/// Minimum time between casts from one resonator, in milliseconds.
const RESONATOR_COOLDOWN_MS: u64 = 3000;

/// One required block in a sigil pattern. Offsets are in cells of the
/// resonator's scale, relative to the resonator cell; blocks are looked up in
/// namespace 0 (builtin) or the content namespace depending on the constant.
struct SigilCell {
    offset: [i32; 4],
    block_type: u32,
}

struct SigilDef {
    name: &'static str,
    cells: &'static [SigilCell],
}

const fn cell(x: i32, y: i32, z: i32, w: i32, block_type: u32) -> SigilCell {
    SigilCell {
        offset: [x, y, z, w],
        block_type,
    }
}

/// Gate sigil: a ring of obsidian around the resonator in the z–w plane —
/// a genuinely 4D drawing (its plane includes the hidden axis). Casting it
/// translates the caster along +w.
const GATE_SIGIL: SigilDef = SigilDef {
    name: "Gate",
    cells: &[
        cell(0, 0, -1, -1, BLOCK_OBSIDIAN),
        cell(0, 0, -1, 0, BLOCK_OBSIDIAN),
        cell(0, 0, -1, 1, BLOCK_OBSIDIAN),
        cell(0, 0, 0, -1, BLOCK_OBSIDIAN),
        cell(0, 0, 0, 1, BLOCK_OBSIDIAN),
        cell(0, 0, 1, -1, BLOCK_OBSIDIAN),
        cell(0, 0, 1, 0, BLOCK_OBSIDIAN),
        cell(0, 0, 1, 1, BLOCK_OBSIDIAN),
    ],
};

/// Beacon sigil: the eight axis-neighbors of the resonator in glowstone —
/// the vertex figure of the 16-cell (the 4D octahedron analog). Casting it
/// erects a pillar of light above the sigil.
const BEACON_SIGIL: SigilDef = SigilDef {
    name: "Beacon",
    cells: &[
        cell(-1, 0, 0, 0, BLOCK_GLOWSTONE),
        cell(1, 0, 0, 0, BLOCK_GLOWSTONE),
        cell(0, -1, 0, 0, BLOCK_GLOWSTONE),
        cell(0, 1, 0, 0, BLOCK_GLOWSTONE),
        cell(0, 0, -1, 0, BLOCK_GLOWSTONE),
        cell(0, 0, 1, 0, BLOCK_GLOWSTONE),
        cell(0, 0, 0, -1, BLOCK_GLOWSTONE),
        cell(0, 0, 0, 1, BLOCK_GLOWSTONE),
    ],
};

/// Summon sigil: four crystal-lattice corners in the x–z plane. Casting it
/// calls seekers to the sigil.
const SUMMON_SIGIL: SigilDef = SigilDef {
    name: "Summon",
    cells: &[
        cell(-1, 0, -1, 0, BLOCK_CRYSTAL_LATTICE),
        cell(-1, 0, 1, 0, BLOCK_CRYSTAL_LATTICE),
        cell(1, 0, -1, 0, BLOCK_CRYSTAL_LATTICE),
        cell(1, 0, 1, 0, BLOCK_CRYSTAL_LATTICE),
    ],
};

const SIGIL_REGISTRY: &[&SigilDef] = &[&GATE_SIGIL, &BEACON_SIGIL, &SUMMON_SIGIL];

/// Match the snapshot against the registry. Every pattern cell must hold an
/// exact-fit block of the required type at the resonator's scale; extra
/// blocks elsewhere are ignored. Pattern block types are content-plugin
/// blocks, so they live in the content namespace.
///
/// Returns all matching sigils; more than one means the build is ambiguous
/// and the caster gets interference instead of an arbitrary pick.
fn match_sigils(snapshot: &[SnapshotBlock]) -> Vec<&'static SigilDef> {
    SIGIL_REGISTRY
        .iter()
        .copied()
        .filter(|sigil| {
            sigil.cells.iter().all(|required| {
                snapshot.iter().any(|b| {
                    b.offset == required.offset
                        && b.namespace == CONTENT_NS
                        && b.block_type == required.block_type
                })
            })
        })
        .collect()
}

fn match_sigil(snapshot: &[SnapshotBlock]) -> Option<&'static SigilDef> {
    let matches = match_sigils(snapshot);
    if matches.len() == 1 {
        Some(matches[0])
    } else {
        None
    }
}

/// Effect magnitude multiplier from the sigil's voxel scale: scale 0 → 1,
/// scale −1 → 2, scale −2 → 4 (denser sigils cast stronger effects).
fn tier_multiplier(scale_exp: i8) -> i32 {
    if scale_exp < 0 {
        1i32 << (-scale_exp).min(6) as u32
    } else {
        1
    }
}

/// Side length of one cell at the given scale, in world units.
fn cell_size(scale_exp: i8) -> f32 {
    if scale_exp >= 0 {
        (1i32 << scale_exp.min(6) as u32) as f32
    } else {
        1.0 / (1i32 << (-scale_exp).min(6) as u32) as f32
    }
}

fn cell_coord(cells: i32, scale_exp: i8) -> ChunkCoord {
    let base = ChunkCoord::from_num(cells);
    if scale_exp >= 0 {
        base << scale_exp as u32
    } else {
        base >> (-scale_exp) as u32
    }
}

/// A box of uniform blocks spanning `size_cells` cells at `scale_exp`,
/// with its minimum corner at the tree-local origin. `block_type` is a
/// content-namespace block.
fn uniform_box_tree(size_cells: [i32; 4], block_type: u32, scale_exp: i8) -> RegionTreeCore {
    let mut block = BlockData::simple(CONTENT_NS, block_type);
    block.scale_exp = scale_exp;
    RegionTreeCore {
        bounds: Aabb4 {
            min: [ChunkCoord::ZERO; 4],
            max: [
                cell_coord(size_cells[0], scale_exp),
                cell_coord(size_cells[1], scale_exp),
                cell_coord(size_cells[2], scale_exp),
                cell_coord(size_cells[3], scale_exp),
            ],
        },
        kind: RegionNodeKind::Uniform(block),
        generator_version_hash: 0,
    }
}

/// A single-cell uniform node at `cell` (in cells of `scale_exp`).
fn single_cell_node(cell: [i32; 4], namespace: u32, block_type: u32, scale_exp: i8) -> RegionTreeCore {
    let mut block = BlockData::simple(namespace, block_type);
    block.scale_exp = scale_exp;
    RegionTreeCore {
        bounds: Aabb4 {
            min: [
                cell_coord(cell[0], scale_exp),
                cell_coord(cell[1], scale_exp),
                cell_coord(cell[2], scale_exp),
                cell_coord(cell[3], scale_exp),
            ],
            max: [
                cell_coord(cell[0] + 1, scale_exp),
                cell_coord(cell[1] + 1, scale_exp),
                cell_coord(cell[2] + 1, scale_exp),
                cell_coord(cell[3] + 1, scale_exp),
            ],
        },
        kind: RegionNodeKind::Uniform(block),
        generator_version_hash: 0,
    }
}

/// A complete, ready-to-cast sigil as a placeable blueprint tree: the sigil's
/// pattern blocks plus its Resonator, all at `scale_exp`. Placing the
/// blueprint stamps the whole "hex" in one action — the densification curve's
/// delivery mechanism. Tree-local origin is the sigil's minimum corner; the
/// resonator sits at the center cell.
pub fn sigil_blueprint_tree(sigil_name: &str, scale_exp: i8) -> Option<RegionTreeCore> {
    let sigil = SIGIL_REGISTRY
        .iter()
        .copied()
        .find(|s| s.name == sigil_name)?;
    let mut min = [i32::MAX; 4];
    let mut max = [i32::MIN; 4];
    for c in sigil.cells.iter().map(|c| c.offset).chain([[0; 4]]) {
        for axis in 0..4 {
            min[axis] = min[axis].min(c[axis]);
            max[axis] = max[axis].max(c[axis]);
        }
    }
    let shift = min;
    let mut children = Vec::with_capacity(sigil.cells.len() + 1);
    children.push(single_cell_node(
        [-shift[0], -shift[1], -shift[2], -shift[3]],
        CONTENT_NS,
        BLOCK_RESONATOR,
        scale_exp,
    ));
    for c in sigil.cells {
        children.push(single_cell_node(
            [
                c.offset[0] - shift[0],
                c.offset[1] - shift[1],
                c.offset[2] - shift[2],
                c.offset[3] - shift[3],
            ],
            CONTENT_NS,
            c.block_type,
            scale_exp,
        ));
    }
    Some(RegionTreeCore {
        bounds: Aabb4 {
            min: [ChunkCoord::ZERO; 4],
            max: [
                cell_coord(max[0] - min[0] + 1, scale_exp),
                cell_coord(max[1] - min[1] + 1, scale_exp),
                cell_coord(max[2] - min[2] + 1, scale_exp),
                cell_coord(max[3] - min[3] + 1, scale_exp),
            ],
        },
        kind: RegionNodeKind::Branch(children),
        generator_version_hash: 0,
    })
}

fn sigil_effects(sigil: &SigilDef, scale_exp: i8) -> (Vec<SideEffect>, String) {
    let mult = tier_multiplier(scale_exp);
    let cell = cell_size(scale_exp);
    match sigil.name {
        "Gate" => {
            let distance = 6.0 * mult as f32;
            (
                alloc::vec![SideEffect::TeleportPlayer {
                    delta: [0.0, 0.0, 0.0, distance],
                }],
                format!("Gate sigil: shifted {distance:+.0} along w"),
            )
        }
        "Beacon" => {
            // Pillar of light starting two cells above the resonator
            // (clearing the sigil's +y block), 8 × mult² cells tall.
            let height_cells = 8 * mult * mult;
            (
                alloc::vec![SideEffect::EditWorldTree {
                    offset_cells: [0, 2, 0, 0],
                    tree: uniform_box_tree([1, height_cells, 1, 1], BLOCK_LIGHT, scale_exp),
                }],
                format!("Beacon sigil: pillar of light ({height_cells} cells)"),
            )
        }
        "Summon" => {
            let count = 2 * mult;
            let ring = 2.0 * cell;
            let mut effects = Vec::new();
            for i in 0..count {
                let (dx, dz) = match i % 4 {
                    0 => (ring, ring),
                    1 => (-ring, ring),
                    2 => (ring, -ring),
                    _ => (-ring, -ring),
                };
                effects.push(SideEffect::SpawnEntity {
                    entity_type_ns: CONTENT_NS,
                    entity_type: ENTITY_SEEKER,
                    offset: [dx, 1.5, dz, 0.4 * (i / 4) as f32],
                });
            }
            (effects, format!("Summon sigil: called {count} seekers"))
        }
        _ => (Vec::new(), String::from("Unknown sigil")),
    }
}

fn decode_last_cast_ms(metadata: &[u8]) -> u64 {
    if metadata.len() >= 8 {
        u64::from_le_bytes(metadata[0..8].try_into().unwrap())
    } else {
        0
    }
}

pub fn resonator_interact(input: &BlockInteractInput) -> WasmCallResult<BlockInteractOutput> {
    let last_cast_ms = decode_last_cast_ms(&input.metadata);
    // now < last_cast means the host clock restarted (metadata persists in the
    // save; the client clock is per-session) — treat the cooldown as expired.
    if input.now_ms >= last_cast_ms
        && input.now_ms < last_cast_ms.saturating_add(RESONATOR_COOLDOWN_MS)
    {
        return WasmCallResult::with_effects(
            BlockInteractOutput::Nothing,
            alloc::vec![SideEffect::StatusMessage {
                text: String::from("The resonator is still recharging"),
            }],
        );
    }

    let matches = match_sigils(&input.structure_snapshot);
    if matches.len() > 1 {
        return WasmCallResult::with_effects(
            BlockInteractOutput::Nothing,
            alloc::vec![SideEffect::StatusMessage {
                text: String::from("The resonator wavers between sigils — interference"),
            }],
        );
    }
    let Some(&sigil) = matches.first() else {
        return WasmCallResult::with_effects(
            BlockInteractOutput::Nothing,
            alloc::vec![SideEffect::StatusMessage {
                text: String::from("The resonator hums, but no sigil answers"),
            }],
        );
    };

    let mult = tier_multiplier(input.block_scale_exp);
    let (mut effects, message) = sigil_effects(sigil, input.block_scale_exp);
    effects.push(SideEffect::UpdateBlockMetadata {
        metadata: input.now_ms.to_le_bytes().to_vec(),
    });
    effects.push(SideEffect::StatusMessage {
        text: format!("{message} [tier x{mult}]"),
    });
    WasmCallResult::with_effects(BlockInteractOutput::Nothing, effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(cells: &[([i32; 4], u32)]) -> Vec<SnapshotBlock> {
        cells
            .iter()
            .map(|&(offset, block_type)| SnapshotBlock {
                offset,
                namespace: CONTENT_NS,
                block_type,
            })
            .collect()
    }

    fn gate_cells() -> Vec<([i32; 4], u32)> {
        GATE_SIGIL
            .cells
            .iter()
            .map(|c| (c.offset, c.block_type))
            .collect()
    }

    #[test]
    fn complete_gate_sigil_matches() {
        let snapshot = snap(&gate_cells());
        let matched = match_sigil(&snapshot).expect("gate should match");
        assert_eq!(matched.name, "Gate");
    }

    #[test]
    fn gate_sigil_tolerates_extra_blocks() {
        let mut cells = gate_cells();
        cells.push(([2, 0, 0, 0], BLOCK_STONE));
        cells.push(([0, -1, 0, 0], BLOCK_DIRT));
        let matched = match_sigil(&snap(&cells)).expect("superset should match");
        assert_eq!(matched.name, "Gate");
    }

    #[test]
    fn partial_gate_sigil_does_not_match() {
        let mut cells = gate_cells();
        cells.pop();
        assert!(match_sigil(&snap(&cells)).is_none());
    }

    #[test]
    fn wrong_material_does_not_match() {
        let cells: Vec<_> = gate_cells()
            .into_iter()
            .map(|(offset, _)| (offset, BLOCK_STONE))
            .collect();
        assert!(match_sigil(&snap(&cells)).is_none());
    }

    #[test]
    fn wrong_namespace_does_not_match() {
        let snapshot: Vec<SnapshotBlock> = GATE_SIGIL
            .cells
            .iter()
            .map(|c| SnapshotBlock {
                offset: c.offset,
                namespace: 0,
                block_type: c.block_type,
            })
            .collect();
        assert!(match_sigil(&snapshot).is_none());
    }

    #[test]
    fn coexisting_disjoint_sigils_interfere_instead_of_shadowing() {
        // Beacon (axis neighbors) and Summon (x-z corners) occupy disjoint
        // cells, so both can be complete around one resonator. That must
        // read as ambiguous — not silently resolve by registry order.
        let mut cells: Vec<([i32; 4], u32)> = BEACON_SIGIL
            .cells
            .iter()
            .map(|c| (c.offset, c.block_type))
            .collect();
        cells.extend(SUMMON_SIGIL.cells.iter().map(|c| (c.offset, c.block_type)));
        let snapshot = snap(&cells);
        assert_eq!(match_sigils(&snapshot).len(), 2);
        assert!(match_sigil(&snapshot).is_none());
    }

    #[test]
    fn summon_sigil_matches() {
        let cells: Vec<_> = SUMMON_SIGIL
            .cells
            .iter()
            .map(|c| (c.offset, c.block_type))
            .collect();
        let matched = match_sigil(&snap(&cells)).expect("summon should match");
        assert_eq!(matched.name, "Summon");
    }

    #[test]
    fn tier_multiplier_doubles_per_scale_halving() {
        assert_eq!(tier_multiplier(0), 1);
        assert_eq!(tier_multiplier(-1), 2);
        assert_eq!(tier_multiplier(-2), 4);
        assert_eq!(tier_multiplier(1), 1);
    }

    #[test]
    fn gate_effect_scales_teleport_distance_by_tier() {
        let (tier1, _) = sigil_effects(&GATE_SIGIL, 0);
        let (tier2, _) = sigil_effects(&GATE_SIGIL, -1);
        let delta_of = |effects: &[SideEffect]| match effects[0] {
            SideEffect::TeleportPlayer { delta } => delta[3],
            _ => panic!("expected TeleportPlayer"),
        };
        assert_eq!(delta_of(&tier1), 6.0);
        assert_eq!(delta_of(&tier2), 12.0);
    }

    #[test]
    fn beacon_tree_bounds_match_scale() {
        let (effects, _) = sigil_effects(&BEACON_SIGIL, -1);
        let SideEffect::EditWorldTree { tree, offset_cells } = &effects[0] else {
            panic!("expected EditWorldTree");
        };
        assert_eq!(*offset_cells, [0, 2, 0, 0]);
        // 32 cells at scale −1 → 16 world units tall, 0.5 wide.
        assert_eq!(tree.bounds.max[1], ChunkCoord::from_num(16));
        assert_eq!(tree.bounds.max[0], ChunkCoord::from_num(0.5));
    }

    /// Decompose a blueprint tree back into (cell, ns, type) tuples.
    fn blueprint_cells(tree: &RegionTreeCore, scale_exp: i8) -> Vec<([i32; 4], u32, u32)> {
        let RegionNodeKind::Branch(children) = &tree.kind else {
            panic!("expected Branch blueprint tree");
        };
        children
            .iter()
            .map(|child| {
                let RegionNodeKind::Uniform(block) = &child.kind else {
                    panic!("expected Uniform child");
                };
                assert_eq!(block.scale_exp, scale_exp);
                let cell = core::array::from_fn(|i| {
                    let unit = cell_coord(1, scale_exp);
                    (child.bounds.min[i] / unit).to_num::<i32>()
                });
                (cell, block.namespace, block.block_type)
            })
            .collect()
    }

    #[test]
    fn gate_blueprint_satisfies_its_own_sigil() {
        for scale_exp in [0i8, -1] {
            let tree = sigil_blueprint_tree("Gate", scale_exp).expect("gate blueprint");
            let cells = blueprint_cells(&tree, scale_exp);
            assert_eq!(cells.len(), 9, "resonator + 8 ring cells");

            let (resonator_cell, _, _) = *cells
                .iter()
                .find(|(_, ns, ty)| *ns == CONTENT_NS && *ty == BLOCK_RESONATOR)
                .expect("blueprint contains a resonator");

            // View the placed blueprint as a snapshot around its resonator.
            let snapshot: Vec<SnapshotBlock> = cells
                .iter()
                .map(|(cell, ns, ty)| SnapshotBlock {
                    offset: core::array::from_fn(|i| cell[i] - resonator_cell[i]),
                    namespace: *ns,
                    block_type: *ty,
                })
                .collect();
            let matched = match_sigil(&snapshot).expect("blueprint must satisfy its sigil");
            assert_eq!(matched.name, "Gate");
        }
    }

    #[test]
    fn gate_blueprint_tier2_packs_same_sigil_at_half_scale() {
        let tier1 = sigil_blueprint_tree("Gate", 0).expect("tier 1");
        let tier2 = sigil_blueprint_tree("Gate", -1).expect("tier 2");
        // Same footprint shape (1 x 1 x 3 x 3 cells), half the world extent.
        for axis in 0..4 {
            assert_eq!(tier1.bounds.max[axis], tier2.bounds.max[axis] * 2);
        }
        assert_eq!(tier1.bounds.max[2], ChunkCoord::from_num(3));
        assert_eq!(tier2.bounds.max[2], ChunkCoord::from_num(1.5f32));
    }

    #[test]
    fn cooldown_blocks_recast_and_cast_updates_metadata() {
        let gate_snapshot = snap(&gate_cells());
        let mut input = BlockInteractInput {
            block_ns: CONTENT_NS,
            block_type: BLOCK_RESONATOR,
            now_ms: 10_000,
            structure_snapshot: gate_snapshot,
            ..Default::default()
        };
        let result = resonator_interact(&input);
        let updated = result.side_effects.iter().find_map(|e| match e {
            SideEffect::UpdateBlockMetadata { metadata } => Some(metadata.clone()),
            _ => None,
        });
        let metadata = updated.expect("cast should stamp cooldown metadata");
        assert_eq!(decode_last_cast_ms(&metadata), 10_000);
        assert!(result
            .side_effects
            .iter()
            .any(|e| matches!(e, SideEffect::TeleportPlayer { .. })));

        // Recast 1s later: only a status message, no teleport.
        input.metadata = metadata.clone();
        input.now_ms = 11_000;
        let blocked = resonator_interact(&input);
        assert!(!blocked
            .side_effects
            .iter()
            .any(|e| matches!(e, SideEffect::TeleportPlayer { .. })));

        // Clock reset (new session): stale future metadata must not lock the
        // resonator — the cast should go through.
        input.metadata = metadata;
        input.now_ms = 500;
        let after_restart = resonator_interact(&input);
        assert!(after_restart
            .side_effects
            .iter()
            .any(|e| matches!(e, SideEffect::TeleportPlayer { .. })));
    }
}
