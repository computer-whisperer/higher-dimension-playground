use crate::migration::legacy_voxel::Chunk as LegacyChunk;
use crate::migration::legacy_voxel::RegionChunkWorld;
use crate::migration::legacy_world_io::load_world;
use crate::migration::save_v3;
use crate::save_v4::{
    self, PersistedEntityRecord, PlayerEntityHint, PlayerRecord, SaveChunkPayloadRequest,
    SaveResult, DEFAULT_REGION_CHUNK_EDGE,
};
use crate::shared::chunk_payload::{ChunkPayload as FieldChunkPayload, ResolvedChunkPayload};
use crate::migration::save_v3::{EntityClass, EntityKind};
use crate::shared::entity_types;
use crate::shared::protocol::{Entity, EntityPose};
use crate::shared::region_tree::ChunkKey;
use crate::shared::voxel::BlockData;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
struct LegacySidecarBlob {
    version: u32,
    entities: Vec<LegacySidecarEntity>,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)] // Fields retained for deserialization of legacy sidecar JSON
struct LegacySidecarEntity {
    class: EntityClass,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    mob: Option<serde_json::Value>,
}

fn entity_kind_to_type_key(kind: EntityKind) -> (u32, u32) {
    match kind {
        EntityKind::PlayerAvatar => entity_types::ENTITY_PLAYER_AVATAR,
        EntityKind::TestCube => entity_types::ENTITY_TEST_CUBE,
        EntityKind::TestRotor => entity_types::ENTITY_TEST_ROTOR,
        EntityKind::TestDrifter => entity_types::ENTITY_TEST_DRIFTER,
        EntityKind::MobSeeker => entity_types::ENTITY_MOB_SEEKER,
        EntityKind::MobCreeper4d => entity_types::ENTITY_MOB_CREEPER4D,
        EntityKind::MobPhaseSpider => entity_types::ENTITY_MOB_PHASE_SPIDER,
    }
}

/// Build a legacy block palette for resolving old material tokens (u16) → BlockData.
/// Index 0 = air; indices 1..68 correspond to the polychora-content blocks in registration order.
fn build_legacy_block_palette() -> Vec<BlockData> {
    use polychora_plugin_api::content_ids::*;
    let ordered: &[u32] = &[
        BLOCK_RED, BLOCK_ORANGE, BLOCK_YELLOW_GREEN, BLOCK_GREEN,
        BLOCK_CYAN, BLOCK_BLUE, BLOCK_PURPLE, BLOCK_MAGENTA,
        BLOCK_RAINBOW, BLOCK_BROWN, BLOCK_GRID_FLOOR, BLOCK_WHITE,
        BLOCK_LIGHT, BLOCK_MIRROR, BLOCK_LAVA_VEINED_BASALT, BLOCK_CRYSTAL_LATTICE,
        BLOCK_MARBLE, BLOCK_OXIDIZED_METAL, BLOCK_BIO_SPORE_MOSS, BLOCK_VOID_MIRROR,
        BLOCK_AVATAR_MARKER, BLOCK_HOLOGRAPHIC_LAMINATE, BLOCK_TIDAL_GLASS, BLOCK_CIRCUIT_WEAVE,
        BLOCK_AURORA_STONE, BLOCK_HAZARD_CHEVRONS, BLOCK_STONE, BLOCK_COBBLESTONE,
        BLOCK_DIRT, BLOCK_COARSE_DIRT, BLOCK_OAK_PLANKS, BLOCK_SPRUCE_PLANKS,
        BLOCK_LOG_BARK, BLOCK_LOG_END_RINGS, BLOCK_SAND, BLOCK_GRAVEL,
        BLOCK_CLAY, BLOCK_GRASS_BLOCK, BLOCK_SNOW, BLOCK_ICE,
        BLOCK_COAL_ORE, BLOCK_IRON_ORE, BLOCK_GOLD_ORE, BLOCK_DIAMOND_ORE,
        BLOCK_REDSTONE_ORE, BLOCK_BIRCH_PLANKS, BLOCK_BRICKS, BLOCK_SANDSTONE,
        BLOCK_GLASS, BLOCK_GLOWSTONE, BLOCK_OBSIDIAN, BLOCK_PRISMARINE,
        BLOCK_TERRACOTTA, BLOCK_WOOL_WHITE, BLOCK_BASALT_TILES, BLOCK_COPPER_WEAVE,
        BLOCK_NEBULA_STRATA, BLOCK_STARFORGED_CORE, BLOCK_CRYO_CIRCUIT, BLOCK_SMOKED_GLASS,
        BLOCK_IVORY_MARBLE, BLOCK_RUNIC_ALLOY, BLOCK_HYPERPHASE_GEL, BLOCK_SINGULARITY_CORE,
        BLOCK_CHRONO_BLOOM, BLOCK_TESSERACT_WEAVE, BLOCK_EVENTIDE_ALLOY, BLOCK_BEACON_MATRIX,
    ];
    let mut palette = vec![BlockData::AIR]; // index 0 = air
    for &block_type in ordered {
        palette.push(BlockData::simple(CONTENT_NS, block_type));
    }
    palette
}

/// Resolve a legacy ChunkPayload (where u16 values are raw material tokens) into a
/// ResolvedChunkPayload with a proper block palette.
fn resolve_legacy_payload(payload: FieldChunkPayload, palette: &[BlockData]) -> ResolvedChunkPayload {
    ResolvedChunkPayload {
        payload,
        block_palette: palette.to_vec(),
    }
}

fn world_chunk_payloads(world: &RegionChunkWorld) -> Vec<(ChunkKey, i8, ResolvedChunkPayload)> {
    fn payload_from_legacy_chunk(chunk: &LegacyChunk) -> FieldChunkPayload {
        if chunk.is_empty() {
            return FieldChunkPayload::Empty;
        }
        let first = u16::from(chunk.voxels[0].0);
        if chunk.voxels.iter().all(|voxel| u16::from(voxel.0) == first) {
            return FieldChunkPayload::Uniform(first);
        }
        FieldChunkPayload::Dense16 {
            materials: chunk
                .voxels
                .iter()
                .map(|voxel| u16::from(voxel.0))
                .collect(),
        }
    }

    let legacy_palette = build_legacy_block_palette();
    let mut chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)> = world
        .chunks
        .iter()
        .map(|(&chunk_pos, chunk)| {
            (
                crate::shared::spatial::chunk_key_from_lattice(chunk_pos, 0),
                0i8,
                resolve_legacy_payload(payload_from_legacy_chunk(chunk), &legacy_palette),
            )
        })
        .collect();
    chunk_payloads.sort_unstable_by(|(ka, sa, _), (kb, sb, _)| (*sa, *ka).cmp(&(*sb, *kb)));
    chunk_payloads
}

fn all_block_regions_from_chunk_payloads(
    chunk_payloads: &[(ChunkKey, i8, ResolvedChunkPayload)],
    region_chunk_edge: i32,
) -> HashSet<[i32; 4]> {
    chunk_payloads
        .iter()
        .map(|(key, _, _)| save_v4::region_from_chunk_key(*key, region_chunk_edge))
        .collect()
}

pub fn load_legacy_sidecar_entities(
    path: &Path,
    now_ms: u64,
) -> io::Result<Vec<PersistedEntityRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed: LegacySidecarBlob = serde_json::from_reader(reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if parsed.version == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid legacy sidecar version 0",
        ));
    }

    let mut next_entity_id = 1u64;
    let mut out = Vec::with_capacity(parsed.entities.len());
    for entity in parsed.entities {
        let payload = if let Some(mob) = entity.mob {
            serde_json::to_vec(&mob)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        } else {
            Vec::new()
        };
        let (namespace, entity_type) = entity_kind_to_type_key(entity.kind);
        out.push(PersistedEntityRecord {
            entity_id: next_entity_id,
            entity: Entity {
                namespace,
                entity_type,
                pose: EntityPose {
                    position: entity.position,
                    orientation: entity.orientation,
                    velocity: [0.0, 0.0, 0.0, 0.0],
                    scale: entity.scale,
                },
                data: payload,
            },
            display_name: entity.display_name,
            tags: Vec::new(),
            last_saved_ms: now_ms,
        });
        next_entity_id = next_entity_id.saturating_add(1);
    }
    Ok(out)
}

pub fn migrate_legacy_world_to_v4(
    legacy_world: &Path,
    legacy_sidecar: Option<&Path>,
    output_root: &Path,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<SaveResult> {
    let file = File::open(legacy_world)?;
    let mut reader = BufReader::new(file);
    let world = load_world(&mut reader)?;

    let entities = if let Some(sidecar_path) = legacy_sidecar {
        load_legacy_sidecar_entities(sidecar_path, now_ms)?
    } else {
        Vec::new()
    };
    let players: Vec<PlayerRecord> = Vec::new();
    let chunk_payloads = world_chunk_payloads(&world);
    let block_regions =
        all_block_regions_from_chunk_payloads(&chunk_payloads, DEFAULT_REGION_CHUNK_EDGE);
    let entity_regions = save_v4::all_entity_regions(&entities, DEFAULT_REGION_CHUNK_EDGE);
    let next_entity_id = entities
        .iter()
        .map(|entity| entity.entity_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    save_v4::save_state_from_chunk_payloads(
        output_root,
        SaveChunkPayloadRequest {
            base_world_kind: world.base_kind(),
            chunk_payloads,
            entities: &entities,
            players: &players,
            world_seed,
            next_entity_id,
            dirty_block_regions: &block_regions,
            dirty_entity_regions: &entity_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
}

fn verify_v3_migration_equivalence(
    output_root: &Path,
    expected_chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
    expected_entities: &[PersistedEntityRecord],
    expected_players: &[PlayerRecord],
    expected_world_seed: u64,
    expected_next_entity_id: u64,
    expected_player_entity_hints: &[PlayerEntityHint],
    expected_custom_global_payload: &[u8],
) -> io::Result<()> {
    let loaded = save_v4::load_state(output_root)?;

    if loaded.global.world_seed != expected_world_seed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "v3->v4 migration mismatch: world_seed expected={} got={}",
                expected_world_seed, loaded.global.world_seed
            ),
        ));
    }
    if loaded.global.next_entity_id != expected_next_entity_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "v3->v4 migration mismatch: next_entity_id expected={} got={}",
                expected_next_entity_id, loaded.global.next_entity_id
            ),
        ));
    }
    if loaded.global.player_entity_hints != expected_player_entity_hints {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v3->v4 migration mismatch: player_entity_hints differ",
        ));
    }
    if loaded.global.custom_global_payload != expected_custom_global_payload {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v3->v4 migration mismatch: custom_global_payload differs",
        ));
    }

    let mut expected_entities_sorted = expected_entities.to_vec();
    expected_entities_sorted.sort_unstable_by_key(|entity| entity.entity_id);
    let mut loaded_entities_sorted = loaded.entities;
    loaded_entities_sorted.sort_unstable_by_key(|entity| entity.entity_id);
    if expected_entities_sorted != loaded_entities_sorted {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v3->v4 migration mismatch: entities differ",
        ));
    }

    let mut expected_players_sorted = expected_players.to_vec();
    expected_players_sorted.sort_unstable_by_key(|player| player.player_id);
    let mut loaded_players_sorted = loaded.players.players;
    loaded_players_sorted.sort_unstable_by_key(|player| player.player_id);
    if expected_players_sorted != loaded_players_sorted {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v3->v4 migration mismatch: players differ",
        ));
    }

    let mut loaded_chunk_payloads = loaded.world_chunk_payloads;
    loaded_chunk_payloads.sort_unstable_by_key(|(key, _)| *key);
    // Compare semantically: palettes may differ structurally after save/load round-trip
    // (the save process compacts palettes), so compare resolved blocks instead.
    if expected_chunk_payloads.len() != loaded_chunk_payloads.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "v3->v4 migration mismatch: chunk count expected={} got={}",
                expected_chunk_payloads.len(),
                loaded_chunk_payloads.len()
            ),
        ));
    }
    for ((exp_key, _, exp_payload), loaded) in expected_chunk_payloads.iter().zip(loaded_chunk_payloads.iter()) {
        if *exp_key != loaded.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "v3->v4 migration mismatch: chunk position expected={:?} got={:?}",
                    exp_key, loaded.0
                ),
            ));
        }
        if exp_payload.dense_blocks() != loaded.1.dense_blocks() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "v3->v4 migration mismatch: chunk blocks differ at position {:?}",
                    exp_key
                ),
            ));
        }
    }

    Ok(())
}

pub fn migrate_v3_save_to_v4(
    v3_root: &Path,
    output_root: &Path,
    overwrite: bool,
    now_ms: u64,
) -> io::Result<SaveResult> {
    if output_root.exists() {
        if !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output path '{}' already exists (use overwrite to replace)",
                    output_root.display()
                ),
            ));
        }
        if output_root.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output path '{}' is a file", output_root.display()),
            ));
        }
        std::fs::remove_dir_all(output_root)?;
    }

    let loaded = save_v3::load_state(v3_root)?;
    let save_v3::LoadedState {
        global,
        players,
        world,
        entities,
        manifest,
        ..
    } = loaded;

    let entities: Vec<PersistedEntityRecord> = entities
        .into_iter()
        .map(|entity| {
            let (namespace, entity_type) = entity_kind_to_type_key(entity.kind);
            PersistedEntityRecord {
                entity_id: entity.entity_id,
                entity: Entity {
                    namespace,
                    entity_type,
                    pose: EntityPose {
                        position: entity.position,
                        orientation: entity.orientation,
                        velocity: entity.velocity,
                        scale: entity.scale,
                    },
                    data: entity.payload,
                },
                display_name: entity.display_name,
                tags: entity.tags,
                last_saved_ms: entity.last_saved_ms,
            }
        })
        .collect();
    let players: Vec<PlayerRecord> = players
        .players
        .into_iter()
        .map(|player| PlayerRecord {
            player_id: player.player_id,
            position: player.position,
            orientation: player.orientation,
            tags: player.tags,
            inventory_payload: player.inventory_payload,
            last_saved_ms: player.last_saved_ms,
        })
        .collect();
    let player_entity_hints: Vec<PlayerEntityHint> = global
        .player_entity_hints
        .into_iter()
        .map(|hint| PlayerEntityHint {
            player_id: hint.player_id,
            entity_id: hint.entity_id,
            last_region_hint: hint.last_region_hint,
        })
        .collect();

    let region_chunk_edge = manifest.limits.region_chunk_edge.max(1);
    let chunk_payloads = world_chunk_payloads(&world);
    let block_regions = all_block_regions_from_chunk_payloads(&chunk_payloads, region_chunk_edge);
    let entity_regions = save_v4::all_entity_regions(&entities, region_chunk_edge);
    let expected_world_seed = global.world_seed;
    let expected_next_entity_id = global.next_entity_id;
    let expected_custom_global_payload = global.custom_global_payload.clone();
    let expected_player_entity_hints = player_entity_hints.clone();
    let expected_chunk_payloads = chunk_payloads.clone();

    let save_result = save_v4::save_state_from_chunk_payloads(
        output_root,
        SaveChunkPayloadRequest {
            base_world_kind: world.base_kind(),
            chunk_payloads,
            entities: &entities,
            players: &players,
            world_seed: global.world_seed,
            next_entity_id: global.next_entity_id,
            dirty_block_regions: &block_regions,
            dirty_entity_regions: &entity_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: Some(player_entity_hints),
            custom_global_payload: Some(global.custom_global_payload),
            disable_block_persistence: false,
            now_ms,
        },
    )?;

    verify_v3_migration_equivalence(
        output_root,
        expected_chunk_payloads,
        &entities,
        &players,
        expected_world_seed,
        expected_next_entity_id,
        &expected_player_entity_hints,
        &expected_custom_global_payload,
    )?;

    Ok(save_result)
}
