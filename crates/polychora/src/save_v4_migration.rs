use crate::migration::legacy_voxel::Chunk as LegacyChunk;
use crate::migration::legacy_voxel::RegionChunkWorld;
use crate::migration::legacy_world_io::load_world;
use crate::migration::save_v3;
use crate::save_v4::{
    self, PersistedEntityRecord, PlayerEntityHint, PlayerRecord, SaveChunkPayloadRequest,
    SaveResult, DEFAULT_REGION_CHUNK_EDGE,
};
use crate::shared::chunk_payload::ChunkPayload as FieldChunkPayload;
use crate::shared::protocol::{EntityClass, EntityKind};
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

fn world_chunk_payloads(world: &RegionChunkWorld) -> Vec<([i32; 4], FieldChunkPayload)> {
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

    let mut chunk_payloads: Vec<([i32; 4], FieldChunkPayload)> = world
        .chunks
        .iter()
        .map(|(&chunk_pos, chunk)| {
            (
                [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
                payload_from_legacy_chunk(chunk),
            )
        })
        .collect();
    chunk_payloads.sort_unstable_by_key(|(pos, _)| *pos);
    chunk_payloads
}

fn all_block_regions_from_chunk_payloads(
    chunk_payloads: &[([i32; 4], FieldChunkPayload)],
    region_chunk_edge: i32,
) -> HashSet<[i32; 4]> {
    chunk_payloads
        .iter()
        .map(|(chunk_pos, _)| save_v4::region_from_chunk(*chunk_pos, region_chunk_edge))
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
        out.push(PersistedEntityRecord {
            entity_id: next_entity_id,
            class: entity.class,
            kind: entity.kind,
            position: entity.position,
            orientation: entity.orientation,
            velocity: [0.0, 0.0, 0.0, 0.0],
            scale: entity.scale,
            material: entity.material,
            display_name: entity.display_name,
            tags: Vec::new(),
            payload,
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
    expected_chunk_payloads: Vec<([i32; 4], FieldChunkPayload)>,
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
    loaded_chunk_payloads.sort_unstable_by_key(|(pos, _)| *pos);
    if expected_chunk_payloads != loaded_chunk_payloads {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v3->v4 migration mismatch: world chunk payloads differ",
        ));
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
        .map(|entity| PersistedEntityRecord {
            entity_id: entity.entity_id,
            class: entity.class,
            kind: entity.kind,
            position: entity.position,
            orientation: entity.orientation,
            velocity: entity.velocity,
            scale: entity.scale,
            material: entity.material,
            display_name: entity.display_name,
            tags: entity.tags,
            payload: entity.payload,
            last_saved_ms: entity.last_saved_ms,
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
