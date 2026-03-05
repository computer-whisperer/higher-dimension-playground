//! The write/save path for save-v4 world data.

use super::entity_io::{materialize_entities_from_index_filtered, resolve_entities_for_save};
use super::index_tree::*;
use super::load::{
    load_chunk_payload_from_ref, load_or_init_save_metadata,
    materialize_world_chunk_payloads_from_index_filtered,
};
use super::*;

#[cfg(test)]
use crate::migration::legacy_voxel::RegionChunkWorld;
#[cfg(test)]
use crate::shared::region_tree::chunk_key_i32;

#[cfg(test)]
pub(super) fn save_state_from_world(
    root: &Path,
    request: SaveWorldRequest<'_>,
) -> io::Result<SaveResult> {
    let chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)> = request
        .world
        .chunks
        .iter()
        .map(|(&chunk_pos, chunk)| {
            (
                chunk_key_i32(chunk_pos[0], chunk_pos[1], chunk_pos[2], chunk_pos[3]),
                0i8,
                ResolvedChunkPayload::from_payload_with_static_palette(
                    field_chunk_payload_from_legacy_chunk(chunk),
                ),
            )
        })
        .collect();

    save_state_internal(
        root,
        request.world.base_kind(),
        chunk_payloads,
        SaveStateCommon {
            entities: request.entities,
            players: request.players,
            world_seed: request.world_seed,
            next_entity_id: request.next_entity_id,
            dirty_block_regions: request.dirty_block_regions,
            dirty_entity_regions: request.dirty_entity_regions,
            force_full_blocks: request.force_full_blocks,
            force_full_entities: request.force_full_entities,
            player_entity_hints: request.player_entity_hints,
            custom_global_payload: request.custom_global_payload,
            disable_block_persistence: request.disable_block_persistence,
            now_ms: request.now_ms,
        },
    )
}

pub fn save_state_from_chunk_payloads(
    root: &Path,
    request: SaveChunkPayloadRequest<'_>,
) -> io::Result<SaveResult> {
    save_state_internal(
        root,
        request.base_world_kind,
        request.chunk_payloads,
        SaveStateCommon {
            entities: request.entities,
            players: request.players,
            world_seed: request.world_seed,
            next_entity_id: request.next_entity_id,
            dirty_block_regions: request.dirty_block_regions,
            dirty_entity_regions: request.dirty_entity_regions,
            force_full_blocks: request.force_full_blocks,
            force_full_entities: request.force_full_entities,
            player_entity_hints: request.player_entity_hints,
            custom_global_payload: request.custom_global_payload,
            disable_block_persistence: request.disable_block_persistence,
            now_ms: request.now_ms,
        },
    )
}

pub fn save_state_from_chunk_payload_patch(
    root: &Path,
    request: SaveChunkPayloadPatchRequest,
) -> io::Result<Option<SaveResult>> {
    let loaded = load_or_init_save_metadata(
        root,
        request.base_world_kind.clone(),
        request.world_seed,
        request.now_ms,
    )?;
    let LoadedSaveMetadata {
        mut manifest,
        global: loaded_global,
        index: loaded_index,
    } = loaded;
    let mut reuse_index = build_blob_reuse_index(root, &manifest)?;

    let mut dirty_chunk_payloads = HashMap::<ChunkKey, (i8, Option<ResolvedChunkPayload>)>::new();
    for (key, scale_exp, payload) in request.dirty_chunk_payloads {
        dirty_chunk_payloads.insert(key, (scale_exp, payload));
    }
    if dirty_chunk_payloads.is_empty() && request.players.is_none() {
        return Ok(None);
    }

    // Players-only save: update only the players payload file without touching chunks.
    if dirty_chunk_payloads.is_empty() {
        if let Some(player_records) = request.players {
            let next_generation = manifest.current_generation.saturating_add(1);
            let next_players_file = players_generation_path(next_generation);
            let next_players = PlayersPayload {
                players: player_records,
            };
            write_payload_file(
                root.join(&next_players_file),
                PLAYERS_MAGIC,
                PAYLOAD_FILE_VERSION,
                &next_players,
            )?;
            manifest.players_file = next_players_file;
            manifest.current_generation = next_generation;
            manifest.last_modified_ms = request.now_ms;
            save_manifest_atomic(root, &manifest)?;
            return Ok(Some(SaveResult {
                generation: next_generation,
                saved_block_regions: 0,
                saved_entity_regions: 0,
            }));
        }
        return Ok(None);
    }

    let dirty_keys_map: HashMap<ChunkKey, i8> = dirty_chunk_payloads
        .iter()
        .map(|(k, (se, _))| (*k, *se))
        .collect();
    let node_by_id = build_node_lookup(&loaded_index);
    let mut preserved_world_leaves = Vec::<LeafDescriptor>::new();
    let mut reencode_chunk_payloads = Vec::<(ChunkKey, i8, ResolvedChunkPayload)>::new();
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    collect_world_leaf_descriptors_with_chunk_patch(
        root,
        &node_by_id,
        loaded_index.root_node_id,
        &dirty_keys_map,
        &mut payload_cache,
        &mut preserved_world_leaves,
        &mut reencode_chunk_payloads,
    )?;

    let mut dirty_keys: Vec<ChunkKey> = dirty_chunk_payloads.keys().copied().collect();
    dirty_keys.sort_unstable();
    for key in dirty_keys {
        if let Some((scale_exp, Some(payload))) = dirty_chunk_payloads.get(&key) {
            reencode_chunk_payloads.push((key, *scale_exp, payload.clone()));
        }
    }

    let mut world_leaves = preserved_world_leaves;
    world_leaves.extend(build_world_leaf_descriptors_from_payloads(
        root,
        &mut manifest,
        &mut reuse_index,
        reencode_chunk_payloads,
    )?);

    let empty_wb = Aabb4i::chunk_world_bounds([ChunkCoord::ZERO; 4], 0);
    let mut world_temp = build_temp_tree_from_leaves(&world_leaves)
        .unwrap_or_else(|| make_empty_branch_root(empty_wb.min, empty_wb.max));
    canonicalize_temp_tree(&mut world_temp);

    let mut entity_temp = match loaded_index.entity_root_node_id {
        Some(entity_root) => Some(build_temp_tree_from_index_subtree(
            &node_by_id,
            entity_root,
        )?),
        None => None,
    };
    if let Some(node) = entity_temp.as_mut() {
        canonicalize_temp_tree(node);
    }

    let mut nodes = Vec::<IndexNode>::new();
    let root_node_id = flatten_temp_tree(&world_temp, &mut nodes);
    let entity_root_node_id = entity_temp
        .as_ref()
        .map(|node| flatten_temp_tree(node, &mut nodes));

    let next_generation = manifest.current_generation.saturating_add(1);
    let next_index_file = index_generation_path(next_generation);
    let next_global_file = global_generation_path(next_generation);

    let next_index = IndexPayload {
        generation: next_generation,
        root_node_id,
        entity_root_node_id,
        nodes,
    };
    validate_index_payload(&next_index)?;
    write_index_file(root.join(&next_index_file), &next_index)?;

    let next_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(&request.base_world_kind),
        world_seed: request.world_seed,
        procgen_manifest_hash: loaded_global.procgen_manifest_hash,
        next_entity_id: request.next_entity_id.max(1),
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: request.now_ms,
        player_entity_hints: request
            .player_entity_hints
            .unwrap_or(loaded_global.player_entity_hints),
        custom_global_payload: request
            .custom_global_payload
            .unwrap_or(loaded_global.custom_global_payload),
    };
    write_payload_file(
        root.join(&next_global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_global,
    )?;

    // Optionally update players payload when provided.
    if let Some(player_records) = request.players {
        let next_players_file = players_generation_path(next_generation);
        let next_players = PlayersPayload {
            players: player_records,
        };
        write_payload_file(
            root.join(&next_players_file),
            PLAYERS_MAGIC,
            PAYLOAD_FILE_VERSION,
            &next_players,
        )?;
        manifest.players_file = next_players_file;
    }

    manifest.current_generation = next_generation;
    manifest.last_modified_ms = request.now_ms;
    manifest.index_file = next_index_file;
    manifest.global_file = next_global_file;
    save_manifest_atomic(root, &manifest)?;

    Ok(Some(SaveResult {
        generation: next_generation,
        saved_block_regions: world_leaves.len(),
        saved_entity_regions: 0,
    }))
}

fn save_state_internal(
    root: &Path,
    base_world_kind: BaseWorldKind,
    chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
    common: SaveStateCommon<'_>,
) -> io::Result<SaveResult> {
    let loaded = load_or_init_save_metadata(
        root,
        base_world_kind.clone(),
        common.world_seed,
        common.now_ms,
    )?;
    let LoadedSaveMetadata {
        mut manifest,
        global: loaded_global,
        index: loaded_index,
    } = loaded;
    let mut reuse_index = build_blob_reuse_index(root, &manifest)?;

    let effective_chunk_payloads = if common.disable_block_persistence {
        Vec::new()
    } else {
        let persisted_chunk_payloads = if common.force_full_blocks {
            Vec::new()
        } else if common.dirty_block_regions.is_empty() {
            materialize_world_chunk_payloads_from_index_filtered(
                root,
                &loaded_index,
                None,
                true,
                None,
            )?
        } else {
            materialize_world_chunk_payloads_from_index_filtered(
                root,
                &loaded_index,
                Some(common.dirty_block_regions),
                false,
                None,
            )?
        };
        resolve_world_chunk_payloads_for_save(
            persisted_chunk_payloads,
            chunk_payloads,
            common.dirty_block_regions,
            common.force_full_blocks,
        )
    };

    let world_leaves = if common.disable_block_persistence {
        Vec::new()
    } else {
        build_world_leaf_descriptors_from_payloads(
            root,
            &mut manifest,
            &mut reuse_index,
            effective_chunk_payloads,
        )?
    };

    let persisted_entities = if common.force_full_entities {
        Vec::new()
    } else if common.dirty_entity_regions.is_empty() {
        materialize_entities_from_index_filtered(root, &loaded_index, None, true)?
    } else {
        materialize_entities_from_index_filtered(
            root,
            &loaded_index,
            Some(common.dirty_entity_regions),
            false,
        )?
    };
    let effective_entities = resolve_entities_for_save(
        persisted_entities,
        common.entities,
        common.dirty_entity_regions,
        common.force_full_entities,
    );

    let mut entities_by_chunk = HashMap::<[i32; 4], Vec<PersistedEntityRecord>>::new();
    for entity in effective_entities {
        let chunk = chunk_from_world_position(entity.entity.pose.position);
        entities_by_chunk.entry(chunk).or_default().push(entity);
    }
    for entities in entities_by_chunk.values_mut() {
        entities.sort_unstable_by_key(|entity| entity.entity_id);
    }

    let mut entity_chunks: Vec<[i32; 4]> = entities_by_chunk.keys().copied().collect();
    entity_chunks.sort_unstable();

    let mut entity_leaves = Vec::<LeafDescriptor>::new();
    for chunk in entity_chunks {
        let entities = entities_by_chunk.remove(&chunk).unwrap_or_default();
        if entities.is_empty() {
            continue;
        }

        let entity_blob = EntityBlob {
            volume_min_chunk: chunk,
            volume_max_chunk: chunk,
            entities,
        };
        let entity_blob_bytes = postcard::to_stdvec(&entity_blob)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let entity_ref = append_or_reuse_blob_record(
            root,
            &mut manifest,
            &mut reuse_index,
            BLOB_KIND_ENTITY,
            ENTITY_BLOB_VERSION,
            &entity_blob_bytes,
        )?;

        let ck = chunk.map(ChunkCoord::from_num);
        let entity_wb = Aabb4i::chunk_world_bounds(ck, 0);
        entity_leaves.push(LeafDescriptor {
            min: entity_wb.min,
            max: entity_wb.max,
            scale_exp: 0,
            kind: IndexNodeKind::LeafChunkArray {
                chunk_array_ref: entity_ref,
            },
        });
    }

    let empty_wb = Aabb4i::chunk_world_bounds([ChunkCoord::ZERO; 4], 0);
    let mut world_temp = build_temp_tree_from_leaves(&world_leaves)
        .unwrap_or_else(|| make_empty_branch_root(empty_wb.min, empty_wb.max));
    canonicalize_temp_tree(&mut world_temp);

    let mut entity_temp = build_temp_tree_from_leaves(&entity_leaves);
    if let Some(node) = entity_temp.as_mut() {
        canonicalize_temp_tree(node);
    }

    let mut nodes = Vec::<IndexNode>::new();
    let root_node_id = flatten_temp_tree(&world_temp, &mut nodes);
    let entity_root_node_id = entity_temp
        .as_ref()
        .map(|node| flatten_temp_tree(node, &mut nodes));

    let next_generation = manifest.current_generation.saturating_add(1);
    let next_index_file = index_generation_path(next_generation);
    let next_global_file = global_generation_path(next_generation);
    let next_players_file = players_generation_path(next_generation);

    let next_index = IndexPayload {
        generation: next_generation,
        root_node_id,
        entity_root_node_id,
        nodes,
    };
    validate_index_payload(&next_index)?;
    write_index_file(root.join(&next_index_file), &next_index)?;

    let next_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(&base_world_kind),
        world_seed: common.world_seed,
        procgen_manifest_hash: loaded_global.procgen_manifest_hash,
        next_entity_id: common.next_entity_id,
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: common.now_ms,
        player_entity_hints: common
            .player_entity_hints
            .unwrap_or(loaded_global.player_entity_hints),
        custom_global_payload: common
            .custom_global_payload
            .unwrap_or(loaded_global.custom_global_payload),
    };
    write_payload_file(
        root.join(&next_global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_global,
    )?;

    let next_players = PlayersPayload {
        players: common.players.to_vec(),
    };
    write_payload_file(
        root.join(&next_players_file),
        PLAYERS_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_players,
    )?;

    manifest.current_generation = next_generation;
    manifest.last_modified_ms = common.now_ms;
    manifest.index_file = next_index_file;
    manifest.global_file = next_global_file;
    manifest.players_file = next_players_file;
    save_manifest_atomic(root, &manifest)?;

    Ok(SaveResult {
        generation: next_generation,
        saved_block_regions: world_leaves.len(),
        saved_entity_regions: entity_leaves.len(),
    })
}

fn resolve_world_chunk_payloads_for_save(
    persisted_chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
    chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
    dirty_block_regions: &HashSet<[i32; 4]>,
    force_full_blocks: bool,
) -> Vec<(ChunkKey, i8, ResolvedChunkPayload)> {
    let normalized_current = normalize_chunk_payloads_latest_wins(chunk_payloads);
    if force_full_blocks {
        return normalized_current;
    }

    let mut merged = HashMap::<ChunkKey, (i8, ResolvedChunkPayload)>::new();
    for (key, se, payload) in persisted_chunk_payloads {
        merged.insert(key, (se, payload));
    }
    if dirty_block_regions.is_empty() {
        let mut out: Vec<(ChunkKey, i8, ResolvedChunkPayload)> =
            merged.into_iter().map(|(k, (se, p))| (k, se, p)).collect();
        out.sort_unstable_by(|(ka, sa, _), (kb, sb, _)| (*sa, *ka).cmp(&(*sb, *kb)));
        return out;
    }

    merged.retain(|key, _| {
        !dirty_block_regions.contains(&region_from_chunk(
            key.map(|c| c.to_num::<i32>()),
            DEFAULT_REGION_CHUNK_EDGE,
        ))
    });
    for (key, se, payload) in normalized_current {
        if dirty_block_regions.contains(&region_from_chunk(
            key.map(|c| c.to_num::<i32>()),
            DEFAULT_REGION_CHUNK_EDGE,
        )) {
            merged.insert(key, (se, payload));
        }
    }

    let mut out: Vec<(ChunkKey, i8, ResolvedChunkPayload)> =
        merged.into_iter().map(|(k, (se, p))| (k, se, p)).collect();
    out.sort_unstable_by(|(ka, sa, _), (kb, sb, _)| (*sa, *ka).cmp(&(*sb, *kb)));
    out
}

fn normalize_chunk_payloads_latest_wins(
    chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
) -> Vec<(ChunkKey, i8, ResolvedChunkPayload)> {
    let mut by_key = HashMap::<ChunkKey, (u64, i8, ResolvedChunkPayload)>::new();
    for (idx, (key, se, payload)) in chunk_payloads.into_iter().enumerate() {
        let epoch = idx as u64;
        match by_key.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((epoch, se, payload));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let (current_epoch, _, _) = entry.get();
                if epoch >= *current_epoch {
                    entry.insert((epoch, se, payload));
                }
            }
        }
    }
    let mut out: Vec<(ChunkKey, i8, ResolvedChunkPayload)> = by_key
        .into_iter()
        .map(|(key, (_, se, payload))| (key, se, payload))
        .collect();
    out.sort_unstable_by(|(ka, sa, _), (kb, sb, _)| (*sa, *ka).cmp(&(*sb, *kb)));
    out
}

pub(super) fn build_world_leaf_descriptors_from_payloads(
    root: &Path,
    manifest: &mut Manifest,
    reuse_index: &mut BlobReuseIndex,
    mut chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
) -> io::Result<Vec<LeafDescriptor>> {
    let mut world_leaves = Vec::<LeafDescriptor>::new();
    chunk_payloads.sort_unstable_by(|(key_a, se_a, _), (key_b, se_b, _)| {
        (*se_a, *key_a).cmp(&(*se_b, *key_b))
    });
    // Group by scale_exp so chunks at different scales produce separate ChunkArrayBlobs.
    let mut non_uniform_rows =
        HashMap::<(i8, i32, i32, i32), Vec<(i32, ResolvedChunkPayload)>>::new();
    for (key, scale_exp, resolved) in chunk_payloads {
        let chunk_pos = chunk_key_to_lattice(&key, scale_exp);
        let ck = chunk_key_from_lattice(chunk_pos, scale_exp);
        if let Some(block) = resolved.uniform_block() {
            let wb = Aabb4i::chunk_world_bounds(ck, scale_exp);
            if block.is_air() {
                world_leaves.push(LeafDescriptor {
                    min: wb.min,
                    max: wb.max,
                    scale_exp,
                    kind: IndexNodeKind::LeafEmpty,
                });
            } else {
                world_leaves.push(LeafDescriptor {
                    min: wb.min,
                    max: wb.max,
                    scale_exp,
                    kind: IndexNodeKind::LeafUniform {
                        block: block.clone(),
                    },
                });
            }
        } else {
            non_uniform_rows
                .entry((scale_exp, chunk_pos[1], chunk_pos[2], chunk_pos[3]))
                .or_default()
                .push((chunk_pos[0], resolved));
        }
    }

    let mut row_keys: Vec<(i8, i32, i32, i32)> = non_uniform_rows.keys().copied().collect();
    row_keys.sort_unstable();
    for (row_scale_exp, row_y, row_z, row_w) in row_keys {
        let Some(mut row_payloads) = non_uniform_rows.remove(&(row_scale_exp, row_y, row_z, row_w))
        else {
            continue;
        };
        row_payloads.sort_unstable_by_key(|(x, _)| *x);

        let mut run_start = 0usize;
        while run_start < row_payloads.len() {
            let mut run_end = run_start + 1;
            while run_end < row_payloads.len()
                && row_payloads[run_end].0 == row_payloads[run_end - 1].0 + 1
            {
                run_end += 1;
            }
            let run = &row_payloads[run_start..run_end];
            let lattice_min = [run[0].0, row_y, row_z, row_w];
            let lattice_max = [run[run.len() - 1].0, row_y, row_z, row_w];
            let _min = chunk_key_from_lattice(lattice_min, row_scale_exp);
            let _max = chunk_key_from_lattice(lattice_max, row_scale_exp);

            // Build a unified block_palette across all payloads in this run,
            // remapping each payload's internal u16 indices.
            let mut unified_block_palette = vec![BlockData::AIR];
            let mut block_to_unified = HashMap::<BlockData, u16>::new();
            block_to_unified.insert(BlockData::AIR, 0);

            let mut remapped_payloads = Vec::<FieldChunkPayload>::with_capacity(run.len());
            for (_, resolved) in run {
                let Ok(dense_indices) = resolved.payload.dense_materials() else {
                    remapped_payloads.push(FieldChunkPayload::Empty);
                    continue;
                };
                let mut remapped = Vec::<u16>::with_capacity(dense_indices.len());
                for idx in &dense_indices {
                    let block = resolved
                        .block_palette
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or(BlockData::AIR);
                    let unified_idx = match block_to_unified.get(&block) {
                        Some(&idx) => idx,
                        None => {
                            let new_idx = unified_block_palette.len() as u16;
                            unified_block_palette.push(block.clone());
                            block_to_unified.insert(block, new_idx);
                            new_idx
                        }
                    };
                    remapped.push(unified_idx);
                }
                let payload = FieldChunkPayload::from_dense_materials_compact(&remapped).unwrap_or(
                    FieldChunkPayload::Dense16 {
                        materials: remapped,
                    },
                );
                remapped_payloads.push(payload);
            }

            let mut palette = Vec::<FieldChunkPayload>::new();
            let mut palette_lookup = HashMap::<FieldChunkPayload, u16>::new();
            let mut dense_indices = Vec::<u16>::with_capacity(remapped_payloads.len());
            for payload in &remapped_payloads {
                let palette_idx = if let Some(idx) = palette_lookup.get(payload).copied() {
                    idx
                } else {
                    let idx = u16::try_from(palette.len()).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "chunk array payload palette exceeds DenseU16 capacity",
                        )
                    })?;
                    palette.push(payload.clone());
                    palette_lookup.insert(payload.clone(), idx);
                    idx
                };
                dense_indices.push(palette_idx);
            }

            let mut payload_palette = Vec::<BlobRef>::with_capacity(palette.len());
            for payload in palette {
                let payload_blob = ChunkPayloadBlob { payload };
                let payload_blob_bytes = postcard::to_stdvec(&payload_blob)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                let payload_ref = append_or_reuse_blob_record(
                    root,
                    manifest,
                    reuse_index,
                    BLOB_KIND_CHUNK_PAYLOAD,
                    CHUNK_PAYLOAD_BLOB_VERSION,
                    &payload_blob_bytes,
                )?;
                payload_palette.push(payload_ref);
            }

            let mut index_data = Vec::with_capacity(dense_indices.len() * 2);
            for idx in dense_indices {
                index_data.extend_from_slice(&idx.to_le_bytes());
            }
            let chunk_array_blob = ChunkArrayBlob {
                volume_min_chunk: lattice_min,
                volume_max_chunk: lattice_max,
                scale_exp: row_scale_exp,
                payload_palette,
                block_palette: unified_block_palette,
                index_codec: ChunkArrayIndexCodec::DenseU16,
                index_data,
                default_palette_index: None,
            };
            let chunk_array_blob_bytes = postcard::to_stdvec(&chunk_array_blob)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let chunk_array_ref = append_or_reuse_blob_record(
                root,
                manifest,
                reuse_index,
                BLOB_KIND_CHUNK_ARRAY,
                CHUNK_ARRAY_BLOB_VERSION,
                &chunk_array_blob_bytes,
            )?;

            let run_world_bounds =
                Aabb4i::from_lattice_bounds(lattice_min, lattice_max, row_scale_exp);
            world_leaves.push(LeafDescriptor {
                min: run_world_bounds.min,
                max: run_world_bounds.max,
                scale_exp: row_scale_exp,
                kind: IndexNodeKind::LeafChunkArray { chunk_array_ref },
            });
            run_start = run_end;
        }
    }
    Ok(world_leaves)
}

pub(super) fn collect_world_leaf_descriptors_with_chunk_patch(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    dirty_chunks: &HashMap<ChunkKey, i8>,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    preserved_out: &mut Vec<LeafDescriptor>,
    reencode_payloads_out: &mut Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };
    let bounds = node.bounds_aabb4();
    if !bounds.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid world node bounds while collecting patch leaves",
        ));
    }

    let node_min_key = node.bounds_min_fixed.map(ChunkCoord::from_bits);
    let node_max_key = node.bounds_max_fixed.map(ChunkCoord::from_bits);

    let has_dirty = dirty_chunks.iter().any(|(chunk, &scale)| {
        let dirty_wb = Aabb4i::chunk_world_bounds(*chunk, scale);
        bounds.intersects(&dirty_wb)
    });
    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_world_leaf_descriptors_with_chunk_patch(
                    root,
                    node_by_id,
                    *child_id,
                    dirty_chunks,
                    payload_cache,
                    preserved_out,
                    reencode_payloads_out,
                )?;
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } if !has_dirty => {
            preserved_out.push(LeafDescriptor {
                min: node_min_key,
                max: node_max_key,
                scale_exp: node.scale_exp,
                kind: node.kind.clone(),
            });
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {
            let pieces = carve_bounds_excluding_dirty_chunks(bounds, dirty_chunks);
            for piece in pieces {
                preserved_out.push(LeafDescriptor {
                    min: piece.min,
                    max: piece.max,
                    scale_exp: node.scale_exp,
                    kind: node.kind.clone(),
                });
            }
        }
        IndexNodeKind::LeafChunkArray { .. } if !has_dirty => {
            preserved_out.push(LeafDescriptor {
                min: node_min_key,
                max: node_max_key,
                scale_exp: node.scale_exp,
                kind: node.kind.clone(),
            });
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_CHUNK_ARRAY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "world subtree leaf references non-chunk-array blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob = decode_chunk_array_blob(&payload)?;
            let chunk_array_bounds = Aabb4::from_lattice_bounds(
                blob.volume_min_chunk,
                blob.volume_max_chunk,
                blob.scale_exp,
            );
            if !chunk_array_bounds.is_valid() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array bounds while collecting patch leaves",
                ));
            }

            let palette_len = blob.payload_palette.len().max(1);
            let block_palette = blob.block_palette.clone();
            let chunk_array_data = ChunkArrayData {
                bounds: chunk_array_bounds,
                scale_exp: blob.scale_exp,
                chunk_palette: vec![FieldChunkPayload::Empty; palette_len],
                index_codec: blob.index_codec,
                index_data: blob.index_data.clone(),
                default_chunk_idx: blob.default_palette_index,
                block_palette: block_palette.clone(),
            };
            let indices = chunk_array_data
                .decode_dense_indices()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let extents = chunk_array_bounds
                .chunk_extents_at_scale(blob.scale_exp)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid chunk array extents during patch decode",
                    )
                })?;

            let mut palette = Vec::<FieldChunkPayload>::with_capacity(blob.payload_palette.len());
            for payload_ref in &blob.payload_palette {
                palette.push(load_chunk_payload_from_ref(
                    root,
                    payload_ref,
                    payload_cache,
                )?);
            }

            let (lat_min, lat_max) = chunk_array_bounds.to_chunk_lattice_bounds(blob.scale_exp);
            for w in lat_min[3]..=lat_max[3] {
                for z in lat_min[2]..=lat_max[2] {
                    for y in lat_min[1]..=lat_max[1] {
                        for x in lat_min[0]..=lat_max[0] {
                            let chunk = [x, y, z, w];
                            let chunk_key = chunk_key_from_lattice(chunk, blob.scale_exp);
                            let this_wb = Aabb4i::chunk_world_bounds(chunk_key, blob.scale_exp);
                            let is_dirty = dirty_chunks.iter().any(|(dk, &ds)| {
                                let dirty_wb = Aabb4i::chunk_world_bounds(*dk, ds);
                                this_wb.intersects(&dirty_wb)
                            });
                            if is_dirty {
                                continue;
                            }
                            let local = [
                                (x - lat_min[0]) as usize,
                                (y - lat_min[1]) as usize,
                                (z - lat_min[2]) as usize,
                                (w - lat_min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let palette_idx = indices.get(linear).copied().ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "chunk array decoded index out of bounds during patch decode",
                                )
                            })?;
                            let payload =
                                palette.get(palette_idx as usize).cloned().ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "chunk array palette index out of bounds during patch decode",
                                    )
                                })?;
                            reencode_payloads_out.push((
                                chunk_key,
                                blob.scale_exp,
                                ResolvedChunkPayload {
                                    payload,
                                    block_palette: block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn subtract_aabb(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    if !outer.intersects(&inner) {
        return vec![outer];
    }
    let intersection = Aabb4i::new(
        [
            outer.min[0].max(inner.min[0]),
            outer.min[1].max(inner.min[1]),
            outer.min[2].max(inner.min[2]),
            outer.min[3].max(inner.min[3]),
        ],
        [
            outer.max[0].min(inner.max[0]),
            outer.max[1].min(inner.max[1]),
            outer.max[2].min(inner.max[2]),
            outer.max[3].min(inner.max[3]),
        ],
    );
    if !intersection.is_valid() {
        return vec![outer];
    }
    if intersection == outer {
        return Vec::new();
    }

    // Half-open subtraction: no ±1 adjustment needed.
    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    for axis in 0..4 {
        if core.min[axis] < intersection.min[axis] {
            let mut piece = core;
            piece.max[axis] = intersection.min[axis];
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = intersection.min[axis];
        }
        if core.max[axis] > intersection.max[axis] {
            let mut piece = core;
            piece.min[axis] = intersection.max[axis];
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = intersection.max[axis];
        }
    }
    pieces
}

fn carve_bounds_excluding_dirty_chunks(
    bounds: Aabb4i,
    dirty_chunks: &HashMap<ChunkKey, i8>,
) -> Vec<Aabb4i> {
    let mut dirty_bounds: Vec<Aabb4i> = dirty_chunks
        .iter()
        .map(|(chunk, &scale)| Aabb4i::chunk_world_bounds(*chunk, scale))
        .filter(|cb| bounds.intersects(cb))
        .collect();
    if dirty_bounds.is_empty() {
        return vec![bounds];
    }
    dirty_bounds.sort_unstable_by_key(|b| (b.min, b.max));

    let mut pieces = vec![bounds];
    for chunk_bounds in dirty_bounds {
        let mut next = Vec::new();
        for piece in pieces {
            next.extend(subtract_aabb(piece, chunk_bounds));
        }
        pieces = next;
        if pieces.is_empty() {
            break;
        }
    }
    pieces
}

pub(super) fn build_blob_reuse_index(
    root: &Path,
    manifest: &Manifest,
) -> io::Result<BlobReuseIndex> {
    let mut out = BlobReuseIndex::default();
    let max_data_file_id = manifest.data_file_count.max(manifest.active_data_file_id);
    if max_data_file_id == 0 {
        return Ok(out);
    }

    for data_file_id in 1..=max_data_file_id {
        let path = data_file_path(root, data_file_id);
        if !path.exists() {
            continue;
        }

        let mut reader = BufReader::new(File::open(path)?);
        let mut header = [0u8; 12];
        reader.read_exact(&mut header)?;
        if &header[0..4] != DATA_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid data file magic while building reuse index",
            ));
        }
        let data_version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if data_version != DATA_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported data file version {data_version}"),
            ));
        }
        let header_file_id = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if header_file_id != data_file_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "data file id mismatch while building reuse index: header={} path={}",
                    header_file_id, data_file_id
                ),
            ));
        }

        loop {
            let record_offset = reader.stream_position()?;
            let mut record_magic = [0u8; 4];
            match reader.read_exact(&mut record_magic) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            if &record_magic != RECORD_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid record magic at offset {}", record_offset),
                ));
            }

            let mut record_rest = [0u8; 11];
            reader.read_exact(&mut record_rest)?;
            let blob_type = record_rest[0];
            let blob_version = u16::from_le_bytes([record_rest[1], record_rest[2]]);
            let payload_len = u32::from_le_bytes([
                record_rest[3],
                record_rest[4],
                record_rest[5],
                record_rest[6],
            ]);
            let payload_crc32 = u32::from_le_bytes([
                record_rest[7],
                record_rest[8],
                record_rest[9],
                record_rest[10],
            ]);
            reader.seek(SeekFrom::Current(i64::from(payload_len)))?;

            let record_len = RECORD_HEADER_LEN.checked_add(payload_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "blob record length overflow")
            })?;
            let blob_ref = BlobRef {
                data_file_id,
                record_offset,
                record_len,
                record_crc32: payload_crc32,
                blob_type,
                blob_version,
            };
            let key = blob_ref_to_reuse_key(&blob_ref)?;
            out.refs_by_key.entry(key).or_default().push(blob_ref);
        }
    }

    Ok(out)
}

pub(super) fn ensure_data_file(root: &Path, data_file_id: u32) -> io::Result<()> {
    std::fs::create_dir_all(root.join(DATA_DIR))?;
    let path = data_file_path(root, data_file_id);
    if path.exists() {
        return Ok(());
    }
    let mut file = BufWriter::new(File::create(&path)?);
    file.write_all(DATA_MAGIC)?;
    file.write_all(&DATA_FILE_VERSION.to_le_bytes())?;
    file.write_all(&data_file_id.to_le_bytes())?;
    file.flush()?;
    let file = file.into_inner().map_err(io::Error::other)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn blob_ref_to_reuse_key(blob_ref: &BlobRef) -> io::Result<BlobReuseKey> {
    if blob_ref.record_len < RECORD_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "blob record_len {} too small for header {}",
                blob_ref.record_len, RECORD_HEADER_LEN
            ),
        ));
    }
    Ok(BlobReuseKey {
        blob_type: blob_ref.blob_type,
        blob_version: blob_ref.blob_version,
        payload_len: blob_ref.record_len.saturating_sub(RECORD_HEADER_LEN),
        payload_crc32: blob_ref.record_crc32,
    })
}
