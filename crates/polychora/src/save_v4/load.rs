//! The read/load path for save-v4 world data.

use super::entity_io::materialize_entities_from_index;
use super::*;

pub fn load_or_init_state(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedState> {
    let metadata = load_or_init_state_metadata(root, default_base, world_seed, now_ms)?;
    let world_chunk_payloads =
        strip_scale_exp(materialize_world_chunk_payloads_from_index_filtered(
            root,
            &metadata.index,
            None,
            true,
            None,
        )?);
    Ok(LoadedState {
        manifest: metadata.manifest,
        global: metadata.global,
        players: metadata.players,
        index: metadata.index,
        world_chunk_payloads,
        entities: metadata.entities,
    })
}

pub fn load_or_init_state_metadata(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedStateMetadata> {
    if !is_v4_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    load_state_metadata(root)
}

pub(super) fn load_or_init_save_metadata(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedSaveMetadata> {
    if !is_v4_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    Ok(LoadedSaveMetadata {
        manifest,
        global,
        index,
    })
}

pub fn load_state(root: &Path) -> io::Result<LoadedState> {
    let metadata = load_state_metadata(root)?;
    let world_chunk_payloads =
        strip_scale_exp(materialize_world_chunk_payloads_from_index_filtered(
            root,
            &metadata.index,
            None,
            true,
            None,
        )?);
    Ok(LoadedState {
        manifest: metadata.manifest,
        global: metadata.global,
        players: metadata.players,
        index: metadata.index,
        world_chunk_payloads,
        entities: metadata.entities,
    })
}

pub fn load_state_metadata(root: &Path) -> io::Result<LoadedStateMetadata> {
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let players: PlayersPayload =
        read_payload_file(root.join(&manifest.players_file), PLAYERS_MAGIC)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    let mut entities = materialize_entities_from_index(root, &index)?;
    entities.sort_unstable_by_key(|entity| entity.entity_id);

    Ok(LoadedStateMetadata {
        manifest,
        global,
        players,
        index,
        entities,
    })
}

pub fn load_world_chunk_payloads_for_regions(
    root: &Path,
    regions: &HashSet<[i32; 4]>,
) -> io::Result<Vec<(ChunkKey, ResolvedChunkPayload)>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    Ok(strip_scale_exp(
        materialize_world_chunk_payloads_from_index_filtered(
            root,
            &index,
            Some(regions),
            true,
            None,
        )?,
    ))
}

pub fn load_world_chunk_payloads_for_bounds(
    root: &Path,
    bounds: Aabb4i,
) -> io::Result<Vec<(ChunkKey, ResolvedChunkPayload)>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    load_world_chunk_payloads_for_bounds_from_index(root, &index, bounds)
}

pub fn load_world_chunk_payloads_for_bounds_from_index(
    root: &Path,
    index: &IndexPayload,
    bounds: Aabb4i,
) -> io::Result<Vec<(ChunkKey, ResolvedChunkPayload)>> {
    if !bounds.is_valid() {
        return Ok(Vec::new());
    }
    Ok(strip_scale_exp(
        materialize_world_chunk_payloads_from_index_filtered(
            root,
            index,
            None,
            true,
            Some(bounds),
        )?,
    ))
}

pub fn load_world_subtree_core_for_bounds_from_index(
    root: &Path,
    index: &IndexPayload,
    bounds: Aabb4i,
) -> io::Result<RegionTreeCore> {
    if !bounds.is_valid() {
        return Ok(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        });
    }

    let node_by_id = build_node_lookup(index);
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    let mut chunk_array_cache = HashMap::<(u32, u64, u32, u32, u8, u16), ChunkArrayData>::new();
    let root_core = load_world_subtree_core_from_index_node_filtered(
        root,
        &node_by_id,
        index.root_node_id,
        bounds,
        &mut payload_cache,
        &mut chunk_array_cache,
    )?
    .unwrap_or(RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    });
    Ok(slice_region_core_in_bounds(&root_core, bounds))
}

fn load_world_subtree_core_from_index_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    bounds_filter: Aabb4i,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    chunk_array_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), ChunkArrayData>,
) -> io::Result<Option<RegionTreeCore>> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };

    let node_bounds = node.bounds_aabb4();
    if !node_bounds.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index world node {node_id} has invalid bounds"),
        ));
    }
    if !node_bounds.intersects(&bounds_filter) {
        return Ok(None);
    }

    let kind = match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            let mut children = Vec::new();
            for child_id in child_node_ids {
                if let Some(child_core) = load_world_subtree_core_from_index_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    bounds_filter,
                    payload_cache,
                    chunk_array_cache,
                )? {
                    children.push(child_core);
                }
            }
            if children.is_empty() {
                RegionNodeKind::Empty
            } else {
                RegionNodeKind::Branch(children)
            }
        }
        IndexNodeKind::LeafEmpty => RegionNodeKind::Empty,
        IndexNodeKind::LeafUniform { block } => RegionNodeKind::Uniform(block.clone()),
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
            let key = blob_ref_identity(chunk_array_ref);
            let chunk_array = if let Some(cached) = chunk_array_cache.get(&key) {
                cached.clone()
            } else {
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
                        "invalid chunk array bounds",
                    ));
                }
                let mut chunk_palette =
                    Vec::<FieldChunkPayload>::with_capacity(blob.payload_palette.len());
                for payload_ref in &blob.payload_palette {
                    chunk_palette.push(load_chunk_payload_from_ref(
                        root,
                        payload_ref,
                        payload_cache,
                    )?);
                }
                let decoded = ChunkArrayData {
                    bounds: chunk_array_bounds,
                    scale_exp: blob.scale_exp,
                    chunk_palette,
                    index_codec: blob.index_codec,
                    index_data: blob.index_data.clone(),
                    default_chunk_idx: blob.default_palette_index,
                    block_palette: blob.block_palette.clone(),
                };
                chunk_array_cache.insert(key, decoded.clone());
                decoded
            };
            RegionNodeKind::ChunkArray(chunk_array)
        }
    };

    Ok(Some(RegionTreeCore {
        bounds: node_bounds,
        kind,
        generator_version_hash: 0,
    }))
}

pub(super) fn load_chunk_payload_from_ref(
    root: &Path,
    blob_ref: &BlobRef,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
) -> io::Result<FieldChunkPayload> {
    if blob_ref.blob_type != BLOB_KIND_CHUNK_PAYLOAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chunk payload ref has invalid blob type {}, expected {}",
                blob_ref.blob_type, BLOB_KIND_CHUNK_PAYLOAD
            ),
        ));
    }

    let key = blob_ref_identity(blob_ref);
    if let Some(payload) = payload_cache.get(&key) {
        return Ok(payload.clone());
    }

    let payload = read_blob_payload(root, blob_ref)?;
    let payload_blob: ChunkPayloadBlob = postcard::from_bytes(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    payload_cache.insert(key, payload_blob.payload.clone());
    Ok(payload_blob.payload)
}

pub(super) fn materialize_world_chunk_payloads_from_index_filtered(
    root: &Path,
    index: &IndexPayload,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
    bounds_filter: Option<Aabb4i>,
) -> io::Result<Vec<(ChunkKey, i8, ResolvedChunkPayload)>> {
    let node_by_id = build_node_lookup(index);
    let mut out = Vec::<(ChunkKey, i8, ResolvedChunkPayload)>::new();
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    collect_world_chunk_payloads_from_node_filtered(
        root,
        &node_by_id,
        index.root_node_id,
        region_filter,
        include_matches,
        bounds_filter,
        &mut payload_cache,
        &mut out,
    )?;
    out.sort_unstable_by(|(ka, sa, _), (kb, sb, _)| (*sa, *ka).cmp(&(*sb, *kb)));
    Ok(out)
}

pub(super) fn collect_world_chunk_payloads_from_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
    bounds_filter: Option<Aabb4i>,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    out: &mut Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };

    if let Some(filter_bounds) = bounds_filter {
        if !node.bounds_aabb4().intersects(&filter_bounds) {
            return Ok(());
        }
    }

    let node_bounds = node.bounds_aabb4();
    let leaf_scale = node.scale_exp;

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_world_chunk_payloads_from_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    region_filter,
                    include_matches,
                    bounds_filter,
                    payload_cache,
                    out,
                )?;
            }
        }
        IndexNodeKind::LeafEmpty => {
            // After patch saves carve dirty regions out of Uniform/Empty
            // leaves, the residual pieces may have non-lattice-aligned bounds.
            // Use the coarsest scale where the bounds round-trip exactly.
            let emit_scale = coarsest_lattice_aligned_scale(&node_bounds, leaf_scale);
            let (lat_min, lat_max) = node_bounds.to_chunk_lattice_bounds(emit_scale);
            for_each_chunk_in_bounds(lat_min, lat_max, |chunk| {
                let chunk_key = chunk_key_from_lattice(chunk, emit_scale);
                let in_bounds = bounds_filter
                    .map(|filter| filter.contains_chunk_world_min(chunk_key))
                    .unwrap_or(true);
                if in_bounds && include_chunk_for_filter(chunk, region_filter, include_matches) {
                    out.push((chunk_key, emit_scale, ResolvedChunkPayload::empty()));
                }
            });
        }
        IndexNodeKind::LeafUniform { block } => {
            let emit_scale = coarsest_lattice_aligned_scale(&node_bounds, leaf_scale);
            let (lat_min, lat_max) = node_bounds.to_chunk_lattice_bounds(emit_scale);
            for_each_chunk_in_bounds(lat_min, lat_max, |chunk| {
                let chunk_key = chunk_key_from_lattice(chunk, emit_scale);
                let in_bounds = bounds_filter
                    .map(|filter| filter.contains_chunk_world_min(chunk_key))
                    .unwrap_or(true);
                if in_bounds && include_chunk_for_filter(chunk, region_filter, include_matches) {
                    out.push((
                        chunk_key,
                        emit_scale,
                        ResolvedChunkPayload::uniform(block.clone()),
                    ));
                }
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
            let bounds = Aabb4::from_lattice_bounds(
                blob.volume_min_chunk,
                blob.volume_max_chunk,
                blob.scale_exp,
            );
            if !bounds.is_valid() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array bounds",
                ));
            }

            let palette_len = blob.payload_palette.len().max(1);
            let dummy_palette = vec![FieldChunkPayload::Empty; palette_len];
            let block_palette = blob.block_palette.clone();
            let chunk_array_data = ChunkArrayData {
                bounds,
                scale_exp: blob.scale_exp,
                chunk_palette: dummy_palette,
                index_codec: blob.index_codec,
                index_data: blob.index_data.clone(),
                default_chunk_idx: blob.default_palette_index,
                block_palette: block_palette.clone(),
            };
            let indices = chunk_array_data
                .decode_dense_indices()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let extents = bounds
                .chunk_extents_at_scale(blob.scale_exp)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid chunk array extents during decode",
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

            let (lat_min, lat_max) = bounds.to_chunk_lattice_bounds(blob.scale_exp);
            for w in lat_min[3]..=lat_max[3] {
                for z in lat_min[2]..=lat_max[2] {
                    for y in lat_min[1]..=lat_max[1] {
                        for x in lat_min[0]..=lat_max[0] {
                            let chunk = [x, y, z, w];
                            let chunk_key = chunk_key_from_lattice(chunk, blob.scale_exp);
                            if bounds_filter
                                .map(|filter| !filter.contains_chunk_world_min(chunk_key))
                                .unwrap_or(false)
                            {
                                continue;
                            }
                            if !include_chunk_for_filter(chunk, region_filter, include_matches) {
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
                                    "chunk array decoded index out of bounds",
                                )
                            })?;
                            let payload =
                                palette.get(palette_idx as usize).cloned().ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "chunk array palette index out of bounds",
                                    )
                                })?;
                            out.push((
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
