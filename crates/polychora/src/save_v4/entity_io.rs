//! Entity persistence: loading and saving entity records from/to the save-v4 index.

use super::*;

pub fn load_entities_for_regions(
    root: &Path,
    regions: &HashSet<[i32; 4]>,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    materialize_entities_from_index_filtered(root, &index, Some(regions), true)
}

pub(super) fn materialize_entities_from_index(
    root: &Path,
    index: &IndexPayload,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let Some(entity_root) = index.entity_root_node_id else {
        return Ok(Vec::new());
    };

    let node_by_id = build_node_lookup(index);
    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();
    apply_entity_node(root, &node_by_id, entity_root, &mut entities_by_id)?;
    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);
    Ok(entities)
}

pub(super) fn materialize_entities_from_index_filtered(
    root: &Path,
    index: &IndexPayload,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let Some(entity_root) = index.entity_root_node_id else {
        return Ok(Vec::new());
    };
    let node_by_id = build_node_lookup(index);
    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();
    collect_entities_from_node_filtered(
        root,
        &node_by_id,
        entity_root,
        region_filter,
        include_matches,
        &mut entities_by_id,
    )?;

    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);
    Ok(entities)
}

pub(super) fn collect_entities_from_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
    entities_by_id: &mut HashMap<u64, PersistedEntityRecord>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing entity node id {node_id}"),
        ));
    };

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_entities_from_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    region_filter,
                    include_matches,
                    entities_by_id,
                )?;
            }
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_ENTITY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "entity subtree leaf references non-entity blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob: EntityBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            for entity in blob.entities {
                let region = region_from_chunk(
                    chunk_from_world_position(entity.entity.pose.position),
                    DEFAULT_REGION_CHUNK_EDGE,
                );
                if include_region_for_filter(region, region_filter, include_matches) {
                    entities_by_id.insert(entity.entity_id, entity);
                }
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {}
    }

    Ok(())
}

pub(super) fn apply_entity_node(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    entities_by_id: &mut HashMap<u64, PersistedEntityRecord>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing entity node id {node_id}"),
        ));
    };

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                apply_entity_node(root, node_by_id, *child_id, entities_by_id)?;
            }
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_ENTITY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "entity subtree leaf references non-entity blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob: EntityBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            for entity in blob.entities {
                entities_by_id.insert(entity.entity_id, entity);
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {}
    }

    Ok(())
}

pub(super) fn resolve_entities_for_save(
    loaded_entities: Vec<PersistedEntityRecord>,
    current_entities: &[PersistedEntityRecord],
    dirty_entity_regions: &HashSet<[i32; 4]>,
    force_full_entities: bool,
) -> Vec<PersistedEntityRecord> {
    if force_full_entities {
        let mut out = current_entities.to_vec();
        out.sort_unstable_by_key(|entity| entity.entity_id);
        return out;
    }

    if dirty_entity_regions.is_empty() {
        let mut out = loaded_entities;
        out.sort_unstable_by_key(|entity| entity.entity_id);
        return out;
    }

    let mut merged = HashMap::<u64, PersistedEntityRecord>::new();
    for entity in loaded_entities {
        let chunk = chunk_from_world_position(entity.entity.pose.position);
        let region = region_from_chunk(chunk, DEFAULT_REGION_CHUNK_EDGE);
        if !dirty_entity_regions.contains(&region) {
            merged.insert(entity.entity_id, entity);
        }
    }
    for entity in current_entities {
        let chunk = chunk_from_world_position(entity.entity.pose.position);
        let region = region_from_chunk(chunk, DEFAULT_REGION_CHUNK_EDGE);
        if dirty_entity_regions.contains(&region) {
            merged.insert(entity.entity_id, entity.clone());
        }
    }

    let mut out: Vec<PersistedEntityRecord> = merged.into_values().collect();
    out.sort_unstable_by_key(|entity| entity.entity_id);
    out
}
