use crate::migration::legacy_voxel::RegionChunkWorld;
use crate::migration::legacy_world_io::{load_world, save_world};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyTrimResult {
    pub dropped_overrides: usize,
    pub non_empty_chunks: usize,
}

pub fn validate_chunk_bounds(min_chunk: [i32; 4], max_chunk: [i32; 4]) -> Result<(), String> {
    for axis in 0..4 {
        if min_chunk[axis] > max_chunk[axis] {
            return Err(format!(
                "invalid keep chunk bounds: min axis {} ({}) exceeds max ({})",
                axis, min_chunk[axis], max_chunk[axis]
            ));
        }
    }
    Ok(())
}

pub fn drop_overrides_outside_chunk_bounds(
    world: &mut RegionChunkWorld,
    min_chunk: [i32; 4],
    max_chunk: [i32; 4],
) -> usize {
    let positions: Vec<[i32; 4]> = world.chunks.keys().copied().collect();
    let mut dropped = 0usize;
    for pos in positions {
        let outside = pos[0] < min_chunk[0]
            || pos[0] > max_chunk[0]
            || pos[1] < min_chunk[1]
            || pos[1] > max_chunk[1]
            || pos[2] < min_chunk[2]
            || pos[2] > max_chunk[2]
            || pos[3] < min_chunk[3]
            || pos[3] > max_chunk[3];
        if outside && world.remove_chunk_override(pos) {
            dropped = dropped.saturating_add(1);
        }
    }
    dropped
}

pub fn trim_legacy_world_keep_bounds(
    input: &Path,
    output: &Path,
    min_chunk: [i32; 4],
    max_chunk: [i32; 4],
) -> io::Result<LegacyTrimResult> {
    validate_chunk_bounds(min_chunk, max_chunk)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;

    let file = File::open(input)?;
    let mut reader = BufReader::new(file);
    let mut world = load_world(&mut reader)?;
    let dropped_overrides = drop_overrides_outside_chunk_bounds(&mut world, min_chunk, max_chunk);

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    save_world(&world, &mut writer)?;
    writer.flush()?;

    Ok(LegacyTrimResult {
        dropped_overrides,
        non_empty_chunks: world.non_empty_chunk_count(),
    })
}
