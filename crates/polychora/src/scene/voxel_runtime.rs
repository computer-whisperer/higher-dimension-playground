use super::*;
use polychora::shared::chunk_payload::{ChunkPayload, ResolvedChunkPayload};
use polychora::shared::spatial::Aabb4i;

impl Scene {
    pub(super) fn summarize_chunk_payload_compact(
        resolved: Option<&ResolvedChunkPayload>,
    ) -> String {
        let Some(resolved) = resolved else {
            return "None".to_string();
        };
        match &resolved.payload {
            ChunkPayload::Empty => "Empty".to_string(),
            ChunkPayload::Virgin => "Virgin".to_string(),
            ChunkPayload::Uniform(idx) => {
                let block = resolved.block_palette.get(*idx as usize);
                match block {
                    Some(b) if b.is_air() => "Uniform(air)".to_string(),
                    Some(b) => format!("Uniform({}:{})", b.namespace, b.block_type),
                    None => format!("Uniform(idx={})", idx),
                }
            }
            ChunkPayload::Dense16 { materials } => {
                let non_empty = materials
                    .iter()
                    .filter(|idx| {
                        resolved
                            .block_palette
                            .get(**idx as usize)
                            .map(|b| !b.is_air())
                            .unwrap_or(false)
                    })
                    .count();
                format!("Dense16(nz={non_empty})")
            }
            ChunkPayload::PalettePacked {
                palette, bit_width, ..
            } => format!(
                "PalettePacked(palette={},bits={})",
                palette.len(),
                bit_width
            ),
        }
    }

    pub(super) fn summarize_chunk_payload_list_compact(
        payloads: &[ResolvedChunkPayload],
    ) -> String {
        if payloads.is_empty() {
            return "[]".to_string();
        }
        let mut out = String::from("[");
        for (idx, resolved) in payloads.iter().take(3).enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            out.push_str(&Self::summarize_chunk_payload_compact(Some(resolved)));
        }
        if payloads.len() > 3 {
            out.push_str(", ...");
        }
        out.push(']');
        out
    }

    pub(super) fn voxel_frame_root_is_valid(&self) -> bool {
        let root = self.active_config.frame_data.region_bvh_root_index;
        root != higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE
            && (root as usize) < self.active_config.frame_data.region_bvh_nodes.len()
    }

    pub(super) fn log_voxel_snapshot_rebuild(
        &self,
        bounds: Aabb4i,
        reason: &str,
        applied_deltas: usize,
        pending_deltas: usize,
        frame_root: u32,
        cpu_root: u32,
    ) {
        eprintln!(
            "[vte-voxel-snapshot-rebuild] reason={} bounds={:?}->{:?} applied_deltas={} pending_deltas={} frame_root={} cpu_root={}",
            reason,
            bounds.min,
            bounds.max,
            applied_deltas,
            pending_deltas,
            frame_root,
            cpu_root,
        );
    }

    pub(super) fn camera_chunk_key(cam_pos: [f32; 4]) -> [i32; 4] {
        let cs = CHUNK_SIZE as i32;
        [
            (cam_pos[0].floor() as i32).div_euclid(cs),
            (cam_pos[1].floor() as i32).div_euclid(cs),
            (cam_pos[2].floor() as i32).div_euclid(cs),
            (cam_pos[3].floor() as i32).div_euclid(cs),
        ]
    }
}
