use common::MatN;
use std::collections::HashMap;

use super::chunk::Chunk;
use super::world::BaseWorldKind;
use super::{ChunkPos, Face4D, VoxelType, CHUNK_SIZE};
use polychora::shared::voxel::world_to_chunk;
use polychora::shared::worldfield::RegionChunkTree;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

/// Compact per-voxel surface data (18 bytes).
pub struct SurfaceVoxel {
    pub position: [i32; 4],
    pub material: u8,
    /// Bitmask of exposed faces (bits match Face4D::cell_id()).
    pub exposed_faces: u8,
}

/// Per-chunk metadata for fast culling.
pub struct ChunkSurface {
    pub aabb_min: [i32; 4],
    pub aabb_max: [i32; 4],
    /// Range [start..end) into SurfaceData::voxels.
    pub voxel_start: u32,
    pub voxel_end: u32,
    /// Union of all exposed_faces in this chunk.
    pub exposed_faces_union: u8,
}

/// Cached surface topology — rebuilt only when the world changes.
pub struct SurfaceData {
    pub chunks: Vec<ChunkSurface>,
    pub voxels: Vec<SurfaceVoxel>,
}

fn base_chunk_for_pos(
    base_kind: BaseWorldKind,
    flat_floor_chunk: &Chunk,
    pos: ChunkPos,
) -> Option<&Chunk> {
    match base_kind {
        BaseWorldKind::Empty => None,
        BaseWorldKind::FlatFloor { .. } if pos.y == FLAT_FLOOR_CHUNK_Y => Some(flat_floor_chunk),
        BaseWorldKind::FlatFloor { .. } => None,
    }
}

fn world_voxel_at(
    explicit_chunks: &HashMap<ChunkPos, Chunk>,
    base_kind: BaseWorldKind,
    flat_floor_chunk: &Chunk,
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
) -> VoxelType {
    let (chunk_pos, idx) = world_to_chunk(wx, wy, wz, ww);
    if let Some(chunk) = explicit_chunks.get(&chunk_pos) {
        return chunk.voxels[idx];
    }
    base_chunk_for_pos(base_kind, flat_floor_chunk, chunk_pos)
        .map(|chunk| chunk.voxels[idx])
        .unwrap_or(VoxelType::AIR)
}

/// Scan the chunk tree and extract surface voxels + chunk metadata.
pub fn extract_surfaces(
    world_tree: &RegionChunkTree,
    base_kind: BaseWorldKind,
    flat_floor_chunk: &Chunk,
) -> SurfaceData {
    let mut chunks = Vec::new();
    let mut voxels = Vec::new();
    let mut explicit_chunks = HashMap::<ChunkPos, Chunk>::new();

    for (key, payload) in world_tree.collect_chunks() {
        let chunk_pos = key.to_chunk_pos();
        match payload.to_voxel_chunk() {
            Ok(chunk) => {
                explicit_chunks.insert(chunk_pos, chunk);
            }
            Err(error) => {
                eprintln!(
                    "Skipping malformed chunk payload during surface extraction at ({}, {}, {}, {}): {}",
                    chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w, error
                );
            }
        }
    }

    for (&chunk_pos, chunk) in &explicit_chunks {
        if chunk.is_empty() {
            continue;
        }

        let base_x = chunk_pos.x * CHUNK_SIZE as i32;
        let base_y = chunk_pos.y * CHUNK_SIZE as i32;
        let base_z = chunk_pos.z * CHUNK_SIZE as i32;
        let base_w = chunk_pos.w * CHUNK_SIZE as i32;

        let voxel_start = voxels.len() as u32;
        let mut faces_union: u8 = 0;
        let mut aabb_min = [i32::MAX; 4];
        let mut aabb_max = [i32::MIN; 4];

        for lw in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let voxel = chunk.get(lx, ly, lz, lw);
                        if voxel.is_air() {
                            continue;
                        }

                        let wx = base_x + lx as i32;
                        let wy = base_y + ly as i32;
                        let wz = base_z + lz as i32;
                        let ww = base_w + lw as i32;

                        let mut exposed: u8 = 0;
                        for face in Face4D::ALL {
                            let [dx, dy, dz, dw] = face.neighbor_offset();
                            let neighbor = world_voxel_at(
                                &explicit_chunks,
                                base_kind,
                                flat_floor_chunk,
                                wx + dx,
                                wy + dy,
                                wz + dz,
                                ww + dw,
                            );
                            if neighbor.is_air() {
                                exposed |= 1 << face.cell_id();
                            }
                        }

                        if exposed == 0 {
                            continue;
                        }

                        let pos = [wx, wy, wz, ww];
                        for d in 0..4 {
                            aabb_min[d] = aabb_min[d].min(pos[d]);
                            aabb_max[d] = aabb_max[d].max(pos[d] + 1);
                        }
                        faces_union |= exposed;

                        voxels.push(SurfaceVoxel {
                            position: pos,
                            material: voxel.0,
                            exposed_faces: exposed,
                        });
                    }
                }
            }
        }

        let voxel_end = voxels.len() as u32;
        if voxel_start == voxel_end {
            continue;
        }

        chunks.push(ChunkSurface {
            aabb_min,
            aabb_max,
            voxel_start,
            voxel_end,
            exposed_faces_union: faces_union,
        });
    }

    SurfaceData { chunks, voxels }
}

/// Per-frame: cull chunks and voxels, write surviving ModelInstances into `out`.
pub fn cull_and_build(
    surface: &SurfaceData,
    cam_pos: [f32; 4],
    render_dist: f32,
    out: &mut Vec<common::ModelInstance>,
) {
    let render_dist_sq = render_dist * render_dist;

    for cs in &surface.chunks {
        // --- Chunk-level distance cull ---
        let mut dist_sq: f32 = 0.0;
        for d in 0..4 {
            let lo = cs.aabb_min[d] as f32;
            let hi = cs.aabb_max[d] as f32;
            let c = cam_pos[d];
            if c < lo {
                dist_sq += (lo - c) * (lo - c);
            } else if c > hi {
                dist_sq += (c - hi) * (c - hi);
            }
        }
        if dist_sq > render_dist_sq {
            continue;
        }

        // --- Chunk-level backface cull ---
        // Compute which face directions are definitely hidden for the whole chunk.
        let chunk_hidden = chunk_hidden_mask(cam_pos, cs.aabb_min, cs.aabb_max);
        if cs.exposed_faces_union & !chunk_hidden == 0 {
            continue;
        }

        // --- Per-voxel cull ---
        for vi in cs.voxel_start..cs.voxel_end {
            let sv = &surface.voxels[vi as usize];

            // Use world-space backface test only to skip entire voxel.
            // Don't zero individual face materials — the 4D view rotation
            // means world-space "backfacing" faces may be visible after projection.
            let visible = visible_faces_mask(cam_pos, sv.position) & sv.exposed_faces;
            if visible == 0 {
                continue;
            }

            let mat = sv.material as u32;
            let mut cell_material_ids = [0u32; 8];
            for bit in 0..8u8 {
                if sv.exposed_faces & (1 << bit) != 0 {
                    cell_material_ids[bit as usize] = mat;
                }
            }

            let model_transform = translation_matn5(
                sv.position[0] as f32,
                sv.position[1] as f32,
                sv.position[2] as f32,
                sv.position[3] as f32,
            );

            out.push(common::ModelInstance {
                model_transform,
                cell_material_ids,
            });
        }
    }
}

/// Bitmask of face directions that are definitely hidden for a whole chunk AABB.
/// A positive-axis face is hidden if the camera is below the AABB min on that axis;
/// a negative-axis face is hidden if the camera is above the AABB max.
fn chunk_hidden_mask(cam: [f32; 4], aabb_min: [i32; 4], aabb_max: [i32; 4]) -> u8 {
    let mut mask: u8 = 0;
    // Face4D ordering: NegW=0, NegZ=1, NegY=2, NegX=3, PosW=4, PosZ=5, PosY=6, PosX=7
    // axis_index: X=0, Y=1, Z=2, W=3
    const NEG_BITS: [u8; 4] = [3, 2, 1, 0]; // NegX, NegY, NegZ, NegW
    const POS_BITS: [u8; 4] = [7, 6, 5, 4]; // PosX, PosY, PosZ, PosW

    for d in 0..4 {
        let lo = aabb_min[d] as f32;
        let hi = aabb_max[d] as f32;
        // Camera at or below chunk min → can't see any PosAxis face
        if cam[d] <= lo {
            mask |= 1 << POS_BITS[d];
        }
        // Camera at or above chunk max → can't see any NegAxis face
        if cam[d] >= hi {
            mask |= 1 << NEG_BITS[d];
        }
    }
    mask
}

/// Bitmask of which faces of a voxel at `pos` are visible from `cam`.
/// A face is visible iff the camera is on the outward side:
///   PosX: cam[0] > pos[0] + 1
///   NegX: cam[0] < pos[0]
///   (same pattern for Y, Z, W)
fn visible_faces_mask(cam: [f32; 4], pos: [i32; 4]) -> u8 {
    let mut mask: u8 = 0;
    const NEG_BITS: [u8; 4] = [3, 2, 1, 0]; // NegX, NegY, NegZ, NegW
    const POS_BITS: [u8; 4] = [7, 6, 5, 4]; // PosX, PosY, PosZ, PosW

    for d in 0..4 {
        let lo = pos[d] as f32;
        let hi = lo + 1.0;
        if cam[d] >= hi {
            mask |= 1 << POS_BITS[d];
        }
        if cam[d] <= lo {
            mask |= 1 << NEG_BITS[d];
        }
    }
    mask
}

/// Build a 5×5 translation matrix directly, avoiding ndarray allocation.
fn translation_matn5(x: f32, y: f32, z: f32, w: f32) -> MatN<5> {
    let mut m = MatN::<5>::identity();
    m[[0, 4]] = x;
    m[[1, 4]] = y;
    m[[2, 4]] = z;
    m[[3, 4]] = w;
    m
}

/// Report mesh statistics: (num_instances, num_tetrahedra).
pub fn mesh_stats(instances: &[common::ModelInstance]) -> (usize, usize) {
    let num_instances = instances.len();
    let num_tets: usize = instances
        .iter()
        .map(|inst| inst.cell_material_ids.iter().filter(|&&id| id != 0).count() * 6)
        .sum();
    (num_instances, num_tets)
}
