use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use polychora_plugin_api::procgen_abi::{
    ProcgenGenerateInput, ProcgenGenerateOutput, ProcgenPrepareInput, ProcgenPrepareOutput,
    StructureDeclaration,
};
use polychora_plugin_api::region_tree::{
    Aabb4, BlockData, ChunkArrayData, ChunkCoord, ChunkPayload, RegionNodeKind, RegionTreeCore,
    TesseractOrientation,
};
use serde::Deserialize;

const CHUNK_SIZE: usize = 8;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

fn default_spawn_weight() -> u32 {
    1
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureBlueprintFile {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_spawn_weight")]
    spawn_weight: u32,
    #[serde(default)]
    ground_offset_y: i32,
    #[serde(default)]
    fills: Vec<StructureFill>,
    #[serde(default)]
    voxels: Vec<StructureVoxel>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureFill {
    min: [i32; 4],
    size: [i32; 4],
    block: BlockData,
    #[serde(skip)]
    palette_idx: u16,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureVoxel {
    offset: [i32; 4],
    block: BlockData,
    #[serde(skip)]
    palette_idx: u16,
}

struct StructureBlueprint {
    id: u32,
    name: String,
    spawn_weight: u32,
    ground_offset_y: i32,
    fills: Vec<StructureFill>,
    voxels: Vec<StructureVoxel>,
    min_offset: [i32; 4],
    max_offset: [i32; 4],
    block_palette: Vec<BlockData>,
}

// 4! = 24 permutations of [X, Y, Z, W] axes.
const PERMUTATIONS: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

fn apply_orientation(orientation: TesseractOrientation, offset: [i32; 4]) -> [i32; 4] {
    let perm = &PERMUTATIONS[orientation.permutation_index() as usize];
    let signs = orientation.sign_bits();
    let mut result = [0i32; 4];
    for i in 0..4 {
        let val = offset[perm[i] as usize];
        result[i] = if signs & (1 << i) != 0 { -val } else { val };
    }
    result
}

fn intern_block(palette: &mut Vec<BlockData>, block: &BlockData) -> u16 {
    if block.is_air() {
        return 0;
    }
    if let Some(idx) = palette.iter().position(|b| b == block) {
        return idx as u16;
    }
    let idx = palette.len() as u16;
    palette.push(block.clone());
    idx
}

impl StructureBlueprint {
    fn from_json(id: u32, json: &[u8]) -> Self {
        let parsed: StructureBlueprintFile =
            serde_json::from_slice(json).expect("invalid structure blueprint");

        let mut min_offset = [0i32; 4];
        let mut max_offset = [0i32; 4];
        let mut initialized = false;

        let mut include_point = |point: [i32; 4]| {
            if !initialized {
                min_offset = point;
                max_offset = point;
                initialized = true;
                return;
            }
            for axis in 0..4 {
                min_offset[axis] = min_offset[axis].min(point[axis]);
                max_offset[axis] = max_offset[axis].max(point[axis]);
            }
        };

        for fill in &parsed.fills {
            if fill.block.is_air() {
                continue;
            }
            let fill_max = [
                fill.min[0] + fill.size[0] - 1,
                fill.min[1] + fill.size[1] - 1,
                fill.min[2] + fill.size[2] - 1,
                fill.min[3] + fill.size[3] - 1,
            ];
            include_point(fill.min);
            include_point(fill_max);
        }

        for voxel in &parsed.voxels {
            if voxel.block.is_air() {
                continue;
            }
            include_point(voxel.offset);
        }

        let mut block_palette = vec![BlockData::AIR];
        let mut fills = parsed.fills;
        let mut voxels = parsed.voxels;
        for fill in &mut fills {
            fill.palette_idx = intern_block(&mut block_palette, &fill.block);
        }
        for voxel in &mut voxels {
            voxel.palette_idx = intern_block(&mut block_palette, &voxel.block);
        }

        Self {
            id,
            name: parsed.name,
            spawn_weight: parsed.spawn_weight,
            ground_offset_y: parsed.ground_offset_y,
            fills,
            voxels,
            min_offset,
            max_offset,
            block_palette,
        }
    }

    fn oriented_world_bounds(
        &self,
        origin: [i32; 4],
        orientation: TesseractOrientation,
    ) -> ([i32; 4], [i32; 4]) {
        let x_vals = [self.min_offset[0], self.max_offset[0]];
        let y_vals = [self.min_offset[1], self.max_offset[1]];
        let z_vals = [self.min_offset[2], self.max_offset[2]];
        let w_vals = [self.min_offset[3], self.max_offset[3]];
        let mut min = [i32::MAX; 4];
        let mut max = [i32::MIN; 4];
        for x in x_vals {
            for y in y_vals {
                for z in z_vals {
                    for w in w_vals {
                        let rotated = apply_orientation(orientation, [x, y, z, w]);
                        for axis in 0..4 {
                            min[axis] = min[axis].min(origin[axis] + rotated[axis]);
                            max[axis] = max[axis].max(origin[axis] + rotated[axis]);
                        }
                    }
                }
            }
        }
        (min, max)
    }

    fn rasterize(
        &self,
        origin: [i32; 4],
        orientation: TesseractOrientation,
    ) -> RegionTreeCore {
        let (world_min, world_max) = self.oriented_world_bounds(origin, orientation);
        let cs = CHUNK_SIZE as i32;

        let chunk_min = [
            world_min[0].div_euclid(cs),
            world_min[1].div_euclid(cs),
            world_min[2].div_euclid(cs),
            world_min[3].div_euclid(cs),
        ];
        let chunk_max = [
            world_max[0].div_euclid(cs),
            world_max[1].div_euclid(cs),
            world_max[2].div_euclid(cs),
            world_max[3].div_euclid(cs),
        ];

        let bounds = aabb4_from_chunk_lattice(chunk_min, chunk_max);
        let dims = [
            (chunk_max[0] - chunk_min[0] + 1) as usize,
            (chunk_max[1] - chunk_min[1] + 1) as usize,
            (chunk_max[2] - chunk_min[2] + 1) as usize,
            (chunk_max[3] - chunk_min[3] + 1) as usize,
        ];

        let total_chunks = dims[0] * dims[1] * dims[2] * dims[3];
        let mut chunk_data: Vec<Vec<u16>> = Vec::with_capacity(total_chunks);
        for _ in 0..total_chunks {
            chunk_data.push(vec![0u16; CHUNK_VOLUME]);
        }

        // Rasterize fills
        for fill in &self.fills {
            if fill.palette_idx == 0 {
                continue;
            }
            let fill_min_rot = apply_orientation(orientation, fill.min);
            let fill_max_local = [
                fill.min[0] + fill.size[0] - 1,
                fill.min[1] + fill.size[1] - 1,
                fill.min[2] + fill.size[2] - 1,
                fill.min[3] + fill.size[3] - 1,
            ];
            let fill_max_rot = apply_orientation(orientation, fill_max_local);
            let fill_world_min = [
                origin[0] + fill_min_rot[0].min(fill_max_rot[0]),
                origin[1] + fill_min_rot[1].min(fill_max_rot[1]),
                origin[2] + fill_min_rot[2].min(fill_max_rot[2]),
                origin[3] + fill_min_rot[3].min(fill_max_rot[3]),
            ];
            let fill_world_max = [
                origin[0] + fill_min_rot[0].max(fill_max_rot[0]),
                origin[1] + fill_min_rot[1].max(fill_max_rot[1]),
                origin[2] + fill_min_rot[2].max(fill_max_rot[2]),
                origin[3] + fill_min_rot[3].max(fill_max_rot[3]),
            ];

            for wx in fill_world_min[0]..=fill_world_max[0] {
                for wy in fill_world_min[1]..=fill_world_max[1] {
                    for wz in fill_world_min[2]..=fill_world_max[2] {
                        for ww in fill_world_min[3]..=fill_world_max[3] {
                            set_voxel(
                                &mut chunk_data,
                                dims,
                                chunk_min,
                                [wx, wy, wz, ww],
                                fill.palette_idx,
                            );
                        }
                    }
                }
            }
        }

        // Rasterize individual voxels
        for voxel in &self.voxels {
            if voxel.palette_idx == 0 {
                continue;
            }
            let rotated = apply_orientation(orientation, voxel.offset);
            let world_pos = [
                origin[0] + rotated[0],
                origin[1] + rotated[1],
                origin[2] + rotated[2],
                origin[3] + rotated[3],
            ];
            set_voxel(
                &mut chunk_data,
                dims,
                chunk_min,
                world_pos,
                voxel.palette_idx,
            );
        }

        // Convert to ChunkArrayData — deduplicate by linear scan to avoid
        // cloning full chunk vectors into a BTreeMap.
        let mut chunk_palette: Vec<ChunkPayload> = vec![ChunkPayload::Empty];
        let empty_chunk = vec![0u16; CHUNK_VOLUME];
        let mut dense_indices = Vec::with_capacity(total_chunks);

        for chunk in chunk_data.drain(..) {
            if chunk == empty_chunk {
                dense_indices.push(0u16);
                continue;
            }
            let mut found = false;
            for (idx, existing) in chunk_palette.iter().enumerate() {
                if let ChunkPayload::Dense16 { materials } = existing {
                    if materials == &chunk {
                        dense_indices.push(idx as u16);
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                let idx = chunk_palette.len() as u16;
                chunk_palette.push(ChunkPayload::Dense16 { materials: chunk });
                dense_indices.push(idx);
            }
        }

        // Check if everything is empty
        if dense_indices.iter().all(|&idx| idx == 0) {
            return RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            };
        }

        RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(ChunkArrayData {
                bounds,
                scale_exp: 0,
                chunk_palette,
                dense_indices,
                block_palette: self.block_palette.clone(),
            }),
            generator_version_hash: 0,
        }
    }
}

fn set_voxel(
    chunk_data: &mut [Vec<u16>],
    dims: [usize; 4],
    chunk_min: [i32; 4],
    world_pos: [i32; 4],
    material: u16,
) {
    let cs = CHUNK_SIZE as i32;
    let cx = world_pos[0].div_euclid(cs) - chunk_min[0];
    let cy = world_pos[1].div_euclid(cs) - chunk_min[1];
    let cz = world_pos[2].div_euclid(cs) - chunk_min[2];
    let cw = world_pos[3].div_euclid(cs) - chunk_min[3];
    if cx < 0 || cy < 0 || cz < 0 || cw < 0 {
        return;
    }
    let (cx, cy, cz, cw) = (cx as usize, cy as usize, cz as usize, cw as usize);
    if cx >= dims[0] || cy >= dims[1] || cz >= dims[2] || cw >= dims[3] {
        return;
    }

    let chunk_idx = cx + dims[0] * (cy + dims[1] * (cz + dims[2] * cw));
    let lx = world_pos[0].rem_euclid(cs) as usize;
    let ly = world_pos[1].rem_euclid(cs) as usize;
    let lz = world_pos[2].rem_euclid(cs) as usize;
    let lw = world_pos[3].rem_euclid(cs) as usize;
    let voxel_idx =
        lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + lz * CHUNK_SIZE * CHUNK_SIZE + ly * CHUNK_SIZE + lx;
    chunk_data[chunk_idx][voxel_idx] = material;
}

pub(crate) fn aabb4_from_chunk_lattice(chunk_min: [i32; 4], chunk_max: [i32; 4]) -> Aabb4 {
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    Aabb4 {
        min: [
            ChunkCoord::from_num(chunk_min[0]).saturating_mul(cs),
            ChunkCoord::from_num(chunk_min[1]).saturating_mul(cs),
            ChunkCoord::from_num(chunk_min[2]).saturating_mul(cs),
            ChunkCoord::from_num(chunk_min[3]).saturating_mul(cs),
        ],
        max: [
            ChunkCoord::from_num(chunk_max[0] + 1).saturating_mul(cs),
            ChunkCoord::from_num(chunk_max[1] + 1).saturating_mul(cs),
            ChunkCoord::from_num(chunk_max[2] + 1).saturating_mul(cs),
            ChunkCoord::from_num(chunk_max[3] + 1).saturating_mul(cs),
        ],
    }
}

pub struct StructureGenerator {
    blueprints: Vec<StructureBlueprint>,
}

impl StructureGenerator {
    pub fn new() -> Self {
        let sources: [(&str, u32, &[u8]); 26] = [
            ("cross_shrine", 1, include_bytes!("../../assets/structures/cross_shrine.json")),
            ("hyper_arch", 2, include_bytes!("../../assets/structures/hyper_arch.json")),
            ("duoprism_exchange", 3, include_bytes!("../../assets/structures/duoprism_exchange.json")),
            ("tetra_spire", 4, include_bytes!("../../assets/structures/tetra_spire.json")),
            ("sky_bridge", 5, include_bytes!("../../assets/structures/sky_bridge.json")),
            ("ring_keep", 6, include_bytes!("../../assets/structures/ring_keep.json")),
            ("terraced_pyramid", 7, include_bytes!("../../assets/structures/terraced_pyramid.json")),
            ("cathedral_spine", 8, include_bytes!("../../assets/structures/cathedral_spine.json")),
            ("lattice_hall", 9, include_bytes!("../../assets/structures/lattice_hall.json")),
            ("shard_garden", 10, include_bytes!("../../assets/structures/shard_garden.json")),
            ("portal_axis", 11, include_bytes!("../../assets/structures/portal_axis.json")),
            ("resonance_forge", 12, include_bytes!("../../assets/structures/resonance_forge.json")),
            ("void_colonnade", 13, include_bytes!("../../assets/structures/void_colonnade.json")),
            ("lumen_orrery", 14, include_bytes!("../../assets/structures/lumen_orrery.json")),
            ("clifford_atrium", 15, include_bytes!("../../assets/structures/clifford_atrium.json")),
            ("braided_transit", 16, include_bytes!("../../assets/structures/braided_transit.json")),
            ("phase_ladder", 17, include_bytes!("../../assets/structures/phase_ladder.json")),
            ("phase_cloister", 18, include_bytes!("../../assets/structures/phase_cloister.json")),
            ("pentachord_spindle", 19, include_bytes!("../../assets/structures/pentachord_spindle.json")),
            ("orthoplex_nexus", 20, include_bytes!("../../assets/structures/orthoplex_nexus.json")),
            ("ore_cairn", 21, include_bytes!("../../assets/structures/ore_cairn.json")),
            ("frozen_spire", 22, include_bytes!("../../assets/structures/frozen_spire.json")),
            ("moss_ruins", 23, include_bytes!("../../assets/structures/moss_ruins.json")),
            ("sandstone_obelisk", 24, include_bytes!("../../assets/structures/sandstone_obelisk.json")),
            ("prism_fountain", 25, include_bytes!("../../assets/structures/prism_fountain.json")),
            ("hypercube_frame", 26, include_bytes!("../../assets/structures/hypercube_frame.json")),
        ];

        let blueprints: Vec<_> = sources
            .iter()
            .map(|(_, id, json)| StructureBlueprint::from_json(*id, json))
            .collect();

        Self { blueprints }
    }

    pub fn declarations(&self) -> Vec<StructureDeclaration> {
        self.blueprints
            .iter()
            .map(|bp| StructureDeclaration {
                id: bp.id,
                name: bp.name.clone(),
                spawn_weight: bp.spawn_weight,
            })
            .collect()
    }

    pub fn prepare(&self, input: &ProcgenPrepareInput) -> ProcgenPrepareOutput {
        let bp = self
            .blueprints
            .iter()
            .find(|bp| bp.id == input.structure_id)
            .expect("unknown structure_id");

        let orientation = input.orientation;
        // The host passes origin with Y=0; we add the blueprint's ground_offset_y here.
        let origin = [
            input.origin[0],
            input.origin[1] + bp.ground_offset_y,
            input.origin[2],
            input.origin[3],
        ];
        let (world_min, world_max) = bp.oriented_world_bounds(origin, orientation);
        let cs = CHUNK_SIZE as i32;
        let bounds = aabb4_from_chunk_lattice(
            [
                world_min[0].div_euclid(cs),
                world_min[1].div_euclid(cs),
                world_min[2].div_euclid(cs),
                world_min[3].div_euclid(cs),
            ],
            [
                world_max[0].div_euclid(cs),
                world_max[1].div_euclid(cs),
                world_max[2].div_euclid(cs),
                world_max[3].div_euclid(cs),
            ],
        );

        ProcgenPrepareOutput {
            bounds,
            state: Vec::new(),
        }
    }

    pub fn generate(&self, input: &ProcgenGenerateInput) -> ProcgenGenerateOutput {
        let bp = self
            .blueprints
            .iter()
            .find(|bp| bp.id == input.structure_id)
            .expect("unknown structure_id");

        // The host passes origin with Y=0; we add the blueprint's ground_offset_y here.
        let origin = [
            input.origin[0],
            input.origin[1] + bp.ground_offset_y,
            input.origin[2],
            input.origin[3],
        ];
        let tree = bp.rasterize(origin, input.orientation);
        ProcgenGenerateOutput { tree }
    }
}
