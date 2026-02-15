mod entities;
mod procgen;

use self::entities::EntityStore;
use crate::shared::protocol::{
    ClientMessage, EntityKind, PlayerSnapshot, ServerMessage,
    WorldChunkCoordPayload, WorldChunkPayload, WorldSnapshotPayload, WorldSummary,
    WORLD_CHUNK_LOD_FAR, WORLD_CHUNK_LOD_MID, WORLD_CHUNK_LOD_NEAR,
};
use crate::shared::voxel::{
    self, load_world, save_world, BaseWorldKind, ChunkPos, VoxelType, VoxelWorld, CHUNK_SIZE,
};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

const STREAM_MID_LOD_SCALE: i32 = 2;
const STREAM_FAR_LOD_SCALE: i32 = 4;
const STREAM_MESSAGE_CHUNK_LIMIT: usize = 256;
const SERVER_CPU_PROFILE_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: String,
    pub world_file: PathBuf,
    pub tick_hz: f32,
    pub save_interval_secs: u64,
    pub snapshot_on_join: bool,
    pub procgen_structures: bool,
    pub procgen_near_chunk_radius: i32,
    pub procgen_mid_chunk_radius: i32,
    pub procgen_far_chunk_radius: i32,
    pub procgen_keepout_from_existing_world: bool,
    pub procgen_keepout_padding_chunks: i32,
    pub world_seed: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:4000".to_string(),
            world_file: PathBuf::from("saves/world.v4dw"),
            tick_hz: 10.0,
            save_interval_secs: 5,
            snapshot_on_join: true,
            procgen_structures: true,
            procgen_near_chunk_radius: 6,
            procgen_mid_chunk_radius: 10,
            procgen_far_chunk_radius: 6,
            procgen_keepout_from_existing_world: true,
            procgen_keepout_padding_chunks: 1,
            world_seed: 1337,
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}

#[derive(Clone, Debug)]
struct PlayerState {
    client_id: u64,
    name: String,
    position: [f32; 4],
    look: [f32; 4],
    last_update_ms: u64,
    streamed_chunks: HashSet<StreamChunkKey>,
    last_stream_center_chunk: Option<[i32; 4]>,
    last_stream_world_revision: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct StreamChunkKey {
    lod_level: u8,
    chunk_pos: ChunkPos,
}

#[derive(Clone, Debug)]
struct ServerCpuProfile {
    window_start: Instant,
    message_samples: u64,
    message_cpu_ms_sum: f64,
    message_cpu_ms_max: f64,
    tick_samples: u64,
    tick_cpu_ms_sum: f64,
    tick_cpu_ms_max: f64,
}

#[derive(Copy, Clone, Debug)]
struct ServerCpuProfileReport {
    message_samples: u64,
    message_cpu_ms_sum: f64,
    message_cpu_ms_max: f64,
    tick_samples: u64,
    tick_cpu_ms_sum: f64,
    tick_cpu_ms_max: f64,
}

impl ServerCpuProfile {
    fn new(now: Instant) -> Self {
        Self {
            window_start: now,
            message_samples: 0,
            message_cpu_ms_sum: 0.0,
            message_cpu_ms_max: 0.0,
            tick_samples: 0,
            tick_cpu_ms_sum: 0.0,
            tick_cpu_ms_max: 0.0,
        }
    }

    fn record_message_sample(&mut self, elapsed: Duration) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.message_samples = self.message_samples.saturating_add(1);
        self.message_cpu_ms_sum += elapsed_ms;
        self.message_cpu_ms_max = self.message_cpu_ms_max.max(elapsed_ms);
    }

    fn record_tick_sample(&mut self, elapsed: Duration) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.tick_samples = self.tick_samples.saturating_add(1);
        self.tick_cpu_ms_sum += elapsed_ms;
        self.tick_cpu_ms_max = self.tick_cpu_ms_max.max(elapsed_ms);
    }

    fn take_report_if_due(&mut self, now: Instant) -> Option<ServerCpuProfileReport> {
        if now.duration_since(self.window_start) < SERVER_CPU_PROFILE_INTERVAL {
            return None;
        }

        let report = ServerCpuProfileReport {
            message_samples: self.message_samples,
            message_cpu_ms_sum: self.message_cpu_ms_sum,
            message_cpu_ms_max: self.message_cpu_ms_max,
            tick_samples: self.tick_samples,
            tick_cpu_ms_sum: self.tick_cpu_ms_sum,
            tick_cpu_ms_max: self.tick_cpu_ms_max,
        };

        self.window_start = now;
        self.message_samples = 0;
        self.message_cpu_ms_sum = 0.0;
        self.message_cpu_ms_max = 0.0;
        self.tick_samples = 0;
        self.tick_cpu_ms_sum = 0.0;
        self.tick_cpu_ms_max = 0.0;

        if report.message_samples == 0 && report.tick_samples == 0 {
            None
        } else {
            Some(report)
        }
    }
}

#[derive(Debug)]
struct ServerState {
    next_client_id: u64,
    entity_store: EntityStore,
    world: VoxelWorld,
    world_revision: u64,
    procgen_blocked_cells: HashSet<procgen::StructureCell>,
    stream_worldgen_has_content_cache: HashMap<StreamChunkKey, bool>,
    players: HashMap<u64, PlayerState>,
    clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
    cpu_profile: ServerCpuProfile,
}

type SharedState = Arc<Mutex<ServerState>>;

fn monotonic_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn record_server_cpu_sample(
    state: &SharedState,
    message_elapsed: Option<Duration>,
    tick_elapsed: Option<Duration>,
) {
    let maybe_report = {
        let mut guard = state.lock().expect("server state lock poisoned");
        if let Some(elapsed) = message_elapsed {
            guard.cpu_profile.record_message_sample(elapsed);
        }
        if let Some(elapsed) = tick_elapsed {
            guard.cpu_profile.record_tick_sample(elapsed);
        }
        guard
            .cpu_profile
            .take_report_if_due(Instant::now())
            .map(|report| (report, guard.players.len(), guard.entity_store.len()))
    };

    let Some((report, player_count, entity_count)) = maybe_report else {
        return;
    };

    let msg_avg_ms = if report.message_samples > 0 {
        report.message_cpu_ms_sum / report.message_samples as f64
    } else {
        0.0
    };
    let tick_avg_ms = if report.tick_samples > 0 {
        report.tick_cpu_ms_sum / report.tick_samples as f64
    } else {
        0.0
    };

    eprintln!(
        "profile server-cpu msg_avg={:.3}ms msg_max={:.3}ms msg_samples={} tick_avg={:.3}ms tick_max={:.3}ms tick_samples={} players={} entities={}",
        msg_avg_ms,
        report.message_cpu_ms_max,
        report.message_samples,
        tick_avg_ms,
        report.tick_cpu_ms_max,
        report.tick_samples,
        player_count,
        entity_count,
    );
}

fn sanitize_player_name(name: &str, client_id: u64) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("player-{client_id}");
    }
    trimmed.chars().take(32).collect()
}

fn player_snapshot(player: &PlayerState) -> PlayerSnapshot {
    PlayerSnapshot {
        client_id: player.client_id,
        name: player.name.clone(),
        position: player.position,
        look: player.look,
        last_update_ms: player.last_update_ms,
    }
}

fn world_chunk_from_position(position: [f32; 4]) -> [i32; 4] {
    let cs = CHUNK_SIZE as i32;
    [
        (position[0].floor() as i32).div_euclid(cs),
        (position[1].floor() as i32).div_euclid(cs),
        (position[2].floor() as i32).div_euclid(cs),
        (position[3].floor() as i32).div_euclid(cs),
    ]
}

fn encode_world_chunk_payload(
    lod_level: u8,
    chunk_pos: ChunkPos,
    chunk: &voxel::Chunk,
) -> WorldChunkPayload {
    WorldChunkPayload {
        lod_level,
        chunk_pos: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        voxels: chunk.voxels.iter().map(|v| v.0).collect(),
    }
}

fn chunks_equal(a: &voxel::Chunk, b: &voxel::Chunk) -> bool {
    a.solid_count == b.solid_count && a.voxels[..] == b.voxels[..]
}

fn merge_non_air_voxels(dst: &mut voxel::Chunk, src: &voxel::Chunk) {
    for (idx, voxel) in src.voxels.iter().enumerate() {
        if voxel.is_air() {
            continue;
        }
        dst.voxels[idx] = *voxel;
    }
    dst.solid_count = dst.voxels.iter().filter(|v| v.is_solid()).count() as u32;
    dst.dirty = true;
}

fn build_flat_floor_scaled_chunk(material: VoxelType) -> Option<voxel::Chunk> {
    if material.is_air() {
        return None;
    }
    let mut chunk = voxel::Chunk::new();
    let y = CHUNK_SIZE.saturating_sub(1);
    for w in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.set(x, y, z, w, material);
            }
        }
    }
    (!chunk.is_empty()).then_some(chunk)
}

fn build_virgin_chunk_for_edit(
    world: &VoxelWorld,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: Option<&HashSet<procgen::StructureCell>>,
) -> Option<voxel::Chunk> {
    let mut chunk = world.clone_base_chunk_or_empty_at(chunk_pos);
    if procgen_structures {
        if let Some(structure_chunk) =
            procgen::generate_structure_chunk_with_keepout(world_seed, chunk_pos, blocked_cells)
        {
            merge_non_air_voxels(&mut chunk, &structure_chunk);
        }
    }
    if chunk.is_empty() {
        None
    } else {
        Some(chunk)
    }
}

fn build_effective_stream_chunk_l0(
    state: &ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) -> Option<voxel::Chunk> {
    if let Some(override_chunk) = state.world.override_chunk_at(chunk_pos) {
        // Preserve explicit empty overrides (carved holes) instead of falling back
        // to virtual base generation on the client.
        return Some(override_chunk.clone());
    }

    let mut chunk = state.world.clone_base_chunk_or_empty_at(chunk_pos);
    if procgen_structures {
        if let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
            world_seed,
            chunk_pos,
            Some(&state.procgen_blocked_cells),
        ) {
            merge_non_air_voxels(&mut chunk, &structure_chunk);
        }
    }
    (!chunk.is_empty()).then_some(chunk)
}

fn build_effective_l0_chunk_for_sampling(
    state: &ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) -> Option<voxel::Chunk> {
    if let Some(override_chunk) = state.world.override_chunk_at(chunk_pos) {
        return (!override_chunk.is_empty()).then(|| override_chunk.clone());
    }

    let mut chunk = state.world.clone_base_chunk_or_empty_at(chunk_pos);
    if procgen_structures {
        if let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
            world_seed,
            chunk_pos,
            Some(&state.procgen_blocked_cells),
        ) {
            merge_non_air_voxels(&mut chunk, &structure_chunk);
        }
    }
    (!chunk.is_empty()).then_some(chunk)
}

fn stream_chunk_distance2(chunk_pos: ChunkPos, center_chunk: [i32; 4]) -> i64 {
    let dx = (chunk_pos.x - center_chunk[0]) as i64;
    let dy = (chunk_pos.y - center_chunk[1]) as i64;
    let dz = (chunk_pos.z - center_chunk[2]) as i64;
    let dw = (chunk_pos.w - center_chunk[3]) as i64;
    dx * dx + dy * dy + dz * dz + dw * dw
}

fn scaled_lod_child_l0_chunk_pos(parent_chunk_pos: ChunkPos, lod_scale: i32, child: [i32; 4]) -> ChunkPos {
    ChunkPos::new(
        parent_chunk_pos.x * lod_scale + child[0],
        parent_chunk_pos.y * lod_scale + child[1],
        parent_chunk_pos.z * lod_scale + child[2],
        parent_chunk_pos.w * lod_scale + child[3],
    )
}

fn l0_chunk_to_scaled_chunk(pos: ChunkPos, lod_scale: i32) -> ChunkPos {
    ChunkPos::new(
        pos.x.div_euclid(lod_scale),
        pos.y.div_euclid(lod_scale),
        pos.z.div_euclid(lod_scale),
        pos.w.div_euclid(lod_scale),
    )
}

fn stream_lod_scale(lod_level: u8) -> Option<i32> {
    match lod_level {
        WORLD_CHUNK_LOD_NEAR => Some(1),
        WORLD_CHUNK_LOD_MID => Some(STREAM_MID_LOD_SCALE),
        WORLD_CHUNK_LOD_FAR => Some(STREAM_FAR_LOD_SCALE),
        _ => None,
    }
}

fn stream_lod_worldgen_y_bounds(
    state: &ServerState,
    lod_level: u8,
    procgen_structures: bool,
) -> Option<(i32, i32)> {
    let scale = stream_lod_scale(lod_level)?;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut found = false;

    if let Some((base_min_y, base_max_y)) = state.world.base_chunk_y_bounds_for_scale(scale) {
        min_y = min_y.min(base_min_y);
        max_y = max_y.max(base_max_y);
        found = true;
    }
    if procgen_structures {
        let (procgen_min_y, procgen_max_y) = procgen::structure_chunk_y_bounds_for_scale(scale);
        min_y = min_y.min(procgen_min_y);
        max_y = max_y.max(procgen_max_y);
        found = true;
    }

    found.then_some((min_y, max_y))
}

fn stream_lod_worldgen_has_content(
    state: &mut ServerState,
    lod_level: u8,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) -> bool {
    let cache_key = StreamChunkKey {
        lod_level,
        chunk_pos,
    };
    if let Some(cached) = state.stream_worldgen_has_content_cache.get(&cache_key) {
        return *cached;
    }

    let Some(scale) = stream_lod_scale(lod_level) else {
        return false;
    };

    let has_content = if state
        .world
        .base_chunk_has_content_for_scale(chunk_pos, scale)
    {
        true
    } else if procgen_structures
        && procgen::structure_chunk_has_content_for_scale_with_keepout(
            world_seed,
            chunk_pos,
            scale,
            Some(&state.procgen_blocked_cells),
        )
    {
        true
    } else {
        false
    };

    state
        .stream_worldgen_has_content_cache
        .insert(cache_key, has_content);
    has_content
}

fn build_effective_stream_chunk_scaled_lod(
    state: &ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
    lod_scale: i32,
) -> Option<voxel::Chunk> {
    if lod_scale <= 1 {
        return build_effective_stream_chunk_l0(state, chunk_pos, world_seed, procgen_structures);
    }

    if let BaseWorldKind::FlatFloor { material } = state.world.base_kind() {
        let base_has_content = state
            .world
            .base_chunk_has_content_for_scale(chunk_pos, lod_scale);
        if base_has_content {
            let has_override_child = state.world.chunks.keys().any(|&child_pos| {
                l0_chunk_to_scaled_chunk(child_pos, lod_scale) == chunk_pos
            });
            if !has_override_child {
                let has_structure_child = procgen_structures
                    && procgen::structure_chunk_has_content_for_scale_with_keepout(
                        world_seed,
                        chunk_pos,
                        lod_scale,
                        Some(&state.procgen_blocked_cells),
                    );
                if !has_structure_child {
                    return build_flat_floor_scaled_chunk(material);
                }
            }
        }
    }

    let scale = lod_scale as usize;
    let child_count = scale * scale * scale * scale;
    let mut child_chunks: Vec<Option<voxel::Chunk>> = vec![None; child_count];

    for cw in 0..lod_scale {
        for cz in 0..lod_scale {
            for cy in 0..lod_scale {
                for cx in 0..lod_scale {
                    let child_idx = (((cw as usize * scale + cz as usize) * scale + cy as usize)
                        * scale
                        + cx as usize) as usize;
                    let child_pos = scaled_lod_child_l0_chunk_pos(chunk_pos, lod_scale, [cx, cy, cz, cw]);
                    child_chunks[child_idx] = build_effective_l0_chunk_for_sampling(
                        state,
                        child_pos,
                        world_seed,
                        procgen_structures,
                    );
                }
            }
        }
    }

    let mut out = voxel::Chunk::new();
    let mut material_counts = [0u16; 256];

    for w in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    material_counts.fill(0);
                    let mut best_material = 0u8;
                    let mut best_count = 0u16;

                    for dw in 0..lod_scale {
                        for dz in 0..lod_scale {
                            for dy in 0..lod_scale {
                                for dx in 0..lod_scale {
                                    let fx = (x as i32) * lod_scale + dx;
                                    let fy = (y as i32) * lod_scale + dy;
                                    let fz = (z as i32) * lod_scale + dz;
                                    let fw = (w as i32) * lod_scale + dw;
                                    let ccx = fx / CHUNK_SIZE as i32;
                                    let ccy = fy / CHUNK_SIZE as i32;
                                    let ccz = fz / CHUNK_SIZE as i32;
                                    let ccw = fw / CHUNK_SIZE as i32;
                                    let lx = (fx % CHUNK_SIZE as i32) as usize;
                                    let ly = (fy % CHUNK_SIZE as i32) as usize;
                                    let lz = (fz % CHUNK_SIZE as i32) as usize;
                                    let lw = (fw % CHUNK_SIZE as i32) as usize;
                                    let child_idx = ((((ccw as usize) * scale + (ccz as usize))
                                        * scale
                                        + (ccy as usize))
                                        * scale
                                        + (ccx as usize))
                                        as usize;
                                    let Some(child_chunk) = child_chunks[child_idx].as_ref() else {
                                        continue;
                                    };
                                    let voxel = child_chunk.get(lx, ly, lz, lw);
                                    if voxel.is_air() {
                                        continue;
                                    }
                                    let material = voxel.0 as usize;
                                    material_counts[material] = material_counts[material].saturating_add(1);
                                    if material_counts[material] > best_count {
                                        best_count = material_counts[material];
                                        best_material = voxel.0;
                                    }
                                }
                            }
                        }
                    }

                    if best_material != VoxelType::AIR.0 {
                        out.set(x, y, z, w, VoxelType(best_material));
                    }
                }
            }
        }
    }

    (!out.is_empty()).then_some(out)
}

fn build_effective_stream_chunk_l1(
    state: &ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) -> Option<voxel::Chunk> {
    build_effective_stream_chunk_scaled_lod(
        state,
        chunk_pos,
        world_seed,
        procgen_structures,
        STREAM_MID_LOD_SCALE,
    )
}

fn build_effective_stream_chunk_l2(
    state: &ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) -> Option<voxel::Chunk> {
    build_effective_stream_chunk_scaled_lod(
        state,
        chunk_pos,
        world_seed,
        procgen_structures,
        STREAM_FAR_LOD_SCALE,
    )
}

fn plan_stream_window_update(
    state: &mut ServerState,
    client_id: u64,
    center_chunk: [i32; 4],
    world_seed: u64,
    procgen_structures: bool,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
) -> Option<(u64, Vec<WorldChunkPayload>, Vec<WorldChunkCoordPayload>)> {
    let near_radius = near_chunk_radius.max(0);
    let min_chunk = [
        center_chunk[0] - near_radius,
        center_chunk[1] - near_radius,
        center_chunk[2] - near_radius,
        center_chunk[3] - near_radius,
    ];
    let max_chunk = [
        center_chunk[0] + near_radius,
        center_chunk[1] + near_radius,
        center_chunk[2] + near_radius,
        center_chunk[3] + near_radius,
    ];

    let mut desired_set = HashSet::<StreamChunkKey>::new();
    if let Some((worldgen_min_near_y, worldgen_max_near_y)) =
        stream_lod_worldgen_y_bounds(state, WORLD_CHUNK_LOD_NEAR, procgen_structures)
    {
        let scan_min_near_y = worldgen_min_near_y.max(min_chunk[1]);
        let scan_max_near_y = worldgen_max_near_y.min(max_chunk[1]);
        if scan_min_near_y <= scan_max_near_y {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                for chunk_z in min_chunk[2]..=max_chunk[2] {
                    for chunk_w in min_chunk[3]..=max_chunk[3] {
                        for chunk_y in scan_min_near_y..=scan_max_near_y {
                            let chunk_pos = ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w);
                            if stream_lod_worldgen_has_content(
                                state,
                                WORLD_CHUNK_LOD_NEAR,
                                chunk_pos,
                                world_seed,
                                procgen_structures,
                            ) {
                                desired_set.insert(StreamChunkKey {
                                    lod_level: WORLD_CHUNK_LOD_NEAR,
                                    chunk_pos,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    for (&chunk_pos, chunk) in &state.world.chunks {
        if chunk_pos.x < min_chunk[0]
            || chunk_pos.x > max_chunk[0]
            || chunk_pos.y < min_chunk[1]
            || chunk_pos.y > max_chunk[1]
            || chunk_pos.z < min_chunk[2]
            || chunk_pos.z > max_chunk[2]
            || chunk_pos.w < min_chunk[3]
            || chunk_pos.w > max_chunk[3]
            || chunk.is_empty()
        {
            continue;
        }
        desired_set.insert(StreamChunkKey {
            lod_level: WORLD_CHUNK_LOD_NEAR,
            chunk_pos,
        });
    }

    let center_mid_chunk = [
        center_chunk[0].div_euclid(STREAM_MID_LOD_SCALE),
        center_chunk[1].div_euclid(STREAM_MID_LOD_SCALE),
        center_chunk[2].div_euclid(STREAM_MID_LOD_SCALE),
        center_chunk[3].div_euclid(STREAM_MID_LOD_SCALE),
    ];
    let mid_radius = mid_chunk_radius.max(near_radius).max(1);
    let min_mid_chunk = [
        center_mid_chunk[0] - mid_radius,
        center_mid_chunk[1] - mid_radius,
        center_mid_chunk[2] - mid_radius,
        center_mid_chunk[3] - mid_radius,
    ];
    let max_mid_chunk = [
        center_mid_chunk[0] + mid_radius,
        center_mid_chunk[1] + mid_radius,
        center_mid_chunk[2] + mid_radius,
        center_mid_chunk[3] + mid_radius,
    ];

    if let Some((worldgen_min_mid_y, worldgen_max_mid_y)) =
        stream_lod_worldgen_y_bounds(state, WORLD_CHUNK_LOD_MID, procgen_structures)
    {
        let scan_min_mid_y = worldgen_min_mid_y.max(min_mid_chunk[1]);
        let scan_max_mid_y = worldgen_max_mid_y.min(max_mid_chunk[1]);
        if scan_min_mid_y <= scan_max_mid_y {
            for chunk_x in min_mid_chunk[0]..=max_mid_chunk[0] {
                for chunk_z in min_mid_chunk[2]..=max_mid_chunk[2] {
                    for chunk_w in min_mid_chunk[3]..=max_mid_chunk[3] {
                        for chunk_y in scan_min_mid_y..=scan_max_mid_y {
                            let chunk_pos = ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w);
                            if stream_lod_worldgen_has_content(
                                state,
                                WORLD_CHUNK_LOD_MID,
                                chunk_pos,
                                world_seed,
                                procgen_structures,
                            ) {
                                desired_set.insert(StreamChunkKey {
                                    lod_level: WORLD_CHUNK_LOD_MID,
                                    chunk_pos,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    for (&chunk_pos, chunk) in &state.world.chunks {
        if chunk.is_empty() {
            continue;
        }
        let mid_pos = l0_chunk_to_scaled_chunk(chunk_pos, STREAM_MID_LOD_SCALE);
        if mid_pos.x < min_mid_chunk[0]
            || mid_pos.x > max_mid_chunk[0]
            || mid_pos.y < min_mid_chunk[1]
            || mid_pos.y > max_mid_chunk[1]
            || mid_pos.z < min_mid_chunk[2]
            || mid_pos.z > max_mid_chunk[2]
            || mid_pos.w < min_mid_chunk[3]
            || mid_pos.w > max_mid_chunk[3]
        {
            continue;
        }
        desired_set.insert(StreamChunkKey {
            lod_level: WORLD_CHUNK_LOD_MID,
            chunk_pos: mid_pos,
        });
    }

    let center_far_chunk = [
        center_chunk[0].div_euclid(STREAM_FAR_LOD_SCALE),
        center_chunk[1].div_euclid(STREAM_FAR_LOD_SCALE),
        center_chunk[2].div_euclid(STREAM_FAR_LOD_SCALE),
        center_chunk[3].div_euclid(STREAM_FAR_LOD_SCALE),
    ];
    let min_far_radius = (mid_radius + 1).div_euclid(2).max(1);
    let far_radius = far_chunk_radius.max(min_far_radius).max(1);
    let min_far_chunk = [
        center_far_chunk[0] - far_radius,
        center_far_chunk[1] - far_radius,
        center_far_chunk[2] - far_radius,
        center_far_chunk[3] - far_radius,
    ];
    let max_far_chunk = [
        center_far_chunk[0] + far_radius,
        center_far_chunk[1] + far_radius,
        center_far_chunk[2] + far_radius,
        center_far_chunk[3] + far_radius,
    ];

    if let Some((worldgen_min_far_y, worldgen_max_far_y)) =
        stream_lod_worldgen_y_bounds(state, WORLD_CHUNK_LOD_FAR, procgen_structures)
    {
        let scan_min_far_y = worldgen_min_far_y.max(min_far_chunk[1]);
        let scan_max_far_y = worldgen_max_far_y.min(max_far_chunk[1]);
        if scan_min_far_y <= scan_max_far_y {
            for chunk_x in min_far_chunk[0]..=max_far_chunk[0] {
                for chunk_z in min_far_chunk[2]..=max_far_chunk[2] {
                    for chunk_w in min_far_chunk[3]..=max_far_chunk[3] {
                        for chunk_y in scan_min_far_y..=scan_max_far_y {
                            let chunk_pos = ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w);
                            if stream_lod_worldgen_has_content(
                                state,
                                WORLD_CHUNK_LOD_FAR,
                                chunk_pos,
                                world_seed,
                                procgen_structures,
                            ) {
                                desired_set.insert(StreamChunkKey {
                                    lod_level: WORLD_CHUNK_LOD_FAR,
                                    chunk_pos,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    for (&chunk_pos, chunk) in &state.world.chunks {
        if chunk.is_empty() {
            continue;
        }
        let far_pos = l0_chunk_to_scaled_chunk(chunk_pos, STREAM_FAR_LOD_SCALE);
        if far_pos.x < min_far_chunk[0]
            || far_pos.x > max_far_chunk[0]
            || far_pos.y < min_far_chunk[1]
            || far_pos.y > max_far_chunk[1]
            || far_pos.z < min_far_chunk[2]
            || far_pos.z > max_far_chunk[2]
            || far_pos.w < min_far_chunk[3]
            || far_pos.w > max_far_chunk[3]
        {
            continue;
        }
        desired_set.insert(StreamChunkKey {
            lod_level: WORLD_CHUNK_LOD_FAR,
            chunk_pos: far_pos,
        });
    }

    let old_streamed = {
        let player = state.players.get_mut(&client_id)?;
        std::mem::take(&mut player.streamed_chunks)
    };

    let mut load_chunk_keys: Vec<StreamChunkKey> =
        desired_set.difference(&old_streamed).copied().collect();
    load_chunk_keys.sort_unstable_by_key(|key| {
        let center = match key.lod_level {
            WORLD_CHUNK_LOD_NEAR => center_chunk,
            WORLD_CHUNK_LOD_MID => center_mid_chunk,
            WORLD_CHUNK_LOD_FAR => center_far_chunk,
            _ => center_chunk,
        };
        (
            key.lod_level,
            stream_chunk_distance2(key.chunk_pos, center),
            key.chunk_pos.w,
            key.chunk_pos.z,
            key.chunk_pos.y,
            key.chunk_pos.x,
        )
    });

    let mut realized_desired = desired_set;
    let mut load_payloads = Vec::with_capacity(load_chunk_keys.len());
    for key in load_chunk_keys {
        let chunk = match key.lod_level {
            WORLD_CHUNK_LOD_NEAR => build_effective_stream_chunk_l0(
                state,
                key.chunk_pos,
                world_seed,
                procgen_structures,
            ),
            WORLD_CHUNK_LOD_MID => build_effective_stream_chunk_l1(
                state,
                key.chunk_pos,
                world_seed,
                procgen_structures,
            ),
            WORLD_CHUNK_LOD_FAR => build_effective_stream_chunk_l2(
                state,
                key.chunk_pos,
                world_seed,
                procgen_structures,
            ),
            _ => None,
        };
        if let Some(chunk) = chunk {
            load_payloads.push(encode_world_chunk_payload(
                key.lod_level,
                key.chunk_pos,
                &chunk,
            ));
        } else {
            realized_desired.remove(&key);
        }
    }

    let mut unload_chunks: Vec<WorldChunkCoordPayload> = old_streamed
        .difference(&realized_desired)
        .copied()
        .map(|key| WorldChunkCoordPayload {
            lod_level: key.lod_level,
            chunk_pos: [
                key.chunk_pos.x,
                key.chunk_pos.y,
                key.chunk_pos.z,
                key.chunk_pos.w,
            ],
        })
        .collect();
    unload_chunks.sort_unstable_by_key(|payload| {
        (
            payload.lod_level,
            payload.chunk_pos[3],
            payload.chunk_pos[2],
            payload.chunk_pos[1],
            payload.chunk_pos[0],
        )
    });

    if let Some(player) = state.players.get_mut(&client_id) {
        player.streamed_chunks = realized_desired;
    }
    Some((state.world_revision, load_payloads, unload_chunks))
}

fn send_to_client(state: &SharedState, client_id: u64, message: ServerMessage) {
    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    if let Some(tx) = sender {
        let _ = tx.send(message);
    }
}

fn broadcast(state: &SharedState, message: ServerMessage) {
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .map(|(&client_id, tx)| (client_id, tx.clone()))
            .collect()
    };

    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    if !stale.is_empty() {
        let mut guard = state.lock().expect("server state lock poisoned");
        for client_id in stale {
            guard.clients.remove(&client_id);
            guard.players.remove(&client_id);
        }
    }
}

fn send_world_voxel_set_to_streamed_clients(
    state: &SharedState,
    position: [i32; 4],
    material: u8,
    source_client_id: Option<u64>,
    client_edit_id: Option<u64>,
    revision: u64,
) {
    let (chunk_pos, _) = voxel::world_to_chunk(position[0], position[1], position[2], position[3]);
    let stream_key = StreamChunkKey {
        lod_level: WORLD_CHUNK_LOD_NEAR,
        chunk_pos,
    };
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .filter_map(|(&client_id, tx)| {
                let is_source = source_client_id == Some(client_id);
                let is_streaming_chunk = guard
                    .players
                    .get(&client_id)
                    .map(|player| player.streamed_chunks.contains(&stream_key))
                    .unwrap_or(false);
                if is_source || is_streaming_chunk {
                    Some((client_id, tx.clone()))
                } else {
                    None
                }
            })
            .collect()
    };

    let message = ServerMessage::WorldVoxelSet {
        position,
        material,
        source_client_id,
        client_edit_id,
        revision,
    };
    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    if !stale.is_empty() {
        let mut guard = state.lock().expect("server state lock poisoned");
        for client_id in stale {
            guard.clients.remove(&client_id);
            guard.players.remove(&client_id);
        }
    }
}

fn sync_streamed_chunks_for_client(
    state: &SharedState,
    client_id: u64,
    center_chunk: [i32; 4],
    world_seed: u64,
    procgen_structures: bool,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    force: bool,
) {
    let update = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let world_revision = guard.world_revision;
        let should_sync = match guard.players.get_mut(&client_id) {
            Some(player) => {
                let changed = player.last_stream_center_chunk != Some(center_chunk);
                let world_changed = player.last_stream_world_revision != world_revision;
                if force || changed || world_changed {
                    player.last_stream_center_chunk = Some(center_chunk);
                    true
                } else {
                    false
                }
            }
            None => false,
        };

        if should_sync {
            let update = plan_stream_window_update(
                &mut guard,
                client_id,
                center_chunk,
                world_seed,
                procgen_structures,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
            );
            if let Some((revision, _, _)) = update.as_ref() {
                if let Some(player) = guard.players.get_mut(&client_id) {
                    player.last_stream_world_revision = *revision;
                }
            }
            update
        } else {
            None
        }
    };

    let Some((revision, chunk_loads, chunk_unloads)) = update else {
        return;
    };

    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    let Some(sender) = sender else {
        return;
    };

    if !chunk_loads.is_empty() {
        let mut batch = Vec::with_capacity(STREAM_MESSAGE_CHUNK_LIMIT);
        for chunk in chunk_loads {
            batch.push(chunk);
            if batch.len() == STREAM_MESSAGE_CHUNK_LIMIT {
                let _ = sender.send(ServerMessage::WorldChunkBatch {
                    revision,
                    chunks: std::mem::take(&mut batch),
                });
            }
        }
        if !batch.is_empty() {
            let _ = sender.send(ServerMessage::WorldChunkBatch {
                revision,
                chunks: batch,
            });
        }
    }

    if !chunk_unloads.is_empty() {
        let mut batch = Vec::with_capacity(STREAM_MESSAGE_CHUNK_LIMIT);
        for chunk in chunk_unloads {
            batch.push(chunk);
            if batch.len() == STREAM_MESSAGE_CHUNK_LIMIT {
                let _ = sender.send(ServerMessage::WorldChunkUnloadBatch {
                    revision,
                    chunks: std::mem::take(&mut batch),
                });
            }
        }
        if !batch.is_empty() {
            let _ = sender.send(ServerMessage::WorldChunkUnloadBatch {
                revision,
                chunks: batch,
            });
        }
    }
}

fn ensure_chunk_materialized_for_edit(
    state: &mut ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) {
    if state.world.has_chunk_override(chunk_pos) {
        return;
    }
    if let Some(chunk) = build_virgin_chunk_for_edit(
        &state.world,
        chunk_pos,
        world_seed,
        procgen_structures,
        Some(&state.procgen_blocked_cells),
    ) {
        state.world.insert_chunk(chunk_pos, chunk);
    }
}

fn trim_chunk_override_if_virgin(
    state: &mut ServerState,
    chunk_pos: ChunkPos,
    world_seed: u64,
    procgen_structures: bool,
) {
    let Some(override_chunk) = state.world.override_chunk_at(chunk_pos).cloned() else {
        return;
    };
    let virgin_chunk = build_virgin_chunk_for_edit(
        &state.world,
        chunk_pos,
        world_seed,
        procgen_structures,
        Some(&state.procgen_blocked_cells),
    );
    let matches_virgin = match virgin_chunk {
        Some(virgin) => chunks_equal(&override_chunk, &virgin),
        None => override_chunk.is_empty(),
    };
    if matches_virgin {
        let _ = state.world.remove_chunk_override(chunk_pos);
    }
}

fn prune_virgin_overrides(
    world: &mut VoxelWorld,
    world_seed: u64,
    procgen_structures: bool,
) -> usize {
    let positions: Vec<ChunkPos> = world.chunks.keys().copied().collect();
    let mut pruned = 0usize;
    for pos in positions {
        let Some(override_chunk) = world.override_chunk_at(pos).cloned() else {
            continue;
        };
        let virgin_chunk =
            build_virgin_chunk_for_edit(world, pos, world_seed, procgen_structures, None);
        let matches_virgin = match virgin_chunk {
            Some(virgin) => chunks_equal(&override_chunk, &virgin),
            None => override_chunk.is_empty(),
        };
        if matches_virgin && world.remove_chunk_override(pos) {
            pruned += 1;
        }
    }
    pruned
}

fn gather_override_chunks_with_padding(
    world: &VoxelWorld,
    padding_chunks: i32,
) -> HashSet<ChunkPos> {
    let padding = padding_chunks.max(0);
    let mut out = HashSet::new();
    for (&pos, chunk) in &world.chunks {
        if chunk.is_empty() {
            continue;
        }
        for dx in -padding..=padding {
            for dy in -padding..=padding {
                for dz in -padding..=padding {
                    for dw in -padding..=padding {
                        out.insert(ChunkPos::new(
                            pos.x + dx,
                            pos.y + dy,
                            pos.z + dz,
                            pos.w + dw,
                        ));
                    }
                }
            }
        }
    }
    out
}

fn build_procgen_keepout_cells(
    world: &VoxelWorld,
    world_seed: u64,
    padding_chunks: i32,
) -> HashSet<procgen::StructureCell> {
    let keepout_chunks = gather_override_chunks_with_padding(world, padding_chunks);
    let mut blocked_cells = HashSet::new();
    for chunk_pos in keepout_chunks {
        for cell in procgen::structure_cells_affecting_chunk(world_seed, chunk_pos) {
            blocked_cells.insert(cell);
        }
    }
    blocked_cells
}

fn load_world_from_path(path: &Path) -> io::Result<VoxelWorld> {
    if !path.exists() {
        return Ok(VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        }));
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_world(&mut reader)
}

fn save_world_to_path(path: &Path, world: &VoxelWorld) -> io::Result<usize> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_world(world, &mut writer)?;
    writer.flush()?;
    Ok(world.non_empty_chunk_count())
}

fn build_world_snapshot_payload(state: &ServerState) -> io::Result<WorldSnapshotPayload> {
    let mut bytes = Vec::new();
    save_world(&state.world, &mut bytes)?;
    Ok(WorldSnapshotPayload {
        format: "v4dw".to_string(),
        non_empty_chunks: state.world.non_empty_chunk_count(),
        revision: state.world_revision,
        bytes,
    })
}

fn start_broadcast_thread(
    state: SharedState,
    tick_hz: f32,
    start: Instant,
    shutdown: Arc<AtomicBool>,
) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let tick_cpu_start = Instant::now();
            let (players, entities) = {
                let guard = state.lock().expect("server state lock poisoned");
                let players = if guard.players.is_empty() {
                    None
                } else {
                    let mut all: Vec<_> = guard.players.values().map(player_snapshot).collect();
                    all.sort_by_key(|p| p.client_id);
                    Some(all)
                };
                let entities = if guard.entity_store.is_empty() {
                    None
                } else {
                    Some(guard.entity_store.sorted_snapshots())
                };
                (players, entities)
            };
            let did_broadcast = players.is_some() || entities.is_some();
            let now = monotonic_ms(start);
            if let Some(players) = players {
                broadcast(
                    &state,
                    ServerMessage::PlayerPositions {
                        server_time_ms: now,
                        players,
                    },
                );
            }
            if let Some(entities) = entities {
                broadcast(
                    &state,
                    ServerMessage::EntityPositions {
                        server_time_ms: now,
                        entities,
                    },
                );
            }
            if did_broadcast {
                record_server_cpu_sample(&state, None, Some(tick_cpu_start.elapsed()));
            }
        }
    });
}

fn start_autosave_thread(
    state: SharedState,
    world_file: PathBuf,
    save_interval_secs: u64,
    shutdown: Arc<AtomicBool>,
) {
    if save_interval_secs == 0 {
        return;
    }

    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(save_interval_secs));
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let save_result = {
                let guard = state.lock().expect("server state lock poisoned");
                save_world_to_path(&world_file, &guard.world)
                    .map(|chunks| (chunks, guard.world_revision))
            };

            match save_result {
                Ok((chunk_count, revision)) => {
                    eprintln!(
                        "autosave: world revision {} ({} non-empty chunks) -> {}",
                        revision,
                        chunk_count,
                        world_file.display()
                    );
                }
                Err(error) => {
                    eprintln!("autosave failed ({}): {}", world_file.display(), error);
                }
            }
        }
    });
}

fn remove_client(state: &SharedState, client_id: u64) {
    let had_player = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let _ = guard.clients.remove(&client_id);
        guard.players.remove(&client_id).is_some()
    };
    if had_player {
        broadcast(state, ServerMessage::PlayerLeft { client_id });
    }
}

fn install_or_update_player(
    state: &SharedState,
    client_id: u64,
    name: Option<String>,
    position: Option<[f32; 4]>,
    look: Option<[f32; 4]>,
    start: Instant,
) -> PlayerSnapshot {
    let now = monotonic_ms(start);
    let mut guard = state.lock().expect("server state lock poisoned");
    let player = guard
        .players
        .entry(client_id)
        .or_insert_with(|| PlayerState {
            client_id,
            name: format!("player-{client_id}"),
            position: [0.0, 0.0, 0.0, 0.0],
            look: [
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            last_update_ms: now,
            streamed_chunks: HashSet::new(),
            last_stream_center_chunk: None,
            last_stream_world_revision: 0,
        });

    if let Some(n) = name {
        player.name = sanitize_player_name(&n, client_id);
    }
    if let Some(pos) = position {
        player.position = pos;
    }
    if let Some(dir) = look {
        player.look = dir;
    }
    player.last_update_ms = now;
    player_snapshot(player)
}

fn handle_message(
    state: &SharedState,
    client_id: u64,
    message: ClientMessage,
    snapshot_on_join: bool,
    tick_hz: f32,
    procgen_structures: bool,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    world_seed: u64,
    start: Instant,
) {
    let message_cpu_start = Instant::now();
    match message {
        ClientMessage::Hello { name } => {
            let snapshot =
                install_or_update_player(state, client_id, Some(name), None, None, start);
            send_to_client(
                state,
                client_id,
                ServerMessage::Welcome {
                    client_id,
                    server_time_ms: monotonic_ms(start),
                    tick_hz,
                    world: {
                        let guard = state.lock().expect("server state lock poisoned");
                        WorldSummary {
                            non_empty_chunks: guard.world.non_empty_chunk_count(),
                            revision: guard.world_revision,
                        }
                    },
                },
            );
            if snapshot_on_join {
                let snapshot_message = {
                    let guard = state.lock().expect("server state lock poisoned");
                    build_world_snapshot_payload(&guard)
                };
                match snapshot_message {
                    Ok(world) => {
                        send_to_client(state, client_id, ServerMessage::WorldSnapshot { world });
                    }
                    Err(error) => {
                        send_to_client(
                            state,
                            client_id,
                            ServerMessage::Error {
                                message: format!("failed to build world snapshot: {error}"),
                            },
                        );
                    }
                }
            }
            {
                let guard = state.lock().expect("server state lock poisoned");
                let entities = guard.entity_store.snapshots();
                if let Some(tx) = guard.clients.get(&client_id) {
                    for entity in entities {
                        let _ = tx.send(ServerMessage::EntitySpawned { entity });
                    }
                }
            }
            let center_chunk = world_chunk_from_position(snapshot.position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                world_seed,
                procgen_structures,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                true,
            );
            broadcast(state, ServerMessage::PlayerJoined { player: snapshot });
        }
        ClientMessage::UpdatePlayer { position, look } => {
            let _snapshot =
                install_or_update_player(state, client_id, None, Some(position), Some(look), start);
            let center_chunk = world_chunk_from_position(position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                world_seed,
                procgen_structures,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                false,
            );
        }
        ClientMessage::SetVoxel {
            position,
            material,
            client_edit_id,
        } => {
            let revision = {
                let mut guard = state.lock().expect("server state lock poisoned");
                let (chunk_pos, _) =
                    voxel::world_to_chunk(position[0], position[1], position[2], position[3]);
                ensure_chunk_materialized_for_edit(
                    &mut guard,
                    chunk_pos,
                    world_seed,
                    procgen_structures,
                );
                guard.world.set_voxel(
                    position[0],
                    position[1],
                    position[2],
                    position[3],
                    VoxelType(material),
                );
                trim_chunk_override_if_virgin(
                    &mut guard,
                    chunk_pos,
                    world_seed,
                    procgen_structures,
                );
                guard.world_revision = guard.world_revision.wrapping_add(1);
                guard.world_revision
            };

            send_world_voxel_set_to_streamed_clients(
                state,
                position,
                material,
                Some(client_id),
                client_edit_id,
                revision,
            );
        }
        ClientMessage::RequestWorldSnapshot => {
            let snapshot_message = {
                let guard = state.lock().expect("server state lock poisoned");
                build_world_snapshot_payload(&guard)
            };
            match snapshot_message {
                Ok(world) => {
                    send_to_client(state, client_id, ServerMessage::WorldSnapshot { world })
                }
                Err(error) => send_to_client(
                    state,
                    client_id,
                    ServerMessage::Error {
                        message: format!("failed to build world snapshot: {error}"),
                    },
                ),
            }
            let center_chunk = {
                let guard = state.lock().expect("server state lock poisoned");
                guard
                    .players
                    .get(&client_id)
                    .map(|player| world_chunk_from_position(player.position))
                    .unwrap_or([0, 0, 0, 0])
            };
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                world_seed,
                procgen_structures,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                true,
            );
        }
        ClientMessage::Ping { nonce } => {
            send_to_client(state, client_id, ServerMessage::Pong { nonce });
        }
    }
    record_server_cpu_sample(state, Some(message_cpu_start.elapsed()), None);
}

fn spawn_client_thread(
    stream: TcpStream,
    state: SharedState,
    snapshot_on_join: bool,
    tick_hz: f32,
    procgen_structures: bool,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    world_seed: u64,
    start: Instant,
) {
    let peer_label = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let (tx, rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = guard.next_client_id;
        guard.next_client_id = guard.next_client_id.wrapping_add(1).max(1);
        guard.clients.insert(id, tx.clone());
        id
    };

    let writer_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(error) => {
            eprintln!("failed to clone stream for {}: {}", peer_label, error);
            remove_client(&state, client_id);
            return;
        }
    };

    thread::spawn(move || {
        let mut writer = BufWriter::new(writer_stream);
        while let Ok(message) = rx.recv() {
            let Ok(encoded) = postcard::to_stdvec(&message) else {
                continue;
            };
            let len = (encoded.len() as u32).to_le_bytes();
            if writer.write_all(&len).is_err() {
                break;
            }
            if writer.write_all(&encoded).is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    thread::spawn(move || {
        eprintln!("client {} connected from {}", client_id, peer_label);
        let mut reader = stream;
        let mut len_buf = [0u8; 4];

        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 100_000_000 {
                eprintln!(
                    "client {} sent oversized message ({} bytes)",
                    client_id, len
                );
                break;
            }

            let mut msg_buf = vec![0u8; len];
            match reader.read_exact(&mut msg_buf) {
                Ok(()) => {}
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let parsed = postcard::from_bytes::<ClientMessage>(&msg_buf);
            match parsed {
                Ok(message) => {
                    handle_message(
                        &state,
                        client_id,
                        message,
                        snapshot_on_join,
                        tick_hz,
                        procgen_structures,
                        near_chunk_radius,
                        mid_chunk_radius,
                        far_chunk_radius,
                        world_seed,
                        start,
                    );
                }
                Err(error) => {
                    send_to_client(
                        &state,
                        client_id,
                        ServerMessage::Error {
                            message: format!("invalid message: {error}"),
                        },
                    );
                }
            }
        }

        remove_client(&state, client_id);
        eprintln!("client {} disconnected", client_id);
    });
}

fn spawn_entity(
    state: &SharedState,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    start: Instant,
) -> u64 {
    let mut guard = state.lock().expect("server state lock poisoned");
    guard.entity_store.spawn(
        kind,
        position,
        orientation,
        scale,
        material,
        monotonic_ms(start),
    )
}

fn spawn_default_test_entities(state: &SharedState, start: Instant) {
    let test_positions: [[f32; 4]; 3] = [
        [3.0, 2.0, 3.0, 0.0],
        [-2.0, 1.5, 5.0, 1.0],
        [0.0, 3.0, -4.0, 2.0],
    ];
    let test_materials: [u8; 3] = [12, 8, 15];
    for (i, (pos, mat)) in test_positions.iter().zip(test_materials.iter()).enumerate() {
        let id = spawn_entity(
            state,
            EntityKind::TestCube,
            *pos,
            [0.0, 0.0, 1.0, 0.0],
            0.5,
            *mat,
            start,
        );
        eprintln!("spawned test entity {} (id={}) at {:?}", i, id, pos);
    }
}

fn initialize_state(
    config: &RuntimeConfig,
    shutdown: Arc<AtomicBool>,
) -> io::Result<(SharedState, Instant)> {
    let start = Instant::now();
    let mut initial_world = load_world_from_path(&config.world_file)?;
    let pruned = prune_virgin_overrides(
        &mut initial_world,
        config.world_seed,
        config.procgen_structures,
    );
    if pruned > 0 {
        eprintln!(
            "pruned {} virgin chunk overrides from {}",
            pruned,
            config.world_file.display()
        );
    }
    let procgen_blocked_cells =
        if config.procgen_structures && config.procgen_keepout_from_existing_world {
            let blocked = build_procgen_keepout_cells(
                &initial_world,
                config.world_seed,
                config.procgen_keepout_padding_chunks,
            );
            if !blocked.is_empty() {
                eprintln!(
                    "procgen keepout: {} blocked structure cells (padding={} chunks)",
                    blocked.len(),
                    config.procgen_keepout_padding_chunks.max(0)
                );
            }
            blocked
        } else {
            HashSet::new()
        };
    initial_world.clear_dirty();
    let _ = initial_world.drain_pending_chunk_updates();
    let initial_chunks = initial_world.non_empty_chunk_count();
    eprintln!(
        "loaded world {} ({} non-empty chunks)",
        config.world_file.display(),
        initial_chunks
    );

    let state = Arc::new(Mutex::new(ServerState {
        next_client_id: 1,
        entity_store: EntityStore::new(),
        world: initial_world,
        world_revision: 0,
        procgen_blocked_cells,
        stream_worldgen_has_content_cache: HashMap::new(),
        players: HashMap::new(),
        clients: HashMap::new(),
        cpu_profile: ServerCpuProfile::new(start),
    }));

    spawn_default_test_entities(&state, start);
    start_broadcast_thread(state.clone(), config.tick_hz, start, shutdown.clone());
    start_autosave_thread(
        state.clone(),
        config.world_file.clone(),
        config.save_interval_secs,
        shutdown,
    );

    Ok((state, start))
}

pub fn connect_local_client(config: &RuntimeConfig) -> io::Result<LocalConnection> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown.clone())?;
    let (client_to_server_tx, client_to_server_rx) = mpsc::channel::<ClientMessage>();
    let (server_to_client_tx, server_to_client_rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = guard.next_client_id;
        guard.next_client_id = guard.next_client_id.wrapping_add(1).max(1);
        guard.clients.insert(id, server_to_client_tx);
        id
    };

    let state_for_client = state.clone();
    let cfg = config.clone();
    thread::spawn(move || {
        while let Ok(message) = client_to_server_rx.recv() {
            handle_message(
                &state_for_client,
                client_id,
                message,
                cfg.snapshot_on_join,
                cfg.tick_hz.max(0.1),
                cfg.procgen_structures,
                cfg.procgen_near_chunk_radius.max(0),
                cfg.procgen_mid_chunk_radius.max(1),
                cfg.procgen_far_chunk_radius.max(1),
                cfg.world_seed,
                start,
            );
        }
        remove_client(&state_for_client, client_id);
        shutdown.store(true, Ordering::Relaxed);
    });

    Ok(LocalConnection {
        outgoing: client_to_server_tx,
        incoming: server_to_client_rx,
    })
}

pub fn run_tcp_server(config: &RuntimeConfig) -> io::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown)?;

    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, autosave {}s, procgen={}, seed={}, near_radius={} chunks, mid_radius={} chunks, far_radius={} chunks, keepout={}, keepout_padding={} chunks)",
        config.bind,
        config.tick_hz.max(0.1),
        config.save_interval_secs,
        config.procgen_structures,
        config.world_seed,
        config.procgen_near_chunk_radius.max(0),
        config.procgen_mid_chunk_radius.max(1),
        config.procgen_far_chunk_radius.max(1),
        config.procgen_keepout_from_existing_world,
        config.procgen_keepout_padding_chunks.max(0),
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                spawn_client_thread(
                    stream,
                    state.clone(),
                    config.snapshot_on_join,
                    config.tick_hz.max(0.1),
                    config.procgen_structures,
                    config.procgen_near_chunk_radius.max(0),
                    config.procgen_mid_chunk_radius.max(1),
                    config.procgen_far_chunk_radius.max(1),
                    config.world_seed,
                    start,
                );
            }
            Err(error) => {
                eprintln!("accept failed: {}", error);
            }
        }
    }

    Ok(())
}
