use super::*;
use higher_dimension_playground::render::OVERLAY_EDGE_TAG_MASK;

pub(super) fn latest_framebuffer_screenshot_path() -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir("frames").ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        if !(name.starts_with("framebuffer_") && name.ends_with(".webp")) {
            continue;
        }
        let modified = entry.metadata().ok()?.modified().ok()?;
        let replace = best
            .as_ref()
            .map(|(best_time, _)| modified > *best_time)
            .unwrap_or(true);
        if replace {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path)
}

pub(super) fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

pub(super) fn env_flag_enabled_or(name: &str, default_enabled: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => default_enabled,
    }
}

pub(super) fn env_usize_or(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_value)
}

pub(super) fn normalize_server_addr(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return format!("127.0.0.1:{MULTIPLAYER_DEFAULT_PORT}");
    }
    if trimmed.contains(':') || trimmed.starts_with('[') {
        trimmed.to_string()
    } else {
        format!("{trimmed}:{MULTIPLAYER_DEFAULT_PORT}")
    }
}

pub(super) fn make_menu_camera() -> Camera4D {
    let mut camera = Camera4D::new();
    apply_menu_camera_orbit_pose(&mut camera, 0.0);
    camera
}

pub(super) fn apply_menu_camera_orbit_pose(camera: &mut Camera4D, time_s: f32) {
    let orbit_phase_xz = time_s * MENU_ORBIT_RATE_XZ;
    let orbit_phase_w = time_s * MENU_ORBIT_RATE_W;
    let orbit_phase_y = time_s * MENU_ORBIT_RATE_Y;

    camera.position = [
        MENU_ORBIT_CENTER[0] + MENU_ORBIT_RADIUS_XZ * orbit_phase_xz.cos(),
        MENU_ORBIT_HEIGHT_BASE + MENU_ORBIT_HEIGHT_BOB * orbit_phase_y.sin(),
        MENU_ORBIT_CENTER[2] + MENU_ORBIT_RADIUS_XZ * orbit_phase_xz.sin(),
        MENU_ORBIT_CENTER[3] + MENU_ORBIT_RADIUS_W * orbit_phase_w.sin(),
    ];

    let orbit_target = [
        MENU_ORBIT_CENTER[0],
        MENU_ORBIT_CENTER[1] + MENU_ORBIT_TARGET_Y_OFFSET,
        MENU_ORBIT_CENTER[2],
        MENU_ORBIT_CENTER[3],
    ];
    let target_dir = [
        orbit_target[0] - camera.position[0],
        orbit_target[1] - camera.position[1],
        orbit_target[2] - camera.position[2],
        orbit_target[3] - camera.position[3],
    ];
    let (yaw, pitch, xw_angle, zw_angle) = Camera4D::angles_for_direction_upright(target_dir);
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.xw_angle = xw_angle;
    camera.zw_angle = zw_angle;
    camera.yw_deviation = 0.0;
}

pub(super) fn build_singleplayer_runtime_config(
    args: &Args,
    world_file: PathBuf,
    world_generator: polychora::server::WorldGeneratorKind,
    content_registry: std::sync::Arc<polychora::content_registry::ContentRegistry>,
    wasm_manager: Option<polychora::shared::wasm::WasmPluginManager>,
) -> polychora::server::RuntimeConfig {
    polychora::server::RuntimeConfig {
        bind: "127.0.0.1:0".to_string(),
        world_file,
        world_generator,
        tick_hz: args.singleplayer_tick_hz.max(0.1),
        entity_sim_hz: args.singleplayer_entity_sim_hz.max(0.1),
        save_interval_secs: args.singleplayer_save_interval_secs,
        procgen_structures: args.singleplayer_procgen_structures,
        procgen_near_chunk_radius: args.singleplayer_procgen_chunk_radius,
        procgen_mid_chunk_radius: args.singleplayer_procgen_mid_chunk_radius,
        procgen_far_chunk_radius: args.singleplayer_procgen_far_chunk_radius,
        procgen_keepout_from_existing_world: args.singleplayer_procgen_keepout_from_existing_world,
        procgen_keepout_padding_chunks: args.singleplayer_procgen_keepout_padding_chunks,
        world_seed: args.singleplayer_world_seed,
        content_registry,
        wasm_manager,
    }
}

pub(super) fn generate_new_singleplayer_world_path() -> PathBuf {
    let saves_dir = PathBuf::from("saves");
    let _ = std::fs::create_dir_all(&saves_dir);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    for suffix in 0u32..1000 {
        let file_name = if suffix == 0 {
            format!("world-{timestamp}")
        } else {
            format!("world-{timestamp}-{suffix}")
        };
        let candidate = saves_dir.join(file_name);
        if !candidate.exists() {
            return candidate;
        }
    }

    saves_dir.join("world-new")
}

pub(super) fn default_multiplayer_player_name() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .ok()
        .map(|v| {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                "player".to_string()
            } else {
                trimmed.to_string()
            }
        })
        .unwrap_or_else(|| "player".to_string())
}

pub(super) fn is_escape_pressed(event: &winit::event::KeyEvent) -> bool {
    if event.state.is_pressed() && !event.repeat {
        if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
            return true;
        }
        if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
            return true;
        }
    }
    false
}

pub(super) fn project_world_point_to_ndc_with_depth(
    view_matrix: &ndarray::Array2<f32>,
    world_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<([f32; 2], f32)> {
    let input = [
        world_point[0],
        world_point[1],
        world_point[2],
        world_point[3],
        1.0,
    ];
    let mut view_h = [0.0f32; 5];
    for row in 0..5 {
        for col in 0..5 {
            view_h[row] += view_matrix[[row, col]] * input[col];
        }
    }

    let inv_w = if view_h[4].abs() > 1e-6 {
        view_h[4].recip()
    } else {
        1.0
    };
    let view = [
        view_h[0] * inv_w,
        view_h[1] * inv_w,
        view_h[2] * inv_w,
        view_h[3] * inv_w,
    ];

    let depth = (view[2] * view[2] + view[3] * view[3]).sqrt();
    if !depth.is_finite() || depth < 1e-4 {
        return None;
    }
    let projection_divisor = depth / focal_length_xy.max(1e-4);
    let x = view[0] / projection_divisor;
    let y = aspect * (-view[1]) / projection_divisor;
    if x.is_finite() && y.is_finite() {
        Some(([x, y], depth))
    } else {
        None
    }
}

fn overlay_edge_material_ids(tag: u32) -> [u32; 8] {
    let encoded_tag = OVERLAY_EDGE_TAG_MASK | (tag & !OVERLAY_EDGE_TAG_MASK);
    [encoded_tag; 8]
}

pub(super) fn append_axis_aligned_outline_edge_instance(
    overlay_edge_instances: &mut Vec<common::ModelInstance>,
    min_world: [f32; 4],
    max_world: [f32; 4],
    edge_tag: u32,
) {
    let mut axis_scale = [0.0f32; 4];
    let mut center = [0.0f32; 4];
    for axis in 0..4 {
        if !min_world[axis].is_finite() || !max_world[axis].is_finite() {
            return;
        }
        let extent = max_world[axis] - min_world[axis];
        if extent <= 0.0 || !extent.is_finite() {
            return;
        }
        axis_scale[axis] = extent;
        center[axis] = 0.5 * (min_world[axis] + max_world[axis]);
    }

    let basis = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    overlay_edge_instances.push(build_centered_model_instance(
        center,
        &basis,
        axis_scale,
        overlay_edge_material_ids(edge_tag),
    ));
}

pub(super) fn append_voxel_outline_edge_instance(
    overlay_edge_instances: &mut Vec<common::ModelInstance>,
    voxel: [i32; 4],
    edge_tag: u32,
) {
    let min_world = [
        voxel[0] as f32,
        voxel[1] as f32,
        voxel[2] as f32,
        voxel[3] as f32,
    ];
    let max_world = [
        (voxel[0] as f32) + 1.0,
        (voxel[1] as f32) + 1.0,
        (voxel[2] as f32) + 1.0,
        (voxel[3] as f32) + 1.0,
    ];
    append_axis_aligned_outline_edge_instance(
        overlay_edge_instances,
        min_world,
        max_world,
        edge_tag,
    );
}

pub(super) fn append_chunk_bounds_outline_edge_instance(
    overlay_edge_instances: &mut Vec<common::ModelInstance>,
    min_chunk: polychora::shared::region_tree::ChunkKey,
    max_chunk: polychora::shared::region_tree::ChunkKey,
    edge_tag: u32,
) {
    if min_chunk
        .iter()
        .zip(max_chunk.iter())
        .any(|(min, max)| min > max)
    {
        return;
    }

    let chunk_size = voxel::CHUNK_SIZE as i32;
    let min_world_i32: [i32; 4] = std::array::from_fn(|i| min_chunk[i].to_num::<i32>().saturating_mul(chunk_size));
    let max_world_i32: [i32; 4] = std::array::from_fn(|i| max_chunk[i].to_num::<i32>().saturating_add(1).saturating_mul(chunk_size));

    let min_world = [
        min_world_i32[0] as f32,
        min_world_i32[1] as f32,
        min_world_i32[2] as f32,
        min_world_i32[3] as f32,
    ];
    let max_world = [
        max_world_i32[0] as f32,
        max_world_i32[1] as f32,
        max_world_i32[2] as f32,
        max_world_i32[3] as f32,
    ];
    append_axis_aligned_outline_edge_instance(
        overlay_edge_instances,
        min_world,
        max_world,
        edge_tag,
    );
}

pub(super) fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 {
        return v;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

pub(super) fn rotate_basis_plane(
    basis: &mut [[f32; 4]; 4],
    axis_a: usize,
    axis_b: usize,
    angle: f32,
) {
    let c = angle.cos();
    let s = angle.sin();
    let old_a = basis[axis_a];
    let old_b = basis[axis_b];
    for i in 0..4 {
        basis[axis_a][i] = c * old_a[i] + s * old_b[i];
        basis[axis_b][i] = -s * old_a[i] + c * old_b[i];
    }
}

pub(super) fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

pub(super) fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

pub(super) fn distance4(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    let d3 = a[3] - b[3];
    (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt()
}

pub(super) fn normalize4_with_fallback(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = dot4(v, v);
    if len_sq <= 1e-8 || !len_sq.is_finite() {
        return fallback;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

pub(super) fn orthonormal_basis_from_forward(forward: [f32; 4]) -> [[f32; 4]; 4] {
    let forward = normalize4_with_fallback(forward, [0.0, 0.0, 1.0, 0.0]);
    let mut ortho = [[0.0; 4]; 4];
    ortho[0] = forward;
    let mut count = 1usize;

    let candidates = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ];

    for candidate in candidates {
        if count >= 4 {
            break;
        }
        let mut v = candidate;
        for basis_idx in 0..count {
            let projection = dot4(v, ortho[basis_idx]);
            for axis in 0..4 {
                v[axis] -= projection * ortho[basis_idx][axis];
            }
        }

        let len_sq = dot4(v, v);
        if len_sq <= 1e-6 {
            continue;
        }

        let inv_len = len_sq.sqrt().recip();
        for axis in 0..4 {
            v[axis] *= inv_len;
        }
        ortho[count] = v;
        count += 1;
    }

    if count < 4 {
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    // Return basis as [right, up, forward, side].
    // `ortho[1]` is world-up for planar forward inputs, so keep it in the up slot.
    [ortho[2], ortho[1], ortho[0], ortho[3]]
}

pub(super) fn offset_point_along_basis(
    origin: [f32; 4],
    basis: &[[f32; 4]; 4],
    local_offset: [f32; 4],
) -> [f32; 4] {
    let mut p = origin;
    for row in 0..4 {
        p[row] += basis[0][row] * local_offset[0]
            + basis[1][row] * local_offset[1]
            + basis[2][row] * local_offset[2]
            + basis[3][row] * local_offset[3];
    }
    p
}

pub(super) fn build_centered_model_instance(
    center: [f32; 4],
    basis: &[[f32; 4]; 4],
    axis_scale: [f32; 4],
    cell_material_ids: [u32; 8],
) -> common::ModelInstance {
    let mut model_transform = common::MatN::<5>::identity();
    for row in 0..4 {
        model_transform[[row, 0]] = basis[0][row] * axis_scale[0];
        model_transform[[row, 1]] = basis[1][row] * axis_scale[1];
        model_transform[[row, 2]] = basis[2][row] * axis_scale[2];
        model_transform[[row, 3]] = basis[3][row] * axis_scale[3];

        let center_offset = 0.5
            * (model_transform[[row, 0]]
                + model_transform[[row, 1]]
                + model_transform[[row, 2]]
                + model_transform[[row, 3]]);
        model_transform[[row, 4]] = center[row] - center_offset;
    }

    common::ModelInstance {
        model_transform,
        cell_material_ids,
    }
}

pub(super) fn avatar_cell_mask(cell_indices: &[usize]) -> [u32; 8] {
    let mut ids = [0u32; 8];
    for &cell in cell_indices {
        if cell < ids.len() {
            ids[cell] = AVATAR_MATERIAL_ID;
        }
    }
    ids
}

pub(super) fn stable_name_hash(name: &str) -> u32 {
    let mut hash = 0x811C_9DC5u32;
    for b in name.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

pub(super) fn build_place_preview_instance(
    camera: &Camera4D,
    block: &polychora::shared::voxel::BlockData,
    time_s: f32,
    control_scheme: ControlScheme,
    aspect: f32,
    material_resolver: &polychora::content_registry::MaterialResolver,
) -> common::ModelInstance {
    let material_token = material_resolver.resolve_block(block.namespace, block.block_type);
    let preview_material = material_token as u32 | PREVIEW_MATERIAL_FLAG;
    let (right, up, view_z, view_w) = match control_scheme {
        ControlScheme::IntuitiveUpright => camera.view_basis_upright(),
        ControlScheme::LookTransport | ControlScheme::RotorFree => camera.view_basis_look_frame(),
        ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
            camera.view_basis()
        }
    };

    let center_forward = normalize4([
        view_z[0] + view_w[0],
        view_z[1] + view_w[1],
        view_z[2] + view_w[2],
        view_z[3] + view_w[3],
    ]);
    let side_w = normalize4([
        view_w[0] - view_z[0],
        view_w[1] - view_z[1],
        view_w[2] - view_z[2],
        view_w[3] - view_z[3],
    ]);
    let aspect = aspect.max(0.25);
    let up_bias = -0.56 * ((16.0 / 9.0) / aspect).clamp(0.72, 1.35);

    let anchor = [
        camera.position[0]
            + 0.88 * right[0]
            + up_bias * up[0]
            + 1.20 * center_forward[0]
            + 0.46 * side_w[0],
        camera.position[1]
            + 0.88 * right[1]
            + up_bias * up[1]
            + 1.20 * center_forward[1]
            + 0.46 * side_w[1],
        camera.position[2]
            + 0.88 * right[2]
            + up_bias * up[2]
            + 1.20 * center_forward[2]
            + 0.46 * side_w[2],
        camera.position[3]
            + 0.88 * right[3]
            + up_bias * up[3]
            + 1.20 * center_forward[3]
            + 0.46 * side_w[3],
    ];

    let mut basis = [right, up, center_forward, side_w];
    rotate_basis_plane(&mut basis, 0, 2, time_s * 0.85 + 0.2);
    rotate_basis_plane(&mut basis, 0, 3, time_s * 0.55 + 0.9);

    build_centered_model_instance(
        anchor,
        &basis,
        [0.35, 0.35, 0.38, 0.38],
        [preview_material; 8],
    )
}

pub(super) fn build_remote_player_avatar_instances(
    client_id: u64,
    name_hash: u32,
    position: [f32; 4],
    look: [f32; 4],
    time_s: f32,
) -> Vec<common::ModelInstance> {
    let mut instances = Vec::with_capacity(REMOTE_AVATAR_PART_COUNT_ESTIMATE);

    let look_phase = (name_hash as f32) * 0.0078125;
    let id_phase = ((client_id as f32) * 0.173205 + look_phase).rem_euclid(std::f32::consts::TAU);
    let idle_bob = (time_s * 1.9 + id_phase).sin() * 0.04;

    let planar_forward =
        normalize4_with_fallback([look[0], 0.0, look[2], look[3]], [0.0, 0.0, 1.0, 0.0]);
    let full_forward = normalize4_with_fallback(look, planar_forward);
    let mut avatar_basis = orthonormal_basis_from_forward(planar_forward);
    rotate_basis_plane(&mut avatar_basis, 0, 3, 0.18 * id_phase.sin());

    let head_center = offset_point_along_basis(
        position,
        &avatar_basis,
        [0.0, 0.05 + idle_bob * 0.4, 0.0, 0.0],
    );
    let mut head_basis = avatar_basis;
    rotate_basis_plane(&mut head_basis, 0, 2, time_s * 2.6 + id_phase * 0.6);
    rotate_basis_plane(&mut head_basis, 1, 3, time_s * 1.7 + id_phase);
    instances.push(build_centered_model_instance(
        head_center,
        &head_basis,
        [
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
        ],
        [AVATAR_MATERIAL_ID; 8],
    ));

    let body_parts: [([f32; 4], [f32; 4], &[usize]); 6] = [
        (
            [0.0, -0.34 * PLAYER_HEIGHT + idle_bob, 0.03, 0.0],
            [
                0.22 * AVATAR_THICKNESS_SCALE,
                0.20 * AVATAR_THICKNESS_SCALE,
                0.18 * AVATAR_THICKNESS_SCALE,
                0.16 * AVATAR_THICKNESS_SCALE,
            ],
            &[0, 6],
        ),
        (
            [-0.20, -0.40 * PLAYER_HEIGHT + idle_bob, 0.0, 0.10],
            [
                0.13 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
            ],
            &[2],
        ),
        (
            [0.20, -0.40 * PLAYER_HEIGHT + idle_bob, 0.0, -0.10],
            [
                0.13 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
            ],
            &[5],
        ),
        (
            [-0.12, -0.72 * PLAYER_HEIGHT + idle_bob, 0.02, 0.08],
            [
                0.14 * AVATAR_THICKNESS_SCALE,
                0.19 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
            ],
            &[7],
        ),
        (
            [0.12, -0.72 * PLAYER_HEIGHT + idle_bob, -0.02, -0.08],
            [
                0.14 * AVATAR_THICKNESS_SCALE,
                0.19 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
            ],
            &[1],
        ),
        (
            [0.0, -0.58 * PLAYER_HEIGHT + idle_bob, -0.02, 0.0],
            [
                0.17 * AVATAR_THICKNESS_SCALE,
                0.16 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
            ],
            &[3, 4],
        ),
    ];

    for (part_index, (offset, scales, cells)) in body_parts.iter().enumerate() {
        let mut part_basis = avatar_basis;
        let part_phase = id_phase + part_index as f32 * 0.7;
        rotate_basis_plane(&mut part_basis, 0, 2, 0.22 * part_phase.sin());
        rotate_basis_plane(&mut part_basis, 2, 3, 0.17 * part_phase.cos());
        instances.push(build_centered_model_instance(
            offset_point_along_basis(position, &avatar_basis, *offset),
            &part_basis,
            *scales,
            avatar_cell_mask(cells),
        ));
    }

    for fragment_idx in 0..AVATAR_FORWARD_FRAGMENT_COUNT {
        let fragment_phase = id_phase + fragment_idx as f32 * 0.83;
        let fragment_distance =
            (0.54 + fragment_idx as f32 * 0.30) * AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE;
        let swirl = time_s * 3.1 + fragment_phase;
        let lateral_offset = [
            0.10 * swirl.cos(),
            -0.04 + 0.05 * (swirl * 0.9).sin(),
            0.10 * (swirl + 1.1).sin(),
        ];
        let forward_distance =
            fragment_distance + 0.06 * AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE * (swirl * 1.2).sin();
        let mut fragment_center = head_center;
        for axis in 0..4 {
            fragment_center[axis] += avatar_basis[0][axis] * lateral_offset[0]
                + avatar_basis[1][axis] * lateral_offset[1]
                + avatar_basis[3][axis] * lateral_offset[2]
                + full_forward[axis] * forward_distance;
        }

        let mut fragment_basis = orthonormal_basis_from_forward(full_forward);
        rotate_basis_plane(&mut fragment_basis, 0, 3, time_s * 1.7 + fragment_phase);
        rotate_basis_plane(
            &mut fragment_basis,
            1,
            2,
            time_s * 2.3 + fragment_phase * 0.7,
        );

        let fragment_cells: &[usize] = match fragment_idx % 4 {
            0 => &[0],
            1 => &[2],
            2 => &[5],
            _ => &[7],
        };

        let depth_scale = 1.0 - fragment_idx as f32 * 0.12;
        instances.push(build_centered_model_instance(
            fragment_center,
            &fragment_basis,
            [
                0.10 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.09 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.13 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.09 * AVATAR_THICKNESS_SCALE * depth_scale,
            ],
            avatar_cell_mask(fragment_cells),
        ));
    }

    instances
}

// ---------------------------------------------------------------------------
// Entity hit-testing for WAILA
// ---------------------------------------------------------------------------

pub(super) struct EntityHitResult {
    pub entity_id: u64,
    pub distance: f32,
}

const ENTITY_HIT_BASE_RADIUS: f32 = 0.6;

pub(super) fn find_targeted_entity(
    camera_pos: [f32; 4],
    look_dir: [f32; 4],
    max_distance: f32,
    entities: &HashMap<u64, RemoteEntityState>,
    players: &HashMap<u64, RemotePlayerState>,
) -> Option<EntityHitResult> {
    let mut best: Option<EntityHitResult> = None;

    for (&eid, ent) in entities {
        let diff = [
            ent.render_position[0] - camera_pos[0],
            ent.render_position[1] - camera_pos[1],
            ent.render_position[2] - camera_pos[2],
            ent.render_position[3] - camera_pos[3],
        ];
        let t = dot4(diff, look_dir);
        if t <= 0.0 || t > max_distance {
            continue;
        }
        let perp_sq = dot4(diff, diff) - t * t;
        let radius = ent.scale * ENTITY_HIT_BASE_RADIUS;
        if perp_sq > radius * radius {
            continue;
        }
        if best.as_ref().map_or(true, |b| t < b.distance) {
            best = Some(EntityHitResult {
                entity_id: eid,
                distance: t,
            });
        }
    }

    for (&eid, player) in players {
        let diff = [
            player.render_position[0] - camera_pos[0],
            player.render_position[1] - camera_pos[1],
            player.render_position[2] - camera_pos[2],
            player.render_position[3] - camera_pos[3],
        ];
        let t = dot4(diff, look_dir);
        if t <= 0.0 || t > max_distance {
            continue;
        }
        let perp_sq = dot4(diff, diff) - t * t;
        let radius = ENTITY_HIT_BASE_RADIUS; // players use scale 1.0
        if perp_sq > radius * radius {
            continue;
        }
        if best.as_ref().map_or(true, |b| t < b.distance) {
            best = Some(EntityHitResult {
                entity_id: eid,
                distance: t,
            });
        }
    }

    best
}

// ---------------------------------------------------------------------------
// CBOR display decoder for WAILA
// ---------------------------------------------------------------------------

pub(super) fn format_cbor_for_display(data: &[u8]) -> Option<String> {
    if data.is_empty() {
        return None;
    }
    match ciborium::from_reader::<ciborium::Value, _>(data) {
        Ok(value) => {
            let mut out = String::new();
            format_cbor_value(&value, &mut out, 0, 2);
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        Err(_) => {
            // Fall back to hex dump
            let hex: String = data
                .iter()
                .take(32)
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ");
            if data.len() > 32 {
                Some(format!("{} ... ({} bytes)", hex, data.len()))
            } else {
                Some(hex)
            }
        }
    }
}

fn format_cbor_value(value: &ciborium::Value, out: &mut String, depth: usize, max_depth: usize) {
    use std::fmt::Write;
    if depth > max_depth {
        out.push_str("...");
        return;
    }
    match value {
        ciborium::Value::Integer(i) => {
            let _ = write!(out, "{}", i128::from(*i));
        }
        ciborium::Value::Float(f) => {
            let _ = write!(out, "{:.3}", f);
        }
        ciborium::Value::Text(s) => {
            let truncated: String = s.chars().take(64).collect();
            let _ = write!(out, "\"{}\"", truncated);
            if s.len() > 64 {
                out.push_str("...");
            }
        }
        ciborium::Value::Bool(b) => {
            let _ = write!(out, "{}", b);
        }
        ciborium::Value::Null => out.push_str("null"),
        ciborium::Value::Bytes(b) => {
            let _ = write!(out, "<{} bytes>", b.len());
        }
        ciborium::Value::Array(arr) => {
            out.push('[');
            let limit = 8;
            for (i, item) in arr.iter().take(limit).enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                format_cbor_value(item, out, depth + 1, max_depth);
            }
            if arr.len() > limit {
                let _ = write!(out, ", ... +{}", arr.len() - limit);
            }
            out.push(']');
        }
        ciborium::Value::Map(map) => {
            out.push('{');
            let limit = 6;
            for (i, (k, v)) in map.iter().take(limit).enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                format_cbor_value(k, out, depth + 1, max_depth);
                out.push_str(": ");
                format_cbor_value(v, out, depth + 1, max_depth);
            }
            if map.len() > limit {
                let _ = write!(out, ", ... +{}", map.len() - limit);
            }
            out.push('}');
        }
        ciborium::Value::Tag(tag, inner) => {
            let _ = write!(out, "tag({})", tag);
            out.push(' ');
            format_cbor_value(inner, out, depth + 1, max_depth);
        }
        _ => out.push_str("?"),
    }
}
