use super::*;
use crate::shared::region_tree::chunk_key_i32;
use std::collections::HashSet;

fn ck(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    chunk_key_i32(x, y, z, w)
}

fn find_chunk_with_maze(seed: u64) -> Option<ChunkKey> {
    for cell_x in -8..=8 {
        for cell_z in -8..=8 {
            for cell_w in -8..=8 {
                let cell_hash = hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                    continue;
                }
                let shape = maze_shape_from_cell_hash(cell_hash);

                let origin_x = cell_x * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_X_SALT,
                        MAZE_CELL_JITTER,
                    );
                let origin_z = cell_z * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_Z_SALT,
                        MAZE_CELL_JITTER,
                    );
                let origin_w = cell_w * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_W_SALT,
                        MAZE_CELL_JITTER,
                    );
                if origin_x.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                    && origin_z.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                    && origin_w.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                {
                    continue;
                }

                let chunk_key = [
                    origin_x.div_euclid(CHUNK_SIZE as i32),
                    shape.world_y_min.div_euclid(CHUNK_SIZE as i32),
                    origin_z.div_euclid(CHUNK_SIZE as i32),
                    origin_w.div_euclid(CHUNK_SIZE as i32),
                ]
                .map(ChunkCoord::from_num);
                if !collect_maze_placements_for_chunk(seed, chunk_key).is_empty() {
                    return Some(chunk_key);
                }
            }
        }
    }
    None
}

#[test]
fn procedural_maze_bounds_vary_in_all_dimensions() {
    let seed = 0x4b31_c7a9_529d_17f2u64;
    let mut span_x_values = HashSet::new();
    let mut span_y_values = HashSet::new();
    let mut span_z_values = HashSet::new();
    let mut span_w_values = HashSet::new();

    'scan: for cell_x in -16..=16 {
        for cell_z in -16..=16 {
            for cell_w in -16..=16 {
                let cell_hash = hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                    continue;
                }
                let shape = maze_shape_from_cell_hash(cell_hash);
                span_x_values.insert(shape.span[0]);
                span_y_values.insert(shape.span[1]);
                span_z_values.insert(shape.span[2]);
                span_w_values.insert(shape.span[3]);

                if span_x_values.len() > 1
                    && span_y_values.len() > 1
                    && span_z_values.len() > 1
                    && span_w_values.len() > 1
                {
                    break 'scan;
                }
            }
        }
    }

    assert!(
        span_x_values.len() > 1,
        "expected procedural mazes to vary x span"
    );
    assert!(
        span_y_values.len() > 1,
        "expected procedural mazes to vary y span"
    );
    assert!(
        span_z_values.len() > 1,
        "expected procedural mazes to vary z span"
    );
    assert!(
        span_w_values.len() > 1,
        "expected procedural mazes to vary w span"
    );
}

#[test]
fn procedural_maze_topology_uses_y_dimension() {
    let layout_seed = 0x5cf2_9b44_1de3_a871u64;
    let shape = MazeShape {
        grid_cells: [9, 5, 9, 9],
        span: [
            9 * MAZE_STRIDE + 1,
            5 * MAZE_LEVEL_HEIGHT + 1,
            9 * MAZE_STRIDE + 1,
            9 * MAZE_STRIDE + 1,
        ],
        half_span_xzw: [0, 0, 0],
        world_y_min: 0,
        variant: MazeVariant::Vertical,
    };
    let topology = maze_build_topology(layout_seed, shape);
    let mut open_vertical_edges = 0usize;
    for x in 0..shape.grid_cells[0] {
        for y in 0..(shape.grid_cells[1] - 1) {
            for z in 0..shape.grid_cells[2] {
                for w in 0..shape.grid_cells[3] {
                    if topology.edge_open([x, y, z, w], [x, y + 1, z, w]) {
                        open_vertical_edges += 1;
                    }
                }
            }
        }
    }
    assert!(
        open_vertical_edges > 0,
        "expected topology to include vertical connections"
    );
}

#[test]
fn procedural_maze_variants_are_deterministic_and_diverse() {
    let seed = 0x2cab_4d10_3fc8_9912u64;
    let mut variants = HashSet::new();

    'scan: for cell_x in -20..=20 {
        for cell_z in -20..=20 {
            for cell_w in -20..=20 {
                let cell_hash = hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                    continue;
                }
                let variant_a = maze_shape_from_cell_hash(cell_hash).variant;
                let variant_b = maze_shape_from_cell_hash(cell_hash).variant;
                assert_eq!(variant_a, variant_b, "maze variant must be deterministic");
                variants.insert(variant_a);
                if variants.len() == 3 {
                    break 'scan;
                }
            }
        }
    }

    assert_eq!(
        variants.len(),
        3,
        "expected all maze variant presets to appear across sampled spawn cells"
    );
}

#[test]
fn structure_chunk_generation_is_seed_deterministic() {
    let chunk_a = generate_structure_chunk(42, ck(6, 0, 4, -3));
    let chunk_b = generate_structure_chunk(42, ck(6, 0, 4, -3));

    assert_eq!(chunk_a.is_some(), chunk_b.is_some());
    if let (Some(a), Some(b)) = (chunk_a, chunk_b) {
        assert_eq!(a[..], b[..]);
    }
}

#[test]
fn structure_generation_produces_non_empty_chunks() {
    let mut found_non_empty = false;
    let (min_y, max_y) = structure_chunk_y_bounds();

    for x in -25..=25 {
        for z in -25..=25 {
            for w in -25..=25 {
                for y in min_y..=max_y {
                    if generate_structure_chunk(1337, ck(x, y, z, w)).is_some() {
                        found_non_empty = true;
                        break;
                    }
                }
                if found_non_empty {
                    break;
                }
            }
            if found_non_empty {
                break;
            }
        }
        if found_non_empty {
            break;
        }
    }

    assert!(
        found_non_empty,
        "expected at least one structure chunk to be generated"
    );
}

#[test]
fn structure_blueprints_are_loaded() {
    let set = structure_set();
    assert!(!set.blueprints.is_empty());
}

#[test]
fn procedural_mazes_generate_chunks() {
    let seed = 0x5eed_1234_5678_9abc;
    let chunk_pos = find_chunk_with_maze(seed).expect("expected to find at least one maze chunk");
    let chunk =
        generate_structure_chunk(seed, chunk_pos).expect("maze placement should generate chunk");
    let set = structure_set();
    let maze_indices = [
        set.maze_ceiling_idx,
        set.maze_wall_idx,
        set.maze_gate_frame_idx,
        set.maze_beacon_idx,
    ];
    let has_maze_material = chunk.iter().any(|&mat| maze_indices.contains(&mat));
    assert!(
        has_maze_material,
        "expected generated chunk to contain procedural maze materials"
    );
}

#[test]
fn procedural_maze_generation_is_seed_deterministic() {
    let seed = 0x9d6f_21a5_7784_0031;
    let chunk_pos = find_chunk_with_maze(seed).expect("expected to find at least one maze chunk");
    let chunk_a =
        generate_structure_chunk(seed, chunk_pos).expect("maze placement should generate chunk");
    let chunk_b =
        generate_structure_chunk(seed, chunk_pos).expect("maze placement should generate chunk");
    assert_eq!(chunk_a[..], chunk_b[..]);
}

#[test]
fn blueprints_can_span_multiple_chunks() {
    let set = structure_set();
    let mut found_spanning = false;

    for blueprint in &set.blueprints {
        let span_x = blueprint.max_offset[0] - blueprint.min_offset[0] + 1;
        if span_x <= CHUNK_SIZE as i32 {
            continue;
        }

        // Place one boundary voxel-plane at x=8 so content can hit both chunk x=0 and x=1.
        let origin = [
            8 - blueprint.max_offset[0],
            -blueprint.min_offset[1],
            -blueprint.min_offset[2],
            -blueprint.min_offset[3],
        ];

        let mut left_chunk = dense_chunk_new();
        let mut right_chunk = dense_chunk_new();
        blueprint.place_into_chunk_oriented(origin, 0, [0, 0, 0, 0], &mut left_chunk);
        blueprint.place_into_chunk_oriented(
            origin,
            0,
            [CHUNK_SIZE as i32, 0, 0, 0],
            &mut right_chunk,
        );

        if !dense_chunk_is_empty(&left_chunk) && !dense_chunk_is_empty(&right_chunk) {
            found_spanning = true;
            break;
        }
    }

    assert!(
        found_spanning,
        "expected at least one blueprint to place non-empty voxels in adjacent chunks"
    );
}

#[test]
fn generated_placements_span_adjacent_chunks() {
    let (min_y, max_y) = structure_chunk_y_bounds();
    let mut found_spanning = false;

    'scan: for x in -35..=35 {
        for z in -35..=35 {
            for w in -35..=35 {
                for y in min_y..=max_y {
                    let left = ck(x, y, z, w);
                    let right = ck(x + 1, y, z, w);
                    let left_placements = collect_structure_placements_for_chunk(2026, left);
                    if left_placements.is_empty() {
                        continue;
                    }
                    let right_placements = collect_structure_placements_for_chunk(2026, right);
                    if right_placements.is_empty() {
                        continue;
                    }

                    if left_placements
                        .iter()
                        .any(|placement| right_placements.contains(placement))
                    {
                        found_spanning = true;
                        break 'scan;
                    }
                }
            }
        }
    }

    assert!(
        found_spanning,
        "expected at least one generated placement to intersect adjacent chunks"
    );
}

#[test]
fn chunk_has_content_matches_chunk_generation() {
    let seed = 777_u64;
    let (min_y, max_y) = structure_chunk_y_bounds();
    for x in -8..=8 {
        for z in -8..=8 {
            for w in -8..=8 {
                for y in min_y..=max_y {
                    let chunk_pos = ck(x, y, z, w);
                    let has_content = structure_chunk_has_content(seed, chunk_pos);
                    let generated = generate_structure_chunk(seed, chunk_pos).is_some();
                    assert_eq!(
                        has_content, generated,
                        "content mismatch for chunk ({x}, {y}, {z}, {w})"
                    );
                }
            }
        }
    }
}

#[test]
fn structure_chunk_positions_for_bounds_matches_bruteforce() {
    let seed = 0x4e73_9ac1_2f07_118du64;
    let bounds = Aabb4i::from_lattice_bounds([-3, -2, -3, -3], [3, 3, 3, 3], 0);

    let mut brute = Vec::new();
    let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
    for w in bmin[3]..=bmax[3] {
        for z in bmin[2]..=bmax[2] {
            for y in bmin[1]..=bmax[1] {
                for x in bmin[0]..=bmax[0] {
                    let pos = ck(x, y, z, w);
                    if structure_chunk_has_content_with_keepout(seed, pos, None) {
                        brute.push(pos);
                    }
                }
            }
        }
    }
    brute.sort_unstable();

    let fast = structure_chunk_positions_for_bounds_with_keepout(seed, bounds, None);
    assert_eq!(fast, brute);
}

#[test]
fn structure_chunk_positions_for_bounds_respects_keepout_cells() {
    let seed = 0x2a19_6e8d_0cb4_7fd1u64;
    let (min_y, max_y) = structure_chunk_y_bounds();
    let mut target = None::<(ChunkKey, StructureCell)>;
    'search: for x in -30..=30 {
        for z in -30..=30 {
            for w in -30..=30 {
                for y in min_y..=max_y {
                    let pos = ck(x, y, z, w);
                    if generate_structure_chunk(seed, pos).is_none() {
                        continue;
                    }
                    let cells = structure_cells_affecting_chunk(seed, pos);
                    if let Some(cell) = cells.first().copied() {
                        target = Some((pos, cell));
                        break 'search;
                    }
                }
            }
        }
    }
    let (target_pos, blocked_cell) = target.expect("expected target structure chunk");
    let tp = [
        target_pos[0].to_num::<i32>(),
        target_pos[1].to_num::<i32>(),
        target_pos[2].to_num::<i32>(),
        target_pos[3].to_num::<i32>(),
    ];
    let bounds = Aabb4i::from_lattice_bounds(
        [tp[0] - 2, tp[1] - 2, tp[2] - 2, tp[3] - 2],
        [tp[0] + 2, tp[1] + 2, tp[2] + 2, tp[3] + 2],
        0,
    );
    let mut blocked = HashSet::new();
    blocked.insert(blocked_cell);

    let mut brute = Vec::new();
    let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
    for w in bmin[3]..=bmax[3] {
        for z in bmin[2]..=bmax[2] {
            for y in bmin[1]..=bmax[1] {
                for x in bmin[0]..=bmax[0] {
                    let pos = ck(x, y, z, w);
                    if structure_chunk_has_content_with_keepout(seed, pos, Some(&blocked)) {
                        brute.push(pos);
                    }
                }
            }
        }
    }
    brute.sort_unstable();

    let fast = structure_chunk_positions_for_bounds_with_keepout(seed, bounds, Some(&blocked));
    assert_eq!(fast, brute);
}

#[test]
fn chunk_has_content_for_scale_matches_child_chunks() {
    let seed = 1717_u64;
    let scale = 2;
    let (min_y, max_y) = structure_chunk_y_bounds_for_scale(scale);

    for x in -6..=6 {
        for z in -6..=6 {
            for w in -6..=6 {
                for y in min_y..=max_y {
                    let coarse = ck(x, y, z, w);
                    let mut expected = false;
                    for dw in 0..scale {
                        for dz in 0..scale {
                            for dy in 0..scale {
                                for dx in 0..scale {
                                    let child = ck(
                                        x * scale + dx,
                                        y * scale + dy,
                                        z * scale + dz,
                                        w * scale + dw,
                                    );
                                    if structure_chunk_has_content(seed, child) {
                                        expected = true;
                                        break;
                                    }
                                }
                                if expected {
                                    break;
                                }
                            }
                            if expected {
                                break;
                            }
                        }
                        if expected {
                            break;
                        }
                    }

                    assert_eq!(
                        structure_chunk_has_content_for_scale(seed, coarse, scale),
                        expected,
                        "scaled content mismatch for chunk ({x}, {y}, {z}, {w})"
                    );
                }
            }
        }
    }
}

#[test]
fn keepout_cells_block_generation_for_affected_chunk() {
    let seed = 4242_u64;
    let (min_y, max_y) = structure_chunk_y_bounds();

    let mut target_chunk = None;
    'search: for x in -30..=30 {
        for z in -30..=30 {
            for w in -30..=30 {
                for y in min_y..=max_y {
                    let chunk_pos = ck(x, y, z, w);
                    let cells = structure_cells_affecting_chunk(seed, chunk_pos);
                    if !cells.is_empty() && generate_structure_chunk(seed, chunk_pos).is_some() {
                        target_chunk = Some((chunk_pos, cells));
                        break 'search;
                    }
                }
            }
        }
    }

    let (chunk_pos, cells) = target_chunk.expect("expected at least one structure chunk");
    let mut blocked = HashSet::new();
    blocked.insert(cells[0]);

    assert!(
        !structure_chunk_has_content_with_keepout(seed, chunk_pos, Some(&blocked)),
        "expected keepout cell to suppress structure content for chunk {:?}",
        chunk_pos
    );
    assert!(
        generate_structure_chunk_with_keepout(seed, chunk_pos, Some(&blocked)).is_none(),
        "expected keepout cell to suppress chunk generation for chunk {:?}",
        chunk_pos
    );
}

/// Manual-run seed scanner for finding seeds with structures/mazes near perf suite camera poses.
///
/// Run: `cargo test -p polychora scan_seeds_for_perf_scenarios -- --ignored --nocapture`
#[test]
#[ignore]
fn scan_seeds_for_perf_scenarios() {
    let candidate_positions: &[[f32; 4]] = &[
        [0.0, 8.0, -24.0, -4.0],   // platform-surface
        [0.0, 64.0, 0.0, -4.0],    // open-sky
        [96.0, 20.0, 96.0, -4.0],  // corridor
        [-64.0, 14.0, 32.0, -4.0], // far-oblique
        [0.0, 8.0, 0.0, 0.0],      // origin area
    ];
    let scan_radius = 80; // voxels around each position

    for seed in 0..10_000u64 {
        for &pos in candidate_positions {
            let center = [
                pos[0] as i32,
                pos[1] as i32,
                pos[2] as i32,
                pos[3] as i32,
            ];
            let bounds = Aabb4i::from_lattice_bounds(
                [
                    (center[0] - scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[1] - scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[2] - scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[3] - scan_radius).div_euclid(CHUNK_SIZE as i32),
                ],
                [
                    (center[0] + scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[1] + scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[2] + scan_radius).div_euclid(CHUNK_SIZE as i32),
                    (center[3] + scan_radius).div_euclid(CHUNK_SIZE as i32),
                ],
                0,
            );

            let structures =
                collect_structure_placements_for_chunk_bounds(seed, bounds, None);
            let mazes = collect_maze_placements_for_chunk_bounds(seed, bounds);

            if !structures.is_empty() || !mazes.is_empty() {
                println!(
                    "seed={} pos=[{:.0},{:.0},{:.0},{:.0}] structures={} mazes={}",
                    seed, pos[0], pos[1], pos[2], pos[3],
                    structures.len(), mazes.len(),
                );
                for s in &structures {
                    println!(
                        "  structure: blueprint_idx={} origin={:?} orientation={}",
                        s.blueprint_idx, s.origin, s.orientation
                    );
                }
                for m in &mazes {
                    println!(
                        "  maze: origin={:?} grid_cells={:?}",
                        m.origin, m.shape.grid_cells
                    );
                }
            }
        }
    }
}
