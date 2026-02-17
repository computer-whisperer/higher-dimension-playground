struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 for deterministic, repeatable test vectors.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn first_bit_high_u32(v: u32) -> i32 {
    if v == 0 {
        -1
    } else {
        (31 - v.leading_zeros()) as i32
    }
}

fn lbvh_delta(codes: &[u64], i: i32, j: i32) -> i32 {
    if j < 0 || j as usize >= codes.len() {
        return -1;
    }

    let iu = i as usize;
    let ju = j as usize;
    let code_i = codes[iu];
    let code_j = codes[ju];

    if code_i == code_j {
        let xor_idx = (iu as u32) ^ (ju as u32);
        return 64 + (31 - first_bit_high_u32(xor_idx));
    }

    let xor_val = code_i ^ code_j;
    let high = (xor_val >> 32) as u32;
    let low = xor_val as u32;

    if high != 0 {
        31 - first_bit_high_u32(high)
    } else if low != 0 {
        63 - first_bit_high_u32(low)
    } else {
        64
    }
}

fn build_lbvh_children(codes: &[u64]) -> Vec<(u32, u32)> {
    let num_leaves = codes.len();
    if num_leaves <= 1 {
        return Vec::new();
    }

    let num_internal = num_leaves - 1;
    let mut children = vec![(0u32, 0u32); num_internal];

    for idx in 0..num_internal {
        let idx_i = idx as i32;

        let d = if lbvh_delta(codes, idx_i, idx_i + 1) > lbvh_delta(codes, idx_i, idx_i - 1) {
            1
        } else {
            -1
        };

        let delta_min = lbvh_delta(codes, idx_i, idx_i - d);

        let mut l_max = 2;
        while lbvh_delta(codes, idx_i, idx_i + l_max * d) > delta_min {
            l_max *= 2;
        }

        let mut l = 0;
        let mut t = l_max / 2;
        while t >= 1 {
            if lbvh_delta(codes, idx_i, idx_i + (l + t) * d) > delta_min {
                l += t;
            }
            t /= 2;
        }

        let j = idx_i + l * d;

        let delta_node = lbvh_delta(codes, idx_i, j);
        let mut s = 0;
        let mut t2 = l;
        while t2 > 1 {
            t2 = (t2 + 1) / 2;
            let split_idx = idx_i + (s + t2) * d;
            if lbvh_delta(codes, idx_i, split_idx) > delta_node {
                s += t2;
            }
        }

        let gamma = idx_i + s * d + i32::min(d, 0);

        let left = if i32::min(idx_i, j) == gamma {
            (num_internal as u32) + (gamma as u32)
        } else {
            gamma as u32
        };

        let right = if i32::max(idx_i, j) == gamma + 1 {
            (num_internal as u32) + ((gamma + 1) as u32)
        } else {
            (gamma + 1) as u32
        };

        children[idx] = (left, right);
    }

    children
}

fn assert_topology_invariants(num_leaves: usize, children: &[(u32, u32)]) {
    if num_leaves <= 1 {
        assert!(children.is_empty());
        return;
    }

    let num_internal = num_leaves - 1;
    let total_nodes = num_internal + num_leaves;
    assert_eq!(children.len(), num_internal);

    let mut parent_counts = vec![0u32; total_nodes];
    for &(left, right) in children {
        assert!(
            left != right,
            "internal node references the same child twice"
        );
        assert!(
            (left as usize) < total_nodes,
            "left child index out of range: {left} >= {total_nodes}"
        );
        assert!(
            (right as usize) < total_nodes,
            "right child index out of range: {right} >= {total_nodes}"
        );
        parent_counts[left as usize] = parent_counts[left as usize].saturating_add(1);
        parent_counts[right as usize] = parent_counts[right as usize].saturating_add(1);
    }

    for (idx, &count) in parent_counts.iter().take(num_internal).enumerate() {
        let expected = if idx == 0 { 0 } else { 1 };
        assert_eq!(
            count, expected,
            "internal parent-count mismatch at node {idx}: expected {expected}, got {count}"
        );
    }

    for (leaf_idx, &count) in parent_counts.iter().enumerate().skip(num_internal) {
        assert_eq!(
            count, 1,
            "leaf parent-count mismatch at node {leaf_idx}: expected 1, got {count}"
        );
    }

    let mut visited = vec![false; total_nodes];
    let mut stack = vec![0usize];
    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        if node < num_internal {
            let (left, right) = children[node];
            stack.push(left as usize);
            stack.push(right as usize);
        }
    }

    let unreachable = visited
        .iter()
        .enumerate()
        .filter_map(|(idx, seen)| if *seen { None } else { Some(idx) })
        .collect::<Vec<_>>();
    assert!(
        unreachable.is_empty(),
        "unreachable BVH nodes: {:?}",
        unreachable
    );
}

fn sorted_codes(seed: u64, num_leaves: usize, quantized_bits: Option<u32>) -> Vec<u64> {
    let mut rng = TestRng::new(seed);
    let mut codes = Vec::with_capacity(num_leaves);
    let mask = quantized_bits.map(|bits| {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    });
    for _ in 0..num_leaves {
        let mut code = rng.next_u64();
        if let Some(m) = mask {
            code &= m;
        }
        codes.push(code);
    }
    codes.sort_unstable();
    codes
}

#[test]
fn lbvh_topology_regression_case_for_split_search() {
    // This deterministic case reproduced duplicate-internal-child topology
    // with the previous split-search implementation.
    let codes = vec![
        1_471_933_190_870_113_679,
        2_240_419_046_629_026_028,
        6_541_710_774_527_076_405,
        11_604_522_375_613_553_107,
    ];
    let children = build_lbvh_children(&codes);
    assert_topology_invariants(codes.len(), &children);
}

#[test]
fn lbvh_topology_invariants_hold_for_random_unique_codes() {
    for &num_leaves in &[2usize, 3, 4, 5, 8, 16, 32, 64] {
        for case_idx in 0..256u64 {
            let seed = ((num_leaves as u64) << 32) ^ case_idx ^ 0xA531_2F6D_9C77_BA01;
            let codes = sorted_codes(seed, num_leaves, None);
            let children = build_lbvh_children(&codes);
            assert_topology_invariants(num_leaves, &children);
        }
    }
}

#[test]
fn lbvh_topology_invariants_hold_for_duplicate_heavy_codes() {
    for &num_leaves in &[2usize, 3, 4, 5, 8, 16, 32, 64] {
        for case_idx in 0..256u64 {
            let seed = ((num_leaves as u64) << 32) ^ case_idx ^ 0x7C2A_91D4_EB60_143F;
            // 8-bit quantization forces many duplicate Morton codes.
            let codes = sorted_codes(seed, num_leaves, Some(8));
            let children = build_lbvh_children(&codes);
            assert_topology_invariants(num_leaves, &children);
        }
    }
}
