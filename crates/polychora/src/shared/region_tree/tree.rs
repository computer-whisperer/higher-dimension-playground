use super::*;

#[derive(Clone, Debug, Default)]
pub struct RegionChunkTree {
    root: Option<Box<RegionTreeCore>>,
}

impl RegionChunkTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_chunks<I>(chunks: I) -> Self
    where
        I: IntoIterator<Item = (ChunkKey, ChunkPayload)>,
    {
        let mut tree = Self::new();
        for (key, payload) in chunks {
            let _ = tree.set_chunk(key, Some(payload));
        }
        tree
    }

    pub fn root(&self) -> Option<&RegionTreeCore> {
        self.root.as_deref()
    }

    pub fn has_chunk(&self, key: ChunkKey) -> bool {
        self.chunk_payload(key).is_some()
    }

    pub fn chunk_payload(&self, key: ChunkKey) -> Option<ChunkPayload> {
        self.root
            .as_ref()
            .and_then(|node| query_chunk_payload_in_node(node, key))
    }

    pub fn set_chunk(&mut self, key: ChunkKey, payload: Option<ChunkPayload>) -> bool {
        let payload = payload.map(canonicalize_chunk_payload);
        if self.root.is_none() {
            let Some(payload) = payload else {
                return false;
            };
            let bounds = Aabb4i::new(key, key);
            self.root = Some(Box::new(RegionTreeCore {
                bounds,
                kind: kind_from_chunk_value(bounds, Some(payload)),
                generator_version_hash: 0,
            }));
            return true;
        }

        if payload.is_none()
            && self
                .root
                .as_ref()
                .map(|root| !root.bounds.contains_chunk(key))
                .unwrap_or(false)
        {
            // Deleting outside the represented bounds cannot change tree contents.
            return false;
        }

        while self
            .root
            .as_ref()
            .map(|root| !root.bounds.contains_chunk(key))
            .unwrap_or(false)
        {
            let Some(root) = self.root.take() else {
                break;
            };
            self.root = Some(expand_root_once(root, key));
        }

        let changed = if let Some(root) = self.root.as_mut() {
            set_chunk_recursive(root, key, payload)
        } else {
            false
        };

        if changed
            && self
                .root
                .as_ref()
                .map(|root| matches!(root.kind, RegionNodeKind::Empty))
                .unwrap_or(false)
        {
            self.root = None;
        }

        changed
    }

    pub fn remove_chunk(&mut self, key: ChunkKey) -> bool {
        self.set_chunk(key, None)
    }

    pub fn any_non_empty_chunk_in_bounds(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        self.root
            .as_ref()
            .map(|node| kind_has_non_empty_chunk_intersection(&node.kind, node.bounds, bounds))
            .unwrap_or(false)
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.root
            .as_ref()
            .map(|node| count_non_empty_chunks(&node.kind, node.bounds))
            .unwrap_or(0)
    }

    pub fn collect_chunks(&self) -> Vec<(ChunkKey, ChunkPayload)> {
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunks_from_kind(&root.kind, root.bounds, &mut out);
        }
        out
    }

    pub fn collect_chunks_in_bounds(&self, bounds: Aabb4i) -> Vec<(ChunkKey, ChunkPayload)> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunks_from_kind_in_bounds(&root.kind, root.bounds, bounds, &mut out);
        }
        out
    }

    pub fn collect_chunk_keys_in_bounds(&self, bounds: Aabb4i) -> Vec<ChunkKey> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunk_keys_from_kind_in_bounds(&root.kind, root.bounds, bounds, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn collect_non_empty_chunk_keys_in_bounds(&self, bounds: Aabb4i) -> Vec<ChunkKey> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_non_empty_chunk_keys_from_kind_in_bounds(
                &root.kind,
                root.bounds,
                bounds,
                &mut out,
            );
        }
        out.sort_unstable_by_key(|key| *key);
        out
    }

    pub fn slice_core_in_bounds(&self, bounds: Aabb4i) -> RegionTreeCore {
        let Some(root) = self.root.as_ref() else {
            return empty_core_for_bounds(bounds);
        };
        slice_region_core_in_bounds(root.as_ref(), bounds)
    }

    pub fn slice_non_empty_core_in_bounds(&self, bounds: Aabb4i) -> RegionTreeCore {
        let Some(root) = self.root.as_ref() else {
            return empty_core_for_bounds(bounds);
        };
        slice_non_empty_region_core_in_bounds(root.as_ref(), bounds)
    }

    pub fn take_non_empty_core_in_bounds(&mut self, bounds: Aabb4i) -> RegionTreeCore {
        if !bounds.is_valid() {
            return empty_core_for_bounds(bounds);
        }

        let extracted = self.slice_non_empty_core_in_bounds(bounds);
        if matches!(extracted.kind, RegionNodeKind::Empty) {
            return extracted;
        }

        let empty_replacement = empty_core_for_bounds(bounds);
        let _ = self.splice_non_empty_core_in_bounds(bounds, &empty_replacement);
        extracted
    }

    pub fn lazy_drop_outside_bounds(
        &mut self,
        keep_bounds: Aabb4i,
        max_subtree_drops: usize,
    ) -> Option<Aabb4i> {
        if max_subtree_drops == 0 {
            return None;
        }
        let Some(root) = self.root.as_mut() else {
            return None;
        };

        let mut budget = max_subtree_drops;
        let changed_bounds = lazy_drop_outside_node(root, keep_bounds, &mut budget);
        if changed_bounds.is_some() {
            normalize_chunk_node(root);
        }
        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }
        changed_bounds
    }

    pub fn splice_non_empty_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        core: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let replacement = slice_non_empty_region_core_in_bounds(core, bounds);
        let replacement_is_empty = matches!(replacement.kind, RegionNodeKind::Empty);

        if self.root.is_none() {
            if replacement_is_empty {
                return None;
            }
            self.root = Some(Box::new(replacement));
            return Some(bounds);
        }

        if !replacement_is_empty {
            self.ensure_root_contains_bounds(bounds);
        }

        let changed_bounds = if let Some(root) = self.root.as_mut() {
            splice_node_with_non_empty_core(root, bounds, &replacement)
        } else {
            None
        };

        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }

        changed_bounds
    }

    pub fn splice_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        core: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let replacement = slice_region_core_in_bounds(core, bounds);
        let replacement_is_empty = matches!(replacement.kind, RegionNodeKind::Empty);

        if self.root.is_none() {
            if replacement_is_empty {
                return None;
            }
            self.root = Some(Box::new(replacement));
            return Some(bounds);
        }

        if !replacement_is_empty {
            self.ensure_root_contains_bounds(bounds);
        }

        let changed_bounds = if let Some(root) = self.root.as_mut() {
            splice_node_with_core(root, bounds, &replacement)
        } else {
            None
        };

        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }

        changed_bounds
    }

    pub fn overlay_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        overlay: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let overlay_slice = slice_region_core_in_bounds(overlay, bounds);
        if matches!(overlay_slice.kind, RegionNodeKind::Empty) {
            return None;
        }

        let mut changed = None::<Aabb4i>;
        overlay_non_empty_leaves(&overlay_slice, &mut |leaf| {
            if let Some(splice_changed) = self.splice_core_in_bounds(leaf.bounds, leaf) {
                changed = Some(match changed {
                    Some(existing) => merge_aabb(existing, splice_changed),
                    None => splice_changed,
                });
            }
        });
        changed
    }

    fn ensure_root_contains_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            return;
        }

        if self.root.is_none() {
            self.root = Some(Box::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            }));
            return;
        }

        while self
            .root
            .as_ref()
            .map(|root| {
                !root.bounds.contains_chunk(bounds.min) || !root.bounds.contains_chunk(bounds.max)
            })
            .unwrap_or(false)
        {
            let grow_key = self
                .root
                .as_ref()
                .map(|root| {
                    if !root.bounds.contains_chunk(bounds.min) {
                        bounds.min
                    } else {
                        bounds.max
                    }
                })
                .unwrap_or(bounds.min);
            let Some(root) = self.root.take() else {
                break;
            };
            self.root = Some(expand_root_once(root, grow_key));
        }
    }
}

pub fn collect_non_empty_chunks_from_core_in_bounds(
    core: &RegionTreeCore,
    bounds: Aabb4i,
) -> Vec<(ChunkKey, ChunkPayload)> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let mut out = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(&core.kind, core.bounds, bounds, &mut out);
    out.sort_unstable_by_key(|(key, _)| *key);
    out
}

pub fn slice_region_core_in_bounds(core: &RegionTreeCore, bounds: Aabb4i) -> RegionTreeCore {
    if !bounds.is_valid() {
        return empty_core_for_bounds(bounds);
    }

    let Some(intersection) = intersect_aabb(core.bounds, bounds) else {
        return empty_core_for_bounds(bounds);
    };
    let Some(clipped) = slice_node_to_bounds(core, intersection) else {
        return empty_core_for_bounds(bounds);
    };

    if clipped.bounds == bounds {
        clipped
    } else {
        RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Branch(vec![clipped]),
            generator_version_hash: core.generator_version_hash,
        }
    }
}

pub fn slice_non_empty_region_core_in_bounds(
    core: &RegionTreeCore,
    bounds: Aabb4i,
) -> RegionTreeCore {
    let mut sliced = slice_region_core_in_bounds(core, bounds);
    prune_empty_subtrees(&mut sliced);
    sliced
}

fn empty_core_for_bounds(bounds: Aabb4i) -> RegionTreeCore {
    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    }
}

fn slice_node_to_bounds(node: &RegionTreeCore, bounds: Aabb4i) -> Option<RegionTreeCore> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let kind = match &node.kind {
        RegionNodeKind::Empty => RegionNodeKind::Empty,
        RegionNodeKind::Uniform(material) => RegionNodeKind::Uniform(*material),
        RegionNodeKind::ProceduralRef(generator_ref) => {
            RegionNodeKind::ProceduralRef(generator_ref.clone())
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let sliced = if intersection == chunk_array.bounds {
                chunk_array.clone()
            } else {
                slice_chunk_array_to_bounds(chunk_array, intersection)?
            };
            RegionNodeKind::ChunkArray(sliced)
        }
        RegionNodeKind::Branch(children) => {
            let mut clipped_children = Vec::new();
            for child in children {
                if let Some(clipped) = slice_node_to_bounds(child, intersection) {
                    clipped_children.push(clipped);
                }
            }
            if clipped_children.is_empty() {
                RegionNodeKind::Empty
            } else if clipped_children.len() == 1 && clipped_children[0].bounds == intersection {
                return Some(clipped_children.pop().expect("single clipped child"));
            } else {
                RegionNodeKind::Branch(clipped_children)
            }
        }
    };

    Some(RegionTreeCore {
        bounds: intersection,
        kind,
        generator_version_hash: node.generator_version_hash,
    })
}

fn slice_chunk_array_to_bounds(
    chunk_array: &ChunkArrayData,
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = intersect_aabb(chunk_array.bounds, bounds)?;
    let source_indices = chunk_array.decode_dense_indices().ok()?;
    slice_chunk_array_to_bounds_with_dense_indices(chunk_array, &source_indices, intersection)
}

fn slice_chunk_array_to_bounds_with_dense_indices(
    chunk_array: &ChunkArrayData,
    source_indices: &[u16],
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = intersect_aabb(chunk_array.bounds, bounds)?;
    let source_extents = chunk_array.bounds.chunk_extents()?;
    let target_extents = intersection.chunk_extents()?;
    let target_cell_count = target_extents[0]
        .checked_mul(target_extents[1])?
        .checked_mul(target_extents[2])?
        .checked_mul(target_extents[3])?;
    let mut target_indices = Vec::with_capacity(target_cell_count);

    for w in intersection.min[3]..=intersection.max[3] {
        for z in intersection.min[2]..=intersection.max[2] {
            for y in intersection.min[1]..=intersection.max[1] {
                for x in intersection.min[0]..=intersection.max[0] {
                    let source_local = [
                        usize::try_from(x - chunk_array.bounds.min[0]).ok()?,
                        usize::try_from(y - chunk_array.bounds.min[1]).ok()?,
                        usize::try_from(z - chunk_array.bounds.min[2]).ok()?,
                        usize::try_from(w - chunk_array.bounds.min[3]).ok()?,
                    ];
                    let source_linear = linear_cell_index(source_local, source_extents);
                    target_indices.push(*source_indices.get(source_linear)?);
                }
            }
        }
    }

    ChunkArrayData::from_dense_indices(
        intersection,
        chunk_array.chunk_palette.clone(),
        target_indices,
        None,
    )
    .ok()
}

fn prune_empty_subtrees(core: &mut RegionTreeCore) -> bool {
    match &mut core.kind {
        RegionNodeKind::Empty => false,
        RegionNodeKind::Uniform(material) => *material != 0,
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_has_non_empty_intersection(chunk_array, chunk_array.bounds)
        }
        RegionNodeKind::Branch(children) => {
            children.retain_mut(prune_empty_subtrees);
            if children.is_empty() {
                core.kind = RegionNodeKind::Empty;
                false
            } else if children.len() == 1 && children[0].bounds == core.bounds {
                let child = children.pop().expect("single child");
                core.kind = child.kind;
                true
            } else {
                true
            }
        }
    }
}

fn splice_node_with_non_empty_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let replacement_slice = slice_non_empty_region_core_in_bounds(replacement, intersection);

    if matches!(replacement_slice.kind, RegionNodeKind::Empty)
        && !kind_has_non_empty_chunk_intersection(&node.kind, node.bounds, intersection)
    {
        return None;
    }

    match (&node.kind, &replacement_slice.kind) {
        (RegionNodeKind::Empty, RegionNodeKind::Empty) => return None,
        (RegionNodeKind::Uniform(existing), RegionNodeKind::Uniform(incoming))
            if existing == incoming =>
        {
            return None;
        }
        (RegionNodeKind::Uniform(existing), RegionNodeKind::Empty) if *existing == 0 => {
            return None;
        }
        (RegionNodeKind::ProceduralRef(existing), RegionNodeKind::ProceduralRef(incoming))
            if existing == incoming =>
        {
            return None;
        }
        _ => {}
    }

    if matches!(
        node.kind,
        RegionNodeKind::Branch(_) | RegionNodeKind::ChunkArray(_)
    ) {
        let existing_slice = slice_non_empty_region_core_in_bounds(node, intersection);
        if existing_slice.kind == replacement_slice.kind {
            return None;
        }
    }
    if non_empty_kinds_semantically_equal_in_bounds(
        &node.kind,
        node.bounds,
        &replacement_slice.kind,
        replacement_slice.bounds,
        intersection,
    ) {
        return None;
    }

    if intersection == node.bounds {
        if node.kind == replacement_slice.kind {
            return None;
        }
        *node = replacement_slice;
        return Some(node.bounds);
    }

    let mut changed = clear_node_region(node, intersection);
    if !matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        changed |= insert_replacement_slice_into_node(node, replacement_slice);
    }

    if !changed {
        return None;
    }
    normalize_chunk_node(node);
    Some(intersection)
}

fn splice_node_with_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let replacement_slice = slice_region_core_in_bounds(replacement, intersection);
    let existing_slice = slice_region_core_in_bounds(node, intersection);
    if existing_slice.kind == replacement_slice.kind {
        return None;
    }

    if intersection == node.bounds {
        if node.kind == replacement_slice.kind {
            return None;
        }
        *node = replacement_slice;
        return Some(node.bounds);
    }

    let mut changed = clear_node_region(node, intersection);
    if !matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        changed |= insert_replacement_slice_into_node(node, replacement_slice);
    }

    if !changed {
        return None;
    }
    normalize_chunk_node(node);
    Some(intersection)
}

fn clear_node_region(node: &mut RegionTreeCore, clear_bounds: Aabb4i) -> bool {
    let Some(intersection) = intersect_aabb(node.bounds, clear_bounds) else {
        return false;
    };

    if matches!(node.kind, RegionNodeKind::Empty) {
        return false;
    }

    if intersection == node.bounds {
        node.kind = RegionNodeKind::Empty;
        return true;
    }

    match &mut node.kind {
        RegionNodeKind::Branch(children) => {
            let mut changed = false;
            let mut retained = Vec::with_capacity(children.len());
            for mut child in std::mem::take(children) {
                if !child.bounds.intersects(&intersection) {
                    retained.push(child);
                    continue;
                }
                if aabb_contains_aabb(intersection, child.bounds) {
                    changed = true;
                    continue;
                }
                if clear_node_region(&mut child, intersection) {
                    changed = true;
                }
                if !matches!(child.kind, RegionNodeKind::Empty) {
                    retained.push(child);
                }
            }
            *children = retained;
            if changed {
                normalize_chunk_node(node);
            }
            changed
        }
        _ => {
            let old_kind = node.kind.clone();
            let mut pieces = Vec::new();
            for piece_bounds in subtract_aabb(node.bounds, intersection) {
                let projected = project_node_to_bounds(
                    &old_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    pieces.push(projected);
                }
            }
            node.kind = RegionNodeKind::Branch(pieces);
            normalize_chunk_node(node);
            true
        }
    }
}

fn insert_replacement_slice_into_node(
    node: &mut RegionTreeCore,
    replacement_slice: RegionTreeCore,
) -> bool {
    if matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        return false;
    }

    if replacement_slice.bounds == node.bounds {
        if node.kind == replacement_slice.kind {
            return false;
        }
        *node = replacement_slice;
        return true;
    }

    let replacement_bounds = replacement_slice.bounds;
    match &mut node.kind {
        RegionNodeKind::Empty => {
            node.kind = RegionNodeKind::Branch(vec![replacement_slice]);
            normalize_chunk_node(node);
            return !matches!(node.kind, RegionNodeKind::Empty);
        }
        RegionNodeKind::Branch(children)
            if !children
                .iter()
                .any(|child| child.bounds.intersects(&replacement_bounds)) =>
        {
            children.push(replacement_slice);
            normalize_chunk_node(node);
            return true;
        }
        _ => {}
    }

    insert_replacement_slice_into_node_with_rebuild(node, replacement_slice)
}

fn insert_replacement_slice_into_node_with_rebuild(
    node: &mut RegionTreeCore,
    replacement_slice: RegionTreeCore,
) -> bool {
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let old_kind_snapshot = old_kind.clone();
    let mut children = Vec::new();
    match old_kind {
        RegionNodeKind::Empty => {}
        RegionNodeKind::Branch(existing_children) => {
            for child in existing_children {
                for piece_bounds in subtract_aabb(child.bounds, replacement_slice.bounds) {
                    let projected = project_node_to_bounds(
                        &child.kind,
                        child.bounds,
                        piece_bounds,
                        node.generator_version_hash,
                    );
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        children.push(projected);
                    }
                }
            }
        }
        other_kind => {
            for piece_bounds in subtract_aabb(node.bounds, replacement_slice.bounds) {
                let projected = project_node_to_bounds(
                    &other_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
    }
    children.push(replacement_slice);
    node.kind = RegionNodeKind::Branch(children);
    normalize_chunk_node(node);
    node.kind != old_kind_snapshot
}

fn lazy_drop_outside_node(
    node: &mut RegionTreeCore,
    keep_bounds: Aabb4i,
    budget: &mut usize,
) -> Option<Aabb4i> {
    if *budget == 0 {
        return None;
    }
    if !keep_bounds.is_valid() || !node.bounds.intersects(&keep_bounds) {
        if matches!(node.kind, RegionNodeKind::Empty) {
            return None;
        }
        *budget -= 1;
        let changed = node.bounds;
        node.kind = RegionNodeKind::Empty;
        return Some(changed);
    }
    if aabb_contains_aabb(keep_bounds, node.bounds) || is_single_chunk_bounds(node.bounds) {
        return None;
    }

    ensure_binary_children(node);
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return None;
    };

    let mut changed = None::<Aabb4i>;
    for child in children.iter_mut() {
        if *budget == 0 {
            break;
        }
        if let Some(child_changed) = lazy_drop_outside_node(child, keep_bounds, budget) {
            changed = Some(match changed {
                Some(acc) => merge_aabb(acc, child_changed),
                None => child_changed,
            });
        }
    }

    normalize_chunk_node(node);
    changed
}

fn overlay_non_empty_leaves<F>(core: &RegionTreeCore, visit: &mut F)
where
    F: FnMut(&RegionTreeCore),
{
    match &core.kind {
        RegionNodeKind::Empty => {}
        RegionNodeKind::Branch(children) => {
            for child in children {
                overlay_non_empty_leaves(child, visit);
            }
        }
        _ => visit(core),
    }
}

fn aabb_contains_aabb(outer: Aabb4i, inner: Aabb4i) -> bool {
    outer.is_valid()
        && inner.is_valid()
        && outer.min[0] <= inner.min[0]
        && outer.min[1] <= inner.min[1]
        && outer.min[2] <= inner.min[2]
        && outer.min[3] <= inner.min[3]
        && outer.max[0] >= inner.max[0]
        && outer.max[1] >= inner.max[1]
        && outer.max[2] >= inner.max[2]
        && outer.max[3] >= inner.max[3]
}

fn merge_aabb(a: Aabb4i, b: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
            a.min[3].min(b.min[3]),
        ],
        [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
            a.max[3].max(b.max[3]),
        ],
    )
}

fn subtract_aabb(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    let Some(inner) = intersect_aabb(outer, inner) else {
        return vec![outer];
    };
    if inner == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;

    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis] - 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis] + 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = inner.max[axis];
        }
    }

    pieces
}

fn collect_non_empty_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ChunkPayload)>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            if *material == 0 {
                return;
            }
            let payload = ChunkPayload::Uniform(*material);
            for w in intersection.min[3]..=intersection.max[3] {
                for z in intersection.min[2]..=intersection.max[2] {
                    for y in intersection.min[1]..=intersection.max[1] {
                        for x in intersection.min[0]..=intersection.max[0] {
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            for w in chunk_array_intersection.min[3]..=chunk_array_intersection.max[3] {
                for z in chunk_array_intersection.min[2]..=chunk_array_intersection.max[2] {
                    for y in chunk_array_intersection.min[1]..=chunk_array_intersection.max[1] {
                        for x in chunk_array_intersection.min[0]..=chunk_array_intersection.max[0] {
                            let local = [
                                (x - chunk_array.bounds.min[0]) as usize,
                                (y - chunk_array.bounds.min[1]) as usize,
                                (z - chunk_array.bounds.min[2]) as usize,
                                (w - chunk_array.bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let palette_idx = *palette_idx as usize;
                            if !palette_non_empty.get(palette_idx).copied().unwrap_or(true) {
                                continue;
                            }
                            let Some(payload) = chunk_array.chunk_palette.get(palette_idx) else {
                                continue;
                            };
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_chunks_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn collect_non_empty_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            if *material == 0 {
                return;
            }
            for w in intersection.min[3]..=intersection.max[3] {
                for z in intersection.min[2]..=intersection.max[2] {
                    for y in intersection.min[1]..=intersection.max[1] {
                        for x in intersection.min[0]..=intersection.max[0] {
                            out.push([x, y, z, w]);
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            for w in chunk_array_intersection.min[3]..=chunk_array_intersection.max[3] {
                for z in chunk_array_intersection.min[2]..=chunk_array_intersection.max[2] {
                    for y in chunk_array_intersection.min[1]..=chunk_array_intersection.max[1] {
                        for x in chunk_array_intersection.min[0]..=chunk_array_intersection.max[0] {
                            let local = [
                                (x - chunk_array.bounds.min[0]) as usize,
                                (y - chunk_array.bounds.min[1]) as usize,
                                (z - chunk_array.bounds.min[2]) as usize,
                                (w - chunk_array.bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let palette_idx = *palette_idx as usize;
                            if !palette_non_empty.get(palette_idx).copied().unwrap_or(true) {
                                continue;
                            }
                            out.push([x, y, z, w]);
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_chunk_keys_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn set_chunk_recursive(
    node: &mut RegionTreeCore,
    key_pos: [i32; 4],
    payload: Option<ChunkPayload>,
) -> bool {
    if !node.bounds.contains_chunk(key_pos) {
        return false;
    }

    if is_single_chunk_bounds(node.bounds) {
        let new_kind = kind_from_chunk_value(node.bounds, payload);
        if node.kind == new_kind {
            return false;
        }
        node.kind = new_kind;
        return true;
    }

    if matches!(node.kind, RegionNodeKind::Branch(_)) {
        return set_chunk_recursive_in_branch(node, key_pos, payload);
    }

    carve_leaf_for_chunk_edit(node, key_pos, payload)
}

fn set_chunk_recursive_in_branch(
    node: &mut RegionTreeCore,
    key_pos: [i32; 4],
    payload: Option<ChunkPayload>,
) -> bool {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return false;
    };
    let target_idx = children
        .iter()
        .position(|child| child.bounds.contains_chunk(key_pos));
    let changed = if let Some(target_idx) = target_idx {
        set_chunk_recursive(&mut children[target_idx], key_pos, payload)
    } else if let Some(payload) = payload {
        let chunk_bounds = Aabb4i::new(key_pos, key_pos);
        children.push(RegionTreeCore {
            bounds: chunk_bounds,
            kind: kind_from_chunk_value(chunk_bounds, Some(payload)),
            generator_version_hash: node.generator_version_hash,
        });
        true
    } else {
        false
    };

    if changed {
        normalize_chunk_node(node);
    }
    changed
}

fn carve_leaf_for_chunk_edit(
    node: &mut RegionTreeCore,
    key_pos: [i32; 4],
    payload: Option<ChunkPayload>,
) -> bool {
    let chunk_bounds = Aabb4i::new(key_pos, key_pos);
    let source_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);

    let no_change = match &source_kind {
        RegionNodeKind::Empty => payload_option_is_semantically_empty(payload.as_ref()),
        RegionNodeKind::Uniform(material) => {
            payload_option_matches_uniform(*material, payload.as_ref())
        }
        RegionNodeKind::ChunkArray(_) => false,
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::Branch(_) => false,
    };
    if no_change {
        node.kind = source_kind;
        return false;
    }

    let mut children = Vec::with_capacity(9);
    if let RegionNodeKind::ChunkArray(chunk_array) = &source_kind {
        if let Ok(source_indices) = chunk_array.decode_dense_indices() {
            if payload_option_matches_existing(
                chunk_array_payload_at_with_dense_indices(chunk_array, &source_indices, key_pos),
                payload.as_ref(),
            ) {
                node.kind = source_kind;
                return false;
            }
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
                let Some(chunk_array_piece) = slice_chunk_array_to_bounds_with_dense_indices(
                    chunk_array,
                    &source_indices,
                    piece_bounds,
                ) else {
                    continue;
                };
                children.push(RegionTreeCore {
                    bounds: piece_bounds,
                    kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                    generator_version_hash: node.generator_version_hash,
                });
            }
        } else {
            if payload_option_matches_existing(
                chunk_array_payload_at(chunk_array, key_pos),
                payload.as_ref(),
            ) {
                node.kind = source_kind;
                return false;
            }
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
                let projected = project_node_to_bounds(
                    &source_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
    } else {
        for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
            let projected = project_node_to_bounds(
                &source_kind,
                node.bounds,
                piece_bounds,
                node.generator_version_hash,
            );
            if !matches!(projected.kind, RegionNodeKind::Empty) {
                children.push(projected);
            }
        }
    }

    if let Some(payload) = payload {
        children.push(RegionTreeCore {
            bounds: chunk_bounds,
            kind: kind_from_chunk_value(chunk_bounds, Some(payload)),
            generator_version_hash: node.generator_version_hash,
        });
    }

    node.kind = RegionNodeKind::Branch(children);
    normalize_chunk_node(node);
    true
}

fn ensure_binary_children(node: &mut RegionTreeCore) {
    if is_single_chunk_bounds(node.bounds) {
        return;
    }

    if let RegionNodeKind::Branch(children) = &mut node.kind {
        if branch_matches_split(node.bounds, children) {
            sort_children_canonical(children);
            return;
        }
    }

    let Some((left_bounds, right_bounds)) = split_bounds_longest_axis(node.bounds) else {
        return;
    };

    let generator_version_hash = node.generator_version_hash;
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let left = project_node_to_bounds(&old_kind, node.bounds, left_bounds, generator_version_hash);
    let right =
        project_node_to_bounds(&old_kind, node.bounds, right_bounds, generator_version_hash);
    let mut children = vec![left, right];
    sort_children_canonical(&mut children);
    node.kind = RegionNodeKind::Branch(children);
}

fn normalize_chunk_node(node: &mut RegionTreeCore) {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return;
    };
    for child in children.iter_mut() {
        normalize_chunk_node(child);
    }
    children.retain(|child| !matches!(child.kind, RegionNodeKind::Empty));
    if children.is_empty() {
        node.kind = RegionNodeKind::Empty;
        return;
    }
    if children.len() == 1 && children[0].bounds == node.bounds {
        let child = children.pop().expect("single child");
        node.kind = child.kind;
        return;
    }

    sort_children_canonical(children);
    if children.len() != 2 {
        return;
    }
    if !branch_matches_split(node.bounds, children) {
        // Only collapse two-child branches when they are the canonical binary partition
        // of this node's bounds. Two arbitrary non-overlapping children with the same kind
        // must not fill gaps in the parent region.
        return;
    }

    sort_children_canonical(children);
    let left_kind = children[0].kind.clone();
    let right_kind = children[1].kind.clone();
    match (left_kind, right_kind) {
        (RegionNodeKind::Empty, RegionNodeKind::Empty) => {
            node.kind = RegionNodeKind::Empty;
        }
        (RegionNodeKind::Uniform(a), RegionNodeKind::Uniform(b)) if a == b => {
            node.kind = RegionNodeKind::Uniform(a);
        }
        (RegionNodeKind::ProceduralRef(a), RegionNodeKind::ProceduralRef(b)) if a == b => {
            node.kind = RegionNodeKind::ProceduralRef(a);
        }
        _ => {}
    }
}

fn project_node_to_bounds(
    source_kind: &RegionNodeKind,
    source_bounds: Aabb4i,
    target_bounds: Aabb4i,
    generator_version_hash: u64,
) -> RegionTreeCore {
    if !target_bounds.is_valid() || !source_bounds.intersects(&target_bounds) {
        return RegionTreeCore {
            bounds: target_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash,
        };
    }

    let kind = match source_kind {
        RegionNodeKind::Empty => RegionNodeKind::Empty,
        RegionNodeKind::Uniform(material) => RegionNodeKind::Uniform(*material),
        RegionNodeKind::ProceduralRef(generator_ref) => {
            RegionNodeKind::ProceduralRef(generator_ref.clone())
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            match slice_chunk_array_to_bounds(chunk_array, target_bounds) {
                Some(sliced) => RegionNodeKind::ChunkArray(sliced),
                None => RegionNodeKind::Empty,
            }
        }
        RegionNodeKind::Branch(children) => {
            let mut clipped_children = Vec::new();
            for child in children {
                let Some(clipped) = slice_node_to_bounds(child, target_bounds) else {
                    continue;
                };
                if matches!(clipped.kind, RegionNodeKind::Empty) {
                    continue;
                }
                clipped_children.push(clipped);
            }

            if clipped_children.is_empty() {
                RegionNodeKind::Empty
            } else if clipped_children.len() == 1 && clipped_children[0].bounds == target_bounds {
                return clipped_children.pop().expect("single projected child");
            } else {
                RegionNodeKind::Branch(clipped_children)
            }
        }
    };

    let mut projected = RegionTreeCore {
        bounds: target_bounds,
        kind,
        generator_version_hash,
    };
    if matches!(projected.kind, RegionNodeKind::Branch(_)) {
        normalize_chunk_node(&mut projected);
    }
    projected
}

fn query_chunk_payload_in_node(node: &RegionTreeCore, key_pos: [i32; 4]) -> Option<ChunkPayload> {
    query_chunk_payload_in_kind(&node.kind, node.bounds, key_pos)
}

fn query_chunk_payload_in_kind(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    key_pos: [i32; 4],
) -> Option<ChunkPayload> {
    if !bounds.contains_chunk(key_pos) {
        return None;
    }
    match kind {
        RegionNodeKind::Empty => None,
        RegionNodeKind::Uniform(material) => Some(ChunkPayload::Uniform(*material)),
        RegionNodeKind::ProceduralRef(_) => None,
        RegionNodeKind::ChunkArray(chunk_array) => chunk_array_payload_at(chunk_array, key_pos),
        RegionNodeKind::Branch(children) => {
            for child in children {
                if child.bounds.contains_chunk(key_pos) {
                    return query_chunk_payload_in_kind(&child.kind, child.bounds, key_pos);
                }
            }
            None
        }
    }
}

fn kind_has_non_empty_chunk_intersection(
    kind: &RegionNodeKind,
    kind_bounds: Aabb4i,
    query_bounds: Aabb4i,
) -> bool {
    if !kind_bounds.intersects(&query_bounds) {
        return false;
    }
    match kind {
        RegionNodeKind::Empty => false,
        RegionNodeKind::Uniform(material) => *material != 0,
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_has_non_empty_intersection(chunk_array, query_bounds)
        }
        RegionNodeKind::Branch(children) => children.iter().any(|child| {
            kind_has_non_empty_chunk_intersection(&child.kind, child.bounds, query_bounds)
        }),
    }
}

fn count_non_empty_chunks(kind: &RegionNodeKind, bounds: Aabb4i) -> usize {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => 0,
        RegionNodeKind::Uniform(material) => {
            if *material == 0 {
                0
            } else {
                bounds.chunk_cell_count().unwrap_or(0)
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return 0;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            indices
                .into_iter()
                .filter(|idx| {
                    palette_non_empty
                        .get(*idx as usize)
                        .copied()
                        .unwrap_or(true)
                })
                .count()
        }
        RegionNodeKind::Branch(children) => children
            .iter()
            .map(|child| count_non_empty_chunks(&child.kind, child.bounds))
            .sum(),
    }
}

fn collect_chunks_from_kind(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ChunkPayload)>,
) {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            let payload = ChunkPayload::Uniform(*material);
            for w in bounds.min[3]..=bounds.max[3] {
                for z in bounds.min[2]..=bounds.max[2] {
                    for y in bounds.min[1]..=bounds.max[1] {
                        for x in bounds.min[0]..=bounds.max[0] {
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            for w in chunk_array.bounds.min[3]..=chunk_array.bounds.max[3] {
                for z in chunk_array.bounds.min[2]..=chunk_array.bounds.max[2] {
                    for y in chunk_array.bounds.min[1]..=chunk_array.bounds.max[1] {
                        for x in chunk_array.bounds.min[0]..=chunk_array.bounds.max[0] {
                            let local = [
                                (x - chunk_array.bounds.min[0]) as usize,
                                (y - chunk_array.bounds.min[1]) as usize,
                                (z - chunk_array.bounds.min[2]) as usize,
                                (w - chunk_array.bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let Some(payload) =
                                chunk_array.chunk_palette.get(*palette_idx as usize)
                            else {
                                continue;
                            };
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunks_from_kind(&child.kind, child.bounds, out);
            }
        }
    }
}

fn collect_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ChunkPayload)>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            let payload = ChunkPayload::Uniform(*material);
            for w in intersection.min[3]..=intersection.max[3] {
                for z in intersection.min[2]..=intersection.max[2] {
                    for y in intersection.min[1]..=intersection.max[1] {
                        for x in intersection.min[0]..=intersection.max[0] {
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            for w in chunk_array_intersection.min[3]..=chunk_array_intersection.max[3] {
                for z in chunk_array_intersection.min[2]..=chunk_array_intersection.max[2] {
                    for y in chunk_array_intersection.min[1]..=chunk_array_intersection.max[1] {
                        for x in chunk_array_intersection.min[0]..=chunk_array_intersection.max[0] {
                            let local = [
                                (x - chunk_array.bounds.min[0]) as usize,
                                (y - chunk_array.bounds.min[1]) as usize,
                                (z - chunk_array.bounds.min[2]) as usize,
                                (w - chunk_array.bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let Some(payload) =
                                chunk_array.chunk_palette.get(*palette_idx as usize)
                            else {
                                continue;
                            };
                            out.push(([x, y, z, w], payload.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunks_from_kind_in_bounds(&child.kind, child.bounds, query_bounds, out);
            }
        }
    }
}

fn collect_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(_) | RegionNodeKind::ChunkArray(_) => {
            for w in intersection.min[3]..=intersection.max[3] {
                for z in intersection.min[2]..=intersection.max[2] {
                    for y in intersection.min[1]..=intersection.max[1] {
                        for x in intersection.min[0]..=intersection.max[0] {
                            out.push([x, y, z, w]);
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunk_keys_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn chunk_array_has_non_empty_intersection(
    chunk_array: &ChunkArrayData,
    query_bounds: Aabb4i,
) -> bool {
    let Some(intersection) = intersect_aabb(chunk_array.bounds, query_bounds) else {
        return false;
    };
    let Ok(indices) = chunk_array.decode_dense_indices() else {
        // Conservatively treat malformed payload as potentially non-empty.
        return true;
    };
    let Some(extents) = chunk_array.bounds.chunk_extents() else {
        return true;
    };
    let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);

    for w in intersection.min[3]..=intersection.max[3] {
        for z in intersection.min[2]..=intersection.max[2] {
            for y in intersection.min[1]..=intersection.max[1] {
                for x in intersection.min[0]..=intersection.max[0] {
                    let local = [
                        (x - chunk_array.bounds.min[0]) as usize,
                        (y - chunk_array.bounds.min[1]) as usize,
                        (z - chunk_array.bounds.min[2]) as usize,
                        (w - chunk_array.bounds.min[3]) as usize,
                    ];
                    let linear = linear_cell_index(local, extents);
                    let Some(palette_idx) = indices.get(linear) else {
                        return true;
                    };
                    if palette_non_empty
                        .get(*palette_idx as usize)
                        .copied()
                        .unwrap_or(true)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn payload_has_solid_material(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(material) => *material != 0,
        ChunkPayload::Dense16 { materials } => materials.iter().any(|m| *m != 0),
        ChunkPayload::PalettePacked { .. } => payload
            .dense_materials()
            .map(|dense| dense.into_iter().any(|m| m != 0))
            .unwrap_or(true),
    }
}

fn non_empty_kinds_semantically_equal_in_bounds(
    lhs_kind: &RegionNodeKind,
    lhs_bounds: Aabb4i,
    rhs_kind: &RegionNodeKind,
    rhs_bounds: Aabb4i,
    bounds: Aabb4i,
) -> bool {
    const MAX_COMPARE_CELLS: usize = 4096;
    if bounds
        .chunk_cell_count()
        .map(|count| count > MAX_COMPARE_CELLS)
        .unwrap_or(true)
    {
        return false;
    }

    let mut lhs_chunks = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(lhs_kind, lhs_bounds, bounds, &mut lhs_chunks);
    lhs_chunks.sort_unstable_by_key(|(key, _)| *key);

    let mut rhs_chunks = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(rhs_kind, rhs_bounds, bounds, &mut rhs_chunks);
    rhs_chunks.sort_unstable_by_key(|(key, _)| *key);

    lhs_chunks == rhs_chunks
}

fn chunk_array_palette_non_empty_mask(chunk_array: &ChunkArrayData) -> Vec<bool> {
    chunk_array
        .chunk_palette
        .iter()
        .map(payload_has_solid_material)
        .collect()
}

fn chunk_array_payload_at(chunk_array: &ChunkArrayData, key_pos: [i32; 4]) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    chunk_array_payload_at_with_dense_indices(chunk_array, &dense_indices, key_pos)
}

fn chunk_array_payload_at_with_dense_indices(
    chunk_array: &ChunkArrayData,
    dense_indices: &[u16],
    key_pos: [i32; 4],
) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk(key_pos) {
        return None;
    }
    let extents = chunk_array.bounds.chunk_extents()?;
    let local = [
        (key_pos[0] - chunk_array.bounds.min[0]) as usize,
        (key_pos[1] - chunk_array.bounds.min[1]) as usize,
        (key_pos[2] - chunk_array.bounds.min[2]) as usize,
        (key_pos[3] - chunk_array.bounds.min[3]) as usize,
    ];
    let linear = linear_cell_index(local, extents);
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
}

fn payload_option_is_semantically_empty(payload: Option<&ChunkPayload>) -> bool {
    payload
        .map(|payload| !payload_has_solid_material(payload))
        .unwrap_or(true)
}

fn payload_option_matches_uniform(material: u16, payload: Option<&ChunkPayload>) -> bool {
    match payload {
        Some(payload) => {
            canonicalize_chunk_payload(payload.clone()) == ChunkPayload::Uniform(material)
        }
        None => material == 0,
    }
}

fn payload_option_matches_existing(
    existing: Option<ChunkPayload>,
    payload: Option<&ChunkPayload>,
) -> bool {
    match payload {
        Some(payload) => {
            existing.map(canonicalize_chunk_payload)
                == Some(canonicalize_chunk_payload(payload.clone()))
        }
        None => existing
            .as_ref()
            .map(|payload| !payload_has_solid_material(payload))
            .unwrap_or(true),
    }
}

fn canonicalize_chunk_payload(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        other => other,
    };
    let Ok(dense) = payload.dense_materials() else {
        return payload;
    };
    if dense.is_empty() {
        return payload;
    }
    let first = dense[0];
    if dense.iter().all(|m| *m == first) {
        ChunkPayload::Uniform(first)
    } else {
        payload
    }
}

fn kind_from_chunk_value(bounds: Aabb4i, value: Option<ChunkPayload>) -> RegionNodeKind {
    let Some(payload) = value else {
        return RegionNodeKind::Empty;
    };
    match canonicalize_chunk_payload(payload) {
        ChunkPayload::Uniform(material) => RegionNodeKind::Uniform(material),
        other => repeated_payload_kind(bounds, other),
    }
}

fn repeated_payload_kind(bounds: Aabb4i, payload: ChunkPayload) -> RegionNodeKind {
    let Some(cell_count) = bounds.chunk_cell_count() else {
        return RegionNodeKind::Empty;
    };
    let indices = vec![0u16; cell_count];
    match ChunkArrayData::from_dense_indices(bounds, vec![payload], indices, Some(0)) {
        Ok(chunk_array) => RegionNodeKind::ChunkArray(chunk_array),
        Err(_) => RegionNodeKind::Empty,
    }
}

fn is_single_chunk_bounds(bounds: Aabb4i) -> bool {
    bounds.min == bounds.max
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn branch_matches_split(bounds: Aabb4i, children: &[RegionTreeCore]) -> bool {
    if children.len() != 2 {
        return false;
    }
    let Some((left, right)) = split_bounds_longest_axis(bounds) else {
        return false;
    };
    (children[0].bounds == left && children[1].bounds == right)
        || (children[0].bounds == right && children[1].bounds == left)
}

fn split_bounds_longest_axis(bounds: Aabb4i) -> Option<(Aabb4i, Aabb4i)> {
    if !bounds.is_valid() {
        return None;
    }

    let spans = [
        bounds.max[0] - bounds.min[0] + 1,
        bounds.max[1] - bounds.min[1] + 1,
        bounds.max[2] - bounds.min[2] + 1,
        bounds.max[3] - bounds.min[3] + 1,
    ];
    let mut axis = 0usize;
    for idx in 1..4 {
        if spans[idx] > spans[axis] {
            axis = idx;
        }
    }
    if spans[axis] <= 1 {
        return None;
    }

    let left_len = spans[axis] / 2;
    let left_max_axis = bounds.min[axis] + left_len - 1;
    let mut left_max = bounds.max;
    left_max[axis] = left_max_axis;

    let mut right_min = bounds.min;
    right_min[axis] = left_max_axis + 1;

    Some((
        Aabb4i::new(bounds.min, left_max),
        Aabb4i::new(right_min, bounds.max),
    ))
}

fn sort_children_canonical(children: &mut [RegionTreeCore]) {
    children.sort_unstable_by_key(|child| {
        (
            child.bounds.min[0],
            child.bounds.min[1],
            child.bounds.min[2],
            child.bounds.min[3],
            child.bounds.max[0],
            child.bounds.max[1],
            child.bounds.max[2],
            child.bounds.max[3],
        )
    });
}

fn expand_root_once(root: Box<RegionTreeCore>, key_pos: [i32; 4]) -> Box<RegionTreeCore> {
    if root.bounds.contains_chunk(key_pos) {
        return root;
    }

    let mut old_root = *root;
    // Root expansion introduces an empty sibling placeholder for the out-of-bounds side.
    // Normalize the carried subtree first so placeholder empties from prior expansions
    // do not accumulate off-path.
    normalize_chunk_node(&mut old_root);
    let old_bounds = old_root.bounds;
    let axis = (0..4)
        .find(|axis| {
            key_pos[*axis] < old_bounds.min[*axis] || key_pos[*axis] > old_bounds.max[*axis]
        })
        .unwrap_or(0);
    let span = (old_bounds.max[axis] - old_bounds.min[axis] + 1).max(1);

    let mut new_bounds = old_bounds;
    let mut sibling_bounds = old_bounds;
    if key_pos[axis] < old_bounds.min[axis] {
        let mut expanded = old_bounds.min[axis].saturating_sub(span);
        if expanded >= old_bounds.min[axis] {
            expanded = key_pos[axis];
        }
        new_bounds.min[axis] = expanded;
        sibling_bounds.min[axis] = expanded;
        sibling_bounds.max[axis] = old_bounds.min[axis] - 1;
    } else {
        let mut expanded = old_bounds.max[axis].saturating_add(span);
        if expanded <= old_bounds.max[axis] {
            expanded = key_pos[axis];
        }
        new_bounds.max[axis] = expanded;
        sibling_bounds.min[axis] = old_bounds.max[axis] + 1;
        sibling_bounds.max[axis] = expanded;
    }

    let sibling = RegionTreeCore {
        bounds: sibling_bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: old_root.generator_version_hash,
    };
    let mut children = vec![old_root, sibling];
    sort_children_canonical(&mut children);
    Box::new(RegionTreeCore {
        bounds: new_bounds,
        kind: RegionNodeKind::Branch(children),
        generator_version_hash: 0,
    })
}

fn intersect_aabb(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    let min = [
        a.min[0].max(b.min[0]),
        a.min[1].max(b.min[1]),
        a.min[2].max(b.min[2]),
        a.min[3].max(b.min[3]),
    ];
    let max = [
        a.max[0].min(b.max[0]),
        a.max[1].min(b.max[1]),
        a.max[2].min(b.max[2]),
        a.max[3].min(b.max[3]),
    ];
    (min[0] <= max[0] && min[1] <= max[1] && min[2] <= max[2] && min[3] <= max[3])
        .then_some(Aabb4i::new(min, max))
}
