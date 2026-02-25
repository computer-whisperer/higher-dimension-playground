use crate::shared::spatial::Aabb4i;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaledBoundsError {
    ScaleResolutionOverflow,
    ArithmeticOverflow,
}

fn resolution_shift_for_pair(a_scale_exp: i8, b_scale_exp: i8) -> Result<u32, ScaledBoundsError> {
    let neg_a = (-(a_scale_exp as i16)).max(0);
    let neg_b = (-(b_scale_exp as i16)).max(0);
    let shift = neg_a.max(neg_b) as u32;
    if shift > 120 {
        return Err(ScaledBoundsError::ScaleResolutionOverflow);
    }
    Ok(shift)
}

fn scaled_cell_size_units(scale_exp: i8, resolution_shift: u32) -> Result<i128, ScaledBoundsError> {
    let total_shift = (scale_exp as i32)
        .checked_add(resolution_shift as i32)
        .ok_or(ScaledBoundsError::ScaleResolutionOverflow)?;
    if total_shift < 0 || total_shift > 120 {
        return Err(ScaledBoundsError::ScaleResolutionOverflow);
    }
    Ok(1i128 << (total_shift as u32))
}

fn axis_interval_world_units(
    bounds: Aabb4i,
    scale_exp: i8,
    axis: usize,
    resolution_shift: u32,
) -> Result<(i128, i128), ScaledBoundsError> {
    let cell_size = scaled_cell_size_units(scale_exp, resolution_shift)?;
    let min = i128::from(bounds.min[axis])
        .checked_mul(cell_size)
        .ok_or(ScaledBoundsError::ArithmeticOverflow)?;
    let max_exclusive = i128::from(bounds.max[axis])
        .checked_add(1)
        .ok_or(ScaledBoundsError::ArithmeticOverflow)?
        .checked_mul(cell_size)
        .ok_or(ScaledBoundsError::ArithmeticOverflow)?;
    Ok((min, max_exclusive))
}

pub fn scaled_bounds_overlap_world(
    a_bounds: Aabb4i,
    a_scale_exp: i8,
    b_bounds: Aabb4i,
    b_scale_exp: i8,
) -> Result<bool, ScaledBoundsError> {
    if !a_bounds.is_valid() || !b_bounds.is_valid() {
        return Ok(false);
    }

    let resolution_shift = resolution_shift_for_pair(a_scale_exp, b_scale_exp)?;
    for axis in 0..4 {
        let (a_min, a_max) =
            axis_interval_world_units(a_bounds, a_scale_exp, axis, resolution_shift)?;
        let (b_min, b_max) =
            axis_interval_world_units(b_bounds, b_scale_exp, axis, resolution_shift)?;
        if a_min >= b_max || b_min >= a_max {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_same_scale_detects_intersection() {
        let a = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let b = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        assert_eq!(scaled_bounds_overlap_world(a, 0, b, 0).unwrap(), true);
    }

    #[test]
    fn overlap_mixed_scale_detects_shared_world_volume() {
        let coarse = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let fine = Aabb4i::new([0, 0, 0, 0], [1, 1, 1, 1]);
        assert_eq!(scaled_bounds_overlap_world(coarse, 0, fine, -1).unwrap(), true);
    }

    #[test]
    fn overlap_mixed_scale_detects_disjoint_world_volume() {
        let coarse = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let fine = Aabb4i::new([2, 0, 0, 0], [3, 1, 1, 1]);
        assert_eq!(scaled_bounds_overlap_world(coarse, 0, fine, -1).unwrap(), false);
    }
}
