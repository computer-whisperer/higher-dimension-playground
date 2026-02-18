use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Aabb4i {
    pub min: [i32; 4],
    pub max: [i32; 4],
}

impl Aabb4i {
    pub fn new(min: [i32; 4], max: [i32; 4]) -> Self {
        Self { min, max }
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0]
            && self.min[1] <= self.max[1]
            && self.min[2] <= self.max[2]
            && self.min[3] <= self.max[3]
    }

    pub fn chunk_extents(&self) -> Option<[usize; 4]> {
        if !self.is_valid() {
            return None;
        }
        let mut extents = [0usize; 4];
        for axis in 0..4 {
            let span = i64::from(self.max[axis]) - i64::from(self.min[axis]) + 1;
            if span <= 0 {
                return None;
            }
            extents[axis] = usize::try_from(span).ok()?;
        }
        Some(extents)
    }

    pub fn chunk_cell_count(&self) -> Option<usize> {
        let extents = self.chunk_extents()?;
        extents[0]
            .checked_mul(extents[1])?
            .checked_mul(extents[2])?
            .checked_mul(extents[3])
    }

    pub fn contains_chunk(&self, pos: [i32; 4]) -> bool {
        self.is_valid()
            && pos[0] >= self.min[0]
            && pos[0] <= self.max[0]
            && pos[1] >= self.min[1]
            && pos[1] <= self.max[1]
            && pos[2] >= self.min[2]
            && pos[2] <= self.max[2]
            && pos[3] >= self.min[3]
            && pos[3] <= self.max[3]
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.is_valid()
            && other.is_valid()
            && self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
            && self.min[3] <= other.max[3]
            && self.max[3] >= other.min[3]
    }
}
