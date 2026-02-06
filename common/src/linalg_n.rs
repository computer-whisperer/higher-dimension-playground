
use crate::mat_n::MatN;
use crate::vec_n::VecN;

// Specialized 4D normal calculation (cross product generalization)
// Takes 3 vectors in 4D space and returns the normal vector
pub fn get_normal_4d(components: &[VecN<4>; 3]) -> VecN<4> {
    let mut output = VecN::<4>::ZERO;
    for i in 0..4 {
        let mut sub_matrix = MatN::<3>::ZERO;
        let mut output_row = 0;
        for j in 0..4 {
            if j == i {
                continue;
            }
            for k in 0..3 {
                sub_matrix[[output_row, k]] = components[k][j]
            }
            output_row += 1;
        }
        output[i] = sub_matrix.determinant_basic();
    }
    output
}

// Re-export as get_normal for backwards compatibility
pub fn get_normal(components: &[VecN<4>; 3]) -> VecN<4> {
    get_normal_4d(components)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_normal() {
        let components = [
            VecN::<4>::new([1.0, 0.0, 0.0, 0.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 0.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 0.0])
        ];
        let result = get_normal(&components);
        let expected_result = VecN::<4>::new([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(result, expected_result);

        let components = [
            VecN::<4>::new([1.0, 1.0, 0.0, 0.0]),
            VecN::<4>::new([0.0, 1.0, 1.0, 0.0]),
            VecN::<4>::new([1.0, 0.0, 1.0, 0.0])
        ];
        let result = get_normal(&components);
        let expected_result = VecN::<4>::new([0.0, 0.0, 0.0, 2.0]);
        assert_eq!(result, expected_result);

        let components = [
            VecN::<4>::new([1.0, 1.0, 0.0, 0.0]),
            VecN::<4>::new([0.0, 1.0, 1.0, 0.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let result = get_normal(&components);
        let expected_result = VecN::<4>::new([1.0, 1.0, 1.0, 1.0]);
        assert_eq!(result, expected_result);
    }
}
