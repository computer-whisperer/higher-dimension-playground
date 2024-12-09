
use glam::{Mat2, Mat3, Mat4};
use crate::factorial;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use crate::mat_n::MatN;
use crate::vec_n::VecN;


// An N-1 blade in an N space, the dot product with this vector should give the magnitude of the external product of this blade and the vector
pub fn get_normal<const N: usize>(components: &[VecN::<N>; N-1]) -> VecN::<N> where [(); N-1]:{
    let mut output = VecN::<N>::ZERO;
    for i in 0..N {
        let mut sub_matrix = MatN::<{N-1}>::ZERO;
        let mut output_row = 0;
        for j in 0..N {
            if j == i {
                continue;
            }
            for k in 0..N-1 {
                sub_matrix[[output_row, k]] = components[k][j]
            }
            output_row += 1;
        }
        output[i] = sub_matrix.determinant_basic();
    }

    output
}

pub fn get_gram_matrix<const N: usize, const K: usize>(vertices: &[VecN::<N>; K]) -> MatN::<{K-1}> {
    let mut output = MatN::<{K-1}>::ZERO;

    for i in 0..K-1 {
        for j in 0..K-1 {
            output[[i, j]] = (vertices[i+1] - vertices[0]) * (vertices[j+1] - vertices[0]);
        }
    }

    output
}

pub fn get_simplex_volume<const N: usize, const K: usize>(vertices: &[VecN::<N>; K]) -> f32 where [(); K-1]: {
    get_gram_matrix::<N, K>(vertices).determinant_basic().sqrt()/factorial(K-1) as f32
}

pub struct PlaneN<const N: usize> where [(); N-1]:{
    vertices: [VecN::<N>; N],
    normal: VecN::<N>,
}

impl<const N: usize> PlaneN<N> where [(); N-1]: {
    const HACK: [VecN::<N>; N-1] = [VecN::<N>::ZERO; N-1];

    pub fn new(vertices: [VecN::<N>; N]) -> PlaneN<N> {
        let mut blade = Self::HACK;
        for i in 0..N-1 {
            blade[i] = vertices[i+1] - vertices[0];
        }
        let normal = get_normal(&blade);
        Self {
            vertices,
            normal
        }
    }
    
    pub fn normal(&self) -> VecN::<N> {self.normal}

    pub fn test_side(&self, point: VecN::<N>) -> bool {
        self.normal*(point - self.vertices[0]) > 0.0
    }

    pub fn clip_line(&self, line: [VecN::<N>; 2]) -> [VecN::<N>; 2] {
        let a_in = self.test_side(line[0]);
        let b_in = self.test_side(line[1]);
        if !a_in && !b_in {
            // Completely out of bounds
            [VecN::ZERO; 2]
        }
        else if ((line[1] - line[0]) * self.normal).abs() < 0.001 {
            // Parallel to plane, skip
            line
        }
        else if !a_in || !b_in {
            let (v_in, v_out) = if a_in { (line[0], line[1]) } else { (line[1], line[0]) };

            let t = -(self.normal*(v_in - self.vertices[0]))/(self.normal*(v_out - v_in));

            let v_intersect = (v_out*t) + (v_in*(1.0-t));

            if a_in {[v_in, v_intersect]} else {[v_intersect, v_in]}
        }
        else {
            // Completely in bounds
            line
        }
        
    }
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vec_conversion() {
        let input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nd_vec = ndarray::Array1::from_vec(input_vec);
        let converted_vec = VecN::<5>::from(&nd_vec);
        let double_converted_vec = ndarray::Array1::from(converted_vec);
        assert_eq!(nd_vec, double_converted_vec);
    }

    #[test]
    fn test_vec_conversion_2() {
        let input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nd_vec = ndarray::Array2::from_shape_vec((5, 1), input_vec).unwrap();
        let converted_vec = VecN::<5>::from(&nd_vec);
        let double_converted_vec = ndarray::Array2::from(converted_vec);
        assert_eq!(nd_vec, double_converted_vec);
    }

    #[test]
    fn test_mat_conversion() {
        let input_mat: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ];
        let nd_mat = ndarray::Array2::from_shape_vec((5, 5), input_mat).unwrap();
        let converted_mat = MatN::<5>::from(&nd_mat);
        let double_converted_mat = ndarray::Array2::from(converted_mat);
        assert_eq!(nd_mat, double_converted_mat);
    }

    #[test]
    fn test_mat_vec_mul() {
        let input_mat: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ];
        let input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nd_mat = ndarray::Array2::from_shape_vec((5, 5), input_mat).unwrap();
        let nd_vec = ndarray::Array2::from_shape_vec((5, 1), input_vec).unwrap();
        let converted_mat = MatN::<5>::from(&nd_mat);
        let converted_vec = VecN::<5>::from(&nd_vec);
        let result_vec = converted_mat * converted_vec;
        let back_converted_result = ndarray::Array2::from(result_vec);
        let expected_result = nd_mat.dot(&nd_vec);
        assert_eq!(expected_result, back_converted_result);
    }

    #[test]
    fn test_mat_mat_mul() {
        let input_mat1: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ];
        let input_mat2: Vec<f32> = vec![
            26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0,
            36.0, 37.0, 38.0, 39.0, 40.0,
            41.0, 42.0, 43.0, 44.0, 45.0,
            46.0, 47.0, 48.0, 49.0, 50.0
        ];
        let nd_mat1 = ndarray::Array2::from_shape_vec((5, 5), input_mat1).unwrap();
        let nd_mat2 = ndarray::Array2::from_shape_vec((5, 5), input_mat2).unwrap();
        let converted_mat1 = MatN::<5>::from(&nd_mat1);
        let converted_mat2 = MatN::<5>::from(&nd_mat2);
        let result_mat = converted_mat1 * converted_mat2;
        let back_converted_result = ndarray::Array2::from(result_mat);
        let expected_result = nd_mat1.dot(&nd_mat2);
        assert_eq!(expected_result, back_converted_result);
    }

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
    
    /*
    #[test]
    fn test_get_simplex_volume() {
        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let result = get_simplex_volume::<4, 4>(&vertices);
        assert_eq!(result, 4.0/24.0);

        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 2.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let result = get_simplex_volume::<4, 4>(&vertices);
        assert_eq!(result, 2.0/24.0);

        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let result = get_simplex_volume::<4, 4>(&vertices);
        assert_eq!(result, 1.0/24.0);
    }
    */
    
    #[test]
    fn test_plane_n() {
        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let plane = PlaneN::<4>::new(vertices);
        let normal = plane.normal();
        assert_eq!(normal, VecN::<4>::new([0.0, 0.0, 0.0, 1.0]));
        
        let line_a = [VecN::<4>::new([0.0, 0.0, 0.0, 0.0]), VecN::<4>::new([0.0, 0.0, 0.0, 2.0])];
        let clipped_line_a = plane.clip_line(line_a);
        assert_eq!(clipped_line_a, [VecN::<4>::new([0.0, 0.0, 0.0, 1.0]), VecN::<4>::new([0.0, 0.0, 0.0, 2.0])]);
        
        let line_b = [VecN::<4>::new([0.0, 0.0, 1.0, 0.0]), VecN::<4>::new([0.0, 0.0, 0.0, 4.0])];
        let clipped_line_b = plane.clip_line(line_b);
        assert_eq!(clipped_line_b, [VecN::<4>::new([0.0, 0.0, 0.75, 1.0]), VecN::<4>::new([0.0, 0.0, 0.0, 4.0])]);
    }
}