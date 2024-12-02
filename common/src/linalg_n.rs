
use glam::{Vec4, Mat4, Vec3, Vec2, Mat3, Mat2};
use crate::factorial;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(transparent)]
pub struct VecN<const N: usize> {
    values: [f32; N]
}

unsafe impl<const N: usize> bytemuck::Zeroable for VecN<N> {}

unsafe impl<const N: usize> bytemuck::Pod for VecN<N> {}

impl<const N: usize> VecN<N> {
    pub const ZERO: Self = Self {values: [0f32; N]};

    pub fn new(v: [f32; N]) -> Self {
        let mut values = [0f32; N];
        for i in 0..N {
            values[i] = v[i];
        }
        Self {
            values
        }
    }

    pub fn x(&self) -> f32 {self[0]}
    pub fn y(&self) -> f32 {self[1]}
    pub fn z(&self) -> f32 {self[2]}
    pub fn w(&self) -> f32 {self[3]}
    pub fn v(&self) -> f32 {self[4]}
}

impl<const N: usize> VecN<N>
where
    [(); N+1]:
{
    pub fn extend(&self, v: f32) -> VecN<{N+1}> {
        let mut output = VecN::<{N+1}>::ZERO;
        
        for i in 0..N {
            output[i] = self[i];
        }
        output[N] = v;
        
        output
    }
}

impl <const N: usize> core::ops::Index<usize> for VecN<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl <const N: usize> core::ops::Index<usize> for &VecN<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl <const N: usize> core::ops::IndexMut<usize> for VecN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(transparent)]
pub struct MatN<const N: usize> {
    values: [[f32; N]; N] // [row][col]
}

unsafe impl<const N: usize> bytemuck::Zeroable for MatN<N> {
}

unsafe impl<const N: usize> bytemuck::Pod for MatN<N> {
}

impl <const N: usize> MatN<N> {
    pub const ZERO: Self = Self{ values: [[0f32; N]; N]};

    pub fn new(v: &[&[f32]]) -> Self {
        let mut output = Self::ZERO;
        for i in 0..N {
            for j in 0..N {
                output[[i, j]] = v[i][j];
            }
        }
        output
    }



    pub fn determinant_basic(&self) -> f32 {
        // Compute determinant
        match N {
            1 => self.values[0][0],
            2 => self.values[0][0] * self.values[1][1] - self.values[0][1] * self.values[1][0],
            3 => self.values[0][0] * (self.values[1][1] * self.values[2][2] - self.values[1][2] * self.values[2][1])
                - self.values[0][1] * (self.values[1][0] * self.values[2][2] - self.values[1][2] * self.values[2][0])
                + self.values[0][2] * (self.values[1][0] * self.values[2][1] - self.values[1][1] * self.values[2][0]),
            _ => {unimplemented!()}
        }
    }
 }

impl <const N: usize> MatN<N> where [[(); N-1]; N-1]: {
    pub fn minor(&self, row: usize, col: usize) -> MatN<{N-1}> {
        let mut output = MatN::<{N-1}>::ZERO;

        let mut output_row = 0;
        for j in 0..N {
            if j == row {
                continue;
            }
            let mut output_col = 0;
            for k in 0..N {
                if k == col {
                    continue;
                }
                output[[output_row, output_col]] = self[[j, k]];
                output_col += 1;
            }
            output_row += 1;
        }
        output
    }
}

impl MatN<2> {
    pub fn determinant_native(&self) -> f32 {
        Mat2::from(self).determinant()
    }
}

impl MatN<3> {
    pub fn determinant_native(&self) -> f32 {
        Mat3::from(self).determinant()
    }
}

impl MatN<4> {
    pub fn determinant_native(&self) -> f32 {
        Mat4::from(self).determinant()
    }
}

impl <const N: usize> core::ops::Index<[usize; 2]> for MatN<N> {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.values[index[0]][index[1]]
    }
}

impl <const N: usize> core::ops::Index<[usize; 2]> for &MatN<N> {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.values[index[0]][index[1]]
    }
}

impl <const N: usize> core::ops::IndexMut<[usize; 2]> for MatN<N> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.values[index[0]][index[1]]
    }
}

impl <const N: usize> core::ops::Mul<VecN<N>> for MatN<N> {
    type Output = VecN<N>;

    fn mul(self, rhs: VecN<N>) -> Self::Output {
        let mut outputs = [0f32; N];

        for i in 0..N {
            for j in 0..N {
                outputs[i] += self.values[i][j] * rhs.values[j];
            }
        }

        VecN {
            values: outputs
        }
    }
}

impl <const N: usize> core::ops::Mul<VecN<N>> for VecN<N> {
    type Output = f32;

    fn mul(self, rhs: VecN<N>) -> Self::Output {
        let mut output = 0.0;

        for i in 0..N {
            output += self.values[i] * rhs.values[i];
        }
        
        output
    }
}

impl <const N: usize> core::ops::Add<&VecN<N>> for &VecN<N> {
    type Output = VecN<N>;

    fn add(self, rhs: &VecN<N>) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] + rhs.values[i];
        }
        output
    }
}

impl <const N: usize> core::ops::Sub<&VecN<N>> for &VecN<N> {
    type Output = VecN<N>;

    fn sub(self, rhs: &VecN<N>) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] - rhs.values[i];
        }
        output
    }
}

impl <const N: usize> core::ops::Add<VecN<N>> for VecN<N> {
    type Output = VecN<N>;

    fn add(self, rhs: VecN<N>) -> Self::Output {&self + &rhs}
}

impl <const N: usize> core::ops::Sub<VecN<N>> for VecN<N> {
    type Output = VecN<N>;

    fn sub(self, rhs: VecN<N>) -> Self::Output {&self - &rhs}
}

impl <const N: usize> core::ops::Mul<MatN<N>> for MatN<N> where [(); (N+3)/4]: {
    type Output = MatN<N>;
    fn mul(self, rhs: MatN<N>) -> Self::Output {
        let mut values = [[0f32; N]; N];

        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    values[i][j] += self.values[i][k] * rhs.values[k][j];
                }
            }
        }

        MatN {
            values
        }
    }
}

impl From<Vec4> for VecN<4> {
    fn from(value: Vec4) -> Self {
        Self::new([value.x, value.y, value.z, value.w])
    }
}

impl From<Vec3> for VecN<3> {
    fn from(value: Vec3) -> Self {
        Self::new([value.x, value.y, value.z])
    }
}

impl From<Vec2> for VecN<2> {fn from(value: Vec2) -> Self {
        Self::new([value.x, value.y])
    } }

impl From<&Mat4> for MatN<4> {
    fn from(value: &Mat4) -> Self {
        let mut values = [[0f32; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                values[i][j] = value.col(j)[i];
            }
        }

        MatN {
            values
        }
    }
}

impl From<&Mat3> for MatN<3> {
    fn from(value: &Mat3) -> Self {
        let mut values = [[0f32; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                values[i][j] = value.col(j)[i];
            }
        }

        MatN {
            values
        }
    }
}

impl From<&Mat2> for MatN<2> {
    fn from(value: &Mat2) -> Self {
        let mut values = [[0f32; 2]; 2];

        for i in 0..2 {
            for j in 0..2 {
                values[i][j] = value.col(j)[i];
            }
        }

        MatN {
            values
        }
    }
}

impl From<Mat4> for MatN<4> { fn from(value: Mat4) -> Self {value.into()} }

impl From<Mat3> for MatN<3> { fn from(value: Mat3) -> Self {value.into()} }

impl From<Mat2> for MatN<2> { fn from(value: Mat2) -> Self {value.into()} }

impl From<&MatN::<2>> for Mat2 {
    fn from(value: &MatN<2>) -> Self {
        Mat2::from_cols_array_2d(&value.values)
    }
}

impl From<&MatN::<3>> for Mat3 {
    fn from(value: &MatN<3>) -> Self {
        Mat3::from_cols_array_2d(&value.values)
    }
}

impl From<&MatN::<4>> for Mat4 {
    fn from(value: &MatN<4>) -> Self {
        Mat4::from_cols_array_2d(&value.values)
    }
}

impl From<MatN::<4>> for Mat4 {
    fn from(value: MatN<4>) -> Self {
        value.into()
    }
}


impl From<VecN<4>> for Vec4 {
    fn from(value: VecN<4>) -> Self {
        Self::new(value[0], value[1], value[2], value[3])
    }
}

impl From<VecN<3>> for Vec3 {
    fn from(value: VecN<3>) -> Self {
        Self::new(value.x(), value.y(), value.z())
    }
}

impl From<VecN<2>> for Vec2 {
    fn from(value: VecN<2>) -> Self {
        Self::new(value.x(), value.y())
    }
}

impl From<&VecN<4>> for Vec4 {
    fn from(value: &VecN<4>) -> Self {
        Self::new(value[0], value[1], value[2], value[3])
    }
}

impl From<&VecN<3>> for Vec3 {
    fn from(value: &VecN<3>) -> Self {
        Self::new(value.x(), value.y(), value.z())
    }
}

impl From<&VecN<2>> for Vec2 {
    fn from(value: &VecN<2>) -> Self {
        Self::new(value.x(), value.y())
    }
}

#[cfg(feature = "ndarray")]
impl<'a, const N: usize> From<ndarray::ArrayView1<'a, f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::ArrayView1<'a, f32>) -> Self {
        assert_eq!(value.shape(), &[N]);
        let mut output = VecN::ZERO;
        for i in 0..N {
            output[i] = value[i];
        }
        output
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&ndarray::Array1<f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: &ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<ndarray::Array1<f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<'a, const N: usize> From<ndarray::ArrayView2<'a, f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::ArrayView2<'a, f32>) -> Self {
        assert_eq!(value.shape(), &[N, 1]);
        let mut output = VecN::<N>::ZERO;

        for i in 0..N {
            output[i] = value[[i, 0]];
        }

        output
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&ndarray::Array2<f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<ndarray::Array2<f32>> for VecN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&VecN<N>> for ndarray::Array1<f32> where [(); (N+3)/4]: {
    fn from(value: &VecN<N>) -> Self {
        let mut values = Vec::new();
        for i in 0..N {
            values.push(value[i]);
        }
        ndarray::Array1::from_vec(values)
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<VecN<N>> for ndarray::Array1<f32> where [(); (N+3)/4]: {
    fn from(value: VecN<N>) -> Self {
        Self::from(&value)
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&VecN<N>> for ndarray::Array2<f32> where [(); (N+3)/4]: {
    fn from(value: &VecN<N>) -> Self {
        let mut values = Vec::new();
        for i in 0..N {
            values.push(value[i]);
        }
        Self::from_shape_vec((N, 1), values).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<VecN<N>> for ndarray::Array2<f32> where [(); (N+3)/4]: {
    fn from(value: VecN<N>) -> Self {
        Self::from(&value)
    }
}

#[cfg(feature = "ndarray")]
impl<'a, const N: usize> From<ndarray::ArrayView2<'a, f32>> for MatN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::ArrayView2<'a, f32>) -> Self {
        assert_eq!(value.shape(), &[N, N]);
        let mut output = Self::ZERO;
        for i in 0..N {
            for j in 0..N {
                output[[i, j]] = value[[i, j]];
            }
        }
        output
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&ndarray::Array2<f32>> for MatN<N> where [(); (N+3)/4]: {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<ndarray::Array2<f32>> for MatN<N> where [(); (N+3)/4]: {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<&MatN<N>> for ndarray::Array2<f32> where [(); (N+3)/4]: {
    fn from(value: &MatN<N>) -> Self {
        let mut values = Vec::new();
        for i in 0..N {
            for j in 0..N {
                values.push(value[[i, j]]);
            }
        }
        ndarray::Array2::from_shape_vec((N, N), values).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<const N: usize> From<MatN<N>> for ndarray::Array2<f32> where [(); (N+3)/4]: {
    fn from(value: MatN<N>) -> Self {
        Self::from(&value)
    }
}

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

fn get_gram_matrix<const N: usize, const K: usize>(vertices: &[VecN::<N>; K]) -> MatN::<{K-1}> {
    let mut output = MatN::<{K-1}>::ZERO;
    
    for i in 0..K-1 {
        for j in 0..K-1 {
            output[[i, j]] = (vertices[i+1] - vertices[0]) * (vertices[j+1] - vertices[0]);
        }
    }
    
    output
}

pub fn get_simplex_volume<const N: usize, const K: usize>(vertices: &[VecN::<N>; K]) -> f32 where [(); K-1]: {
    get_gram_matrix::<N, K>(vertices).determinant_basic().sqrt()/factorial(N) as f32
}

pub fn get_pseudo_barycentric<const N: usize, const K: usize>(vertices: &[VecN::<N>; K], point: VecN::<N>) -> VecN::<K> where [(); K-1]: {
    let full_simplex_volume = get_simplex_volume(vertices);
    let mut components = VecN::<K>::ZERO;
    let mut volume_sum = 0.0;
    
    for i in 0..N {
        let mut local_vertices = vertices.clone();
        local_vertices[i] = point;
        components[i] = get_simplex_volume(&local_vertices);
        volume_sum += components[i];
    }
    
    if volume_sum - full_simplex_volume > 0.01 {
        VecN::<K>::ZERO
    }
    else {
        let mut output = VecN::<K>::ZERO;
        for i in 0..N {
            output[i] = components[i]/volume_sum;
        }
        output
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
    
    #[test]
    fn test_get_simplex_volume() {
        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let result = get_simplex_volume::<4, 4>(&vertices);
        assert_eq!(result, 1.0/24.0);

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
    
    #[test]
    fn test_get_pseudo_barycentric() {
        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let pixel_pos = VecN::<4>::new([0.2, 0.2, 0.2, 1.0]);
        let result = get_pseudo_barycentric(&vertices, pixel_pos);
        assert_eq!(result, VecN::<4>::new([0.40000007, 0.19999999, 0.19999999, 0.19999999]));

        let vertices = [
            VecN::<4>::new([0.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([1.0, 0.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 1.0, 0.0, 1.0]),
            VecN::<4>::new([0.0, 0.0, 1.0, 1.0])
        ];
        let pixel_pos = VecN::<4>::new([0.5, 0.9, 0.5, 1.0]);
        let result = get_pseudo_barycentric(&vertices, pixel_pos);
        assert_eq!(result, VecN::<4>::ZERO);
    }
}