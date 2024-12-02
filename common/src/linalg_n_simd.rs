
use glam::{Vec4, Mat4, Vec3, Vec2};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct VecN<const N: usize> where [(); (N+3)/4]: {
    sub_vecs: [Vec4; (N+3)/4]
}

unsafe impl<const N: usize> bytemuck::Zeroable for VecN<N> where [(); (N+3)/4]: {
}

unsafe impl<const N: usize> bytemuck::Pod for VecN<N> where [(); (N+3)/4]: {
}

impl<const N: usize> VecN<N> where [(); (N+3)/4]: {
    const ZERO: Self = Self {sub_vecs: [Vec4::ZERO; (N+3)/4]};

    pub fn new(v: &[f32]) -> Self {
        let mut sub_vecs = [Vec4::ZERO; (N+3)/4];
        for i in 0..N {
            sub_vecs[i/4][i%4] = v[i];
        }
        Self {
            sub_vecs
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
    [(); (N+3)/4]:,
    [(); ({N+1}+3)/4]:
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

impl <const N: usize> core::ops::Index<usize> for VecN<N> where [(); (N+3)/4]: {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.sub_vecs[index/4][index%4]
    }
}

impl <const N: usize> core::ops::Index<usize> for &VecN<N> where [(); (N+3)/4]: {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.sub_vecs[index/4][index%4]
    }
}

impl <const N: usize> core::ops::IndexMut<usize> for VecN<N> where [(); (N+3)/4]: {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.sub_vecs[index/4][index%4]
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct MatN<const N: usize> where [(); (N+3)/4]: {
    sub_mats: [[Mat4; (N+3)/4]; (N+3)/4] // [row][col]
}

unsafe impl<const N: usize> bytemuck::Zeroable for MatN<N> where [(); (N+3)/4]: {
}

unsafe impl<const N: usize> bytemuck::Pod for MatN<N> where [(); (N+3)/4]: {
}

impl <const N: usize> MatN<N> where [(); (N+3)/4]: {
    const ZERO: Self = Self{sub_mats: [[Mat4::ZERO; (N+3)/4]; (N+3)/4]};

    pub fn new(v: &[&[f32]]) -> Self {
        let mut output = Self::ZERO;
        for i in 0..N {
            for j in 0..N {
                output[[i, j]] = v[i][j];
            }
        }
        output
    }
}

impl <const N: usize> core::ops::Index<[usize; 2]> for MatN<N> where [(); (N+3)/4]: {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let sub_mat = &self.sub_mats[index[0]/4][index[1]/4];
        &(match index[1]%4 {
            0 => &sub_mat.x_axis,
            1 => &sub_mat.y_axis,
            2 => &sub_mat.z_axis,
            3 => &sub_mat.w_axis,
            _ => {unreachable!()}
        })[index[0]%4]
    }
}

impl <const N: usize> core::ops::Index<[usize; 2]> for &MatN<N> where [(); (N+3)/4]: {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let sub_mat = &self.sub_mats[index[0]/4][index[1]/4];
        &(match index[1]%4 {
            0 => &sub_mat.x_axis,
            1 => &sub_mat.y_axis,
            2 => &sub_mat.z_axis,
            3 => &sub_mat.w_axis,
            _ => {unreachable!()}
        })[index[0]%4]
    }
}

impl <const N: usize> core::ops::IndexMut<[usize; 2]> for MatN<N> where [(); (N+3)/4]: {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.sub_mats[index[0]/4][index[1]/4].col_mut(index[1]%4)[index[0]%4]
    }
}

impl <const N: usize> core::ops::Mul<VecN<N>> for MatN<N> where [(); (N+3)/4]: {
    type Output = VecN<N>;

    fn mul(self, rhs: VecN<N>) -> Self::Output {
        let mut outputs = [Vec4::ZERO; (N+3)/4];

        for i in 0..(N+3)/4 {
            for j in 0..(N+3)/4 {
                outputs[i] += self.sub_mats[i][j] * rhs.sub_vecs[j];
            }
        }

        VecN {
            sub_vecs: outputs
        }
    }
}

impl <const N: usize> core::ops::Mul<MatN<N>> for MatN<N> where [(); (N+3)/4]: {
    type Output = MatN<N>;
    fn mul(self, rhs: MatN<N>) -> Self::Output {
        let mut sub_mats = [[Mat4::ZERO; (N+3)/4]; (N+3)/4];

        for i in 0..(N+3)/4 {
            for j in 0..(N+3)/4 {
                for k in 0..(N+3)/4 {
                    sub_mats[i][j] += self.sub_mats[i][k] * rhs.sub_mats[k][j];
                }
            }
        }

        MatN {
            sub_mats
        }
    }
}

impl From<Vec4> for VecN<4> {
    fn from(value: Vec4) -> Self {
        Self {
            sub_vecs: [value]
        }
    }
}

impl From<Vec3> for VecN<3> {
    fn from(value: Vec3) -> Self {
        Self::new(&[value.x, value.y, value.z])
    }
}

impl From<Vec2> for VecN<2> {
    fn from(value: Vec2) -> Self {
        Self::new(&[value.x, value.y])
    }
}

impl From<VecN<4>> for Vec4 {
    fn from(value: VecN<4>) -> Self {
        value.sub_vecs[0]
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
        value.sub_vecs[0]
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
}