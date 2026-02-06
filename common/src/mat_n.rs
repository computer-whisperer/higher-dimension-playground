use glam::{Mat2, Mat3, Mat4};
use crate::vec_n::VecN;

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

    pub fn identity() -> Self {
        let mut output = Self::ZERO;
        for i in 0..N {
            output[[i, i]] = 1.0;
        }
        output
    }

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

impl <const N: usize> core::ops::Index<[usize; 2]> for &MatN<N> {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.values[index[0]][index[1]]
    }
}

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
        Mat4::from(&value)
    }
}

// Specialized ndarray conversions for MatN<5> (used for 4D homogeneous transforms)
#[cfg(feature = "ndarray")]
impl From<&MatN<5>> for ndarray::Array2<f32> {
    fn from(value: &MatN<5>) -> Self {
        let mut values = Vec::with_capacity(25);
        for i in 0..5 {
            for j in 0..5 {
                values.push(value[[i, j]]);
            }
        }
        ndarray::Array2::from_shape_vec((5, 5), values).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl From<MatN<5>> for ndarray::Array2<f32> {
    fn from(value: MatN<5>) -> Self {
        Self::from(&value)
    }
}


impl <const N: usize> core::ops::Index<[usize; 2]> for MatN<N> {
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
                outputs[i] += self.values[i][j] * rhs[j];
            }
        }

        VecN::new(outputs)
    }
}

impl <const N: usize> core::ops::Mul<MatN<N>> for MatN<N> {
    type Output = MatN<N>;
    fn mul(self, rhs: MatN<N>) -> Self::Output {
        let mut output = MatN::<N>::ZERO;

        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    output[[i, j]] += self.values[i][k] * rhs.values[k][j];
                }
            }
        }

        output
    }
}

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

impl From<Mat4> for MatN<4> { fn from(value: Mat4) -> Self { MatN::from(&value) } }

impl From<Mat3> for MatN<3> { fn from(value: Mat3) -> Self { MatN::from(&value) } }

impl From<Mat2> for MatN<2> { fn from(value: Mat2) -> Self { MatN::from(&value) } }

#[cfg(feature = "nalgebra")]
use nalgebra::Matrix5;

#[cfg(feature = "nalgebra")]
impl From<nalgebra::Matrix5<f32>> for MatN::<5> {
    fn from(value: Matrix5<f32>) -> Self {
        let mut output = Self::ZERO;

        for x in 0..5 {
            for y in 0..5 {
                output[[x, y]] = value[(x, y)];
            }
        }

        output
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::ArrayView2<'_, f32>> for MatN<5> {
    fn from(value: ndarray::ArrayView2<'_, f32>) -> Self {
        assert_eq!(value.shape(), &[5, 5]);
        let mut output = Self::ZERO;
        for i in 0..5 {
            for j in 0..5 {
                output[[i, j]] = value[[i, j]];
            }
        }
        output
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array2<f32>> for MatN<5> {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array2<f32>> for MatN<5> {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
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
