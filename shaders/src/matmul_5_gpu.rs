use glam::{vec4, Vec4, Mat4};
#[cfg(feature = "ndarray")]
use glam::{mat4};

#[cfg(feature = "ndarray")]
use core::convert::From;

use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vec5GPU {
    first: Vec4,
    last_val: f32,
    padding: [u32; 3]
}

impl Vec5GPU {
    pub fn new(x: f32, y: f32, z: f32, w: f32, v: f32) -> Self {
        Self {
            first: vec4(x, y, z, w),
            last_val: v,
            padding: [0; 3]
        }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0)
    }
    
    pub fn from_4_and_1(first: Vec4, last_val: f32) -> Self {
        Self {
            first,
            last_val,
            padding: [0; 3]
        }
    }
    
    pub fn x(&self) -> f32 {self[0]}
    pub fn y(&self) -> f32 {self[1]}
    pub fn z(&self) -> f32 {self[2]}
    pub fn w(&self) -> f32 {self[3]}
    pub fn v(&self) -> f32 {self.last_val}
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Mat5GPU {
    first: Mat4,
    last_col: Vec4,
    last_row: Vec4,
    last_value: f32,
    padding: [u32; 3]
}

impl core::ops::Index<u32> for Vec5GPU {
    type Output = f32;
    fn index(&self, index: u32) -> &Self::Output {
        match index {
            0 => &self.first.x,
            1 => &self.first.y,
            2 => &self.first.z,
            3 => &self.first.w,
            4 => &self.last_val,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl core::ops::Mul<Vec5GPU> for Mat5GPU {
    type Output = Vec5GPU;

    fn mul(self, rhs: Vec5GPU) -> Self::Output {
        Self::Output {
            first: self.first * rhs.first + self.last_col * rhs.last_val,
            last_val: self.last_row.dot(rhs.first) + self.last_value * rhs.last_val,
            padding: [0; 3]
        }
    }
}

impl core::ops::Mul<Mat5GPU> for Mat5GPU {
    type Output = Mat5GPU;
    fn mul(self, rhs: Mat5GPU) -> Self::Output {
        Mat5GPU {
            first: self.first * rhs.first + Mat4::from_cols(
                self.last_col*rhs.last_row.x,
                self.last_col*rhs.last_row.y,
                self.last_col*rhs.last_row.z,
                self.last_col*rhs.last_row.w,
            ),
            last_col: self.first*rhs.last_col + self.last_col*rhs.last_value,
            last_row: rhs.first.transpose()*self.last_row + rhs.last_row*self.last_value,
            last_value: self.last_row.dot(rhs.last_col) + self.last_value*rhs.last_value,
            padding: [0; 3]
        }
    }
}

#[cfg(feature = "ndarray")]
impl<'a> From<ndarray::ArrayView1<'a, f32>> for Vec5GPU {
    fn from(value: ndarray::ArrayView1<'a, f32>) -> Self {
        Vec5GPU {
            first: vec4(value[0], value[1], value[2], value[3]),
            last_val: value[4],
            padding: [0; 3]
        }
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array1<f32>> for Vec5GPU {
    fn from(value: &ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array1<f32>> for Vec5GPU {
    fn from(value: ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl<'a> From<ndarray::ArrayView2<'a, f32>> for Vec5GPU {
    fn from(value: ndarray::ArrayView2<'a, f32>) -> Self {
        assert_eq!(value.shape(), &[5, 1]);
        Vec5GPU {
            first: vec4(value[(0, 0)], value[(1, 0)], value[(2, 0)], value[(3, 0)]),
            last_val: value[(4, 0)],
            padding: [0; 3]
        }
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array2<f32>> for Vec5GPU {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array2<f32>> for Vec5GPU {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<&Vec5GPU> for ndarray::Array1<f32> {
    fn from(value: &Vec5GPU) -> Self {
        Self::from_vec(vec!(value.first.x, value.first.y, value.first.z, value.first.w, value.last_val))
    }
}

#[cfg(feature = "ndarray")]
impl From<Vec5GPU> for ndarray::Array1<f32> {
    fn from(value: Vec5GPU) -> Self {
        Self::from(&value)
    }
}

#[cfg(feature = "ndarray")]
impl From<&Vec5GPU> for ndarray::Array2<f32> {
    fn from(value: &Vec5GPU) -> Self {
        Self::from_shape_vec((5, 1), vec!(value.first.x, value.first.y, value.first.z, value.first.w, value.last_val)).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl From<Vec5GPU> for ndarray::Array2<f32> {
    fn from(value: Vec5GPU) -> Self {
        Self::from(&value)
    }
}

#[cfg(feature = "ndarray")]
impl<'a> From<ndarray::ArrayView2<'a, f32>> for Mat5GPU {
    fn from(value: ndarray::ArrayView2<'a, f32>) -> Self {
        Mat5GPU {
            first: mat4(
                vec4(value[(0, 0)], value[(1, 0)], value[(2, 0)], value[(3, 0)]),
                vec4(value[(0, 1)], value[(1, 1)], value[(2, 1)], value[(3, 1)]),
                vec4(value[(0, 2)], value[(1, 2)], value[(2, 2)], value[(3, 2)]),
                vec4(value[(0, 3)], value[(1, 3)], value[(2, 3)], value[(3, 3)]),
            ),
            last_row: vec4(value[(4, 0)], value[(4, 1)], value[(4, 2)], value[(4, 3)]),
            last_col: vec4(value[(0, 4)], value[(1, 4)], value[(2, 4)], value[(3, 4)]),
            last_value: value[(4, 4)],
            padding: [0; 3]
        }
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array2<f32>> for Mat5GPU {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array2<f32>> for Mat5GPU {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<&Mat5GPU> for ndarray::Array2<f32> {
    fn from(value: &Mat5GPU) -> Self {
        ndarray::Array2::from_shape_vec((5, 5), vec!(
            value.first.col(0).x, value.first.col(1).x, value.first.col(2).x, value.first.col(3).x, value.last_col.x,
            value.first.col(0).y, value.first.col(1).y, value.first.col(2).y, value.first.col(3).y, value.last_col.y,
            value.first.col(0).z, value.first.col(1).z, value.first.col(2).z, value.first.col(3).z, value.last_col.z,
            value.first.col(0).w, value.first.col(1).w, value.first.col(2).w, value.first.col(3).w, value.last_col.w,
            value.last_row.x, value.last_row.y, value.last_row.z, value.last_row.w, value.last_value
        )).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl From<Mat5GPU> for ndarray::Array2<f32> {
    fn from(value: Mat5GPU) -> Self {
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
        let converted_vec = Vec5GPU::from(&nd_vec);
        let double_converted_vec = ndarray::Array1::from(converted_vec);
        assert_eq!(nd_vec, double_converted_vec);
    }

    #[test]
    fn test_vec_conversion_2() {
        let input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nd_vec = ndarray::Array2::from_shape_vec((5, 1), input_vec).unwrap();
        let converted_vec = Vec5GPU::from(&nd_vec);
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
        let converted_mat = Mat5GPU::from(&nd_mat);
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
        let converted_mat = Mat5GPU::from(&nd_mat);
        let converted_vec = Vec5GPU::from(&nd_vec);
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
        let converted_mat1 = Mat5GPU::from(&nd_mat1);
        let converted_mat2 = Mat5GPU::from(&nd_mat2);
        let result_mat = converted_mat1 * converted_mat2;
        let back_converted_result = ndarray::Array2::from(result_mat);
        let expected_result = nd_mat1.dot(&nd_mat2);
        assert_eq!(expected_result, back_converted_result);
    }
}