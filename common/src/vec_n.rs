use glam::{Vec2, Vec3, Vec4};

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

    pub fn resize<const K: usize>(&self) -> VecN<K> {
        let mut output = VecN::<K>::ZERO;
        for i in 0..N.min(K) {
            output[i] = self[i];
        }
        output
    }
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

impl <const N: usize> core::ops::Mul<f32> for VecN<N> {
    type Output = VecN<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] * rhs;
        }
        output
    }
}

impl <const N: usize> core::ops::Div<f32> for VecN<N> {
    type Output = VecN<N>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] / rhs;
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
}