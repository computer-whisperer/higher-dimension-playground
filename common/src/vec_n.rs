use glam::{Vec2, Vec3, Vec4};

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(transparent)]
pub struct VecN<const N: usize> {
    values: [f32; N],
}

unsafe impl<const N: usize> bytemuck::Zeroable for VecN<N> {}

unsafe impl<const N: usize> bytemuck::Pod for VecN<N> {}

impl<const N: usize> VecN<N> {
    pub const ZERO: Self = Self { values: [0f32; N] };

    pub fn new(v: [f32; N]) -> Self {
        let mut values = [0f32; N];
        for i in 0..N {
            values[i] = v[i];
        }
        Self { values }
    }

    pub fn x(&self) -> f32 {
        self[0]
    }
    pub fn y(&self) -> f32 {
        self[1]
    }
    pub fn z(&self) -> f32 {
        self[2]
    }
    pub fn w(&self) -> f32 {
        self[3]
    }
    pub fn v(&self) -> f32 {
        self[4]
    }

    pub fn resize<const K: usize>(&self) -> VecN<K> {
        let mut output = VecN::<K>::ZERO;
        for i in 0..N.min(K) {
            output[i] = self[i];
        }
        output
    }
}

// Specialized extend for VecN<4> -> VecN<5>
impl VecN<4> {
    pub fn extend(&self, v: f32) -> VecN<5> {
        VecN::<5>::new([self[0], self[1], self[2], self[3], v])
    }
}

impl<const N: usize> core::ops::Index<usize> for VecN<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<const N: usize> core::ops::Index<usize> for &VecN<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<const N: usize> core::ops::IndexMut<usize> for VecN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<const N: usize> core::ops::Mul<VecN<N>> for VecN<N> {
    type Output = f32;

    fn mul(self, rhs: VecN<N>) -> Self::Output {
        let mut output = 0.0;

        for i in 0..N {
            output += self.values[i] * rhs.values[i];
        }

        output
    }
}

impl<const N: usize> core::ops::Add<&VecN<N>> for &VecN<N> {
    type Output = VecN<N>;

    fn add(self, rhs: &VecN<N>) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] + rhs.values[i];
        }
        output
    }
}

impl<const N: usize> core::ops::Sub<&VecN<N>> for &VecN<N> {
    type Output = VecN<N>;

    fn sub(self, rhs: &VecN<N>) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] - rhs.values[i];
        }
        output
    }
}

impl<const N: usize> core::ops::Mul<f32> for VecN<N> {
    type Output = VecN<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] * rhs;
        }
        output
    }
}

impl<const N: usize> core::ops::Div<f32> for VecN<N> {
    type Output = VecN<N>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut output = VecN::<N>::ZERO;
        for i in 0..N {
            output[i] = self.values[i] / rhs;
        }
        output
    }
}

impl<const N: usize> core::ops::Add<VecN<N>> for VecN<N> {
    type Output = VecN<N>;

    fn add(self, rhs: VecN<N>) -> Self::Output {
        &self + &rhs
    }
}

impl<const N: usize> core::ops::Sub<VecN<N>> for VecN<N> {
    type Output = VecN<N>;

    fn sub(self, rhs: VecN<N>) -> Self::Output {
        &self - &rhs
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

impl From<Vec2> for VecN<2> {
    fn from(value: Vec2) -> Self {
        Self::new([value.x, value.y])
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

// Specialized ndarray conversions for VecN<5> (used for 4D homogeneous coordinates)
#[cfg(feature = "ndarray")]
impl From<ndarray::ArrayView1<'_, f32>> for VecN<5> {
    fn from(value: ndarray::ArrayView1<'_, f32>) -> Self {
        assert_eq!(value.shape(), &[5]);
        VecN::new([value[0], value[1], value[2], value[3], value[4]])
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array1<f32>> for VecN<5> {
    fn from(value: &ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array1<f32>> for VecN<5> {
    fn from(value: ndarray::Array1<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::ArrayView2<'_, f32>> for VecN<5> {
    fn from(value: ndarray::ArrayView2<'_, f32>) -> Self {
        assert_eq!(value.shape(), &[5, 1]);
        VecN::new([
            value[[0, 0]],
            value[[1, 0]],
            value[[2, 0]],
            value[[3, 0]],
            value[[4, 0]],
        ])
    }
}

#[cfg(feature = "ndarray")]
impl From<&ndarray::Array2<f32>> for VecN<5> {
    fn from(value: &ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::Array2<f32>> for VecN<5> {
    fn from(value: ndarray::Array2<f32>) -> Self {
        Self::from(value.view())
    }
}

#[cfg(feature = "ndarray")]
impl From<&VecN<5>> for ndarray::Array1<f32> {
    fn from(value: &VecN<5>) -> Self {
        ndarray::Array1::from_vec(vec![value[0], value[1], value[2], value[3], value[4]])
    }
}

#[cfg(feature = "ndarray")]
impl From<VecN<5>> for ndarray::Array1<f32> {
    fn from(value: VecN<5>) -> Self {
        Self::from(&value)
    }
}

#[cfg(feature = "ndarray")]
impl From<&VecN<5>> for ndarray::Array2<f32> {
    fn from(value: &VecN<5>) -> Self {
        Self::from_shape_vec(
            (5, 1),
            vec![value[0], value[1], value[2], value[3], value[4]],
        )
        .unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl From<VecN<5>> for ndarray::Array2<f32> {
    fn from(value: VecN<5>) -> Self {
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
