pub fn identity_matrix<const N: usize>() -> [[f32; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        result[i][i] = 1.0;
    }
    result
}

pub fn flatten_5x5_matrix_for_wgpu(matrix: [[f32; 5]; 5]) -> [f32; 32] {
    let mut result = [0.0; 32];
    for i in 0..5 {
        for j in 0..5 {
            result[i * 5 + j] = matrix[j][i];
        }
    }
    result
}

fn scale_matrix_3d(scale: f32) -> [[f32; 4]; 4] {
    [[scale, 0.0, 0.0, 0.0],
     [0.0, scale, 0.0, 0.0],
     [0.0, 0.0, scale, 0.0],
     [0.0, 0.0,   0.0,  1.0]]
}

pub fn scale_matrix_4d(scale: f32) -> [[f32; 5]; 5] {
    [   [scale, 0.0,   0.0,   0.0,   0.0],
        [0.0,   scale, 0.0,   0.0,   0.0],
        [0.0,   0.0,   scale, 0.0,   0.0],
        [0.0,   0.0,   0.0,   scale, 0.0],
        [0.0,   0.0,   0.0,   0.0,   1.0]]
}

pub fn translate_matrix_4d(x: f32, y: f32, z: f32, w: f32) -> [[f32; 5]; 5] {
    [[1.0, 0.0, 0.0, 0.0, x],
     [0.0, 1.0, 0.0, 0.0, y],
     [0.0, 0.0, 1.0, 0.0, z],
     [0.0, 0.0, 0.0, 1.0, w],
     [0.0, 0.0, 0.0, 0.0, 1.0]
    ]
}

pub fn rotation_matrix_3d_yaw(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[cos_theta, 0.0, sin_theta, 0.0],
     [0.0,       1.0, 0.0, 0.0],
     [-sin_theta, 0.0, cos_theta, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

pub fn rotation_matrix_3d_pitch(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, cos_theta, -sin_theta, 0.0],
     [0.0, sin_theta, cos_theta, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

pub fn rotation_matrix_3d_roll(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[cos_theta, -sin_theta, 0.0, 0.0],
     [sin_theta, cos_theta, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

pub fn rotation_matrix_4d_rotate_0(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, -sin_theta, 0.0, 0.0, 0.0],
        [sin_theta, cos_theta,  0.0, 0.0, 0.0],
        [0.0,       0.0,        1.0, 0.0, 0.0],
        [0.0,       0.0,        0.0, 1.0, 0.0],
        [0.0,       0.0,        0.0, 0.0, 1.0]
    ]
}

pub fn rotation_matrix_4d_rotate_1(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, 0.0, -sin_theta, 0.0, 0.0],
        [0.0,       1.0, 0.0,        0.0, 0.0],
        [sin_theta, 0.0, cos_theta,  0.0, 0.0],
        [0.0,       0.0, 0.0,        1.0, 0.0],
        [0.0,       0.0, 0.0,        0.0, 1.0]
    ]
}

pub fn rotation_matrix_4d_rotate_3(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, 0.0, 0.0,    -sin_theta, 0.0],
        [0.0,       1.0, 0.0,    0.0, 0.0],
        [0.0,       0.0, 1.0,    0.0, 0.0],
        [sin_theta, 0.0, 0.0,    cos_theta, 0.0],
        [0.0,       0.0, 0.0,    0.0, 1.0]
    ]
}

pub fn matrix_multiply<const N: usize> (a: [[f32; N]; N], b: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}