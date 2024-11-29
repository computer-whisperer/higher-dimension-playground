use ndarray::Array2;

#[allow(dead_code)]
pub fn scale_matrix_4d(scale: f32) -> Array2<f32> {
    Array2::from_shape_vec( (5, 5),
    vec![scale, 0.0,   0.0,   0.0,   0.0,
            0.0,   scale, 0.0,   0.0,   0.0,
            0.0,   0.0,   scale, 0.0,   0.0,
            0.0,   0.0,   0.0,   scale, 0.0,
            0.0,   0.0,   0.0,   0.0,   1.0]).unwrap()
}

#[allow(dead_code)]
pub fn translate_matrix_4d(x: f32, y: f32, z: f32, w: f32) -> Array2<f32> {
    Array2::from_shape_vec( (5, 5),
        vec![1.0,   0.0,   0.0,   0.0,   x,
                 0.0,   1.0,   0.0,   0.0,   y,
                 0.0,   0.0,   1.0,   0.0,  z,
                 0.0,   0.0,   0.0,   1.0,  w,
                 0.0,   0.0,   0.0,   0.0,   1.0]).unwrap()
}

#[allow(dead_code)]
pub fn rotation_matrix_4d_rotate_1(angle: f32) -> Array2<f32> {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    Array2::from_shape_vec( (5, 5),
    vec![cos_theta, 0.0, -sin_theta, 0.0, 0.0,
            0.0,       1.0, 0.0,        0.0, 0.0,
            sin_theta, 0.0, cos_theta,  0.0, 0.0,
            0.0,       0.0, 0.0,        1.0, 0.0,
            0.0,       0.0, 0.0,        0.0, 1.0]
    ).unwrap()
}

#[allow(dead_code)]
pub fn rotation_matrix_one_angle(dims: usize, dim_from: usize, dim_to: usize, angle: f32) -> Array2<f32> {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    
    let mut mat = Array2::eye(dims);
    
    mat[(dim_from, dim_from)] = cos_theta;
    mat[(dim_from, dim_to)] = sin_theta;
    mat[(dim_to, dim_from)] = -sin_theta;
    mat[(dim_to, dim_to)] = cos_theta;
    mat
}