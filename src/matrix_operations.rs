use ndarray::Array2;

#[allow(dead_code)]
pub fn scale_matrix_4d(scale: f32) -> Array2<f32> {
    Array2::from_shape_vec(
        (5, 5),
        vec![
            scale, 0.0, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0,
            0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap()
}

#[allow(dead_code)]
pub fn scale_matrix_4d_elementwise(x: f32, y: f32, z: f32, w: f32) -> Array2<f32> {
    Array2::from_shape_vec(
        (5, 5),
        vec![
            x, 0.0, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 0.0, w,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap()
}

#[allow(dead_code)]
pub fn translate_matrix_4d(x: f32, y: f32, z: f32, w: f32) -> Array2<f32> {
    Array2::from_shape_vec(
        (5, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, y, 0.0, 0.0, 1.0, 0.0, z, 0.0, 0.0, 0.0,
            1.0, w, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap()
}

#[allow(dead_code)]
pub fn rotation_matrix_4d_rotate_1(angle: f32) -> Array2<f32> {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    Array2::from_shape_vec(
        (5, 5),
        vec![
            cos_theta, 0.0, -sin_theta, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, sin_theta, 0.0,
            cos_theta, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap()
}

#[allow(dead_code)]
pub fn rotation_matrix_one_angle(
    dims: usize,
    dim_from: usize,
    dim_to: usize,
    angle: f32,
) -> Array2<f32> {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();

    let mut mat = Array2::eye(dims);

    mat[(dim_from, dim_from)] = cos_theta;
    mat[(dim_from, dim_to)] = sin_theta;
    mat[(dim_to, dim_from)] = -sin_theta;
    mat[(dim_to, dim_to)] = cos_theta;
    mat
}

#[allow(dead_code)]
pub fn double_rotation_matrix_4d(
    plane1: [usize; 2],
    angle1: f32,
    plane2: [usize; 2],
    angle2: f32,
) -> Array2<f32> {
    let r1 = rotation_matrix_one_angle(5, plane1[0], plane1[1], angle1);
    let r2 = rotation_matrix_one_angle(5, plane2[0], plane2[1], angle2);
    r1.dot(&r2)
}
