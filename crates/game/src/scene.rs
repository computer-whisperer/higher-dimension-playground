use higher_dimension_playground::matrix_operations::{
    double_rotation_matrix_4d, scale_matrix_4d, scale_matrix_4d_elementwise, translate_matrix_4d,
};

pub fn build_scene_instances(time: f32) -> Vec<common::ModelInstance> {
    let mut instances = Vec::new();

    // 16 outer blocks: 2x2x2x2 grid
    let mut texture_rot = 0u32;
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                for w in 0..2 {
                    let px = (x * 4 - 2) as f32 - 0.5;
                    let py = (y * 4 - 2) as f32 - 0.5;
                    let pz = (z * 4 - 2) as f32 - 0.5;
                    let pw = (w * 4 - 2) as f32 - 0.5;

                    let model_transform =
                        translate_matrix_4d(px, py, pz, pw).dot(&scale_matrix_4d(1.0));

                    instances.push(common::ModelInstance {
                        model_transform: model_transform.into(),
                        cell_material_ids: [texture_rot + 1; 8],
                    });

                    texture_rot = (texture_rot + 1) % 5;
                }
            }
        }
    }

    // Center block with double rotation
    let rot = double_rotation_matrix_4d([0, 1], 0.5 * time, [2, 3], 0.3 * time);
    let model_transform = translate_matrix_4d(0.0, 0.0, 0.0, 0.0)
        .dot(&rot)
        .dot(&translate_matrix_4d(-0.5, -0.5, -0.5, -0.5))
        .dot(&scale_matrix_4d(1.0));

    instances.push(common::ModelInstance {
        model_transform: model_transform.into(),
        cell_material_ids: [13; 8],
    });

    // Floor plane: huge thin tesseract with top surface at Y = -3.0
    let floor_transform = translate_matrix_4d(-100.0, -3.5, -100.0, -100.0)
        .dot(&scale_matrix_4d_elementwise(200.0, 0.5, 200.0, 200.0));
    instances.push(common::ModelInstance {
        model_transform: floor_transform.into(),
        cell_material_ids: [11; 8],
    });

    instances
}
