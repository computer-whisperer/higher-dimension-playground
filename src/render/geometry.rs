use crate::hypercube::{generate_simplexes_for_k_face_3, Hypercube};
use common::{get_normal, ModelTetrahedron};
use glam::{Vec2, Vec4};

pub(super) fn generate_tesseract_tetrahedrons() -> Vec<ModelTetrahedron> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let cube_vertexes = Hypercube::<3, usize>::generate_vertices();
    let tetrahedron_cells = Hypercube::<4, usize>::generate_k_faces_3();

    let mut output_tetrahedrons = Vec::new();
    let texture_position_simplexes =
        generate_simplexes_for_k_face_3::<3>([0b000, 0b001, 0b010, 0b100]);

    for cell_id in 0..tetrahedron_cells.len() {
        let position_simplexes = generate_simplexes_for_k_face_3::<4>(tetrahedron_cells[cell_id]);

        for simplex_id in 0..position_simplexes.len() {
            let texture_simplex = texture_position_simplexes[simplex_id];

            let mut vertex_positions = position_simplexes[simplex_id].map(|i| {
                let vertex = tesseract_vertexes[i];
                glam::Vec4::new(
                    vertex[0] as f32,
                    vertex[1] as f32,
                    vertex[2] as f32,
                    vertex[3] as f32,
                )
            });

            let mut texture_positions = texture_simplex.map(|i| {
                let vertex = cube_vertexes[i];
                glam::Vec4::new(vertex[0] as f32, vertex[1] as f32, vertex[2] as f32, 1.0)
            });

            let normal: Vec4 = get_normal(&[
                (vertex_positions[1] - vertex_positions[0]).into(),
                (vertex_positions[2] - vertex_positions[0]).into(),
                (vertex_positions[3] - vertex_positions[0]).into(),
            ])
            .into();
            let test_vector = Vec4::new(1.0, 1.0, 1.0, 1.0);
            let is_normal_flipped = test_vector.dot(normal.into()) < 0.0;
            let should_be_flipped = tetrahedron_cells[cell_id].contains(&0);
            if should_be_flipped != is_normal_flipped {
                let temp = vertex_positions[1];
                vertex_positions[1] = vertex_positions[2];
                vertex_positions[2] = temp;
                let temp = texture_positions[1];
                texture_positions[1] = texture_positions[2];
                texture_positions[2] = temp;
            }

            output_tetrahedrons.push(common::ModelTetrahedron {
                vertex_positions,
                texture_positions,
                cell_id: cell_id as u32,
                padding: [0; 3],
            })
        }
    }

    output_tetrahedrons
}

pub(super) fn generate_tesseract_edges() -> Vec<common::ModelEdge> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let tesseract_edges = Hypercube::<4, usize>::generate_k_faces_1();

    let mut output_edges = Vec::new();

    for edge in tesseract_edges {
        let vertex_positions = edge.map(|i| {
            let vertex = tesseract_vertexes[i];
            glam::Vec4::new(
                vertex[0] as f32,
                vertex[1] as f32,
                vertex[2] as f32,
                vertex[3] as f32,
            )
        });

        output_edges.push(common::ModelEdge { vertex_positions });
    }

    output_edges
}

pub(super) fn mat5_mul_vec5(mat: &nalgebra::Matrix5<f32>, v: [f32; 5]) -> [f32; 5] {
    let mut out = [0.0; 5];
    for r in 0..5 {
        for c in 0..5 {
            out[r] += mat[(r, c)] * v[c];
        }
    }
    out
}

pub(super) fn transform_model_point(mat: &common::MatN<5>, point: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 5];
    let in_p = [point[0], point[1], point[2], point[3], 1.0];

    for r in 0..5 {
        for c in 0..5 {
            out[r] += mat[[r, c]] * in_p[c];
        }
    }

    let inv_w = if out[4].abs() > 1e-6 {
        1.0 / out[4]
    } else {
        1.0
    };
    [
        out[0] * inv_w,
        out[1] * inv_w,
        out[2] * inv_w,
        out[3] * inv_w,
    ]
}

pub(super) fn project_view_point_to_ndc(
    view_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<Vec2> {
    let depth = (view_point[2] * view_point[2] + view_point[3] * view_point[3]).sqrt();
    if depth < 1e-4 {
        return None;
    }

    let projection_divisor = depth / focal_length_xy.max(1e-4);
    Some(Vec2::new(
        view_point[0] / projection_divisor,
        aspect * (-view_point[1]) / projection_divisor,
    ))
}
