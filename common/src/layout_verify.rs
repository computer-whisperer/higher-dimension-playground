// Layout verification - ensures Rust struct sizes match Slang shader expectations
// Run with: cargo test -p common layout_verify

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn verify_struct_sizes() {
        // Expected sizes from Slang reflection with -fvk-use-scalar-layout
        assert_eq!(
            std::mem::size_of::<Tetrahedron>(),
            176,
            "Tetrahedron size mismatch"
        );
        assert_eq!(
            std::mem::size_of::<ModelTetrahedron>(),
            144,
            "ModelTetrahedron size mismatch"
        );
        assert_eq!(
            std::mem::size_of::<ModelEdge>(),
            32,
            "ModelEdge size mismatch"
        );
        assert_eq!(
            std::mem::size_of::<ModelInstance>(),
            132,
            "ModelInstance size mismatch"
        );
        assert_eq!(
            std::mem::size_of::<WorkingData>(),
            256,
            "WorkingData size mismatch"
        );
        assert_eq!(std::mem::size_of::<MatN<5>>(), 100, "MatN<5> size mismatch");
        assert_eq!(std::mem::size_of::<VecN<5>>(), 20, "VecN<5> size mismatch");
    }

    #[test]
    fn verify_tetrahedron_layout() {
        // Slang: vertexPositions: offset=0, size=64
        // Slang: texturePositions: offset=64, size=64
        // Slang: normal: offset=128, size=16
        // Slang: invProjectionDivisors: offset=144, size=16
        // Slang: materialId: offset=160, size=4
        // Slang: padding: offset=164, size=12
        assert_eq!(std::mem::offset_of!(Tetrahedron, vertex_positions), 0);
        assert_eq!(std::mem::offset_of!(Tetrahedron, texture_positions), 64);
        assert_eq!(std::mem::offset_of!(Tetrahedron, normal), 128);
        assert_eq!(std::mem::offset_of!(Tetrahedron, inv_projection_divisors), 144);
        assert_eq!(std::mem::offset_of!(Tetrahedron, material_id), 160);
        assert_eq!(std::mem::offset_of!(Tetrahedron, padding), 164);
    }

    #[test]
    fn verify_model_tetrahedron_layout() {
        // Slang: vertexPositions: offset=0, size=64
        // Slang: texturePositions: offset=64, size=64
        // Slang: cellId: offset=128, size=4
        // Slang: padding: offset=132, size=12
        assert_eq!(std::mem::offset_of!(ModelTetrahedron, vertex_positions), 0);
        assert_eq!(
            std::mem::offset_of!(ModelTetrahedron, texture_positions),
            64
        );
        assert_eq!(std::mem::offset_of!(ModelTetrahedron, cell_id), 128);
        assert_eq!(std::mem::offset_of!(ModelTetrahedron, padding), 132);
    }

    #[test]
    fn verify_model_instance_layout() {
        // Slang: modelTransform: offset=0, size=100
        // Slang: cellMaterialIds: offset=100, size=32
        assert_eq!(std::mem::offset_of!(ModelInstance, model_transform), 0);
        assert_eq!(std::mem::offset_of!(ModelInstance, cell_material_ids), 100);
    }

    #[test]
    fn verify_working_data_layout() {
        // Slang: renderDimensions: offset=0, size=16
        // Slang: presentDimensions: offset=16, size=8
        // Slang: _raytraceSeedPacked: offset=24, size=8
        // Slang: viewMatrix: offset=32, size=100
        // Slang: viewMatrixInverse: offset=132, size=100
        // Slang: totalNumTetrahedrons: offset=232, size=4
        // Slang: shaderFault: offset=236, size=4
        // Slang: focalLengthXY: offset=240, size=4
        // Slang: focalLengthZW: offset=244, size=4
        // Slang: padding: offset=248, size=8
        assert_eq!(std::mem::offset_of!(WorkingData, render_dimensions), 0);
        assert_eq!(std::mem::offset_of!(WorkingData, present_dimensions), 16);
        assert_eq!(std::mem::offset_of!(WorkingData, raytrace_seed), 24);
        assert_eq!(std::mem::offset_of!(WorkingData, view_matrix), 32);
        assert_eq!(std::mem::offset_of!(WorkingData, view_matrix_inverse), 132);
        assert_eq!(
            std::mem::offset_of!(WorkingData, total_num_tetrahedrons),
            232
        );
        assert_eq!(std::mem::offset_of!(WorkingData, shader_fault), 236);
        assert_eq!(std::mem::offset_of!(WorkingData, focal_length_xy), 240);
        assert_eq!(std::mem::offset_of!(WorkingData, focal_length_zw), 244);
        assert_eq!(std::mem::offset_of!(WorkingData, padding), 248);
    }
}
