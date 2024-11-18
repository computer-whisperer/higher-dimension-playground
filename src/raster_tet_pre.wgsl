struct IntermediateVertex {
    pos: vec4<f32>,
    divisor: f32,
    tex_pos: vec3<f32>,
    tex_id: u32
}

struct RenderMetadata {
    window_width: u32,
    window_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32,
    input_tetrahedrons: u32
}

struct Tetrahedron {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
    v0_tex: vec3<f32>,
    v1_tex: vec3<f32>,
    v2_tex: vec3<f32>,
    v3_tex: vec3<f32>,
    texture_id: u32
}

const NULL_TEXTURE_ID: u32 = 0u;

@group(2) @binding(0) var<storage, read_write> intermediate_vertex_buffer : array<IntermediateVertex>;
@group(2) @binding(4) var<storage, read_write> outgoing_tet_buffer : array<Tetrahedron>;
@group(0) @binding(3) var<uniform> render_metadata : RenderMetadata;

fn apply_projection(v: vec4<f32>, divisor: f32) -> vec4<f32> {
    return vec4<f32>(
        v.x/divisor,
        v.y/divisor,
        v.z,
        v.w
    );
}

fn scale_to_screen(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        ((v.x + 1.0)/2.0)*f32(render_metadata.render_width),
        ((v.y + 1.0)/2.0)*f32(render_metadata.render_height),
        ((v.y + 1.0)/2.0)*f32(render_metadata.depth_factor),
        v.w
    );
}

// Process tets in intermediate_vertex_buffer buffer, clip, cull, shift to screen space, and other fun stuff. Re-emit to intermediate_vertex_buffer, and append extra tets to end

@compute @workgroup_size(256, 1)
fn raster_tet_pre_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tet_id = global_id.x;
    var v0 = intermediate_vertex_buffer[tet_id*4u + 0u];
    var v1 = intermediate_vertex_buffer[tet_id*4u + 1u];
    var v2 = intermediate_vertex_buffer[tet_id*4u + 2u];
    var v3 = intermediate_vertex_buffer[tet_id*4u + 3u];

    // For now just discard tets we don't like

    // Clip against z/w plane first
    if (v0.pos.z < -1.0 || v0.pos.z > 1.0 ||
        v1.pos.z < -1.0 || v1.pos.z > 1.0 ||
        v2.pos.z < -1.0 || v2.pos.z > 1.0 ||
        v3.pos.z < -1.0 || v3.pos.z > 1.0) {
        outgoing_tet_buffer[tet_id].texture_id = NULL_TEXTURE_ID;
        return;
    }

    // Apply projection
    v0.pos = apply_projection(v0.pos, v0.divisor);
    v1.pos = apply_projection(v1.pos, v1.divisor);
    v2.pos = apply_projection(v2.pos, v2.divisor);
    v3.pos = apply_projection(v3.pos, v3.divisor);

    // Clip to screen bounds
    if (v0.pos.x < -1.0 || v0.pos.x > 1.0 || v0.pos.y < -1.0 || v0.pos.y > 1.0 ||
        v1.pos.x < -1.0 || v1.pos.x > 1.0 || v1.pos.y < -1.0 || v1.pos.y > 1.0 ||
        v2.pos.x < -1.0 || v2.pos.x > 1.0 || v2.pos.y < -1.0 || v2.pos.y > 1.0 ||
        v3.pos.x < -1.0 || v3.pos.x > 1.0 || v3.pos.y < -1.0 || v3.pos.y > 1.0 )
    {
        outgoing_tet_buffer[tet_id].texture_id = NULL_TEXTURE_ID;
        return;
    }

    // Scale to screen
    v0.pos = scale_to_screen(v0.pos);
    v1.pos = scale_to_screen(v1.pos);
    v2.pos = scale_to_screen(v2.pos);
    v3.pos = scale_to_screen(v3.pos);


    // Output
    outgoing_tet_buffer[tet_id] = Tetrahedron(
        v0.pos,
        v1.pos,
        v2.pos,
        v3.pos,
        intermediate_vertex_buffer[tet_id*4u + 0u].tex_pos,
        intermediate_vertex_buffer[tet_id*4u + 1u].tex_pos,
        intermediate_vertex_buffer[tet_id*4u + 2u].tex_pos,
        intermediate_vertex_buffer[tet_id*4u + 3u].tex_pos,
        intermediate_vertex_buffer[tet_id*4u + 0u].tex_id,
    );
}