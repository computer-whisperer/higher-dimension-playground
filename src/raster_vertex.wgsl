
struct VertexInput {
    pos: vec4<f32>,
    tex_pos: vec3<f32>,
    cell: u32
}

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

struct Camera {
    view_transform: array<vec4<f32>, 8>
}

struct Instance {
    model_transform: array<vec4<f32>, 8>,
    cell_texture_ids: array<u32, 8>
}

@group(1) @binding(0) var<storage, read> vertex_input_buffer : array<VertexInput>;

@group(2) @binding(0) var<storage, read_write> intermediate_vertex_buffer : array<IntermediateVertex>;
@group(2) @binding(1) var<storage, read> instance_buffer : array<Instance>;
@group(2) @binding(3) var<uniform> camera : Camera;
@group(0) @binding(3) var<uniform> render_metadata : RenderMetadata;

struct Mat5x5 {
    // Store as array of 5 vec4s plus 5 individual floats for last column
    columns: array<vec4<f32>, 5>,
    last_row: array<f32, 5>,
}

// Constructor
fn mat5x5(
    c0: vec4<f32>, c0_last: f32,
    c1: vec4<f32>, c1_last: f32,
    c2: vec4<f32>, c2_last: f32,
    c3: vec4<f32>, c3_last: f32,
    c4: vec4<f32>, c4_last: f32
) -> Mat5x5 {
    return Mat5x5(
        array<vec4<f32>, 5>(c0, c1, c2, c3, c4),
        array<f32, 5>(c0_last, c1_last, c2_last, c3_last, c4_last)
    );
}

// Matrix multiplication
fn mat5x5_mul(a: Mat5x5, b: Mat5x5) -> Mat5x5 {
    var result: Mat5x5;

    for (var col = 0u; col < 5u; col++) {
        var b_col = array<f32, 5>(b.columns[col].x, b.columns[col].y, b.columns[col].z, b.columns[col].w, b.last_row[col]);
        for (var row = 0u; row < 4u; row++) {
            var a_row = array<f32, 5>(a.columns[0][row], a.columns[1][row], a.columns[2][row], a.columns[3][row], a.columns[4][row]);
            result.columns[col][row] = 0.0;
            for (var i = 0u; i < 5u; i++) {
                result.columns[col][row] += a_row[i]*b_col[i];
            }
        }
        result.last_row[col] = 0.0;
        for (var i = 0u; i < 5u; i++) {
            result.last_row[col] += a.last_row[i]*b_col[i];
        }
    }
    return result;
}

// Matrix-vector multiplication (5x5 * vec5)
fn mat5x5_mul_vec5(m: Mat5x5, v: array<f32, 5>) -> array<f32, 5> {
    var result: array<f32, 5>;

    // First 4 components
    for (var i = 0u; i < 4u; i++) {
        result[i] = m.columns[0][i]*v[0] + m.columns[1][i]*v[1] + m.columns[2][i]*v[2] + m.columns[3][i]*v[3] + m.columns[4][i]*v[4];
    }

    // Last component
    result[4] = m.last_row[0]*v[0] + m.last_row[1]*v[1] + m.last_row[2]*v[2] + m.last_row[3]*v[3] + m.last_row[4]*v[4];

    return result;
}

// Create identity matrix
fn mat5x5_identity() -> Mat5x5 {
    return mat5x5(
        vec4<f32>(1.0, 0.0, 0.0, 0.0), 0.0,
        vec4<f32>(0.0, 1.0, 0.0, 0.0), 0.0,
        vec4<f32>(0.0, 0.0, 1.0, 0.0), 0.0,
        vec4<f32>(0.0, 0.0, 0.0, 1.0), 0.0,
        vec4<f32>(0.0, 0.0, 0.0, 0.0), 1.0
    );
}


// Convert from uniform buffer array to Mat5x5
fn mat5x5_unpack(arr: array<vec4<f32>, 8>) -> Mat5x5 {
    var m: Mat5x5;
    var arr_copy = arr;

    // Process each column
    for (var col = 0u; col < 5u; col++) {

        let ix = (col * 5) + 0;
        let iy = (col * 5) + 1;
        let iz = (col * 5) + 2;
        let iw = (col * 5) + 3;
        let iv = (col * 5) + 4;
/*
        let ix = (0 * 5) + col;
        let iy = (1 * 5) + col;
        let iz = (2 * 5) + col;
        let iw = (3 * 5) + col;
        let iv = (4 * 5) + col;
*/
        let x = arr_copy[ix/4][ix%4];
        let y = arr_copy[iy/4][iy%4];
        let z = arr_copy[iz/4][iz%4];
        let w = arr_copy[iw/4][iw%4];
        let v = arr_copy[iv/4][iv%4];
        /*
        let x = 0.0;
        let y = 0.0;
        let z = 0.0;
        let w = 0.0;
        let v = 0.0;*/
        // First 4 rows go into vec4
        m.columns[col] = vec4<f32>(
            x,
            y,
            z,
            w
        );
        // Last row element
        m.last_row[col] = v;
    }

    return m;
}

const VERTEXES_PER_INSTANCE: u32 = 40u*4u;

@compute @workgroup_size(256, 1)
fn raster_vertex_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let instance_id = index/VERTEXES_PER_INSTANCE;
    let vertex_id = index%VERTEXES_PER_INSTANCE;

    let vertex_input = vertex_input_buffer[vertex_id];
    let model_vertex_position = array<f32, 5>(vertex_input.pos.x, vertex_input.pos.y, vertex_input.pos.z, vertex_input.pos.w, 1.0);

    let view_transform = mat5x5_unpack(camera.view_transform);
    let model_transform = mat5x5_unpack(instance_buffer[instance_id].model_transform);

    let pre_camera_transform = mat5x5_mul_vec5(model_transform, model_vertex_position);

    let view_vector = mat5x5_mul_vec5(view_transform, pre_camera_transform);

    // Perspective scaling
    let focal_length = 1.0;
    //var projection_divisor = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3])/focal_length;
    //var projection_divisor = (view_vector[2] + view_vector[3])/focal_length;
    //var projection_divisor = max(view_vector[2], view_vector[3])/focal_length;
    //var projection_divisor = view_vector[2]/focal_length;
    let aspect_ratio = f32(render_metadata.render_width)/f32(render_metadata.render_height);

    let theta = (atan2(view_vector[2], view_vector[3])/3.14159)/2.0;
    let depth = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3]);
    let projection_divisor = depth/focal_length;

    let clip_position = vec4<f32>(
        view_vector[0],
        aspect_ratio * (-view_vector[1]),
        theta,
        depth
    );

    let v0_t = vertex_input_buffer[vertex_id].tex_pos;
    let cell = vertex_input_buffer[vertex_id].cell;

    // For now:
    let texture_id = instance_buffer[instance_id].cell_texture_ids[cell];

    intermediate_vertex_buffer[index] = IntermediateVertex(
                                          clip_position,
                                          projection_divisor,
                                          v0_t,
                                          texture_id
                                      );
}