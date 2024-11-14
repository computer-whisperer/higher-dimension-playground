
struct UniformInput {
    model_transform: array<vec4<f32>, 8>,
    view_transform: array<vec4<f32>, 8>
}

@group(0) @binding(0)
var<uniform> uniform_input: UniformInput;

struct VertexInput {
    @location(0) position: vec4<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) full_position: vec4<f32>
};

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

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var model_vertex_position = array<f32, 5>(model.position.x, model.position.y, model.position.z, model.position.w, 1.0);

    var view_transform = mat5x5_unpack(uniform_input.view_transform);
    var model_transform = mat5x5_unpack(uniform_input.model_transform);

    var pre_camera_transform = mat5x5_mul_vec5(model_transform, model_vertex_position);

    var view_vector = mat5x5_mul_vec5(view_transform, pre_camera_transform);

    // Perspective scaling
    let focal_length = 1.0;
    let near = 0.1;
    let far = 100.0;
    let projected_distance = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3]);
    out.clip_position = vec4<f32>(
        view_vector[0],
        view_vector[1],
        (projected_distance*far - far*near)/(focal_length * (far - near)),
        projected_distance/focal_length);

    out.full_position = vec4<f32>(view_vector[0], view_vector[1], view_vector[2], view_vector[3]);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}