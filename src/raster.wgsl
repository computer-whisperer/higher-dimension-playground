struct ColorBuffer {
  values: array<atomic<u32>>,
}

struct VertexInput {
    pos: vec4<f32>,
    tex_pos: vec3<f32>,
    cell: u32
}

struct VertexInputBuffer {
  values: array<VertexInput>,
}

struct VertexOutput {
    pos: vec4<f32>,
    tex_pos: vec3<f32>,
    tex_id: u32
}

struct VertexOutputBuffer {
  values: array<VertexOutput>,
}

struct ScreenUniform {
    window_width: f32,
    window_height: f32,
    render_width: f32,
    render_height: f32,
    depth_factor: u32
}

struct Camera {
    view_transform: array<vec4<f32>, 8>,
    model_transform: array<vec4<f32>, 8>
}

@group(0) @binding(0) var<storage, read_write> color_buffer : ColorBuffer;
@group(1) @binding(0) var<storage, read_write> depth_buffer : array<atomic<u32>>;
@group(2) @binding(0) var<storage, read> vertex_input_buffer : VertexInputBuffer;
@group(3) @binding(0) var<uniform> screen_dims : ScreenUniform;
@group(4) @binding(0) var<uniform> camera : Camera;
@group(5) @binding(0) var<storage, read_write> vertex_output_buffer : VertexOutputBuffer;


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

fn color_pixel(x: u32, y: u32, r: u32, g: u32, b: u32) {
  let pixelID = u32(x + y * u32(screen_dims.render_width)) * 3u;

  atomicMax(&color_buffer.values[pixelID + 0u], r);
  atomicMax(&color_buffer.values[pixelID + 1u], g);
  atomicMax(&color_buffer.values[pixelID + 2u], b);
}

fn transform_to_screen_coords(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        ((v[0] + 1)/2)*screen_dims.render_width,
        ((v[1] + 1)/2)*screen_dims.render_height
    );
};

fn transform_to_screen_coords_4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        ((v[0] + 1)/2)*screen_dims.render_width,
        ((v[1] + 1)/2)*screen_dims.render_height,
        v[2],
        v[3]
    );
};

fn draw_line(v1: vec2<f32>, v2: vec2<f32>, color: vec3<f32>) {
    let v1_s = transform_to_screen_coords(v1);
    let v2_s = transform_to_screen_coords(v2);
    let dist = i32(distance(v1_s.xy, v2_s.xy));
    for (var i = 0; i < dist; i = i + 1) {
        let x = v1_s.x + (v2_s.x - v1_s.x) * (f32(i) / f32(dist));
        let y = v1_s.y + (v2_s.y - v1_s.y) * (f32(i) / f32(dist));
        // color_pixel(u32(x), u32(y), vec3<f32>(1.0, 1.0, 1.0));
        if (x > 0 && x < screen_dims.render_width && y > 0 && y < screen_dims.render_height)
        {
            color_pixel(u32(x), u32(y), u32(color.r*256.0), u32(color.g*256.0), u32(color.b*256.0));
        }
    }
}

// From: https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling
fn barycentric(v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>, p: vec2<f32>) -> vec3<f32> {
    let u = cross(vec3<f32>(v3.x - v1.x, v2.x - v1.x, v1.x - p.x),
                vec3<f32>(v3.y - v1.y, v2.y - v1.y, v1.y - p.y));

    if (abs(u.z) < 1.0) {
        return vec3<f32>(-1.0, 1.0, 1.0);
    }

    return vec3<f32>(1.0 - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

fn get_min_max(v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>) -> vec4<f32> {
    var min_max = vec4<f32>(0.);
    min_max.x = min(min(v1.x, v2.x), v3.x);
    min_max.y = min(min(v1.y, v2.y), v3.y);
    min_max.z = max(max(v1.x, v2.x), v3.x);
    min_max.w = max(max(v1.y, v2.y), v3.y);

    return min_max;
}


/*
fn draw_triangle(v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>) {
  let min_max = get_min_max(v1, v2, v3);
  let startX = u32(min_max.x);
  let startY = u32(min_max.y);
  let endX = u32(min_max.z);
  let endY = u32(min_max.w);

  for (var x: u32 = startX; x <= endX; x = x + 1u) {
    for (var y : u32 = startY; y <= endY; y = y + 1u) {
      let bc = barycentric(v1, v2, v3, vec2<f32>(f32(x), f32(y)));
      let color = (bc.x * v1.z + bc.y * v2.z + bc.z * v3.z) * 25. - 100.;

      let R = color;
      let G = color;
      let B = color;

      if (bc.x < 0.0 || bc.y < 0.0 || bc.z < 0.0) {
        continue;
      }
      color_pixel(x, y, u32(R), u32(G), u32(B));
    }
  }
}
*/

fn sample_texture(texture_id: u32, texture_pos: vec3<f32>) -> vec3<f32> {
    let color_array = array<vec3<f32>, 8>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0)
    );
    return color_array[texture_id%8];
}

struct FragmentVertex {
    pos: vec4<f32>,
    texture_pos: vec3<f32>
}

fn do_zw_raycast(v0: vec2<f32>, v1: vec2<f32>, i: u32) -> vec2<f32>
{
    //https://gamedev.stackexchange.com/questions/116422/best-way-to-find-line-segment-intersection
    let pi = 3.14159;
    let angle_ratio = ((f32(i)/f32(screen_dims.depth_factor)) - 0.5);
    let theta = angle_ratio*(pi/6.0) + (pi/4.0);
    //let theta = angle_ratio*(pi/2.0) + (pi/4.0);
    //let theta = pi/4.0;
    let sample_ray_y = sin(theta);
    let sample_ray_x = cos(theta);
    let r = ((-v0.y)*(v1.x - v0.x) - (-v0.x)*(v1.y - v0.y)) / ((sample_ray_x)*(v1.y - v0.y) - (sample_ray_y)*(v1.x - v0.x));
    let s = ((-v0.y)*(sample_ray_x) - (-v0.x)*(sample_ray_y))/((sample_ray_x)*(v1.y - v0.y) - (sample_ray_y)*(v1.x - v0.x));
    return vec2<f32>(r, s);
}

fn depth_zw_line(vs: vec2<i32>, v0: vec4<f32>, v1: vec4<f32>){
    for (var i = 0u; i < screen_dims.depth_factor; i += 1u) {
        let res = do_zw_raycast(v0.zw, v1.zw, i);
        if (res.y > 0 && res.y < 1) {
            let depth_index = (u32(vs.x) + u32(vs.y) * u32(screen_dims.render_width)) * screen_dims.depth_factor + i;
            let depth_value = u32(res.x*10000.0);
            atomicMin(&depth_buffer[depth_index], depth_value);
        }
    }
}

fn render_zw_line(vs: vec2<i32>, v0: FragmentVertex, v1: FragmentVertex, texture_id: u32){

    if (false)
    {
        // Wrong, but simple
        let color = sample_texture(texture_id, v0.texture_pos);
        let x = v0.pos.x;
        let y = v0.pos.y;
        if (x > 0 && x < screen_dims.render_width && y > 0 && y < screen_dims.render_height)
        {
            color_pixel(u32(vs.x), u32(vs.y), u32(color.r*256.0), u32(color.g*256.0), u32(color.b*256.0));
        }
    }
    else {
        // 4-d appropriate volumetric shading
        var occlusion_numerator = 0.0;
        var occlusion_denominator = 0.1;
        let color = sample_texture(texture_id, v0.texture_pos);
        for (var i = 0u; i < screen_dims.depth_factor; i += 1u) {
            let res = do_zw_raycast(v0.pos.zw, v1.pos.zw, i);
            if (res.y > 0 && res.y < 1) {
                let depth_index = (u32(vs.x) + u32(vs.y) * u32(screen_dims.render_width)) * screen_dims.depth_factor + i;
                let depth_value = u32(res.x*10000.0);
                if depth_value > depth_buffer[depth_index] + 10u {
                    occlusion_numerator += 1.0;
                }
                occlusion_denominator += 1.0;
            }
        }
        let occlusion_ratio = occlusion_numerator/occlusion_denominator;
        if occlusion_ratio < 0.9 && occlusion_denominator >= 1.0 {
            let delta = v0.pos.zw - v1.pos.zw;
            let dist = sqrt(delta.x*delta.x + delta.y*delta.y);
            let intensity = 1.0;
            //let dist=1.0;
            let pixelID = (u32(vs.x) + u32(vs.y) * u32(screen_dims.render_width)) * 3u;

            atomicAdd(&color_buffer.values[pixelID + 0u], u32(color.r*256.0*intensity));
            atomicAdd(&color_buffer.values[pixelID + 1u], u32(color.g*256.0*intensity));
            atomicAdd(&color_buffer.values[pixelID + 2u], u32(color.b*256.0*intensity));
        }
    }
}

const SLICE_COUNT_X: u32 = 256u;
const SLICE_COUNT_Y: u32 = 1u;
const SLICE_COUNT_TOTAL: u32 = SLICE_COUNT_X*SLICE_COUNT_Y;

@compute @workgroup_size(256, 1)
fn raster_vertex_shader(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    let vertex_input = vertex_input_buffer.values[index];
    let model_vertex_position = array<f32, 5>(vertex_input.pos.x, vertex_input.pos.y, vertex_input.pos.z, vertex_input.pos.w, 1.0);

    let view_transform = mat5x5_unpack(camera.view_transform);
    let model_transform = mat5x5_unpack(camera.model_transform);

    let pre_camera_transform = mat5x5_mul_vec5(model_transform, model_vertex_position);

    let view_vector = mat5x5_mul_vec5(view_transform, pre_camera_transform);

    // Perspective scaling
    let focal_length = 1.0;
    let near = 0.1;
    let far = 100.0;
    let projected_distance = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3]);

    let aspect_ratio = screen_dims.render_width/screen_dims.render_height;

    var clip_position = vec4<f32>(
        view_vector[0] / projected_distance/focal_length,
        aspect_ratio * (-view_vector[1]) / projected_distance/focal_length,
        view_vector[2],
        view_vector[3]
    );

    let v0_t = vertex_input_buffer.values[index].tex_pos;
    let cell = vertex_input_buffer.values[index].cell;

    // For now:
    let texture_id = cell;

    vertex_output_buffer.values[index] = VertexOutput(
                                          clip_position,
                                          v0_t,
                                          texture_id
                                      );
}

@compute @workgroup_size(SLICE_COUNT_TOTAL, 1)
fn raster_depth_shader(@builtin(global_invocation_id) global_id: vec3<u32>){
    let tet_index = global_id.x/(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index = global_id.x%(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index_x = slice_index%SLICE_COUNT_X;
    let slice_index_y = slice_index/SLICE_COUNT_X;
    let v0 = transform_to_screen_coords_4(vertex_output_buffer.values[tet_index*4u + 0u].pos);
    let v1 = transform_to_screen_coords_4(vertex_output_buffer.values[tet_index*4u + 1u].pos);
    let v2 = transform_to_screen_coords_4(vertex_output_buffer.values[tet_index*4u + 2u].pos);
    let v3 = transform_to_screen_coords_4(vertex_output_buffer.values[tet_index*4u + 3u].pos);

    let min_xy = vec2<i32>(
        i32(min(min(min(v0.x, v1.x), v2.x), v3.x)),
        i32(min(min(min(v0.y, v1.y), v2.y), v3.y))
    );
    let max_xy = vec2<i32>(
        i32(max(max(max(v0.x, v1.x), v2.x), v3.x)),
        i32(max(max(max(v0.y, v1.y), v2.y), v3.y))
    );

    for (var x: i32 = min_xy.x + i32(slice_index_x); x <= max_xy.x; x = x + i32(SLICE_COUNT_X)) {
        for (var y : i32 = min_xy.y + i32(slice_index_y); y <= max_xy.y; y = y + i32(SLICE_COUNT_Y)) {
            let screen_v = vec2<i32>(x, y);
            let bc_a = barycentric(v0.xy, v1.xy, v2.xy, vec2<f32>(f32(x), f32(y)));
            let bc_b = barycentric(v0.xy, v1.xy, v3.xy, vec2<f32>(f32(x), f32(y)));
            let bc_c = barycentric(v0.xy, v2.xy, v3.xy, vec2<f32>(f32(x), f32(y)));
            let bc_d = barycentric(v1.xy, v2.xy, v3.xy, vec2<f32>(f32(x), f32(y)));

            let in_a: bool = (bc_a.x > 0.0 && bc_a.y > 0.0 && bc_a.z > 0.0);
            let in_b: bool = (bc_b.x > 0.0 && bc_b.y > 0.0 && bc_b.z > 0.0);
            let in_c: bool = (bc_c.x > 0.0 && bc_c.y > 0.0 && bc_c.z > 0.0);
            let in_d: bool = (bc_d.x > 0.0 && bc_d.y > 0.0 && bc_d.z > 0.0);

            if (!in_a && !in_b && !in_c && !in_d)
            {
                continue;
            }

            let vertex_a = v0*bc_a.x + v1*bc_a.y + v2*bc_a.z;
            let vertex_b = v0*bc_b.x + v1*bc_b.y + v3*bc_b.z;
            let vertex_c = v0*bc_c.x + v2*bc_c.y + v3*bc_c.z;
            let vertex_d = v1*bc_d.x + v2*bc_d.y + v3*bc_d.z;

            let vertex_result_a =
                select(
                    select(
                        vertex_c,
                        vertex_b,
                        in_b
                    ),
                    vertex_a,
                    in_a
                );

            let vertex_result_b =
                select(
                    select(
                        vertex_d,
                        select(
                            vertex_d,
                            vertex_c,
                            in_c
                        ),
                        in_b
                    ),
                    select(
                        select(
                            vertex_d,
                            vertex_c,
                            in_c
                        ),
                        vertex_b,
                        in_b
                    ),
                    in_a
                );

            depth_zw_line(screen_v, vertex_result_a, vertex_result_b);
        }
    }
}


@compute @workgroup_size(SLICE_COUNT_TOTAL, 1)
fn raster_pixel_shader(@builtin(global_invocation_id) global_id: vec3<u32>){
    let tet_index = global_id.x/(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index = global_id.x%(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index_x = slice_index%SLICE_COUNT_X;
    let slice_index_y = slice_index/SLICE_COUNT_X;
    let v0 = FragmentVertex(vertex_output_buffer.values[tet_index*4u + 0u].pos, vertex_output_buffer.values[tet_index*4u + 0u].tex_pos);
    let v1 = FragmentVertex(vertex_output_buffer.values[tet_index*4u + 1u].pos, vertex_output_buffer.values[tet_index*4u + 1u].tex_pos);
    let v2 = FragmentVertex(vertex_output_buffer.values[tet_index*4u + 2u].pos, vertex_output_buffer.values[tet_index*4u + 2u].tex_pos);
    let v3 = FragmentVertex(vertex_output_buffer.values[tet_index*4u + 3u].pos, vertex_output_buffer.values[tet_index*4u + 3u].tex_pos);
    let texture_id = vertex_output_buffer.values[tet_index*4u + 0u].tex_id;
    let color = sample_texture(texture_id, vec3<f32>(0.0, 0.0, 0.0));

    let v0_s = FragmentVertex(transform_to_screen_coords_4(v0.pos), v0.texture_pos);
    let v1_s = FragmentVertex(transform_to_screen_coords_4(v1.pos), v1.texture_pos);
    let v2_s = FragmentVertex(transform_to_screen_coords_4(v2.pos), v2.texture_pos);
    let v3_s = FragmentVertex(transform_to_screen_coords_4(v3.pos), v3.texture_pos);

    let min_xy = vec2<i32>(
        i32(min(min(min(v0_s.pos.x, v1_s.pos.x), v2_s.pos.x), v3_s.pos.x)),
        i32(min(min(min(v0_s.pos.y, v1_s.pos.y), v2_s.pos.y), v3_s.pos.y))
    );
    let max_xy = vec2<i32>(
        i32(max(max(max(v0_s.pos.x, v1_s.pos.x), v2_s.pos.x), v3_s.pos.x)),
        i32(max(max(max(v0_s.pos.y, v1_s.pos.y), v2_s.pos.y), v3_s.pos.y))
    );

    for (var x: i32 = min_xy.x + i32(slice_index_x); x <= max_xy.x; x = x + i32(SLICE_COUNT_X)) {
        for (var y : i32 = min_xy.y + i32(slice_index_y); y <= max_xy.y; y = y + i32(SLICE_COUNT_Y)) {
            let screen_v = vec2<i32>(x, y);
            let bc_a = barycentric(v0_s.pos.xy, v1_s.pos.xy, v2_s.pos.xy, vec2<f32>(f32(x), f32(y)));
            let bc_b = barycentric(v0_s.pos.xy, v1_s.pos.xy, v3_s.pos.xy, vec2<f32>(f32(x), f32(y)));
            let bc_c = barycentric(v0_s.pos.xy, v2_s.pos.xy, v3_s.pos.xy, vec2<f32>(f32(x), f32(y)));
            let bc_d = barycentric(v1_s.pos.xy, v2_s.pos.xy, v3_s.pos.xy, vec2<f32>(f32(x), f32(y)));

            let in_a: bool = (bc_a.x > 0.0 && bc_a.y > 0.0 && bc_a.z > 0.0);
            let in_b: bool = (bc_b.x > 0.0 && bc_b.y > 0.0 && bc_b.z > 0.0);
            let in_c: bool = (bc_c.x > 0.0 && bc_c.y > 0.0 && bc_c.z > 0.0);
            let in_d: bool = (bc_d.x > 0.0 && bc_d.y > 0.0 && bc_d.z > 0.0);

            if (!in_a && !in_b && !in_c && !in_d)
            {
                continue;
            }

            let vertex_a = FragmentVertex(
                v0_s.pos*bc_a.x + v1_s.pos*bc_a.y + v2_s.pos*bc_a.z,
                v0_s.texture_pos*bc_a.x + v1_s.texture_pos*bc_a.y + v2_s.texture_pos*bc_a.z
            );
            let vertex_b = FragmentVertex(
                v0_s.pos*bc_b.x + v1_s.pos*bc_b.y + v3_s.pos*bc_b.z,
                v0_s.texture_pos*bc_b.x + v1_s.texture_pos*bc_b.y + v3_s.texture_pos*bc_b.z
            );
            let vertex_c = FragmentVertex(
                v0_s.pos*bc_c.x + v2_s.pos*bc_c.y + v3_s.pos*bc_c.z,
                v0_s.texture_pos*bc_c.x + v2_s.texture_pos*bc_c.y + v3_s.texture_pos*bc_c.z
            );
            let vertex_d = FragmentVertex(
                v1_s.pos*bc_d.x + v2_s.pos*bc_d.y + v3_s.pos*bc_d.z,
                v1_s.texture_pos*bc_d.x + v2_s.texture_pos*bc_d.y + v3_s.texture_pos*bc_d.z
            );

            let vertex_result_a = FragmentVertex(
                select(
                    select(
                        vertex_c.pos,
                        vertex_b.pos,
                        in_b
                    ),
                    vertex_a.pos,
                    in_a
                ),
                select(
                    select(
                        vertex_c.texture_pos,
                        vertex_b.texture_pos,
                        in_b
                    ),
                    vertex_a.texture_pos,
                    in_a
                )
            );

            let vertex_result_b = FragmentVertex(
                select(
                    select(
                        vertex_d.pos,
                        select(
                            vertex_d.pos,
                            vertex_c.pos,
                            in_c
                        ),
                        in_b
                    ),
                    select(
                        select(
                            vertex_d.pos,
                            vertex_c.pos,
                            in_c
                        ),
                        vertex_b.pos,
                        in_b
                    ),
                    in_a
                ),
                select(
                    select(
                        vertex_d.texture_pos,
                        select(
                            vertex_d.texture_pos,
                            vertex_c.texture_pos,
                            in_c
                        ),
                        in_b
                    ),
                    select(
                        select(
                            vertex_d.texture_pos,
                            vertex_c.texture_pos,
                            in_c
                        ),
                        vertex_b.texture_pos,
                        in_b
                    ),
                    in_a
                )
            );

            render_zw_line(screen_v, vertex_result_a, vertex_result_b, texture_id);
        }
    }
}

@compute @workgroup_size(256, 1)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x * 3u;

    atomicStore(&color_buffer.values[index + 0u], 0u);
    atomicStore(&color_buffer.values[index + 1u], 0u);
    atomicStore(&color_buffer.values[index + 2u], 0u);

    for (var i: u32 = 0u; i <= screen_dims.depth_factor; i = i + 1u)
    {
        atomicStore(&depth_buffer[index*screen_dims.depth_factor + i], 0xFFFFFFFFu);
    }
}
