struct ColorBuffer {
  values: array<atomic<u32>>,
}

struct VertexBuffer {
  values: array<vec4<f32>>,
}

struct ScreenUniform {
  width: f32,
  height: f32,
}


struct Camera {
    view_transform: array<vec4<f32>, 8>,
    model_transform: array<vec4<f32>, 8>
}

@group(0) @binding(0) var<storage, read_write> color_buffer : ColorBuffer;
@group(1) @binding(0) var<storage, read> vertex_buffer : VertexBuffer;
@group(2) @binding(0) var<uniform> screen_dims : ScreenUniform;
@group(3) @binding(0) var<uniform> camera : Camera;


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


fn project(v: vec4<f32>) -> vec2<f32> {

    var model_vertex_position = array<f32, 5>(v.x, v.y, v.z, v.w, 1.0);

    var view_transform = mat5x5_unpack(camera.view_transform);
    var model_transform = mat5x5_unpack(camera.model_transform);

    var pre_camera_transform = mat5x5_mul_vec5(model_transform, model_vertex_position);

    var view_vector = mat5x5_mul_vec5(view_transform, pre_camera_transform);

    // Perspective scaling
    let focal_length = 1.0;
    let near = 0.1;
    let far = 100.0;
    let projected_distance = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3]);

    var clip_position = vec2<f32>(
        view_vector[0] / projected_distance/focal_length,
        -view_vector[1] / projected_distance/focal_length
    );

    clip_position[0] = ((clip_position[0] + 1)/2)*screen_dims.width;
    clip_position[1] = ((clip_position[1] + 1)/2)*screen_dims.height;

    return clip_position.xy;
}

fn color_pixel(x: u32, y: u32, r: u32, g: u32, b: u32) {
  let pixelID = u32(x + y * u32(screen_dims.width)) * 3u;

  atomicMax(&color_buffer.values[pixelID + 0u], r);
  atomicMax(&color_buffer.values[pixelID + 1u], g);
  atomicMax(&color_buffer.values[pixelID + 2u], b);
}

fn draw_line(v1: vec2<f32>, v2: vec2<f32>, r: u32, g: u32, b: u32) {
  let dist = i32(distance(v1.xy, v2.xy));
  for (var i = 0; i < dist; i = i + 1) {
    let x = v1.x + (v2.x - v1.x) * (f32(i) / f32(dist));
    let y = v1.y + (v2.y - v1.y) * (f32(i) / f32(dist));
    // color_pixel(u32(x), u32(y), vec3<f32>(1.0, 1.0, 1.0));
    color_pixel(u32(x), u32(y), r, g, b);
  }
}

// From: https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling
fn barycentric(v1: vec3<f32>, v2: vec3<f32>, v3: vec3<f32>, p: vec2<f32>) -> vec3<f32> {
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
// move it inside the color pix function
fn is_off_screen(v: vec2<f32>) -> bool {
    if (v.x < 0.0 || v.x > screen_dims.width || v.y < 0.0 ||
      v.y > screen_dims.height) {
        return true;
    }

    return false;
}

@compute @workgroup_size(256, 1)
fn raster(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x * 3u;

    let v1 = project(vertex_buffer.values[index + 0u]);
    let v2 = project(vertex_buffer.values[index + 1u]);
    let v3 = project(vertex_buffer.values[index + 2u]);

    if (is_off_screen(v1) || is_off_screen(v2) || is_off_screen(v3)) {
       return;
    }

    //color_pixel(u32(screen_dims.width*1/4), u32(screen_dims.height/2), 255u, 0u, 0u);
    //color_pixel(u32(v2.x), u32(v2.y), 255u, 0u, 0u);
    //color_pixel(u32(v3.x), u32(v3.y), 255u, 0u, 0u);

    //draw_line(vec2<f32>(0, screen_dims.height/2), vec2<f32>(screen_dims.width, screen_dims.height/2), 0u, 255u, 0u);
    //draw_line(vec2<f32>(screen_dims.width/2, 0), vec2<f32>(screen_dims.width/2, screen_dims.height), 0u, 0u, 255u);

    draw_line(v1, v2, 0u, 0u, 255u);
    draw_line(v1, v3, 0u, 0u, 255u);
    draw_line(v2, v3, 0u, 0u, 255u);

    //draw_triangle(v1, v2, v3);
}

@compute @workgroup_size(256, 1)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x * 3u;

    atomicStore(&color_buffer.values[index + 0u], 0u);
    atomicStore(&color_buffer.values[index + 1u], 0u);
    atomicStore(&color_buffer.values[index + 2u], 0u);
}
