

struct VertexInput {
    pos: vec4<f32>,
    tex_pos: vec3<f32>,
    cell: u32
}

struct VertexInputBuffer {
  values: array<VertexInput>,
}

struct IntermediateVertex {
    pos: vec4<f32>,
    tex_pos: vec3<f32>,
    tex_id: u32
}

struct RenderMetadata {
    window_width: u32,
    window_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32
}

struct Camera {
    view_transform: array<vec4<f32>, 8>
}

struct Instance {
    model_transform: array<vec4<f32>, 8>,
    cell_texture_ids: array<u32, 8>
}

@group(0) @binding(0) var<storage, read_write> color_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> depth_buffer : array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> vertex_input_buffer : VertexInputBuffer;

@group(2) @binding(0) var<storage, read_write> intermediate_vertex_buffer : array<IntermediateVertex>;
@group(2) @binding(1) var<storage, read> instance_buffer : array<Instance>;
@group(2) @binding(2) var<uniform> render_metadata : RenderMetadata;
@group(2) @binding(3) var<uniform> camera : Camera;




fn write_pixel(pos: vec2<i32>, i:u32, color: vec3<f32>) {
    if (pos.x < 0 || pos.x > i32(render_metadata.render_width) || pos.y < 0 || pos.y > i32(render_metadata.render_height))
    {
        return;
    }
    let color_u = color*255.0;
    let color_24 = ((u32(color_u.r) & 0xFFu) << 16u) | ((u32(color_u.g) & 0xFFu) << 8u) | (u32(color_u.b) & 0xFFu);
    let idx = (u32(pos.x) + u32(pos.y)*render_metadata.render_width)*render_metadata.depth_factor + i;
    color_buffer[idx] = color_24;
}

/*
fn color_pixel(x: u32, y: u32, r: u32, g: u32, b: u32) {
  let pixelID = u32(x + y * u32(render_metadata.render_width)) * 3u;

  atomicMax(&color_buffer.values[pixelID + 0u], r);
  atomicMax(&color_buffer.values[pixelID + 1u], g);
  atomicMax(&color_buffer.values[pixelID + 2u], b);
}*/

fn transform_to_screen_coords(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        ((v[0] + 1)/2)*f32(render_metadata.render_width),
        ((v[1] + 1)/2)*f32(render_metadata.render_height)
    );
};

fn transform_to_screen_coords_4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        ((v[0] + 1)/2)*f32(render_metadata.render_width),
        ((v[1] + 1)/2)*f32(render_metadata.render_height),
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
        if (x > 0 && x < f32(render_metadata.render_width) && y > 0 && y < f32(render_metadata.render_height))
        {
            for (var j = 0u; j < render_metadata.depth_factor; j += 1u) {
                write_pixel(vec2<i32>(i32(x), i32(y)), j, color);
            }
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
    let color_array = array<vec3<f32>, 12>(
        vec3<f32>(1.0, 0.0, 0.0), //0
        vec3<f32>(1.0, 0.0, 0.0), //1
        vec3<f32>(1.0, 0.8, 0.0), //2
        vec3<f32>(0.5, 1.0, 0.0), //3
        vec3<f32>(0.0, 1.0, 0.2), //4
        vec3<f32>(0.0, 1.0, 1.0), //5
        vec3<f32>(0.0, 0.2, 1.0), //6
        vec3<f32>(0.5, 0.0, 1.0), //7
        vec3<f32>(1.0, 0.0, 0.8), //8
        (texture_pos+vec3<f32>(1.0, 1.0, 1.0))/2.0,  // 9
        vec3<f32>(139.0, 69.0, 19.8)/256.0, // 10
        vec3<f32>(34.0, 139.0, 34.0)/256.0, // 11
        vec3<f32>(0.0, 0.0, 1.0), // 12
    );
    return color_array[texture_id%12];
}

struct FragmentVertex {
    pos: vec4<f32>,
    texture_pos: vec3<f32>
}

fn do_zw_raycast(v0: vec2<f32>, v1: vec2<f32>, i: u32) -> vec2<f32>
{
    //https://gamedev.stackexchange.com/questions/116422/best-way-to-find-line-segment-intersection
    let pi = 3.14159;
    let angle_ratio = ((f32(i)/f32(render_metadata.depth_factor)) - 0.5);
    let theta = angle_ratio*(pi/1.0) + (pi/4.0);
    //let theta = angle_ratio*(pi/2.0) + (pi/4.0);
    //let theta = pi/4.0;
    let sample_ray_y = sin(theta);
    let sample_ray_x = cos(theta);
    let r = ((-v0.y)*(v1.x - v0.x) - (-v0.x)*(v1.y - v0.y)) / ((sample_ray_x)*(v1.y - v0.y) - (sample_ray_y)*(v1.x - v0.x));
    let s = ((-v0.y)*(sample_ray_x) - (-v0.x)*(sample_ray_y))/((sample_ray_x)*(v1.y - v0.y) - (sample_ray_y)*(v1.x - v0.x));
    return vec2<f32>(r, s);
}

const DEPTH_DIVISOR : f32 = 1000.0;

fn write_depth(pos: vec2<i32>, i: u32, depth: u32) {
    let depth_index = (u32(pos.x) + u32(pos.y) * render_metadata.render_width) * render_metadata.depth_factor + i;
    atomicMin(&depth_buffer[depth_index], depth);
}

fn depth_zw_line(vs: vec2<i32>, v0: vec4<f32>, v1: vec4<f32>){
    for (var i = 0u; i < render_metadata.depth_factor; i += 1u) {
        let res = do_zw_raycast(v0.zw, v1.zw, i);
        if (res.y >= 0 && res.y <= 1) {
            write_depth(vs, i, u32(res.x*DEPTH_DIVISOR));
        }
    }
}

fn render_zw_line(vs: vec2<i32>, v0: FragmentVertex, v1: FragmentVertex, texture_id: u32){
    // 4-d appropriate volumetric shading
    var color_added = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < render_metadata.depth_factor; i += 1u) {
        let res = do_zw_raycast(v0.pos.zw, v1.pos.zw, i);
        if (res.y > 0 && res.y < 1 && res.x > 0) {
            let depth_index = (u32(vs.x) + u32(vs.y) * render_metadata.render_width) * render_metadata.depth_factor + i;
            let depth_value = u32(res.x*DEPTH_DIVISOR);
            if depth_value == atomicLoad(&depth_buffer[depth_index]) {
                let intensity = 1.0;
                let texture_pos = v0.texture_pos*(1.0 - res.y) + v1.texture_pos*res.y;
                let color = sample_texture(texture_id, texture_pos);
                write_pixel(vs, i, color*intensity);
            }
        }
    }
}

const SLICE_COUNT_X: u32 = 128u;
const SLICE_COUNT_Y: u32 = 1u;
const SLICE_COUNT_TOTAL: u32 = SLICE_COUNT_X*SLICE_COUNT_Y;


@compute @workgroup_size(SLICE_COUNT_TOTAL, 1)
fn raster_depth_shader(@builtin(global_invocation_id) global_id: vec3<u32>){
    let tet_index = global_id.x/(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index = global_id.x%(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index_x = slice_index%SLICE_COUNT_X;
    let slice_index_y = slice_index/SLICE_COUNT_X;
    let v0 = transform_to_screen_coords_4(intermediate_vertex_buffer[tet_index*4u + 0u].pos);
    let v1 = transform_to_screen_coords_4(intermediate_vertex_buffer[tet_index*4u + 1u].pos);
    let v2 = transform_to_screen_coords_4(intermediate_vertex_buffer[tet_index*4u + 2u].pos);
    let v3 = transform_to_screen_coords_4(intermediate_vertex_buffer[tet_index*4u + 3u].pos);

    let min_xy = vec2<i32>(
        i32(max(min(min(min(v0.x, v1.x), v2.x), v3.x), 0.0)),
        i32(max(min(min(min(v0.y, v1.y), v2.y), v3.y), 0.0))
    );
    let max_xy = vec2<i32>(
        i32(min(max(max(max(v0.x, v1.x), v2.x), v3.x), f32(render_metadata.render_width))),
        i32(min(max(max(max(v0.y, v1.y), v2.y), v3.y), f32(render_metadata.render_height)))
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
            //color_pixel(u32(screen_v.x), u32(screen_v.y), 0u, 0u, 255u);
        }
    }
}


@compute @workgroup_size(SLICE_COUNT_TOTAL, 1)
fn raster_pixel_shader(@builtin(global_invocation_id) global_id: vec3<u32>){
    let tet_index = global_id.x/(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index = global_id.x%(SLICE_COUNT_X*SLICE_COUNT_Y);
    let slice_index_x = slice_index%SLICE_COUNT_X;
    let slice_index_y = slice_index/SLICE_COUNT_X;
    let v0 = FragmentVertex(intermediate_vertex_buffer[tet_index*4u + 0u].pos, intermediate_vertex_buffer[tet_index*4u + 0u].tex_pos);
    let v1 = FragmentVertex(intermediate_vertex_buffer[tet_index*4u + 1u].pos, intermediate_vertex_buffer[tet_index*4u + 1u].tex_pos);
    let v2 = FragmentVertex(intermediate_vertex_buffer[tet_index*4u + 2u].pos, intermediate_vertex_buffer[tet_index*4u + 2u].tex_pos);
    let v3 = FragmentVertex(intermediate_vertex_buffer[tet_index*4u + 3u].pos, intermediate_vertex_buffer[tet_index*4u + 3u].tex_pos);
    let texture_id = intermediate_vertex_buffer[tet_index*4u + 0u].tex_id;
    let color = sample_texture(texture_id, vec3<f32>(0.0, 0.0, 0.0));

    if (true)
    {
        draw_4d_line(v0.pos, v1.pos, vec3<f32>(0.0, 0.0, 1.0));
        draw_4d_line(v0.pos, v2.pos, vec3<f32>(0.0, 0.0, 1.0));
        draw_4d_line(v0.pos, v3.pos, vec3<f32>(0.0, 0.0, 1.0));
        draw_4d_line(v1.pos, v2.pos, vec3<f32>(0.0, 0.0, 1.0));
        draw_4d_line(v1.pos, v3.pos, vec3<f32>(0.0, 0.0, 1.0));
        draw_4d_line(v2.pos, v3.pos, vec3<f32>(0.0, 0.0, 1.0));
    }
    if (false)
    {
        let v0_s = FragmentVertex(transform_to_screen_coords_4(v0.pos), v0.texture_pos);
        let v1_s = FragmentVertex(transform_to_screen_coords_4(v1.pos), v1.texture_pos);
        let v2_s = FragmentVertex(transform_to_screen_coords_4(v2.pos), v2.texture_pos);
        let v3_s = FragmentVertex(transform_to_screen_coords_4(v3.pos), v3.texture_pos);
        let min_xy = vec2<i32>(
            i32(max(min(min(min(v0_s.pos.x, v1_s.pos.x), v2_s.pos.x), v3_s.pos.x), 0.0)),
            i32(max(min(min(min(v0_s.pos.y, v1_s.pos.y), v2_s.pos.y), v3_s.pos.y), 0.0))
        );
        let max_xy = vec2<i32>(
            i32(min(max(max(max(v0_s.pos.x, v1_s.pos.x), v2_s.pos.x), v3_s.pos.x), f32(render_metadata.render_width))),
            i32(min(max(max(max(v0_s.pos.y, v1_s.pos.y), v2_s.pos.y), v3_s.pos.y), f32(render_metadata.render_height)))
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
}

@compute @workgroup_size(256, 1)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    for (var i: u32 = 0u; i <= render_metadata.depth_factor; i = i + 1u)
    {
        color_buffer[global_id.x*render_metadata.depth_factor + i] = 0u;
        atomicStore(&depth_buffer[global_id.x*render_metadata.depth_factor + i], 0xFFFFFFFFu);
    }
}
