struct Pixel {
    r: u32,
    g: u32,
    b: u32,
}

fn pixel_to_vec(p: Pixel) -> vec3<f32> {
    return vec3<f32>(f32(p.r), f32(p.g), f32(p.b)) / (256.0);
}

struct ColorBuffer {
    value: array<Pixel>,
}

struct ScreenUniform {
    window_width: f32,
    window_height: f32,
    render_width: f32,
    render_height: f32,
    depth_factor: u32
}

@group(0) @binding(0) var<storage, read> color_buffer: ColorBuffer;
@group(1) @binding(0) var<uniform> screen_dims : ScreenUniform;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main_quad(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0));

    let out = VertexOutput(vec4<f32>(pos[vertex_idx], 0.0, 1.0));
    return out;
}

@vertex
fn vs_main_trig(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    let uv = vec2<u32>((vertex_idx << 1u) & 2u, vertex_idx & 2u);
    let out = VertexOutput(vec4<f32>(1.0 - 2.0 * vec2<f32>(uv), 0.0, 1.0));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normalized_x = in.pos.x/screen_dims.window_width;
    let normalized_y = in.pos.y/screen_dims.window_height;
    let render_x = normalized_x*screen_dims.render_width;
    let render_y = normalized_y*screen_dims.render_height;
    let x = floor(render_x);
    let y = floor(render_y);
    let index = u32(x + y * screen_dims.render_width);
    let p = color_buffer.value[index];

    let pixel = pixel_to_vec(p);

    let col = vec4<f32>(pixel, 1.0);
    return col;
}