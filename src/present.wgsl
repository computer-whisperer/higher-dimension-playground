

fn pixel_to_vec(p: u32) -> vec3<f32> {
    return vec3<f32>(f32((p>>16)&0xFF), f32((p>>8)&0xFF), f32((p>>0)&0xFF)) / (256.0);
}


struct RenderMetadata {
    window_width: u32,
    window_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32
}

@group(0) @binding(0) var<storage, read> color_buffer: array<u32>;
@group(1) @binding(0) var<uniform> render_metadata : RenderMetadata;

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
    let normalized_x = in.pos.x/f32(render_metadata.window_width);
    let normalized_y = in.pos.y/f32(render_metadata.window_height);
    let render_x = normalized_x*f32(render_metadata.render_width);
    let render_y = normalized_y*f32(render_metadata.render_height);
    let x = u32(floor(render_x));
    let y = u32(floor(render_y));
    var accumulated_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < render_metadata.depth_factor; i += 1u)
    {
        let index = (x + y * render_metadata.render_width)*render_metadata.depth_factor + i;
        let p = color_buffer[index];


        let pixel = pixel_to_vec(p);

        accumulated_color += vec4<f32>(pixel/f32(render_metadata.depth_factor), 0.0);
    }
    return accumulated_color;
}