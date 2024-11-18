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

@group(0) @binding(0) var<storage, read_write> color_buffer : array<u32>;
@group(2) @binding(0) var<storage, read_write> intermediate_vertex_buffer : array<IntermediateVertex>;
@group(0) @binding(2) var<storage, read_write> overlay_buffer : array<atomic<u32>>;
@group(0) @binding(3) var<uniform> render_metadata : RenderMetadata;


// Define region codes for Cohen-Sutherland
const INSIDE = 0;    // 0000
const LEFT = 1;      // 0001
const RIGHT = 2;     // 0010
const BOTTOM = 4;    // 0100
const TOP = 8;       // 1000

fn compute_code(p: vec2<f32>, width: f32, height: f32) -> i32 {
    var code = INSIDE;

    if (p.x < 0.0) {
        code = code | LEFT;
    } else if (p.x > width) {
        code = code | RIGHT;
    }

    if (p.y < 0.0) {
        code = code | BOTTOM;
    } else if (p.y > height) {
        code = code | TOP;
    }

    return code;
}

fn clip_line(p1: ptr<function, vec2<f32>>, p2: ptr<function, vec2<f32>>, width: f32, height: f32) -> bool {
    var x1 = (*p1).x;
    var y1 = (*p1).y;
    var x2 = (*p2).x;
    var y2 = (*p2).y;

    var code1 = compute_code(vec2<f32>(x1, y1), width, height);
    var code2 = compute_code(vec2<f32>(x2, y2), width, height);
    var accept = false;

    loop {
        if ((code1 | code2) == 0) {
            // Both points inside window
            accept = true;
            break;
        } else if ((code1 & code2) != 0) {
            // Both points outside window
            break;
        } else {
            // Line needs clipping
            var code_out = code1;
            if (code1 == INSIDE) {
                code_out = code2;
            }

            var x = 0.0;
            var y = 0.0;

            if ((code_out & TOP) != 0) {
                x = x1 + (x2 - x1) * (height - y1) / (y2 - y1);
                y = height;
            } else if ((code_out & BOTTOM) != 0) {
                x = x1 + (x2 - x1) * (0.0 - y1) / (y2 - y1);
                y = 0.0;
            } else if ((code_out & RIGHT) != 0) {
                y = y1 + (y2 - y1) * (width - x1) / (x2 - x1);
                x = width;
            } else if ((code_out & LEFT) != 0) {
                y = y1 + (y2 - y1) * (0.0 - x1) / (x2 - x1);
                x = 0.0;
            }

            if (code_out == code1) {
                x1 = x;
                y1 = y;
                code1 = compute_code(vec2<f32>(x1, y1), width, height);
            } else {
                x2 = x;
                y2 = y;
                code2 = compute_code(vec2<f32>(x2, y2), width, height);
            }
        }
    }

    if (accept) {
        (*p1).x = x1;
        (*p1).y = y1;
        (*p2).x = x2;
        (*p2).y = y2;
    }

    return accept;
}

fn write_overlay_pixel(pos: vec2<i32>, color: vec3<f32>) {
    if (pos.x < 0 || pos.x > i32(render_metadata.render_width) || pos.y < 0 || pos.y > i32(render_metadata.render_height))
    {
        return;
    }
    let color_u = color*255.0;
    let color_24 = (((0xFFu) << 24u) | (u32(color_u.r) & 0xFFu) << 16u) | ((u32(color_u.g) & 0xFFu) << 8u) | (u32(color_u.b) & 0xFFu);
    let idx = u32(pos.x) + u32(pos.y)*render_metadata.render_width;
    overlay_buffer[idx] = color_24;
}

const SLICE_X: u32 = 32;
const WORKGROUP_SIZE: u32 = SLICE_X;

fn draw_4d_line(v1: vec4<f32>, v2: vec4<f32>, color: vec3<f32>, local_id: u32) {
    let dist = i32(distance(v1.xy, v2.xy));

    for (var i = i32(local_id); i < dist; i = i + i32(SLICE_X)) {
        let pos = v1.xy + (v2.xy - v1.xy) * (f32(i) / f32(dist));
        if (pos.x > 0.0 && pos.x < f32(render_metadata.render_width) && pos.y > 0.0 && pos.y < f32(render_metadata.render_height))
        {
            write_overlay_pixel(vec2<i32>(i32(pos.x), i32(pos.y)), color);
        }
    }
}

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


@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn raster_lines_main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tet_id = workgroup_id.x;
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
        return;
    }


    // Scale to screen
    v0.pos = scale_to_screen(v0.pos);
    v1.pos = scale_to_screen(v1.pos);
    v2.pos = scale_to_screen(v2.pos);
    v3.pos = scale_to_screen(v3.pos);

    // Render lines
    let color = vec3<f32>(0.0, 0.0, 1.0);
    draw_4d_line(v0.pos, v1.pos, color, local_id.x);
    draw_4d_line(v0.pos, v2.pos, color, local_id.x);
    draw_4d_line(v0.pos, v3.pos, color, local_id.x);
    draw_4d_line(v1.pos, v2.pos, color, local_id.x);
    draw_4d_line(v1.pos, v3.pos, color, local_id.x);
    draw_4d_line(v2.pos, v3.pos, color, local_id.x);
}