struct RenderMetadata {
    window_width: u32,
    window_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32
}

@group(0) @binding(0) var<storage, read_write> color_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> depth_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> overlay_buffer : array<u32>;
@group(0) @binding(3) var<uniform> render_metadata : RenderMetadata;

@compute @workgroup_size(256, 1)
fn raster_clear_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    overlay_buffer[global_id.x] = 0x00000000u;
/*
    for (var i: u32 = 0u; i <= render_metadata.depth_factor; i = i + 1u)
    {
        color_buffer[global_id.x*render_metadata.depth_factor + i] = 0u;
        depth_buffer[global_id.x*render_metadata.depth_factor + i] = 0xFFFFFFFFu;
    }*/
}
