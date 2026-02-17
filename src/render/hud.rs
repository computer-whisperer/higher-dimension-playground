use ab_glyph::{point, Font, FontArc, ScaleFont};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub(super) struct LineVertex {
    position: Vec2,
    // padding[0] is used as an optional line style id for present shader.
    _padding: [f32; 2],
    color: Vec4,
}

impl LineVertex {
    pub(super) fn new_with_style(position: Vec2, color: Vec4, style: f32) -> Self {
        Self {
            position,
            _padding: [style, 0.0],
            color,
        }
    }
}

#[derive(Copy, Clone)]
pub(super) struct OverlayLine {
    pub(super) start: Vec2,
    pub(super) end: Vec2,
    pub(super) color: Vec4,
    pub(super) style: f32,
}

pub(super) const HUD_VERTEX_CAPACITY: usize = 64_000;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub(super) struct HudVertex {
    position: Vec2,
    texcoord: Vec2,
    color: Vec4,
}

impl HudVertex {
    pub(super) fn new(position: Vec2, texcoord: Vec2, color: Vec4) -> Self {
        Self {
            position,
            texcoord,
            color,
        }
    }
}

#[derive(Clone)]
struct GlyphInfo {
    uv_min: Vec2,
    uv_max: Vec2,
    size_px: Vec2,
    bearing_px: Vec2,
    advance_px: f32,
}

pub(super) struct FontAtlas {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u8>,
    glyphs: [Option<GlyphInfo>; 128],
    white_uv: Vec2,
    line_height: f32,
    ascent: f32,
}

pub(super) fn build_font_atlas(font: &FontArc, pixel_size: f32) -> FontAtlas {
    let atlas_w: u32 = 512;
    let atlas_h: u32 = 256;
    let pixels = vec![0u8; (atlas_w * atlas_h * 4) as usize];
    let glyphs: [Option<GlyphInfo>; 128] = std::array::from_fn(|_| None);
    let mut atlas = FontAtlas {
        width: atlas_w,
        height: atlas_h,
        pixels,
        glyphs,
        white_uv: Vec2::ZERO,
        line_height: 0.0,
        ascent: 0.0,
    };

    let scaled = font.as_scaled(pixel_size);
    atlas.line_height = scaled.height() + scaled.line_gap();
    atlas.ascent = scaled.ascent();

    // Shelf packing
    let mut cursor_x: u32 = 0;
    let mut cursor_y: u32 = 0;
    let mut row_height: u32 = 0;
    let pad = 2u32; // padding between glyphs

    for ch in 32u8..=126 {
        let glyph_id = scaled.glyph_id(ch as char);
        let glyph = glyph_id.with_scale_and_position(pixel_size, point(0.0, scaled.ascent()));
        let advance = scaled.h_advance(glyph_id);

        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            let gw = bounds.width().ceil() as u32;
            let gh = bounds.height().ceil() as u32;

            if gw == 0 || gh == 0 {
                atlas.glyphs[ch as usize] = Some(GlyphInfo {
                    uv_min: Vec2::ZERO,
                    uv_max: Vec2::ZERO,
                    size_px: Vec2::ZERO,
                    bearing_px: Vec2::new(bounds.min.x, bounds.min.y - scaled.ascent()),
                    advance_px: advance,
                });
                continue;
            }

            // Advance to next row if needed
            if cursor_x + gw + pad > atlas_w {
                cursor_x = 0;
                cursor_y += row_height + pad;
                row_height = 0;
            }
            if cursor_y + gh + pad > atlas_h {
                // Out of atlas space, skip remaining glyphs
                break;
            }

            // Rasterize glyph into atlas (RGBA: white with varying alpha)
            outline.draw(|x, y, c| {
                let px = cursor_x + x;
                let py = cursor_y + y;
                if px < atlas_w && py < atlas_h {
                    let idx = ((py * atlas_w + px) * 4) as usize;
                    let val = (c * 255.0).clamp(0.0, 255.0) as u8;
                    if val > atlas.pixels[idx + 3] {
                        atlas.pixels[idx] = 255;
                        atlas.pixels[idx + 1] = 255;
                        atlas.pixels[idx + 2] = 255;
                        atlas.pixels[idx + 3] = val;
                    }
                }
            });

            atlas.glyphs[ch as usize] = Some(GlyphInfo {
                uv_min: Vec2::new(
                    cursor_x as f32 / atlas_w as f32,
                    cursor_y as f32 / atlas_h as f32,
                ),
                uv_max: Vec2::new(
                    (cursor_x + gw) as f32 / atlas_w as f32,
                    (cursor_y + gh) as f32 / atlas_h as f32,
                ),
                size_px: Vec2::new(gw as f32, gh as f32),
                bearing_px: Vec2::new(bounds.min.x, bounds.min.y - scaled.ascent()),
                advance_px: advance,
            });

            cursor_x += gw + pad;
            row_height = row_height.max(gh);
        } else {
            // No outline (e.g. space character)
            atlas.glyphs[ch as usize] = Some(GlyphInfo {
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ZERO,
                size_px: Vec2::ZERO,
                bearing_px: Vec2::ZERO,
                advance_px: advance,
            });
        }
    }

    // Reserve a 4x4 white block for solid rectangles (RGBA: fully opaque white)
    if cursor_x + 4 + pad > atlas_w {
        cursor_x = 0;
        cursor_y += row_height + pad;
    }
    for dy in 0..4u32 {
        for dx in 0..4u32 {
            let px = cursor_x + dx;
            let py = cursor_y + dy;
            if px < atlas_w && py < atlas_h {
                let idx = ((py * atlas_w + px) * 4) as usize;
                atlas.pixels[idx] = 255;
                atlas.pixels[idx + 1] = 255;
                atlas.pixels[idx + 2] = 255;
                atlas.pixels[idx + 3] = 255;
            }
        }
    }
    atlas.white_uv = Vec2::new(
        (cursor_x as f32 + 2.0) / atlas_w as f32,
        (cursor_y as f32 + 2.0) / atlas_h as f32,
    );

    atlas
}

pub(super) struct HudResources {
    pub(super) font_atlas: FontAtlas,
    pub(super) atlas_view: Arc<ImageView>,
    pub(super) atlas_sampler: Arc<Sampler>,
    #[allow(dead_code)]
    pub(super) hud_descriptor_set_layout: Arc<DescriptorSetLayout>,
}

impl HudResources {
    #[allow(dead_code)]
    pub(super) fn create_per_frame_hud(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    ) -> (Subbuffer<[HudVertex]>, Arc<DescriptorSet>) {
        let hud_vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![HudVertex::zeroed(); HUD_VERTEX_CAPACITY],
        )
        .unwrap();

        let hud_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            self.hud_descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, hud_vertex_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.atlas_view.clone(),
                    self.atlas_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        (hud_vertex_buffer, hud_descriptor_set)
    }
}

pub(super) fn push_text_quads(
    quads: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    text: &str,
    top_left_px: Vec2,
    pixel_size: f32,
    color: Vec4,
    present_size: [u32; 2],
) {
    let scale = pixel_size / 32.0; // atlas was rasterized at 32px
    let line_advance = atlas.line_height * scale;
    let mut caret_x = top_left_px.x;
    // ndc_to_pixels produces Y-up coords (pixel 0 = screen bottom, pixel h = screen top).
    // Baseline sits below the top of the text box.
    let mut baseline_y = top_left_px.y - atlas.ascent * scale;

    for ch in text.chars() {
        if ch == '\n' {
            caret_x = top_left_px.x;
            baseline_y -= line_advance; // next line goes down in Y-up
            continue;
        }

        let idx = ch as usize;
        if idx >= 128 {
            continue;
        }
        let glyph = match atlas.glyphs[idx].as_ref() {
            Some(g) => g,
            None => continue,
        };

        if glyph.size_px.x > 0.0 && glyph.size_px.y > 0.0 {
            let gw = glyph.size_px.x * scale;
            let gh = glyph.size_px.y * scale;
            let x0 = caret_x + glyph.bearing_px.x * scale;
            // bearing_px.y is Y-down offset from baseline to glyph top (negative).
            // In Y-up, negate it to get glyph top, then subtract gh for glyph bottom.
            let y0 = baseline_y - glyph.bearing_px.y * scale - gh;

            let p_min = pixels_to_ndc(Vec2::new(x0, y0), present_size);
            let p_max = pixels_to_ndc(Vec2::new(x0 + gw, y0 + gh), present_size);

            let uv_min = glyph.uv_min;
            let uv_max = glyph.uv_max;

            // Two triangles forming a quad
            // Note: pixels_to_ndc flips Y (pixel Y=0 → NDC Y=+1 = Vulkan bottom),
            // so p_min.y (larger NDC) is the screen bottom and p_max.y (smaller NDC)
            // is the screen top. We must swap UV Y to compensate: atlas top (uv_min.y)
            // maps to screen top (p_max.y), atlas bottom (uv_max.y) maps to screen
            // bottom (p_min.y).
            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_min.y),
                texcoord: Vec2::new(uv_min.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_min.y),
                texcoord: Vec2::new(uv_max.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_max.y),
                texcoord: Vec2::new(uv_max.x, uv_min.y),
                color,
            });

            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_min.y),
                texcoord: Vec2::new(uv_min.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_max.y),
                texcoord: Vec2::new(uv_max.x, uv_min.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_max.y),
                texcoord: Vec2::new(uv_min.x, uv_min.y),
                color,
            });
        }

        caret_x += glyph.advance_px * scale;
    }
}

pub(super) fn text_width_px(atlas: &FontAtlas, text: &str, pixel_size: f32) -> f32 {
    let scale = pixel_size / 32.0;
    let mut width = 0.0f32;
    for ch in text.chars() {
        if ch == '\n' {
            break;
        }
        let idx = ch as usize;
        if idx >= 128 {
            continue;
        }
        if let Some(glyph) = atlas.glyphs[idx].as_ref() {
            width += glyph.advance_px * scale;
        }
    }
    width
}

pub(super) fn push_filled_rect_quads(
    quads: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    min_ndc: Vec2,
    max_ndc: Vec2,
    color: Vec4,
) {
    let uv = atlas.white_uv;
    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });

    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });
}

pub(super) fn push_line(lines: &mut Vec<OverlayLine>, a: Vec2, b: Vec2, color: Vec4) {
    push_line_with_style(lines, a, b, color, 0.0);
}

fn push_line_with_style(lines: &mut Vec<OverlayLine>, a: Vec2, b: Vec2, color: Vec4, style: f32) {
    lines.push(OverlayLine {
        start: a,
        end: b,
        color,
        style,
    });
}

pub(super) fn push_rect(lines: &mut Vec<OverlayLine>, min: Vec2, max: Vec2, color: Vec4) {
    let a = Vec2::new(min.x, min.y);
    let b = Vec2::new(max.x, min.y);
    let c = Vec2::new(max.x, max.y);
    let d = Vec2::new(min.x, max.y);
    push_line(lines, a, b, color);
    push_line(lines, b, c, color);
    push_line(lines, c, d, color);
    push_line(lines, d, a, color);
}

pub(super) fn push_cross(lines: &mut Vec<OverlayLine>, center: Vec2, radius: Vec2, color: Vec4) {
    push_line(
        lines,
        center + Vec2::new(-radius.x, 0.0),
        center + Vec2::new(radius.x, 0.0),
        color,
    );
    push_line(
        lines,
        center + Vec2::new(0.0, -radius.y),
        center + Vec2::new(0.0, radius.y),
        color,
    );
}

pub(super) fn push_minecraft_crosshair(
    lines: &mut Vec<OverlayLine>,
    present_size: [u32; 2],
    center_ndc: Vec2,
    dpi_scale: f32,
    color: Vec4,
    outline_color: Vec4,
) {
    // Crosshair dimensions are authored in logical pixels and scaled by the
    // window DPI factor so it remains consistent on HiDPI Wayland displays.
    const INNER_GAP_PX: f32 = 5.0;
    const ARM_LENGTH_PX: f32 = 12.0;
    const OUTLINE_PAD_PX: f32 = 2.0;
    const INNER_THICKNESS_PX: f32 = 3.0;
    const OUTLINE_THICKNESS_PX: f32 = 5.0;

    let inv_w = 2.0 / present_size[0].max(1) as f32;
    let inv_h = 2.0 / present_size[1].max(1) as f32;
    let px_scale = dpi_scale.max(1.0);

    let inner_x = INNER_GAP_PX * px_scale * inv_w;
    let inner_y = INNER_GAP_PX * px_scale * inv_h;
    let arm_x = ARM_LENGTH_PX * px_scale * inv_w;
    let arm_y = ARM_LENGTH_PX * px_scale * inv_h;
    let pad_x = OUTLINE_PAD_PX * px_scale * inv_w;
    let pad_y = OUTLINE_PAD_PX * px_scale * inv_h;

    let inner_half_steps = (((INNER_THICKNESS_PX * px_scale) - 1.0) * 0.5)
        .max(0.0)
        .round() as i32;
    let outline_half_steps = (((OUTLINE_THICKNESS_PX * px_scale) - 1.0) * 0.5)
        .max(0.0)
        .round() as i32;

    let push_hbar =
        |lines: &mut Vec<OverlayLine>, x0: f32, x1: f32, half_steps: i32, bar_color: Vec4| {
            for step in -half_steps..=half_steps {
                let y_offset = step as f32 * inv_h;
                push_line_with_style(
                    lines,
                    center_ndc + Vec2::new(x0, y_offset),
                    center_ndc + Vec2::new(x1, y_offset),
                    bar_color,
                    1.0,
                );
            }
        };

    let push_vbar =
        |lines: &mut Vec<OverlayLine>, y0: f32, y1: f32, half_steps: i32, bar_color: Vec4| {
            for step in -half_steps..=half_steps {
                let x_offset = step as f32 * inv_w;
                push_line_with_style(
                    lines,
                    center_ndc + Vec2::new(x_offset, y0),
                    center_ndc + Vec2::new(x_offset, y1),
                    bar_color,
                    1.0,
                );
            }
        };

    // Outline pass.
    push_hbar(
        lines,
        -(inner_x + arm_x + pad_x),
        -(inner_x - pad_x),
        outline_half_steps,
        outline_color,
    );
    push_hbar(
        lines,
        inner_x - pad_x,
        inner_x + arm_x + pad_x,
        outline_half_steps,
        outline_color,
    );
    push_vbar(
        lines,
        -(inner_y + arm_y + pad_y),
        -(inner_y - pad_y),
        outline_half_steps,
        outline_color,
    );
    push_vbar(
        lines,
        inner_y - pad_y,
        inner_y + arm_y + pad_y,
        outline_half_steps,
        outline_color,
    );

    // Inner pass.
    push_hbar(lines, -(inner_x + arm_x), -inner_x, inner_half_steps, color);
    push_hbar(lines, inner_x, inner_x + arm_x, inner_half_steps, color);
    push_vbar(lines, -(inner_y + arm_y), -inner_y, inner_half_steps, color);
    push_vbar(lines, inner_y, inner_y + arm_y, inner_half_steps, color);
}

pub(super) fn map_to_panel(center: Vec2, half_size: Vec2, map_range: f32, a: f32, b: f32) -> Vec2 {
    let x = (a / map_range).clamp(-1.0, 1.0) * half_size.x;
    let y = (-b / map_range).clamp(-1.0, 1.0) * half_size.y; // negate for Vulkan +Y=down
    center + Vec2::new(x, y)
}

pub(super) fn ndc_to_pixels(p: Vec2, present_size: [u32; 2]) -> Vec2 {
    let w = present_size[0] as f32;
    let h = present_size[1] as f32;
    Vec2::new((p.x * 0.5 + 0.5) * w, (1.0 - (p.y * 0.5 + 0.5)) * h)
}

pub(super) fn pixels_to_ndc(p: Vec2, present_size: [u32; 2]) -> Vec2 {
    let w = present_size[0] as f32;
    let h = present_size[1] as f32;
    Vec2::new((p.x / w) * 2.0 - 1.0, 1.0 - (p.y / h) * 2.0)
}

pub(super) fn load_hud_font() -> Option<FontArc> {
    const FONT_CANDIDATES: &[&str] = &[
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    ];

    if let Ok(custom_path) = std::env::var("HUD_FONT_PATH") {
        if let Ok(data) = std::fs::read(&custom_path) {
            if let Ok(font) = FontArc::try_from_vec(data) {
                return Some(font);
            }
        }
    }

    for &path in FONT_CANDIDATES {
        if let Ok(data) = std::fs::read(path) {
            if let Ok(font) = FontArc::try_from_vec(data) {
                return Some(font);
            }
        }
    }

    None
}

pub(super) fn push_text_lines(
    lines: &mut Vec<OverlayLine>,
    font: &FontArc,
    text: &str,
    top_left_px: Vec2,
    pixel_size: f32,
    color: Vec4,
    present_size: [u32; 2],
) {
    let scaled = font.as_scaled(pixel_size);
    let h = present_size[1] as f32;
    let mut caret_x = top_left_px.x;
    // top_left_px is in Y-up pixel space (from ndc_to_pixels), but ab_glyph uses Y-down.
    // Convert: Y-down baseline = (h - top_left_y_up) + ascent
    let mut baseline_y = (h - top_left_px.y) + scaled.ascent();
    let line_advance = (scaled.height() + scaled.line_gap()).max(pixel_size * 1.2);
    let mut prev_glyph = None;

    for ch in text.chars() {
        if ch == '\n' {
            caret_x = top_left_px.x;
            baseline_y += line_advance; // Y-down: next line increases y
            prev_glyph = None;
            continue;
        }

        let glyph_id = scaled.glyph_id(ch);
        if let Some(prev) = prev_glyph {
            caret_x += scaled.kern(prev, glyph_id);
        }

        let glyph = glyph_id.with_scale_and_position(pixel_size, point(caret_x, baseline_y));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            let width = bounds.width().max(0.0).ceil() as usize;
            let height = bounds.height().max(0.0).ceil() as usize;

            if width > 0 && height > 0 {
                let mut coverage = vec![0.0f32; width * height];
                outline.draw(|x, y, c| {
                    let idx = y as usize * width + x as usize;
                    if c > coverage[idx] {
                        coverage[idx] = c;
                    }
                });

                for y in 0..height {
                    let mut x = 0usize;
                    while x < width {
                        while x < width && coverage[y * width + x] < 0.35 {
                            x += 1;
                        }
                        if x >= width {
                            break;
                        }

                        let run_start = x;
                        while x < width && coverage[y * width + x] >= 0.35 {
                            x += 1;
                        }

                        // bounds are in Y-down pixels; convert to Y-up for pixels_to_ndc
                        let px_y_down = bounds.min.y + y as f32 + 0.5;
                        let px_y = h - px_y_down;
                        let px_start = Vec2::new(bounds.min.x + run_start as f32, px_y);
                        let px_end = Vec2::new(bounds.min.x + x as f32, px_y);
                        push_line(
                            lines,
                            pixels_to_ndc(px_start, present_size),
                            pixels_to_ndc(px_end, present_size),
                            color,
                        );
                    }
                }
            }
        }

        caret_x += scaled.h_advance(glyph_id);
        prev_glyph = Some(glyph_id);
    }
}
