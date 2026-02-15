use super::*;

impl RenderContext {
    pub fn save_rendered_frame(&mut self, path: &str) {
        self.wait_for_all_frames();

        let result = self.sized_buffers.output_cpu_pixel_buffer.read();
        match result {
            Ok(buffer_content) => {
                let mut buffer_stored = Vec::new();
                buffer_stored.extend_from_slice(&buffer_content[..]);
                let buffer_arc = Arc::new(buffer_stored);
                let mut buffer_dimensions = self.sized_buffers.render_dimensions;
                let logical_depth = buffer_dimensions[2].max(1);
                let storage_depth = self.sized_buffers.pixel_storage_layers.max(1);
                buffer_dimensions[2] = if self.last_backend == RenderBackend::VoxelTraversal {
                    1
                } else {
                    logical_depth.min(storage_depth)
                };
                let mut layers = Vec::new();
                struct HereGetPixel {
                    buffer_arc: Arc<Vec<Vec4>>,
                    dimensions: [u32; 3],
                    z: Option<u32>,
                }
                impl exr::prelude::GetPixel for HereGetPixel {
                    type Pixel = (f32, f32, f32, f32);

                    fn get_pixel(&self, position: exr::math::Vec2<usize>) -> Self::Pixel {
                        // Get cumulative pixel value
                        let mut full_pixel = Vec4::ZERO;
                        for z in 0..self.dimensions[2] {
                            full_pixel += self.buffer_arc[position.x()
                                + position.y() * self.dimensions[0] as usize
                                + (z * self.dimensions[0] * self.dimensions[1]) as usize];
                        }
                        if let Some(z) = self.z {
                            let local_pixel = self.buffer_arc[position.x()
                                + position.y() * self.dimensions[0] as usize
                                + (z * self.dimensions[0] * self.dimensions[1]) as usize];
                            (
                                local_pixel.x / local_pixel.w,
                                local_pixel.y / local_pixel.w,
                                local_pixel.z / local_pixel.w,
                                1.0,
                            )
                        } else {
                            (
                                full_pixel.x / full_pixel.w,
                                full_pixel.y / full_pixel.w,
                                full_pixel.z / full_pixel.w,
                                1.0,
                            )
                        }
                    }
                }
                // Render the full thing in layer 0 (because apparently support for openexr is rather poor)
                let pixel_getter = HereGetPixel {
                    buffer_arc: buffer_arc.clone(),
                    dimensions: buffer_dimensions,
                    z: None,
                };
                layers.push(exr::prelude::Layer::new(
                    (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                    exr::prelude::LayerAttributes {
                        layer_name: Some(exr::prelude::Text::new_or_panic(format!("Full Render"))),
                        ..Default::default()
                    },
                    exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                    exr::prelude::SpecificChannels::rgba(pixel_getter),
                ));
                for z in 0..buffer_dimensions[2] {
                    let pixel_getter = HereGetPixel {
                        buffer_arc: buffer_arc.clone(),
                        dimensions: buffer_dimensions,
                        z: Some(z),
                    };
                    let layer = exr::prelude::Layer::new(
                        (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                        exr::prelude::LayerAttributes {
                            layer_name: Some(exr::prelude::Text::new_or_panic(format!(
                                "ZW Slice {}/{}",
                                z, buffer_dimensions[2]
                            ))),
                            ..Default::default()
                        },
                        exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                        exr::prelude::SpecificChannels::rgba(pixel_getter),
                    );
                    layers.push(layer);
                }

                let image = exr::image::Image::from_layers(
                    exr::prelude::ImageAttributes {
                        display_window: exr::prelude::IntegerBounds::new(
                            (0, 0),
                            (
                                self.sized_buffers.render_dimensions[0] as usize,
                                self.sized_buffers.render_dimensions[1] as usize,
                            ),
                        ),
                        pixel_aspect: 1.0,
                        chromaticities: None,
                        time_code: None,
                        other: Default::default(),
                    },
                    layers,
                );
                image.write().to_file(path).unwrap();
                println!("Saved screenshot to {}", path);
            }
            Err(error) => {
                eprintln!("Error saving screenshot: {:?}", error);
            }
        };
    }

    pub fn save_rendered_frame_png(&mut self, path: &str) {
        self.wait_for_all_frames();

        let result = self.sized_buffers.output_cpu_pixel_buffer.read();
        match result {
            Ok(buffer_content) => {
                fn aces_tone_map(x: f32) -> f32 {
                    let a = 2.51f32;
                    let b = 0.03f32;
                    let c = 2.43f32;
                    let d = 0.59f32;
                    let e = 0.14f32;
                    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
                }

                let w = self.sized_buffers.render_dimensions[0] as u32;
                let h = self.sized_buffers.render_dimensions[1] as u32;
                let vte_collapsed = self.last_backend == RenderBackend::VoxelTraversal;
                let depth = if vte_collapsed {
                    1
                } else {
                    self.sized_buffers.render_dimensions[2]
                        .max(1)
                        .min(self.sized_buffers.pixel_storage_layers.max(1))
                };

                let mut pixels = Vec::with_capacity((w * h * 4) as usize);
                for y in 0..h {
                    for x in 0..w {
                        // Match present shader behavior:
                        // - VTE uses Stage-B-collapsed layer 0.
                        // - Legacy tetra paths accumulate all Z slices.
                        let accum_pixel = if vte_collapsed {
                            let idx = (y * w + x) as usize;
                            buffer_content[idx]
                        } else {
                            let mut full_pixel = Vec4::ZERO;
                            for z in 0..depth {
                                let idx = (z * w * h + y * w + x) as usize;
                                full_pixel += buffer_content[idx];
                            }
                            full_pixel
                        };

                        // Normalize by alpha (denominator) if non-zero.
                        let mut r = 0.0f32;
                        let mut g = 0.0f32;
                        let mut b = 0.0f32;
                        if accum_pixel.w > 0.0 {
                            r = accum_pixel.x / accum_pixel.w;
                            g = accum_pixel.y / accum_pixel.w;
                            b = accum_pixel.z / accum_pixel.w;
                        }

                        // Match present pass tone-map + gamma.
                        r = aces_tone_map(r);
                        g = aces_tone_map(g);
                        b = aces_tone_map(b);
                        let gamma = 1.0 / 2.2;
                        pixels.push((r.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push((g.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push((b.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push(255u8);
                    }
                }

                let image = ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, pixels).unwrap();
                image.save(path).unwrap();
                println!("Saved PNG to {}", path);
            }
            Err(error) => {
                eprintln!("Error saving PNG: {:?}", error);
            }
        };
    }
}
