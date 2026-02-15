use super::*;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window_attrs = {
            let attrs = Window::default_attributes();
            // When explicit window size given, use it.
            // When --gpu-screenshot active but no explicit window size, match render buffer.
            let (w, h) = if self.args.window_width.is_some() || self.args.window_height.is_some() {
                (
                    self.args.window_width.unwrap_or(self.args.width),
                    self.args.window_height.unwrap_or(self.args.height),
                )
            } else if self.args.gpu_screenshot {
                (self.args.width, self.args.height)
            } else {
                (0, 0) // sentinel: use default
            };
            if w > 0 && h > 0 {
                attrs.with_inner_size(LogicalSize::new(w, h))
            } else {
                attrs
            }
        };
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.egui_winit_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        ));

        if self.app_state == AppState::Playing && !self.perf_suite_active() {
            self.grab_mouse(&window);
        }
        let backend = self.args.backend.to_render_backend();
        let vte_mode = self.args.vte_display_mode.to_render_mode();
        let pixel_storage_layers =
            if backend == RenderBackend::VoxelTraversal && vte_mode == VteDisplayMode::Integral {
                Some(1)
            } else {
                None
            };
        self.rcx = Some(RenderContext::new_with_pixel_storage_layers(
            self.device.clone(),
            self.queue.clone(),
            self.instance.clone(),
            Some(window.clone()),
            [self.args.width, self.args.height, self.args.layers],
            pixel_storage_layers,
        ));

        // Generate material icon sprite sheet and upload to GPU
        if self.material_icon_sheet.is_none() {
            let start = Instant::now();
            let model_tets = higher_dimension_playground::render::generate_tesseract_tetrahedrons();
            let sheet = material_icons::generate_material_icon_sheet(&model_tets);
            eprintln!(
                "Generated material icon sprite sheet ({}x{}) in {:.2}s",
                sheet.width,
                sheet.height,
                start.elapsed().as_secs_f32()
            );
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.upload_material_icons_texture(
                    self.queue.clone(),
                    sheet.width,
                    sheet.height,
                    &sheet.pixels,
                );
            }
            // Use User(1) as the egui texture ID for material icons
            self.material_icons_texture_id = Some(egui::TextureId::User(1));
            self.material_icon_sheet = Some(sheet);
        }

        self.last_frame = Instant::now();
        if self.perf_suite_active() {
            self.begin_perf_suite_phase(true);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        let show_egui_overlay = self.app_state == AppState::MainMenu
            || self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open;
        let egui_consumed = if let (Some(egui_state), Some(window)) =
            (self.egui_winit_state.as_mut(), window.as_ref())
        {
            show_egui_overlay && egui_state.on_window_event(window, &event).consumed
        } else {
            false
        };
        let perf_suite_input_locked = self.perf_suite_active();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if perf_suite_input_locked {
                    return;
                }
                if !egui_consumed {
                    self.input.handle_key_event(&event);
                }

                if is_escape_pressed(&event) {
                    if self.app_state == AppState::MainMenu {
                        // In main menu, Escape goes back or quits
                        if self.main_menu_page != MainMenuPage::Root {
                            self.main_menu_page = MainMenuPage::Root;
                            self.main_menu_connect_error = None;
                        } else {
                            event_loop.exit();
                        }
                    } else if self.teleport_dialog_open {
                        self.teleport_dialog_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.inventory_open {
                        self.inventory_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.menu_open {
                        self.menu_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.mouse_grabbed {
                        if let Some(window) = window.as_ref() {
                            self.release_mouse(window);
                        }
                        self.menu_open = true;
                        self.menu_selection = 0;
                    } else {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Left => {
                    if perf_suite_input_locked {
                        return;
                    }
                    if state.is_pressed() {
                        if self.app_state == AppState::MainMenu {
                            // Don't grab mouse in main menu -- egui handles clicks
                        } else if self.mouse_grabbed {
                            if !egui_consumed {
                                self.input.handle_mouse_button(button, state);
                            }
                        } else if self.teleport_dialog_open && !egui_consumed {
                            // Click outside teleport dialog — close it and grab mouse
                            self.toggle_teleport_dialog();
                        } else if !self.menu_open
                            && !self.inventory_open
                            && !self.teleport_dialog_open
                        {
                            if let Some(window) = window.as_ref() {
                                self.grab_mouse(window);
                                self.menu_open = false;
                            }
                        }
                    }
                }
                MouseButton::Middle
                | MouseButton::Right
                | MouseButton::Back
                | MouseButton::Forward => {
                    if perf_suite_input_locked {
                        return;
                    }
                    if !egui_consumed {
                        self.input.handle_mouse_button(button, state);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if perf_suite_input_locked {
                    return;
                }
                if !egui_consumed {
                    let y = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                    };
                    self.input.handle_scroll(y);
                }
            }
            WindowEvent::Focused(false) => {
                if perf_suite_input_locked {
                    return;
                }
                if self.mouse_grabbed {
                    let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                    self.release_mouse(&window);
                    self.menu_open = true;
                    self.menu_selection = 0;
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.should_exit_after_render {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if self.perf_suite_active() {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_grabbed {
                self.input.handle_mouse_motion(delta);
                if let Some(rcx) = self.rcx.as_ref() {
                    if let Some(window) = rcx.window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(rcx) = self.rcx.as_ref() {
            if let Some(window) = rcx.window.as_ref() {
                window.request_redraw();
            }
        }
    }
}
