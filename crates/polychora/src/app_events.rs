#![allow(clippy::collapsible_match)]

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
        // Give the scene the GPU memory allocator so background voxel rebuilds
        // can pre-create GPU buffers off the main thread.
        self.scene
            .set_memory_allocator(self.rcx.as_ref().unwrap().memory_allocator());

        // Generate material icon sprite sheet and upload to GPU
        if self.material_icon_sheet.is_none() {
            let start = Instant::now();
            if let Some(sheet) = material_icons::generate_material_icon_sheet_gpu(
                self.device.clone(),
                self.queue.clone(),
                self.instance.clone(),
                &self.content_registry,
                &self.material_resolver,
                &self.pending_texture_uploads,
            ) {
                eprintln!(
                    "Generated material icon sprite sheet ({}x{}) in {:.2}s",
                    sheet.width,
                    sheet.height,
                    start.elapsed().as_secs_f32()
                );
                self.material_icon_sheet = Some(sheet);
            } else {
                eprintln!(
                    "Failed to generate GPU material icon sprite sheet; using color fallback icons."
                );
            }
        }
        if self.args.damascene_bundle_dump {
            if let Err(error) = self.dump_damascene_overlay_bundle() {
                eprintln!("Failed to dump Damascene overlay bundles: {error}");
            }
            event_loop.exit();
            return;
        }

        // Upload any pending plugin textures to the GPU texture pool.
        self.process_pending_texture_uploads();

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
        let damascene_inventory_open =
            self.app_state == AppState::Playing && self.inventory_open && !self.args.no_hud;
        let damascene_pause_open =
            self.app_state == AppState::Playing && self.menu_open && !self.args.no_hud;
        let damascene_teleport_open =
            self.app_state == AppState::Playing && self.teleport_dialog_open && !self.args.no_hud;
        let damascene_dev_console_open =
            self.app_state == AppState::Playing && self.dev_console_open && !self.args.no_hud;
        let damascene_block_gui_open = self.app_state == AppState::Playing
            && self.block_gui_session.is_some()
            && !self.args.no_hud;
        let damascene_main_menu_open = self.app_state == AppState::MainMenu && !self.args.no_hud;
        let user_input_locked = self.user_input_locked();
        let route_damascene_overlay = !self.args.no_hud
            && !user_input_locked
            && (damascene_main_menu_open
                || (self.app_state == AppState::Playing
                    && !self.mouse_grabbed
                    && (damascene_pause_open
                        || damascene_teleport_open
                        || damascene_dev_console_open
                        || damascene_block_gui_open
                        || damascene_inventory_open
                        || !self.inventory_open)));

        match event {
            WindowEvent::CloseRequested => {
                self.persist_settings_if_needed(true);
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if user_input_locked {
                    return;
                }

                if self.app_state == AppState::Playing
                    && event.state.is_pressed()
                    && !event.repeat
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Backquote))
                {
                    self.toggle_dev_console();
                    return;
                }

                // Tab toggles inventory regardless of overlay focus.
                if self.inventory_open
                    && event.state.is_pressed()
                    && !event.repeat
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Tab))
                {
                    self.inventory_open = false;
                    if let Some(window) = window.as_ref() {
                        self.grab_mouse(window);
                    }
                    return;
                }

                if is_escape_pressed(&event) {
                    if self.app_state == AppState::MainMenu {
                        // In main menu, Escape goes back or quits
                        if self.main_menu_page != MainMenuPage::Root {
                            self.main_menu_page = MainMenuPage::Root;
                            self.main_menu_connect_error = None;
                        } else {
                            self.persist_settings_if_needed(true);
                            event_loop.exit();
                        }
                    } else if self.dev_console_open {
                        self.close_dev_console();
                    } else if self.block_gui_session.is_some() {
                        if let Some(window) = window.as_ref() {
                            self.close_block_gui(window);
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
                    } else {
                        self.persist_settings_if_needed(true);
                        event_loop.exit();
                    }
                    return;
                }

                if route_damascene_overlay && event.state.is_pressed() {
                    if let Some(key) = damascene_ui_key(&event) {
                        let mut events = self
                            .rcx
                            .as_mut()
                            .map(|rcx| {
                                rcx.damascene_key_down(key.clone(), self.damascene_modifiers, event.repeat)
                            })
                            .unwrap_or_default();
                        if let Some(text) = event.text.as_ref() {
                            if let Some(text_event) = self
                                .rcx
                                .as_mut()
                                .and_then(|rcx| rcx.damascene_text_input(text.to_string()))
                            {
                                events.push(text_event);
                            }
                        }
                        let consumed = self.handle_damascene_ui_events(events);
                        if consumed {
                            if let Some(window) = window.as_ref() {
                                window.request_redraw();
                            }
                            return;
                        }

                        if damascene_dev_console_open
                            && self.handle_damascene_dev_console_key_fallback(
                                Some(key),
                                event.text.as_ref().map(ToString::to_string),
                                self.damascene_modifiers,
                                event.repeat,
                            )
                        {
                            if let Some(window) = window.as_ref() {
                                window.request_redraw();
                            }
                            return;
                        }
                    }
                }

                if self.app_state != AppState::MainMenu {
                    self.input.handle_key_event(&event);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if route_damascene_overlay {
                    let scale = window
                        .as_ref()
                        .map(|window| window.scale_factor() as f32)
                        .unwrap_or(1.0);
                    let point = (position.x as f32 / scale, position.y as f32 / scale);
                    self.damascene_last_pointer = Some(point);
                    let (needs_redraw, events) = self
                        .rcx
                        .as_mut()
                        .map(|rcx| rcx.damascene_pointer_moved(point.0, point.1))
                        .unwrap_or_default();
                    let consumed = self.handle_damascene_ui_events(events);
                    if needs_redraw || consumed {
                        if let Some(window) = window.as_ref() {
                            window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.damascene_last_pointer = None;
                if route_damascene_overlay {
                    if let Some(rcx) = self.rcx.as_mut() {
                        rcx.damascene_pointer_left();
                    }
                    if let Some(window) = window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.damascene_modifiers = damascene_key_modifiers(modifiers.state());
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.damascene_set_modifiers(self.damascene_modifiers);
                }
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Left => {
                    if user_input_locked {
                        return;
                    }
                    if route_damascene_overlay {
                        if let (Some(button), Some((x, y))) =
                            (damascene_pointer_button(button), self.damascene_last_pointer)
                        {
                            let events = match state {
                                winit::event::ElementState::Pressed => self
                                    .rcx
                                    .as_mut()
                                    .map(|rcx| rcx.damascene_pointer_down(x, y, button))
                                    .unwrap_or_default(),
                                winit::event::ElementState::Released => self
                                    .rcx
                                    .as_mut()
                                    .map(|rcx| rcx.damascene_pointer_up(x, y, button))
                                    .unwrap_or_default(),
                            };
                            if self.handle_damascene_ui_events(events) {
                                if let Some(window) = window.as_ref() {
                                    window.request_redraw();
                                }
                                return;
                            }
                        }
                    }
                    if state.is_pressed() {
                        if self.app_state == AppState::MainMenu {
                            // Don't grab mouse in main menu.
                        } else if self.mouse_grabbed {
                            self.input.handle_mouse_button(button, state);
                        } else if self.teleport_dialog_open {
                            // Click outside teleport dialog — close it and grab mouse
                            self.toggle_teleport_dialog();
                        } else if !self.menu_open
                            && !self.inventory_open
                            && !self.teleport_dialog_open
                            && !self.dev_console_open
                            && self.block_gui_session.is_none()
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
                    if user_input_locked {
                        return;
                    }
                    if route_damascene_overlay {
                        if let (Some(button), Some((x, y))) =
                            (damascene_pointer_button(button), self.damascene_last_pointer)
                        {
                            let events = match state {
                                winit::event::ElementState::Pressed => self
                                    .rcx
                                    .as_mut()
                                    .map(|rcx| rcx.damascene_pointer_down(x, y, button))
                                    .unwrap_or_default(),
                                winit::event::ElementState::Released => self
                                    .rcx
                                    .as_mut()
                                    .map(|rcx| rcx.damascene_pointer_up(x, y, button))
                                    .unwrap_or_default(),
                            };
                            if self.handle_damascene_ui_events(events) {
                                if let Some(window) = window.as_ref() {
                                    window.request_redraw();
                                }
                                return;
                            }
                        }
                    }
                    if self.app_state != AppState::MainMenu {
                        self.input.handle_mouse_button(button, state);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if user_input_locked {
                    return;
                }
                if route_damascene_overlay {
                    if let Some((x, y)) = self.damascene_last_pointer {
                        let scale = window
                            .as_ref()
                            .map(|window| window.scale_factor() as f32)
                            .unwrap_or(1.0);
                        let dy = match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => -y * 50.0,
                            winit::event::MouseScrollDelta::PixelDelta(pos) => {
                                -(pos.y as f32) / scale
                            }
                        };
                        if self
                            .rcx
                            .as_mut()
                            .map(|rcx| rcx.damascene_pointer_wheel(x, y, dy))
                            .unwrap_or(false)
                        {
                            if let Some(window) = window.as_ref() {
                                window.request_redraw();
                            }
                            return;
                        }
                    }
                }
                if self.app_state != AppState::MainMenu {
                    let y = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                    };
                    self.input.handle_scroll(y);
                }
            }
            WindowEvent::Focused(false) => {
                if user_input_locked {
                    return;
                }
                if self.mouse_grabbed {
                    let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                    self.release_mouse(&window);
                    self.menu_open = true;
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.should_exit_after_render {
                    self.persist_settings_if_needed(true);
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
        if self.user_input_locked() {
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

fn damascene_pointer_button(button: MouseButton) -> Option<damascene_core::PointerButton> {
    match button {
        MouseButton::Left => Some(damascene_core::PointerButton::Primary),
        MouseButton::Right => Some(damascene_core::PointerButton::Secondary),
        MouseButton::Middle => Some(damascene_core::PointerButton::Middle),
        _ => None,
    }
}

fn damascene_key_modifiers(mods: winit::keyboard::ModifiersState) -> damascene_core::KeyModifiers {
    damascene_core::KeyModifiers {
        shift: mods.shift_key(),
        ctrl: mods.control_key(),
        alt: mods.alt_key(),
        logo: mods.super_key(),
    }
}

fn damascene_ui_key(event: &winit::event::KeyEvent) -> Option<damascene_core::UiKey> {
    match &event.logical_key {
        Key::Named(named) => match named {
            NamedKey::Enter => Some(damascene_core::UiKey::Enter),
            NamedKey::Escape => Some(damascene_core::UiKey::Escape),
            NamedKey::Tab => Some(damascene_core::UiKey::Tab),
            NamedKey::Space => Some(damascene_core::UiKey::Space),
            NamedKey::ArrowUp => Some(damascene_core::UiKey::ArrowUp),
            NamedKey::ArrowDown => Some(damascene_core::UiKey::ArrowDown),
            NamedKey::ArrowLeft => Some(damascene_core::UiKey::ArrowLeft),
            NamedKey::ArrowRight => Some(damascene_core::UiKey::ArrowRight),
            NamedKey::Backspace => Some(damascene_core::UiKey::Backspace),
            NamedKey::Delete => Some(damascene_core::UiKey::Delete),
            NamedKey::Home => Some(damascene_core::UiKey::Home),
            NamedKey::End => Some(damascene_core::UiKey::End),
            NamedKey::PageUp => Some(damascene_core::UiKey::PageUp),
            NamedKey::PageDown => Some(damascene_core::UiKey::PageDown),
            _ => Some(damascene_core::UiKey::Other(format!("{named:?}"))),
        },
        Key::Character(text) => Some(damascene_core::UiKey::Character(text.to_string())),
        Key::Unidentified(_) => {
            if let PhysicalKey::Code(code) = event.physical_key {
                Some(damascene_core::UiKey::Other(format!("{code:?}")))
            } else {
                None
            }
        }
        Key::Dead(dead) => Some(damascene_core::UiKey::Other(format!("{dead:?}"))),
    }
}

impl App {
    /// Drain pending plugin texture uploads and push each to the GPU texture pool.
    ///
    /// Texture tokens are already registered in the content registry (before it
    /// was wrapped in Arc), with pre-assigned sequential slot indices.  This
    /// method performs the actual GPU upload and verifies the slot assignment
    /// matches what was pre-registered.
    fn process_pending_texture_uploads(&mut self) {
        let uploads: Vec<_> = self.pending_texture_uploads.drain(..).collect();
        if uploads.is_empty() {
            return;
        }

        let rcx = match self.rcx.as_mut() {
            Some(rcx) => rcx,
            None => return,
        };

        for (i, upload) in uploads.iter().enumerate() {
            match rcx.upload_texture_3d(
                &upload.data,
                upload.width,
                upload.height,
                upload.depth,
                upload.format,
            ) {
                Some(index) => {
                    assert_eq!(
                        index, i as u16,
                        "texture pool slot {index} diverged from pre-registered slot {i}"
                    );
                }
                None => {
                    eprintln!(
                        "Warning: texture pool full, could not upload texture {:#010x}",
                        upload.texture_id,
                    );
                }
            }
        }
    }
}
