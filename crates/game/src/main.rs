mod camera;
mod input;
mod scene;

use clap::Parser;
use higher_dimension_playground::render::{RenderContext, RenderOptions};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::sync::Arc;
use std::time::Instant;
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::Camera4D;
use input::{ControlScheme, InputState, RotationPair};

const MOUSE_SENSITIVITY: f32 = 0.002;

#[derive(Parser, Debug, Clone)]
#[command(version, about = "4D game explorer")]
struct Args {
    /// Render buffer width in pixels
    #[arg(long, short = 'W', default_value_t = 480)]
    width: u32,

    /// Render buffer height in pixels
    #[arg(long, short = 'H', default_value_t = 270)]
    height: u32,

    /// Number of depth layers (supersampling)
    #[arg(long, default_value_t = 4)]
    layers: u32,
}

fn main() {
    let args = Args::parse();
    let event_loop = EventLoop::new().unwrap();
    let (instance, device, queue) = vulkan_setup(Some(&event_loop));

    let mut app = App {
        instance,
        device,
        queue,
        rcx: None,
        camera: Camera4D::new(),
        input: InputState::new(),
        start_time: Instant::now(),
        last_frame: Instant::now(),
        mouse_grabbed: false,
        should_exit_after_render: false,
        args,
        control_scheme: ControlScheme::SideButtonLayers,
        scroll_cycle_pair: RotationPair::Standard,
        move_speed: 5.0,
    };

    event_loop.run_app(&mut app).unwrap();
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    camera: Camera4D,
    input: InputState,
    start_time: Instant,
    last_frame: Instant,
    mouse_grabbed: bool,
    should_exit_after_render: bool,
    args: Args,
    control_scheme: ControlScheme,
    scroll_cycle_pair: RotationPair,
    move_speed: f32,
}

impl App {
    fn grab_mouse(&mut self, window: &Window) {
        let result = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if result.is_ok() {
            window.set_cursor_visible(false);
            self.mouse_grabbed = true;
        }
    }

    fn release_mouse(&mut self, window: &Window) {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.mouse_grabbed = false;
    }

    fn active_rotation_pair(&self) -> RotationPair {
        if self.input.mouse_back_held() && self.input.mouse_forward_held() {
            RotationPair::DoubleRotation
        } else if self.input.mouse_back_held() {
            RotationPair::FourD
        } else {
            match self.control_scheme {
                ControlScheme::SideButtonLayers => RotationPair::Standard,
                ControlScheme::ScrollCycle => self.scroll_cycle_pair,
            }
        }
    }

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Scheme cycle (Tab)
        if self.input.take_scheme_cycle() {
            self.control_scheme = self.control_scheme.next();
            self.scroll_cycle_pair = RotationPair::Standard;
        }

        // Reset orientation (R)
        if self.input.take_reset_orientation() {
            self.camera.reset_orientation();
        }

        // Scroll wheel
        let scroll = self.input.take_scroll();
        if scroll != 0.0 {
            match self.control_scheme {
                ControlScheme::SideButtonLayers => {
                    let factor = 1.1_f32.powf(scroll);
                    self.move_speed = (self.move_speed * factor).clamp(0.5, 100.0);
                }
                ControlScheme::ScrollCycle => {
                    if scroll > 0.0 {
                        self.scroll_cycle_pair = self.scroll_cycle_pair.next();
                    } else {
                        // Reverse cycle
                        self.scroll_cycle_pair = match self.scroll_cycle_pair {
                            RotationPair::Standard => RotationPair::FourD,
                            RotationPair::FourD => RotationPair::Standard,
                            RotationPair::DoubleRotation => RotationPair::Standard,
                        };
                    }
                }
            }
        }

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        // Mouse look
        if self.mouse_grabbed {
            let (dx, dy) = self.input.take_mouse_delta();
            self.camera
                .apply_mouse_look_on(dx, dy, MOUSE_SENSITIVITY, pair.h_target(), pair.v_target());
        } else {
            self.input.take_mouse_delta();
        }

        // Auto-level when not in double rotation
        if pair != RotationPair::DoubleRotation {
            self.camera.auto_level(dt);
        }

        // Toggle flight mode on double-tap space
        if self.input.take_fly_toggle() {
            self.camera.toggle_flying();
        }

        // Jump when in gravity mode, consume jump either way
        if self.camera.is_flying {
            self.input.take_jump();
        } else if self.input.take_jump() {
            self.camera.jump();
        }

        // Movement (vertical zeroed in gravity mode internally)
        let (forward, strafe, vertical, w_axis) = self.input.movement_axes();
        self.camera
            .apply_movement(forward, strafe, vertical, w_axis, dt, self.move_speed);

        // Apply gravity physics
        self.camera.update_physics(dt);

        // Build view matrix and scene
        let view_matrix = self.camera.view_matrix();
        let time = self.start_time.elapsed().as_secs_f32();
        let instances = scene::build_scene_instances(time);

        let take_screenshot = self.input.take_screenshot();
        if take_screenshot {
            let _ = std::fs::create_dir_all("frames");
            self.should_exit_after_render = true;
        }

        let render_options = RenderOptions {
            do_raster: true,
            do_navigation_hud: true,
            take_framebuffer_screenshot: take_screenshot,
            hud_rotation_label: Some({
                let inv = if self.camera.is_y_inverted() {
                    " Y-INV"
                } else {
                    ""
                };
                format!(
                    "{} [{}]  spd:{:.1}{}\n\
                     yaw:{:+.0} pit:{:+.0} xw:{:+.0} zw:{:+.0} yw:{:+.1}",
                    pair.label(),
                    self.control_scheme.label(),
                    self.move_speed,
                    inv,
                    self.camera.yaw.to_degrees(),
                    self.camera.pitch.to_degrees(),
                    self.camera.xw_angle.to_degrees(),
                    self.camera.zw_angle.to_degrees(),
                    self.camera.yw_deviation.to_degrees(),
                )
            }),
            ..Default::default()
        };

        let rcx = self.rcx.as_mut().unwrap();
        rcx.render(
            self.device.clone(),
            self.queue.clone(),
            view_matrix,
            1.0,
            1.0,
            &instances,
            render_options,
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.grab_mouse(&window);
        self.rcx = Some(RenderContext::new(
            self.device.clone(),
            self.queue.clone(),
            self.instance.clone(),
            Some(window),
            [self.args.width, self.args.height, self.args.layers],
        ));
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
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
                self.input.handle_key_event(&event);

                if self.input.take_escape() {
                    if self.mouse_grabbed {
                        let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                        self.release_mouse(&window);
                    } else {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                match button {
                    MouseButton::Left => {
                        if state.is_pressed() && !self.mouse_grabbed {
                            let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                            self.grab_mouse(&window);
                        }
                    }
                    MouseButton::Back | MouseButton::Forward => {
                        self.input.handle_mouse_button(button, state);
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                self.input.handle_scroll(y);
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
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.handle_mouse_motion(delta);
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
