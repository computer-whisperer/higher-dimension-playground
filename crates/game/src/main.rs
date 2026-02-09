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
use input::InputState;

const MOVE_SPEED: f32 = 5.0;
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

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Mouse look
        if self.mouse_grabbed {
            let (dx, dy) = self.input.take_mouse_delta();
            self.camera.apply_mouse_look(dx, dy, MOUSE_SENSITIVITY);
        } else {
            self.input.take_mouse_delta();
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
            .apply_movement(forward, strafe, vertical, w_axis, dt, MOVE_SPEED);

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
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: winit::event::ElementState::Pressed,
                ..
            } => {
                if !self.mouse_grabbed {
                    let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                    self.grab_mouse(&window);
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
