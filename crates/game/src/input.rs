use std::time::Instant;
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

const DOUBLE_TAP_THRESHOLD_MS: u128 = 300;

pub struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    w_pos: bool,
    w_neg: bool,
    mouse_dx: f64,
    mouse_dy: f64,
    escape_pressed: bool,
    last_space_press: Option<Instant>,
    fly_toggle_requested: bool,
    jump_requested: bool,
    screenshot_requested: bool,
}

impl InputState {
    pub fn new() -> Self {
        InputState {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            w_pos: false,
            w_neg: false,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            escape_pressed: false,
            last_space_press: None,
            fly_toggle_requested: false,
            jump_requested: false,
            screenshot_requested: false,
        }
    }

    pub fn handle_key_event(&mut self, event: &KeyEvent) {
        let pressed = event.state.is_pressed();
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::KeyW => self.forward = pressed,
                KeyCode::KeyS => self.backward = pressed,
                KeyCode::KeyA => self.left = pressed,
                KeyCode::KeyD => self.right = pressed,
                KeyCode::Space => {
                    self.up = pressed;
                    if pressed {
                        let now = Instant::now();
                        if let Some(last) = self.last_space_press {
                            if now.duration_since(last).as_millis() < DOUBLE_TAP_THRESHOLD_MS {
                                self.fly_toggle_requested = true;
                            }
                        }
                        self.last_space_press = Some(now);
                        self.jump_requested = true;
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
                KeyCode::KeyQ => self.w_neg = pressed,
                KeyCode::KeyE => self.w_pos = pressed,
                KeyCode::F12 => {
                    if pressed {
                        self.screenshot_requested = true;
                    }
                }
                KeyCode::Escape => {
                    if pressed {
                        self.escape_pressed = true;
                    }
                }
                _ => {}
            }
        }
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_dx += delta.0;
        self.mouse_dy += delta.1;
    }

    pub fn movement_axes(&self) -> (f32, f32, f32, f32) {
        let forward = (self.forward as i32 - self.backward as i32) as f32;
        let strafe = (self.right as i32 - self.left as i32) as f32;
        let vertical = (self.up as i32 - self.down as i32) as f32;
        let w_axis = (self.w_pos as i32 - self.w_neg as i32) as f32;
        (forward, strafe, vertical, w_axis)
    }

    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let d = (self.mouse_dx as f32, self.mouse_dy as f32);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        d
    }

    pub fn take_escape(&mut self) -> bool {
        let v = self.escape_pressed;
        self.escape_pressed = false;
        v
    }

    pub fn take_fly_toggle(&mut self) -> bool {
        let v = self.fly_toggle_requested;
        self.fly_toggle_requested = false;
        v
    }

    pub fn take_jump(&mut self) -> bool {
        let v = self.jump_requested;
        self.jump_requested = false;
        v
    }

    pub fn take_screenshot(&mut self) -> bool {
        let v = self.screenshot_requested;
        self.screenshot_requested = false;
        v
    }
}
