use std::time::Instant;
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};

use crate::camera::AngleTarget;

const DOUBLE_TAP_THRESHOLD_MS: u128 = 300;
const DOUBLE_TAP_MIN_INTERVAL_MS: u128 = 90;
const FLY_TOGGLE_COOLDOWN_MS: u128 = 700;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ControlScheme {
    IntuitiveUpright,
    LookTransport,
    RotorFree,
    LegacySideButtonLayers,
    LegacyScrollCycle,
}

impl ControlScheme {
    pub fn next(self) -> Self {
        match self {
            ControlScheme::IntuitiveUpright => ControlScheme::LookTransport,
            ControlScheme::LookTransport => ControlScheme::RotorFree,
            ControlScheme::RotorFree => ControlScheme::LegacySideButtonLayers,
            ControlScheme::LegacySideButtonLayers => ControlScheme::LegacyScrollCycle,
            ControlScheme::LegacyScrollCycle => ControlScheme::IntuitiveUpright,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ControlScheme::IntuitiveUpright => "UPRIGHT",
            ControlScheme::LookTransport => "LOOK-TR",
            ControlScheme::RotorFree => "ROTOR",
            ControlScheme::LegacySideButtonLayers => "LEG-SIDE",
            ControlScheme::LegacyScrollCycle => "LEG-SCRL",
        }
    }

    pub fn is_upright_primary(self) -> bool {
        matches!(self, ControlScheme::IntuitiveUpright)
    }

    pub fn uses_look_frame(self) -> bool {
        matches!(
            self,
            ControlScheme::LookTransport | ControlScheme::RotorFree
        )
    }

    pub fn uses_scroll_pair_cycle(self) -> bool {
        matches!(self, ControlScheme::LegacyScrollCycle)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RotationPair {
    Standard,
    FourD,
    DoubleRotation,
}

impl RotationPair {
    pub fn h_target(self) -> AngleTarget {
        match self {
            RotationPair::Standard => AngleTarget::Yaw,
            RotationPair::FourD => AngleTarget::XwAngle,
            RotationPair::DoubleRotation => AngleTarget::Yaw,
        }
    }

    pub fn v_target(self) -> AngleTarget {
        match self {
            RotationPair::Standard => AngleTarget::Pitch,
            RotationPair::FourD => AngleTarget::ZwAngle,
            RotationPair::DoubleRotation => AngleTarget::YwDeviation,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RotationPair::Standard => "XZ/ZY",
            RotationPair::FourD => "XW/ZW",
            RotationPair::DoubleRotation => "DOUBLE XZ+YW",
        }
    }

    pub fn next(self) -> Self {
        match self {
            RotationPair::Standard => RotationPair::FourD,
            RotationPair::FourD => RotationPair::Standard,
            RotationPair::DoubleRotation => RotationPair::Standard,
        }
    }
}

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
    last_forward_press: Option<Instant>,
    last_fly_toggle: Option<Instant>,
    fly_toggle_requested: bool,
    sprint_toggle_requested: bool,
    jump_requested: bool,
    screenshot_requested: bool,
    remove_block_requested: bool,
    place_block_requested: bool,
    pick_material_requested: bool,
    place_material_prev_requested: bool,
    place_material_next_requested: bool,
    place_material_digit_requested: Option<u8>,
    save_world_requested: bool,
    load_world_requested: bool,
    mouse_back_held: bool,
    mouse_forward_held: bool,
    scroll_accumulated: f32,
    scheme_cycle_requested: bool,
    reset_orientation_held: bool,
    pull_to_3d_held: bool,
    vte_entities_toggle_requested: bool,
    vte_y_slice_lookup_cache_toggle_requested: bool,
    vte_sweep_requested: bool,
    vte_integral_sky_emissive_toggle_requested: bool,
    vte_integral_log_merge_toggle_requested: bool,
    look_at_requested: bool,
    inventory_toggle_requested: bool,
    menu_left_requested: bool,
    menu_right_requested: bool,
    menu_up_requested: bool,
    menu_down_requested: bool,
    menu_activate_requested: bool,
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
            last_forward_press: None,
            last_fly_toggle: None,
            fly_toggle_requested: false,
            sprint_toggle_requested: false,
            jump_requested: false,
            screenshot_requested: false,
            remove_block_requested: false,
            place_block_requested: false,
            pick_material_requested: false,
            place_material_prev_requested: false,
            place_material_next_requested: false,
            place_material_digit_requested: None,
            save_world_requested: false,
            load_world_requested: false,
            mouse_back_held: false,
            mouse_forward_held: false,
            scroll_accumulated: 0.0,
            scheme_cycle_requested: false,
            reset_orientation_held: false,
            pull_to_3d_held: false,
            vte_entities_toggle_requested: false,
            vte_y_slice_lookup_cache_toggle_requested: false,
            vte_sweep_requested: false,
            vte_integral_sky_emissive_toggle_requested: false,
            vte_integral_log_merge_toggle_requested: false,
            look_at_requested: false,
            inventory_toggle_requested: false,
            menu_left_requested: false,
            menu_right_requested: false,
            menu_up_requested: false,
            menu_down_requested: false,
            menu_activate_requested: false,
        }
    }

    pub fn handle_key_event(&mut self, event: &KeyEvent) {
        let pressed = event.state.is_pressed();
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::KeyW => {
                    let was_forward = self.forward;
                    self.forward = pressed;
                    if pressed && !was_forward {
                        self.handle_forward_tap(Instant::now());
                    }
                }
                KeyCode::KeyS => self.backward = pressed,
                KeyCode::KeyA => self.left = pressed,
                KeyCode::KeyD => self.right = pressed,
                KeyCode::Space => {
                    let was_up = self.up;
                    self.up = pressed;
                    // Treat only the key-down edge as a tap; this filters key-repeat noise.
                    if pressed && !was_up {
                        self.handle_space_tap(Instant::now());
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
                KeyCode::KeyQ => self.w_neg = pressed,
                KeyCode::KeyE => self.w_pos = pressed,
                KeyCode::KeyI => {
                    if pressed && !event.repeat {
                        self.inventory_toggle_requested = true;
                    }
                }
                KeyCode::F12 => {
                    if pressed {
                        self.screenshot_requested = true;
                    }
                }
                KeyCode::F5 => {
                    if pressed {
                        self.save_world_requested = true;
                    }
                }
                KeyCode::F9 => {
                    if pressed {
                        self.load_world_requested = true;
                    }
                }
                KeyCode::BracketLeft => {
                    if pressed {
                        self.place_material_prev_requested = true;
                    }
                }
                KeyCode::BracketRight => {
                    if pressed {
                        self.place_material_next_requested = true;
                    }
                }
                KeyCode::Digit1 => {
                    if pressed {
                        self.place_material_digit_requested = Some(1);
                    }
                }
                KeyCode::Digit2 => {
                    if pressed {
                        self.place_material_digit_requested = Some(2);
                    }
                }
                KeyCode::Digit3 => {
                    if pressed {
                        self.place_material_digit_requested = Some(3);
                    }
                }
                KeyCode::Digit4 => {
                    if pressed {
                        self.place_material_digit_requested = Some(4);
                    }
                }
                KeyCode::Digit5 => {
                    if pressed {
                        self.place_material_digit_requested = Some(5);
                    }
                }
                KeyCode::Digit6 => {
                    if pressed {
                        self.place_material_digit_requested = Some(6);
                    }
                }
                KeyCode::Digit7 => {
                    if pressed {
                        self.place_material_digit_requested = Some(7);
                    }
                }
                KeyCode::Digit8 => {
                    if pressed {
                        self.place_material_digit_requested = Some(8);
                    }
                }
                KeyCode::Digit9 => {
                    if pressed {
                        self.place_material_digit_requested = Some(9);
                    }
                }
                KeyCode::Digit0 => {
                    if pressed {
                        self.place_material_digit_requested = Some(10);
                    }
                }
                KeyCode::Tab => {
                    if pressed {
                        self.inventory_toggle_requested = true;
                    }
                }
                KeyCode::KeyR => {
                    self.reset_orientation_held = pressed;
                }
                KeyCode::KeyF => {
                    self.pull_to_3d_held = pressed;
                }
                KeyCode::KeyG => {
                    if pressed && !event.repeat {
                        self.look_at_requested = true;
                    }
                }
                KeyCode::F6 => {
                    if pressed {
                        self.vte_entities_toggle_requested = true;
                    }
                }
                KeyCode::F7 => {
                    if pressed {
                        self.vte_y_slice_lookup_cache_toggle_requested = true;
                    }
                }
                KeyCode::F8 => {
                    if pressed {
                        self.vte_sweep_requested = true;
                    }
                }
                KeyCode::F10 => {
                    if pressed {
                        self.vte_integral_sky_emissive_toggle_requested = true;
                    }
                }
                KeyCode::F11 => {
                    if pressed {
                        self.vte_integral_log_merge_toggle_requested = true;
                    }
                }
                KeyCode::ArrowUp => {
                    if pressed {
                        self.menu_up_requested = true;
                    }
                }
                KeyCode::ArrowLeft => {
                    if pressed {
                        self.menu_left_requested = true;
                    }
                }
                KeyCode::ArrowRight => {
                    if pressed {
                        self.menu_right_requested = true;
                    }
                }
                KeyCode::ArrowDown => {
                    if pressed {
                        self.menu_down_requested = true;
                    }
                }
                KeyCode::Enter | KeyCode::NumpadEnter => {
                    if pressed {
                        self.menu_activate_requested = true;
                    }
                }
                KeyCode::Escape => {
                    // Ignore key-repeat so Escape cannot toggle menu open/close in one hold.
                    if pressed && !event.repeat {
                        self.escape_pressed = true;
                    }
                }
                _ => {}
            }
        }

        // Fallback for platforms/layouts where Escape may not map to a physical code.
        if pressed && !event.repeat && matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
            self.escape_pressed = true;
        }
    }

    fn handle_space_tap(&mut self, now: Instant) {
        self.jump_requested = true;

        if let Some(last_toggle) = self.last_fly_toggle {
            if now.duration_since(last_toggle).as_millis() < FLY_TOGGLE_COOLDOWN_MS {
                self.last_space_press = Some(now);
                return;
            }
        }

        if let Some(last_press) = self.last_space_press {
            let dt_ms = now.duration_since(last_press).as_millis();
            if dt_ms < DOUBLE_TAP_MIN_INTERVAL_MS {
                // Ignore implausibly fast "second taps" from switch bounce / repeat.
                return;
            }
            if dt_ms <= DOUBLE_TAP_THRESHOLD_MS {
                self.fly_toggle_requested = true;
                self.last_fly_toggle = Some(now);
                self.last_space_press = None;
                return;
            }
        }

        self.last_space_press = Some(now);
    }

    fn handle_forward_tap(&mut self, now: Instant) {
        if let Some(last_press) = self.last_forward_press {
            let dt_ms = now.duration_since(last_press).as_millis();
            if dt_ms < DOUBLE_TAP_MIN_INTERVAL_MS {
                return;
            }
            if dt_ms <= DOUBLE_TAP_THRESHOLD_MS {
                self.sprint_toggle_requested = true;
                self.last_forward_press = None;
                return;
            }
        }
        self.last_forward_press = Some(now);
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let pressed = state.is_pressed();
        match button {
            MouseButton::Left => {
                if pressed {
                    self.remove_block_requested = true;
                }
            }
            MouseButton::Right => {
                if pressed {
                    self.place_block_requested = true;
                }
            }
            MouseButton::Middle => {
                if pressed {
                    self.pick_material_requested = true;
                }
            }
            MouseButton::Back => self.mouse_back_held = pressed,
            MouseButton::Forward => self.mouse_forward_held = pressed,
            _ => {}
        }
    }

    pub fn handle_scroll(&mut self, delta_y: f32) {
        self.scroll_accumulated += delta_y;
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

    pub fn clear_mouse_delta(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
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

    pub fn take_sprint_toggle(&mut self) -> bool {
        let v = self.sprint_toggle_requested;
        self.sprint_toggle_requested = false;
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

    pub fn take_remove_block(&mut self) -> bool {
        let v = self.remove_block_requested;
        self.remove_block_requested = false;
        v
    }

    pub fn take_place_block(&mut self) -> bool {
        let v = self.place_block_requested;
        self.place_block_requested = false;
        v
    }

    pub fn take_pick_material(&mut self) -> bool {
        let v = self.pick_material_requested;
        self.pick_material_requested = false;
        v
    }

    pub fn take_place_material_prev(&mut self) -> bool {
        let v = self.place_material_prev_requested;
        self.place_material_prev_requested = false;
        v
    }

    pub fn take_place_material_next(&mut self) -> bool {
        let v = self.place_material_next_requested;
        self.place_material_next_requested = false;
        v
    }

    pub fn take_place_material_digit(&mut self) -> Option<u8> {
        self.place_material_digit_requested.take()
    }

    pub fn take_save_world(&mut self) -> bool {
        let v = self.save_world_requested;
        self.save_world_requested = false;
        v
    }

    pub fn take_load_world(&mut self) -> bool {
        let v = self.load_world_requested;
        self.load_world_requested = false;
        v
    }

    pub fn take_scroll_steps(&mut self) -> i32 {
        let steps = self.scroll_accumulated.trunc() as i32;
        self.scroll_accumulated -= steps as f32;
        steps
    }

    pub fn take_scheme_cycle(&mut self) -> bool {
        let v = self.scheme_cycle_requested;
        self.scheme_cycle_requested = false;
        v
    }

    pub fn reset_orientation_held(&self) -> bool {
        self.reset_orientation_held
    }

    pub fn pull_to_3d_held(&self) -> bool {
        self.pull_to_3d_held
    }

    pub fn take_vte_entities_toggle(&mut self) -> bool {
        let v = self.vte_entities_toggle_requested;
        self.vte_entities_toggle_requested = false;
        v
    }

    pub fn take_vte_y_slice_lookup_cache_toggle(&mut self) -> bool {
        let v = self.vte_y_slice_lookup_cache_toggle_requested;
        self.vte_y_slice_lookup_cache_toggle_requested = false;
        v
    }

    pub fn take_vte_sweep(&mut self) -> bool {
        let v = self.vte_sweep_requested;
        self.vte_sweep_requested = false;
        v
    }

    pub fn take_vte_integral_sky_emissive_toggle(&mut self) -> bool {
        let v = self.vte_integral_sky_emissive_toggle_requested;
        self.vte_integral_sky_emissive_toggle_requested = false;
        v
    }

    pub fn take_vte_integral_log_merge_toggle(&mut self) -> bool {
        let v = self.vte_integral_log_merge_toggle_requested;
        self.vte_integral_log_merge_toggle_requested = false;
        v
    }

    pub fn take_look_at(&mut self) -> bool {
        let v = self.look_at_requested;
        self.look_at_requested = false;
        v
    }

    pub fn take_inventory_toggle(&mut self) -> bool {
        let v = self.inventory_toggle_requested;
        self.inventory_toggle_requested = false;
        v
    }

    pub fn take_menu_left(&mut self) -> bool {
        let v = self.menu_left_requested;
        self.menu_left_requested = false;
        v
    }

    pub fn take_menu_right(&mut self) -> bool {
        let v = self.menu_right_requested;
        self.menu_right_requested = false;
        v
    }

    pub fn take_menu_up(&mut self) -> bool {
        let v = self.menu_up_requested;
        self.menu_up_requested = false;
        v
    }

    pub fn take_menu_down(&mut self) -> bool {
        let v = self.menu_down_requested;
        self.menu_down_requested = false;
        v
    }

    pub fn take_menu_activate(&mut self) -> bool {
        let v = self.menu_activate_requested;
        self.menu_activate_requested = false;
        v
    }

    pub fn mouse_back_held(&self) -> bool {
        self.mouse_back_held
    }

    pub fn mouse_forward_held(&self) -> bool {
        self.mouse_forward_held
    }
}
