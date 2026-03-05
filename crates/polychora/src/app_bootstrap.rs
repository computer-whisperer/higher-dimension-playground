use super::*;

fn parse_keycode(key: &str) -> Option<KeyCode> {
    match key.to_lowercase().as_str() {
        "escape" => Some(KeyCode::Escape),
        "e" => Some(KeyCode::KeyE),
        "i" => Some(KeyCode::KeyI),
        "w" => Some(KeyCode::KeyW),
        "a" => Some(KeyCode::KeyA),
        "s" => Some(KeyCode::KeyS),
        "d" => Some(KeyCode::KeyD),
        "q" => Some(KeyCode::KeyQ),
        "r" => Some(KeyCode::KeyR),
        "f" => Some(KeyCode::KeyF),
        "g" => Some(KeyCode::KeyG),
        "tab" => Some(KeyCode::Tab),
        "space" => Some(KeyCode::Space),
        "shift" => Some(KeyCode::ShiftLeft),
        "1" => Some(KeyCode::Digit1),
        "2" => Some(KeyCode::Digit2),
        "3" => Some(KeyCode::Digit3),
        "4" => Some(KeyCode::Digit4),
        "5" => Some(KeyCode::Digit5),
        "6" => Some(KeyCode::Digit6),
        "7" => Some(KeyCode::Digit7),
        "8" => Some(KeyCode::Digit8),
        "9" => Some(KeyCode::Digit9),
        "0" => Some(KeyCode::Digit0),
        "f12" => Some(KeyCode::F12),
        "enter" => Some(KeyCode::Enter),
        "up" => Some(KeyCode::ArrowUp),
        "down" => Some(KeyCode::ArrowDown),
        "left" => Some(KeyCode::ArrowLeft),
        "right" => Some(KeyCode::ArrowRight),
        "lbracket" => Some(KeyCode::BracketLeft),
        "rbracket" => Some(KeyCode::BracketRight),
        _ => None,
    }
}

pub(super) fn parse_commands(input: &str) -> Vec<AutoCommand> {
    let mut commands = Vec::new();
    for cmd_str in input.split(';') {
        let cmd_str = cmd_str.trim();
        if cmd_str.is_empty() {
            continue;
        }
        if let Some((cmd_type, arg)) = cmd_str.split_once(':') {
            match cmd_type.trim() {
                "press" => {
                    if let Some(keycode) = parse_keycode(arg.trim()) {
                        commands.push(AutoCommand::Press(keycode));
                    } else {
                        eprintln!("Warning: unknown key '{}'", arg.trim());
                    }
                }
                "wait" => {
                    if let Ok(frames) = arg.trim().parse::<u32>() {
                        commands.push(AutoCommand::Wait(frames));
                    } else {
                        eprintln!("Warning: invalid wait frames '{}'", arg.trim());
                    }
                }
                _ => {
                    eprintln!("Warning: unknown command type '{}'", cmd_type.trim());
                }
            }
        } else if cmd_str == "screenshot" {
            commands.push(AutoCommand::Screenshot);
        } else {
            eprintln!("Warning: invalid command '{}'", cmd_str);
        }
    }
    commands
}
