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

pub(super) fn run_cpu_render(_scene_preset: ScenePreset, args: &Args) {
    use common::MatN;

    if args.load_world {
        eprintln!(
            "--load-world is ignored for client-side CPU render; run the integrated server to load world data."
        );
    }
    let mut camera = Camera4D::new();

    // Debug camera: specific position/orientation where GPU renders incorrectly
    camera.position = [5.44, 0.47, -1.23, -4.00];
    camera.yaw = -0.49;
    camera.pitch = 0.0;
    camera.xw_angle = 0.58;
    camera.zw_angle = 0.65;

    if let Some(pos) = args.screenshot_pos.as_ref() {
        if pos.len() == 4 {
            camera.position = [pos[0], pos[1], pos[2], pos[3]];
        }
    }
    if let Some(angles) = args.screenshot_angles.as_ref() {
        if angles.len() == 4 {
            camera.yaw = angles[0];
            camera.pitch = angles[1];
            camera.xw_angle = angles[2];
            camera.zw_angle = angles[3];
        }
    } else if let Some(angles_deg) = args.screenshot_angles_deg.as_ref() {
        if angles_deg.len() == 4 {
            camera.yaw = angles_deg[0].to_radians();
            camera.pitch = angles_deg[1].to_radians();
            camera.xw_angle = angles_deg[2].to_radians();
            camera.zw_angle = angles_deg[3].to_radians();
        }
    }
    if let Some(yw) = args.screenshot_yw {
        camera.yw_deviation = yw;
    }

    eprintln!("CPU world-geometry raster path removed; rendering empty tetra scene.");
    let instances: Vec<common::ModelInstance> = Vec::new();

    let view_matrix_ndarray = camera.view_matrix();
    let view_matrix: MatN<5> = MatN::from(&view_matrix_ndarray);

    let model_tets = higher_dimension_playground::render::generate_tesseract_tetrahedrons();

    let params = cpu_render::CpuRenderParams {
        view_matrix,
        focal_length_xy: 1.0,
        focal_length_zw: 1.0,
        width: args.width.max(16),
        height: args.height.max(16),
        ..Default::default()
    };

    eprintln!("CPU render: {}x{}", params.width, params.height);
    let start = Instant::now();
    let content_registry = polychora::plugin_loader::create_full_registry();
    let img = cpu_render::cpu_render(&instances, &model_tets, &params, &content_registry);
    let elapsed = start.elapsed();
    eprintln!("CPU render done in {:.2}s", elapsed.as_secs_f32());

    let _ = std::fs::create_dir_all("frames");
    img.save("frames/cpu_render.png").unwrap();
    eprintln!("Saved frames/cpu_render.png");
}
