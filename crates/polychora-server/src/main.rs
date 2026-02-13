mod protocol;
mod voxel;

use base64::Engine;
use clap::Parser;
use protocol::{ClientMessage, PlayerSnapshot, ServerMessage, WorldSnapshotPayload, WorldSummary};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use voxel::{load_world, save_world, VoxelType, VoxelWorld};

#[derive(Parser, Debug)]
#[command(
    name = "polychora-server",
    about = "Basic multiplayer state server for Polychora"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0:4000")]
    bind: String,
    #[arg(long, default_value = "saves/world.v4dw")]
    world_file: PathBuf,
    #[arg(long, default_value_t = 10.0)]
    tick_hz: f32,
    #[arg(long, default_value_t = 5)]
    save_interval_secs: u64,
    #[arg(long, default_value_t = true)]
    snapshot_on_join: bool,
}

#[derive(Clone, Debug)]
struct PlayerState {
    client_id: u64,
    name: String,
    position: [f32; 4],
    look: [f32; 4],
    last_update_ms: u64,
}

#[derive(Debug)]
struct ServerState {
    next_client_id: u64,
    world: VoxelWorld,
    world_revision: u64,
    players: HashMap<u64, PlayerState>,
    clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
}

type SharedState = Arc<Mutex<ServerState>>;

fn monotonic_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn sanitize_player_name(name: &str, client_id: u64) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("player-{client_id}");
    }
    trimmed.chars().take(32).collect()
}

fn player_snapshot(player: &PlayerState) -> PlayerSnapshot {
    PlayerSnapshot {
        client_id: player.client_id,
        name: player.name.clone(),
        position: player.position,
        look: player.look,
        last_update_ms: player.last_update_ms,
    }
}

fn send_to_client(state: &SharedState, client_id: u64, message: ServerMessage) {
    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    if let Some(tx) = sender {
        let _ = tx.send(message);
    }
}

fn broadcast(state: &SharedState, message: ServerMessage) {
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .map(|(&client_id, tx)| (client_id, tx.clone()))
            .collect()
    };

    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    if !stale.is_empty() {
        let mut guard = state.lock().expect("server state lock poisoned");
        for client_id in stale {
            guard.clients.remove(&client_id);
            guard.players.remove(&client_id);
        }
    }
}

fn load_world_from_path(path: &Path) -> io::Result<VoxelWorld> {
    if !path.exists() {
        return Ok(VoxelWorld::new());
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_world(&mut reader)
}

fn save_world_to_path(path: &Path, world: &VoxelWorld) -> io::Result<usize> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_world(world, &mut writer)?;
    writer.flush()?;
    Ok(world.non_empty_chunk_count())
}

fn build_world_snapshot_payload(state: &ServerState) -> io::Result<WorldSnapshotPayload> {
    let mut bytes = Vec::new();
    save_world(&state.world, &mut bytes)?;
    Ok(WorldSnapshotPayload {
        format: "v4dw".to_string(),
        non_empty_chunks: state.world.non_empty_chunk_count(),
        revision: state.world_revision,
        bytes_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn start_broadcast_thread(state: SharedState, tick_hz: f32, start: Instant) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    thread::spawn(move || loop {
        thread::sleep(interval);
        let players = {
            let guard = state.lock().expect("server state lock poisoned");
            if guard.players.is_empty() {
                None
            } else {
                let mut all: Vec<_> = guard.players.values().map(player_snapshot).collect();
                all.sort_by_key(|p| p.client_id);
                Some(all)
            }
        };
        if let Some(players) = players {
            broadcast(
                &state,
                ServerMessage::PlayerPositions {
                    server_time_ms: monotonic_ms(start),
                    players,
                },
            );
        }
    });
}

fn start_autosave_thread(state: SharedState, world_file: PathBuf, save_interval_secs: u64) {
    if save_interval_secs == 0 {
        return;
    }

    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(save_interval_secs));
        let save_result = {
            let guard = state.lock().expect("server state lock poisoned");
            save_world_to_path(&world_file, &guard.world)
                .map(|chunks| (chunks, guard.world_revision))
        };

        match save_result {
            Ok((chunk_count, revision)) => {
                eprintln!(
                    "autosave: world revision {} ({} non-empty chunks) -> {}",
                    revision,
                    chunk_count,
                    world_file.display()
                );
            }
            Err(error) => {
                eprintln!("autosave failed ({}): {}", world_file.display(), error);
            }
        }
    });
}

fn remove_client(state: &SharedState, client_id: u64) {
    let had_player = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let _ = guard.clients.remove(&client_id);
        guard.players.remove(&client_id).is_some()
    };
    if had_player {
        broadcast(state, ServerMessage::PlayerLeft { client_id });
    }
}

fn install_or_update_player(
    state: &SharedState,
    client_id: u64,
    name: Option<String>,
    position: Option<[f32; 4]>,
    look: Option<[f32; 4]>,
    start: Instant,
) -> PlayerSnapshot {
    let now = monotonic_ms(start);
    let mut guard = state.lock().expect("server state lock poisoned");
    let player = guard
        .players
        .entry(client_id)
        .or_insert_with(|| PlayerState {
            client_id,
            name: format!("player-{client_id}"),
            position: [0.0, 0.0, 0.0, 0.0],
            look: [
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            last_update_ms: now,
        });

    if let Some(n) = name {
        player.name = sanitize_player_name(&n, client_id);
    }
    if let Some(pos) = position {
        player.position = pos;
    }
    if let Some(dir) = look {
        player.look = dir;
    }
    player.last_update_ms = now;
    player_snapshot(player)
}

fn handle_message(
    state: &SharedState,
    client_id: u64,
    message: ClientMessage,
    snapshot_on_join: bool,
    tick_hz: f32,
    start: Instant,
) {
    match message {
        ClientMessage::Hello { name } => {
            let snapshot =
                install_or_update_player(state, client_id, Some(name), None, None, start);
            send_to_client(
                state,
                client_id,
                ServerMessage::Welcome {
                    client_id,
                    server_time_ms: monotonic_ms(start),
                    tick_hz,
                    world: {
                        let guard = state.lock().expect("server state lock poisoned");
                        WorldSummary {
                            non_empty_chunks: guard.world.non_empty_chunk_count(),
                            revision: guard.world_revision,
                        }
                    },
                },
            );
            if snapshot_on_join {
                let snapshot_message = {
                    let guard = state.lock().expect("server state lock poisoned");
                    build_world_snapshot_payload(&guard)
                };
                match snapshot_message {
                    Ok(world) => {
                        send_to_client(state, client_id, ServerMessage::WorldSnapshot { world });
                    }
                    Err(error) => {
                        send_to_client(
                            state,
                            client_id,
                            ServerMessage::Error {
                                message: format!("failed to build world snapshot: {error}"),
                            },
                        );
                    }
                }
            }
            broadcast(state, ServerMessage::PlayerJoined { player: snapshot });
        }
        ClientMessage::UpdatePlayer { position, look } => {
            install_or_update_player(state, client_id, None, Some(position), Some(look), start);
        }
        ClientMessage::SetVoxel {
            position,
            material,
            client_edit_id,
        } => {
            let revision = {
                let mut guard = state.lock().expect("server state lock poisoned");
                guard.world.set_voxel(
                    position[0],
                    position[1],
                    position[2],
                    position[3],
                    VoxelType(material),
                );
                guard.world_revision = guard.world_revision.wrapping_add(1);
                guard.world_revision
            };

            broadcast(
                state,
                ServerMessage::WorldVoxelSet {
                    position,
                    material,
                    source_client_id: Some(client_id),
                    client_edit_id,
                    revision,
                },
            );
        }
        ClientMessage::RequestWorldSnapshot => {
            let snapshot_message = {
                let guard = state.lock().expect("server state lock poisoned");
                build_world_snapshot_payload(&guard)
            };
            match snapshot_message {
                Ok(world) => {
                    send_to_client(state, client_id, ServerMessage::WorldSnapshot { world })
                }
                Err(error) => send_to_client(
                    state,
                    client_id,
                    ServerMessage::Error {
                        message: format!("failed to build world snapshot: {error}"),
                    },
                ),
            }
        }
        ClientMessage::Ping { nonce } => {
            send_to_client(state, client_id, ServerMessage::Pong { nonce });
        }
    }
}

fn spawn_client_thread(
    stream: TcpStream,
    state: SharedState,
    snapshot_on_join: bool,
    tick_hz: f32,
    start: Instant,
) {
    let peer_label = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let (tx, rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = guard.next_client_id;
        guard.next_client_id = guard.next_client_id.wrapping_add(1).max(1);
        guard.clients.insert(id, tx.clone());
        id
    };

    let writer_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(error) => {
            eprintln!("failed to clone stream for {}: {}", peer_label, error);
            remove_client(&state, client_id);
            return;
        }
    };

    thread::spawn(move || {
        let mut writer = BufWriter::new(writer_stream);
        while let Ok(message) = rx.recv() {
            let Ok(serialized) = serde_json::to_string(&message) else {
                continue;
            };
            if writeln!(writer, "{serialized}").is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    thread::spawn(move || {
        eprintln!("client {} connected from {}", client_id, peer_label);
        let mut reader = BufReader::new(stream);
        let mut line = String::new();

        loop {
            line.clear();
            let count = match reader.read_line(&mut line) {
                Ok(n) => n,
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            };
            if count == 0 {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parsed = serde_json::from_str::<ClientMessage>(trimmed);
            match parsed {
                Ok(message) => {
                    handle_message(&state, client_id, message, snapshot_on_join, tick_hz, start);
                }
                Err(error) => {
                    send_to_client(
                        &state,
                        client_id,
                        ServerMessage::Error {
                            message: format!("invalid message: {error}"),
                        },
                    );
                }
            }
        }

        remove_client(&state, client_id);
        eprintln!("client {} disconnected", client_id);
    });
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let initial_world = load_world_from_path(&args.world_file)?;
    let initial_chunks = initial_world.non_empty_chunk_count();
    eprintln!(
        "loaded world {} ({} non-empty chunks)",
        args.world_file.display(),
        initial_chunks
    );

    let state = Arc::new(Mutex::new(ServerState {
        next_client_id: 1,
        world: initial_world,
        world_revision: 0,
        players: HashMap::new(),
        clients: HashMap::new(),
    }));

    start_broadcast_thread(state.clone(), args.tick_hz, start);
    start_autosave_thread(
        state.clone(),
        args.world_file.clone(),
        args.save_interval_secs,
    );

    let listener = TcpListener::bind(&args.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, autosave {}s)",
        args.bind,
        args.tick_hz.max(0.1),
        args.save_interval_secs
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                spawn_client_thread(
                    stream,
                    state.clone(),
                    args.snapshot_on_join,
                    args.tick_hz.max(0.1),
                    start,
                );
            }
            Err(error) => {
                eprintln!("accept failed: {}", error);
            }
        }
    }

    Ok(())
}
