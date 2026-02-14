mod procgen;

use crate::shared::protocol::{
    ClientMessage, EntityKind, EntitySnapshot, PlayerSnapshot, ServerMessage, WorldChunkPayload,
    WorldSnapshotPayload, WorldSummary,
};
use crate::shared::voxel::{
    self, load_world, save_world, BaseWorldKind, ChunkPos, VoxelType, VoxelWorld, CHUNK_SIZE,
};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: String,
    pub world_file: PathBuf,
    pub tick_hz: f32,
    pub save_interval_secs: u64,
    pub snapshot_on_join: bool,
    pub procgen_structures: bool,
    pub procgen_chunk_radius: i32,
    pub world_seed: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:4000".to_string(),
            world_file: PathBuf::from("saves/world.v4dw"),
            tick_hz: 10.0,
            save_interval_secs: 5,
            snapshot_on_join: true,
            procgen_structures: true,
            procgen_chunk_radius: 6,
            world_seed: 1337,
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}

#[derive(Clone, Debug)]
struct PlayerState {
    client_id: u64,
    name: String,
    position: [f32; 4],
    look: [f32; 4],
    last_update_ms: u64,
}

#[derive(Clone, Debug)]
struct EntityState {
    entity_id: u64,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    last_update_ms: u64,
}

fn entity_snapshot(entity: &EntityState) -> EntitySnapshot {
    EntitySnapshot {
        entity_id: entity.entity_id,
        kind: entity.kind,
        position: entity.position,
        orientation: entity.orientation,
        scale: entity.scale,
        material: entity.material,
        last_update_ms: entity.last_update_ms,
    }
}

#[derive(Debug)]
struct ServerState {
    next_client_id: u64,
    next_entity_id: u64,
    world: VoxelWorld,
    world_revision: u64,
    generated_procgen_chunks: HashSet<ChunkPos>,
    players: HashMap<u64, PlayerState>,
    clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
    entities: HashMap<u64, EntityState>,
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

fn world_chunk_from_position(position: [f32; 4]) -> [i32; 4] {
    let cs = CHUNK_SIZE as i32;
    [
        (position[0].floor() as i32).div_euclid(cs),
        (position[1].floor() as i32).div_euclid(cs),
        (position[2].floor() as i32).div_euclid(cs),
        (position[3].floor() as i32).div_euclid(cs),
    ]
}

fn encode_world_chunk_payload(chunk_pos: ChunkPos, chunk: &voxel::Chunk) -> WorldChunkPayload {
    WorldChunkPayload {
        chunk_pos: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        voxels: chunk.voxels.iter().map(|v| v.0).collect(),
    }
}

fn generate_procgen_chunks_around(
    state: &mut ServerState,
    center_chunk: [i32; 4],
    world_seed: u64,
    chunk_radius: i32,
) -> Option<(u64, Vec<WorldChunkPayload>)> {
    let radius = chunk_radius.max(0);
    let (min_structure_chunk_y, max_structure_chunk_y) = procgen::structure_chunk_y_bounds();
    let mut chunk_payloads = Vec::new();

    for chunk_x in (center_chunk[0] - radius)..=(center_chunk[0] + radius) {
        for chunk_z in (center_chunk[2] - radius)..=(center_chunk[2] + radius) {
            for chunk_w in (center_chunk[3] - radius)..=(center_chunk[3] + radius) {
                for chunk_y in min_structure_chunk_y..=max_structure_chunk_y {
                    let chunk_pos = ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w);
                    if !state.generated_procgen_chunks.insert(chunk_pos) {
                        continue;
                    }
                    if state.world.chunks.contains_key(&chunk_pos) {
                        continue;
                    }

                    let Some(chunk) = procgen::generate_structure_chunk(world_seed, chunk_pos)
                    else {
                        continue;
                    };

                    let payload = encode_world_chunk_payload(chunk_pos, &chunk);
                    state.world.insert_chunk(chunk_pos, chunk);
                    chunk_payloads.push(payload);
                }
            }
        }
    }

    if chunk_payloads.is_empty() {
        return None;
    }

    state.world_revision = state.world_revision.wrapping_add(1);
    Some((state.world_revision, chunk_payloads))
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
        return Ok(VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        }));
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
        bytes,
    })
}

fn start_broadcast_thread(
    state: SharedState,
    tick_hz: f32,
    start: Instant,
    shutdown: Arc<AtomicBool>,
) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let (players, entities) = {
                let guard = state.lock().expect("server state lock poisoned");
                let players = if guard.players.is_empty() {
                    None
                } else {
                    let mut all: Vec<_> = guard.players.values().map(player_snapshot).collect();
                    all.sort_by_key(|p| p.client_id);
                    Some(all)
                };
                let entities = if guard.entities.is_empty() {
                    None
                } else {
                    let mut all: Vec<_> = guard.entities.values().map(entity_snapshot).collect();
                    all.sort_by_key(|e| e.entity_id);
                    Some(all)
                };
                (players, entities)
            };
            let now = monotonic_ms(start);
            if let Some(players) = players {
                broadcast(
                    &state,
                    ServerMessage::PlayerPositions {
                        server_time_ms: now,
                        players,
                    },
                );
            }
            if let Some(entities) = entities {
                broadcast(
                    &state,
                    ServerMessage::EntityPositions {
                        server_time_ms: now,
                        entities,
                    },
                );
            }
        }
    });
}

fn start_autosave_thread(
    state: SharedState,
    world_file: PathBuf,
    save_interval_secs: u64,
    shutdown: Arc<AtomicBool>,
) {
    if save_interval_secs == 0 {
        return;
    }

    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(save_interval_secs));
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
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
    procgen_structures: bool,
    procgen_chunk_radius: i32,
    world_seed: u64,
    start: Instant,
) {
    match message {
        ClientMessage::Hello { name } => {
            let snapshot =
                install_or_update_player(state, client_id, Some(name), None, None, start);
            if procgen_structures {
                let generated = {
                    let mut guard = state.lock().expect("server state lock poisoned");
                    let center_chunk = world_chunk_from_position(snapshot.position);
                    generate_procgen_chunks_around(
                        &mut guard,
                        center_chunk,
                        world_seed,
                        procgen_chunk_radius,
                    )
                };
                if let Some((revision, chunks)) = generated {
                    broadcast(state, ServerMessage::WorldChunkBatch { revision, chunks });
                }
            }
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
            {
                let guard = state.lock().expect("server state lock poisoned");
                for entity in guard.entities.values() {
                    let _ = guard.clients.get(&client_id).map(|tx| {
                        let _ = tx.send(ServerMessage::EntitySpawned {
                            entity: entity_snapshot(entity),
                        });
                    });
                }
            }
            broadcast(state, ServerMessage::PlayerJoined { player: snapshot });
        }
        ClientMessage::UpdatePlayer { position, look } => {
            install_or_update_player(state, client_id, None, Some(position), Some(look), start);
            if procgen_structures {
                let generated = {
                    let mut guard = state.lock().expect("server state lock poisoned");
                    let center_chunk = world_chunk_from_position(position);
                    generate_procgen_chunks_around(
                        &mut guard,
                        center_chunk,
                        world_seed,
                        procgen_chunk_radius,
                    )
                };
                if let Some((revision, chunks)) = generated {
                    broadcast(state, ServerMessage::WorldChunkBatch { revision, chunks });
                }
            }
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
                let (chunk_pos, _) =
                    voxel::world_to_chunk(position[0], position[1], position[2], position[3]);
                guard.generated_procgen_chunks.insert(chunk_pos);
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
            if procgen_structures {
                let generated = {
                    let mut guard = state.lock().expect("server state lock poisoned");
                    let center_chunk = guard
                        .players
                        .get(&client_id)
                        .map(|player| world_chunk_from_position(player.position))
                        .unwrap_or([0, 0, 0, 0]);
                    generate_procgen_chunks_around(
                        &mut guard,
                        center_chunk,
                        world_seed,
                        procgen_chunk_radius,
                    )
                };
                if let Some((revision, chunks)) = generated {
                    broadcast(state, ServerMessage::WorldChunkBatch { revision, chunks });
                }
            }
            let snapshot_message = {
                let guard = state.lock().expect("server state lock poisoned");
                build_world_snapshot_payload(&guard)
            };
            match snapshot_message {
                Ok(world) => send_to_client(state, client_id, ServerMessage::WorldSnapshot { world }),
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
    procgen_structures: bool,
    procgen_chunk_radius: i32,
    world_seed: u64,
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
            let Ok(encoded) = postcard::to_stdvec(&message) else {
                continue;
            };
            let len = (encoded.len() as u32).to_le_bytes();
            if writer.write_all(&len).is_err() {
                break;
            }
            if writer.write_all(&encoded).is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    thread::spawn(move || {
        eprintln!("client {} connected from {}", client_id, peer_label);
        let mut reader = stream;
        let mut len_buf = [0u8; 4];

        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 100_000_000 {
                eprintln!("client {} sent oversized message ({} bytes)", client_id, len);
                break;
            }

            let mut msg_buf = vec![0u8; len];
            match reader.read_exact(&mut msg_buf) {
                Ok(()) => {}
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let parsed = postcard::from_bytes::<ClientMessage>(&msg_buf);
            match parsed {
                Ok(message) => {
                    handle_message(
                        &state,
                        client_id,
                        message,
                        snapshot_on_join,
                        tick_hz,
                        procgen_structures,
                        procgen_chunk_radius,
                        world_seed,
                        start,
                    );
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

fn spawn_entity(
    state: &SharedState,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    start: Instant,
) -> u64 {
    let mut guard = state.lock().expect("server state lock poisoned");
    let entity_id = guard.next_entity_id;
    guard.next_entity_id = guard.next_entity_id.wrapping_add(1).max(1);
    let entity = EntityState {
        entity_id,
        kind,
        position,
        orientation,
        scale,
        material,
        last_update_ms: monotonic_ms(start),
    };
    guard.entities.insert(entity_id, entity);
    entity_id
}

fn spawn_default_test_entities(state: &SharedState, start: Instant) {
    let test_positions: [[f32; 4]; 3] = [
        [3.0, 2.0, 3.0, 0.0],
        [-2.0, 1.5, 5.0, 1.0],
        [0.0, 3.0, -4.0, 2.0],
    ];
    let test_materials: [u8; 3] = [12, 8, 15];
    for (i, (pos, mat)) in test_positions.iter().zip(test_materials.iter()).enumerate() {
        let id = spawn_entity(
            state,
            EntityKind::TestCube,
            *pos,
            [0.0, 0.0, 1.0, 0.0],
            0.5,
            *mat,
            start,
        );
        eprintln!("spawned test entity {} (id={}) at {:?}", i, id, pos);
    }
}

fn initialize_state(
    config: &RuntimeConfig,
    shutdown: Arc<AtomicBool>,
) -> io::Result<(SharedState, Instant)> {
    let start = Instant::now();
    let initial_world = load_world_from_path(&config.world_file)?;
    let generated_procgen_chunks: HashSet<_> = initial_world.chunks.keys().copied().collect();
    let initial_chunks = initial_world.non_empty_chunk_count();
    eprintln!(
        "loaded world {} ({} non-empty chunks)",
        config.world_file.display(),
        initial_chunks
    );

    let state = Arc::new(Mutex::new(ServerState {
        next_client_id: 1,
        next_entity_id: 1,
        world: initial_world,
        world_revision: 0,
        generated_procgen_chunks,
        players: HashMap::new(),
        clients: HashMap::new(),
        entities: HashMap::new(),
    }));

    spawn_default_test_entities(&state, start);
    start_broadcast_thread(state.clone(), config.tick_hz, start, shutdown.clone());
    start_autosave_thread(
        state.clone(),
        config.world_file.clone(),
        config.save_interval_secs,
        shutdown,
    );

    Ok((state, start))
}

pub fn connect_local_client(config: &RuntimeConfig) -> io::Result<LocalConnection> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown.clone())?;
    let (client_to_server_tx, client_to_server_rx) = mpsc::channel::<ClientMessage>();
    let (server_to_client_tx, server_to_client_rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = guard.next_client_id;
        guard.next_client_id = guard.next_client_id.wrapping_add(1).max(1);
        guard.clients.insert(id, server_to_client_tx);
        id
    };

    let state_for_client = state.clone();
    let cfg = config.clone();
    thread::spawn(move || {
        while let Ok(message) = client_to_server_rx.recv() {
            handle_message(
                &state_for_client,
                client_id,
                message,
                cfg.snapshot_on_join,
                cfg.tick_hz.max(0.1),
                cfg.procgen_structures,
                cfg.procgen_chunk_radius,
                cfg.world_seed,
                start,
            );
        }
        remove_client(&state_for_client, client_id);
        shutdown.store(true, Ordering::Relaxed);
    });

    Ok(LocalConnection {
        outgoing: client_to_server_tx,
        incoming: server_to_client_rx,
    })
}

pub fn run_tcp_server(config: &RuntimeConfig) -> io::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown)?;

    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, autosave {}s, procgen={}, seed={}, radius={} chunks)",
        config.bind,
        config.tick_hz.max(0.1),
        config.save_interval_secs,
        config.procgen_structures,
        config.world_seed,
        config.procgen_chunk_radius.max(0),
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                spawn_client_thread(
                    stream,
                    state.clone(),
                    config.snapshot_on_join,
                    config.tick_hz.max(0.1),
                    config.procgen_structures,
                    config.procgen_chunk_radius,
                    config.world_seed,
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
