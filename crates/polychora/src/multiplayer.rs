use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSummary {
    pub non_empty_chunks: usize,
    pub revision: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSnapshotPayload {
    pub format: String,
    pub non_empty_chunks: usize,
    pub revision: u64,
    pub bytes_base64: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerSnapshot {
    pub client_id: u64,
    pub name: String,
    pub position: [f32; 4],
    pub look: [f32; 4],
    pub last_update_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Hello { name: String },
    UpdatePlayer { position: [f32; 4], look: [f32; 4] },
    SetVoxel { position: [i32; 4], material: u8 },
    RequestWorldSnapshot,
    Ping { nonce: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Welcome {
        client_id: u64,
        server_time_ms: u64,
        tick_hz: f32,
        world: WorldSummary,
    },
    Error {
        message: String,
    },
    PlayerJoined {
        player: PlayerSnapshot,
    },
    PlayerLeft {
        client_id: u64,
    },
    PlayerPositions {
        server_time_ms: u64,
        players: Vec<PlayerSnapshot>,
    },
    WorldVoxelSet {
        position: [i32; 4],
        material: u8,
        source_client_id: Option<u64>,
        revision: u64,
    },
    WorldSnapshot {
        world: WorldSnapshotPayload,
    },
    Pong {
        nonce: u64,
    },
}

#[derive(Debug)]
pub enum MultiplayerEvent {
    Message(ServerMessage),
    Disconnected(String),
}

pub struct MultiplayerClient {
    server_addr: String,
    outgoing: mpsc::Sender<ClientMessage>,
    incoming: mpsc::Receiver<MultiplayerEvent>,
}

impl MultiplayerClient {
    pub fn connect(server_addr: String, player_name: String) -> io::Result<Self> {
        let stream = TcpStream::connect(&server_addr)?;
        let _ = stream.set_nodelay(true);
        let writer_stream = stream.try_clone()?;

        let (outgoing_tx, outgoing_rx) = mpsc::channel::<ClientMessage>();
        let (incoming_tx, incoming_rx) = mpsc::channel::<MultiplayerEvent>();

        {
            let incoming_tx = incoming_tx.clone();
            thread::spawn(move || {
                let mut writer = BufWriter::new(writer_stream);
                while let Ok(message) = outgoing_rx.recv() {
                    let serialized = match serde_json::to_string(&message) {
                        Ok(v) => v,
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "serialize error: {error}"
                            )));
                            break;
                        }
                    };

                    if writeln!(writer, "{serialized}").is_err() {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed while writing".to_string(),
                        ));
                        break;
                    }
                    if writer.flush().is_err() {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed while flushing".to_string(),
                        ));
                        break;
                    }
                }
            });
        }

        {
            let incoming_tx = incoming_tx.clone();
            thread::spawn(move || {
                let mut reader = BufReader::new(stream);
                let mut line = String::new();
                loop {
                    line.clear();
                    let count = match reader.read_line(&mut line) {
                        Ok(n) => n,
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "read error: {error}"
                            )));
                            break;
                        }
                    };
                    if count == 0 {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed".to_string(),
                        ));
                        break;
                    }

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<ServerMessage>(trimmed) {
                        Ok(message) => {
                            if incoming_tx
                                .send(MultiplayerEvent::Message(message))
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "parse error: {error}"
                            )));
                            break;
                        }
                    }
                }
            });
        }

        let client = Self {
            server_addr,
            outgoing: outgoing_tx,
            incoming: incoming_rx,
        };
        client.send(ClientMessage::Hello { name: player_name });
        Ok(client)
    }

    pub fn send(&self, message: ClientMessage) {
        let _ = self.outgoing.send(message);
    }

    pub fn try_recv(&self) -> Option<MultiplayerEvent> {
        self.incoming.try_recv().ok()
    }

    pub fn server_addr(&self) -> &str {
        &self.server_addr
    }
}
