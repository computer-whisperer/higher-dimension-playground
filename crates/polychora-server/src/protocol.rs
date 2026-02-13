use serde::{Deserialize, Serialize};

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
    SetVoxel {
        position: [i32; 4],
        material: u8,
        #[serde(default)]
        client_edit_id: Option<u64>,
    },
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
        #[serde(default)]
        client_edit_id: Option<u64>,
        revision: u64,
    },
    WorldSnapshot {
        world: WorldSnapshotPayload,
    },
    Pong {
        nonce: u64,
    },
}
