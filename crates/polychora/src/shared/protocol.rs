use serde::{Deserialize, Serialize};

pub const WORLD_CHUNK_LOD_NEAR: u8 = 0;
pub const WORLD_CHUNK_LOD_MID: u8 = 1;
pub const WORLD_CHUNK_LOD_FAR: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    TestCube,
    TestRotor,
    TestDrifter,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub entity_id: u64,
    pub kind: EntityKind,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub last_update_ms: u64,
}

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
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldChunkPayload {
    pub lod_level: u8,
    pub chunk_pos: [i32; 4],
    pub voxels: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WorldChunkCoordPayload {
    pub lod_level: u8,
    pub chunk_pos: [i32; 4],
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
pub enum ClientMessage {
    Hello {
        name: String,
    },
    UpdatePlayer {
        position: [f32; 4],
        look: [f32; 4],
    },
    SetVoxel {
        position: [i32; 4],
        material: u8,
        client_edit_id: Option<u64>,
    },
    RequestWorldSnapshot,
    Ping {
        nonce: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        client_edit_id: Option<u64>,
        revision: u64,
    },
    WorldSnapshot {
        world: WorldSnapshotPayload,
    },
    WorldChunkBatch {
        revision: u64,
        chunks: Vec<WorldChunkPayload>,
    },
    WorldChunkUnloadBatch {
        revision: u64,
        chunks: Vec<WorldChunkCoordPayload>,
    },
    Pong {
        nonce: u64,
    },
    EntitySpawned {
        entity: EntitySnapshot,
    },
    EntityDestroyed {
        entity_id: u64,
    },
    EntityPositions {
        server_time_ms: u64,
        entities: Vec<EntitySnapshot>,
    },
}
