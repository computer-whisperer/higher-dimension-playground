use crate::shared::region_tree::RegionTreeCore;
use crate::shared::spatial::Aabb4i;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    PlayerAvatar,
    TestCube,
    TestRotor,
    TestDrifter,
    MobSeeker,
    MobCreeper4d,
    MobPhaseSpider,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityClass {
    Player,
    Accent,
    Mob,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub entity_id: u64,
    pub class: EntityClass,
    pub kind: EntityKind,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub owner_client_id: Option<u64>,
    pub display_name: Option<String>,
    pub last_update_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityTransform {
    pub entity_id: u64,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub last_update_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSummary {
    pub non_empty_chunks: usize,
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
    },
    SpawnEntity {
        kind: EntityKind,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
        material: u8,
    },
    ConsoleCommand {
        command: String,
    },
    // Requested authoritative coverage in chunk space. This is a coverage contract,
    // not a strict payload clipping request: server subtree patches may extend
    // outside this AABB when doing so preserves larger canonical leaves.
    WorldInterestUpdate {
        bounds: Aabb4i,
    },
    WorldChunkSampleRequest {
        request_id: u64,
        chunk: [i32; 4],
    },
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
    // Subtree patch to apply into client world cache. `subtree.bounds` may be
    // larger than the client's requested interest AABB in order to avoid
    // fragmenting canonical leaves (for example, large Uniform spans).
    WorldSubtreePatch {
        subtree: RegionTreeCore,
    },
    WorldChunkSampleResponse {
        request_id: u64,
        chunk: [i32; 4],
        dense_materials: Vec<u16>,
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
    EntityTransforms {
        server_time_ms: u64,
        entities: Vec<EntityTransform>,
    },
    Explosion {
        position: [f32; 4],
        radius: f32,
        source_entity_id: Option<u64>,
    },
    PlayerMovementModifier {
        delta_position: [f32; 4],
        delta_velocity_y: f32,
        source_entity_id: Option<u64>,
    },
}
