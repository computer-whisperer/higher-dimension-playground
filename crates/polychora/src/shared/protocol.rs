use crate::shared::region_tree::RegionTreeCore;
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::BlockData;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Entity pose — spatial state shared by all entities
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityPose {
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 4],
    pub scale: f32,
}

impl Default for EntityPose {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            orientation: [0.0, 0.0, 1.0, 0.0],
            velocity: [0.0; 4],
            scale: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Entity — namespace + type + pose + opaque data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub namespace: u32,
    pub entity_type: u32,
    pub pose: EntityPose,
    pub data: Vec<u8>,
}

impl Entity {
    pub fn simple(namespace: u32, entity_type: u32) -> Self {
        Self {
            namespace,
            entity_type,
            pose: EntityPose::default(),
            data: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Item / ItemStack
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub namespace: u32,
    pub item_type: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemStack {
    pub item: Item,
    pub count: u32,
}

// ---------------------------------------------------------------------------
// Wire types — entity replication
// ---------------------------------------------------------------------------

/// Full entity state sent when an entity first enters a client's interest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub entity_id: u64,
    pub entity: Entity,
    pub owner_client_id: Option<u64>,
    pub display_name: Option<String>,
    pub last_update_ms: u64,
}

/// Pose-only update sent every tick for visible entities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityTransform {
    pub entity_id: u64,
    pub pose: EntityPose,
    pub last_update_ms: u64,
}

// ---------------------------------------------------------------------------
// World metadata
// ---------------------------------------------------------------------------

/// Per-axis hard collision boundaries. `None` means unbounded in that direction.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WorldBounds {
    pub min: [Option<f32>; 4], // X, Y, Z, W lower bounds
    pub max: [Option<f32>; 4], // X, Y, Z, W upper bounds
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSummary {
    pub non_empty_chunks: usize,
    pub bounds: WorldBounds,
}

// ---------------------------------------------------------------------------
// Client → Server messages
// ---------------------------------------------------------------------------

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
        block: BlockData,
    },
    SpawnEntity {
        entity_type_namespace: u32,
        entity_type: u32,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
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

// ---------------------------------------------------------------------------
// Server → Client messages
// ---------------------------------------------------------------------------

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
    // Subtree patch to apply into client world cache.
    //
    // `authoritative_bounds` is the exact region where this patch is a
    // coverage contract. Inside that AABB, the client may treat both presence
    // and absence as authoritative.
    //
    // `subtree.bounds` may be larger than `authoritative_bounds` in order to
    // avoid fragmenting canonical leaves (for example, large Uniform spans).
    // Outside `authoritative_bounds`, subtree data is advisory: present leaves
    // are valid terrain data, but absent leaves must not be interpreted as
    // "known empty coverage".
    WorldSubtreePatch {
        authoritative_bounds: Aabb4i,
        subtree: RegionTreeCore,
    },
    WorldChunkSampleResponse {
        request_id: u64,
        chunk: [i32; 4],
        dense_blocks: Vec<BlockData>,
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

