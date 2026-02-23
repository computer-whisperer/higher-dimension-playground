use crate::content_registry::ContentRegistry;
use crate::shared::protocol::{ClientMessage, ServerMessage};
use crate::shared::voxel::{BaseWorldKind, BlockData};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorldGeneratorKind {
    FlatFloor,
    MassivePlatforms,
}

impl WorldGeneratorKind {
    pub fn default_base_world_kind(self) -> BaseWorldKind {
        match self {
            WorldGeneratorKind::FlatFloor => BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            WorldGeneratorKind::MassivePlatforms => BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: String,
    pub world_file: PathBuf,
    pub world_generator: WorldGeneratorKind,
    pub tick_hz: f32,
    pub entity_sim_hz: f32,
    pub save_interval_secs: u64,
    pub procgen_structures: bool,
    pub procgen_near_chunk_radius: i32,
    pub procgen_mid_chunk_radius: i32,
    pub procgen_far_chunk_radius: i32,
    pub procgen_keepout_from_existing_world: bool,
    pub procgen_keepout_padding_chunks: i32,
    pub world_seed: u64,
    pub content_registry: Arc<ContentRegistry>,
}

impl RuntimeConfig {
    pub fn with_defaults(content_registry: Arc<ContentRegistry>) -> Self {
        Self {
            bind: "0.0.0.0:4000".to_string(),
            world_file: PathBuf::from("saves/world"),
            world_generator: WorldGeneratorKind::FlatFloor,
            tick_hz: 10.0,
            entity_sim_hz: 30.0,
            save_interval_secs: 5,
            procgen_structures: true,
            procgen_near_chunk_radius: 6,
            procgen_mid_chunk_radius: 10,
            procgen_far_chunk_radius: 6,
            procgen_keepout_from_existing_world: true,
            procgen_keepout_padding_chunks: 1,
            world_seed: 1337,
            content_registry,
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}
