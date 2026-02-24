use crate::content_registry::ContentRegistry;
use crate::shared::protocol::{ClientMessage, ServerMessage};
use crate::shared::voxel::{BaseWorldKind, BlockData};
use crate::shared::wasm::WasmPluginManager;
use polychora_plugin_api::content_ids;
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
                material: BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_GRID_FLOOR),
            },
            WorldGeneratorKind::MassivePlatforms => BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_GRID_FLOOR),
            },
        }
    }
}

/// Server runtime configuration.
///
/// `wasm_manager` is separate from the rest because it is moved into the
/// broadcast thread (not stored in `ServerState`). This keeps `WasmPluginManager`
/// out of the mutex and avoids split-borrow issues during mob simulation.
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
    /// WASM plugin manager for mob steering / ability evaluation.
    /// Taken during `initialize_state` and moved into the broadcast thread.
    pub wasm_manager: Option<WasmPluginManager>,
}

impl std::fmt::Debug for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("bind", &self.bind)
            .field("world_file", &self.world_file)
            .field("world_generator", &self.world_generator)
            .field("tick_hz", &self.tick_hz)
            .field("entity_sim_hz", &self.entity_sim_hz)
            .field("save_interval_secs", &self.save_interval_secs)
            .field("procgen_structures", &self.procgen_structures)
            .field("world_seed", &self.world_seed)
            .field("wasm_manager", &self.wasm_manager.as_ref().map(|_| ".."))
            .finish_non_exhaustive()
    }
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
            wasm_manager: None,
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}
