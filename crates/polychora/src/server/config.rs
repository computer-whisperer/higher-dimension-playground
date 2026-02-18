use crate::shared::protocol::{ClientMessage, ServerMessage};
use std::path::PathBuf;
use std::sync::mpsc;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: String,
    pub world_file: PathBuf,
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
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:4000".to_string(),
            world_file: PathBuf::from("saves/world"),
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
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}
