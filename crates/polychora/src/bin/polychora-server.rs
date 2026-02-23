use clap::Parser;
use polychora::server::{run_tcp_server, RuntimeConfig, WorldGeneratorKind};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "polychora-server",
    about = "Multiplayer/server runtime for Polychora"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0:4000")]
    bind: String,
    #[arg(long, default_value = "saves/world")]
    world_file: PathBuf,
    #[arg(long, value_enum, default_value_t = WorldGeneratorArg::Flat)]
    world_generator: WorldGeneratorArg,
    #[arg(long, default_value_t = 10.0)]
    tick_hz: f32,
    #[arg(long, default_value_t = 30.0)]
    entity_sim_hz: f32,
    #[arg(long, default_value_t = 5)]
    save_interval_secs: u64,
    #[arg(long, default_value_t = true)]
    procgen_structures: bool,
    #[arg(long, default_value_t = 6)]
    procgen_chunk_radius: i32,
    #[arg(long, default_value_t = 10)]
    procgen_mid_chunk_radius: i32,
    #[arg(long, default_value_t = 6)]
    procgen_far_chunk_radius: i32,
    #[arg(long, default_value_t = true)]
    procgen_keepout_from_existing_world: bool,
    #[arg(long, default_value_t = 1)]
    procgen_keepout_padding_chunks: i32,
    #[arg(long, default_value_t = 1337)]
    world_seed: u64,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum WorldGeneratorArg {
    Flat,
    MassivePlatforms,
}

impl WorldGeneratorArg {
    fn to_runtime(self) -> WorldGeneratorKind {
        match self {
            WorldGeneratorArg::Flat => WorldGeneratorKind::FlatFloor,
            WorldGeneratorArg::MassivePlatforms => WorldGeneratorKind::MassivePlatforms,
        }
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let content_registry = Arc::new(polychora::plugin_loader::create_full_registry());
    let config = RuntimeConfig {
        bind: args.bind,
        world_file: args.world_file,
        world_generator: args.world_generator.to_runtime(),
        tick_hz: args.tick_hz,
        entity_sim_hz: args.entity_sim_hz,
        save_interval_secs: args.save_interval_secs,
        procgen_structures: args.procgen_structures,
        procgen_near_chunk_radius: args.procgen_chunk_radius,
        procgen_mid_chunk_radius: args.procgen_mid_chunk_radius,
        procgen_far_chunk_radius: args.procgen_far_chunk_radius,
        procgen_keepout_from_existing_world: args.procgen_keepout_from_existing_world,
        procgen_keepout_padding_chunks: args.procgen_keepout_padding_chunks,
        world_seed: args.world_seed,
        content_registry,
    };
    run_tcp_server(&config)
}
