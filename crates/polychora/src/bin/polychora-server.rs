use clap::Parser;
use polychora::server::{run_tcp_server, RuntimeConfig};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "polychora-server",
    about = "Multiplayer/server runtime for Polychora"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0:4000")]
    bind: String,
    #[arg(long, default_value = "saves/world.v4dw")]
    world_file: PathBuf,
    #[arg(long, default_value_t = 10.0)]
    tick_hz: f32,
    #[arg(long, default_value_t = 30.0)]
    entity_sim_hz: f32,
    #[arg(long, default_value_t = 5)]
    save_interval_secs: u64,
    #[arg(long, default_value_t = true)]
    snapshot_on_join: bool,
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

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let config = RuntimeConfig {
        bind: args.bind,
        world_file: args.world_file,
        tick_hz: args.tick_hz,
        entity_sim_hz: args.entity_sim_hz,
        save_interval_secs: args.save_interval_secs,
        snapshot_on_join: args.snapshot_on_join,
        procgen_structures: args.procgen_structures,
        procgen_near_chunk_radius: args.procgen_chunk_radius,
        procgen_mid_chunk_radius: args.procgen_mid_chunk_radius,
        procgen_far_chunk_radius: args.procgen_far_chunk_radius,
        procgen_keepout_from_existing_world: args.procgen_keepout_from_existing_world,
        procgen_keepout_padding_chunks: args.procgen_keepout_padding_chunks,
        world_seed: args.world_seed,
    };
    run_tcp_server(&config)
}
