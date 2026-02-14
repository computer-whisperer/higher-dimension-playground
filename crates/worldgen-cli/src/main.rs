use clap::{Parser, Subcommand};
use polychora_common::voxel::{BaseWorldKind, VoxelType, VoxelWorld, save_world};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "worldgen-cli", about = "Generate .v4dw world files for polychora")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a flat floor world
    Flat {
        /// Material ID for the floor (0-255)
        #[arg(long)]
        material: u8,
        /// Output .v4dw file path
        #[arg(long, short)]
        output: PathBuf,
    },
    /// Generate the demo cube layout
    DemoCubes {
        /// Output .v4dw file path
        #[arg(long, short)]
        output: PathBuf,
    },
    /// Fill a 4D rectangular region with a material
    FilledRegion {
        /// Minimum corner (4 integers: X Y Z W)
        #[arg(long, num_args = 4, value_names = ["X", "Y", "Z", "W"], allow_hyphen_values = true)]
        min: Vec<i32>,
        /// Size of region (4 integers: X Y Z W)
        #[arg(long, num_args = 4, value_names = ["X", "Y", "Z", "W"])]
        size: Vec<i32>,
        /// Material ID to fill with (0-255)
        #[arg(long)]
        material: u8,
        /// Base world type: "flat" or "empty" (default: empty)
        #[arg(long, default_value = "empty")]
        base: String,
        /// Material ID for flat base floor (default: 11)
        #[arg(long, default_value_t = 11)]
        base_material: u8,
        /// Output .v4dw file path
        #[arg(long, short)]
        output: PathBuf,
    },
}

fn fill_hypercube(world: &mut VoxelWorld, min: [i32; 4], size: [i32; 4], material: VoxelType) {
    for x in min[0]..(min[0] + size[0]) {
        for y in min[1]..(min[1] + size[1]) {
            for z in min[2]..(min[2] + size[2]) {
                for w in min[3]..(min[3] + size[3]) {
                    world.set_voxel(x, y, z, w, material);
                }
            }
        }
    }
}

fn generate_demo_cube_layout_world() -> VoxelWorld {
    let mut world = VoxelWorld::new();
    let mut texture_rot = 0u8;

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                for w in 0..2 {
                    let base = [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2];
                    let material = VoxelType((texture_rot % 5) + 1);
                    fill_hypercube(&mut world, base, [2, 2, 2, 2], material);
                    texture_rot = (texture_rot + 1) % 5;
                }
            }
        }
    }

    // Central bright cube
    fill_hypercube(&mut world, [0, 0, 0, 0], [2, 2, 2, 2], VoxelType(13));
    world
}

fn save_world_to_file(world: &VoxelWorld, path: &PathBuf) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_world(world, &mut writer)?;
    println!("Saved world to {}", path.display());
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Flat { material, output } => {
            let world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
                material: VoxelType(material),
            });
            save_world_to_file(&world, &output)
        }
        Command::DemoCubes { output } => {
            let world = generate_demo_cube_layout_world();
            save_world_to_file(&world, &output)
        }
        Command::FilledRegion {
            min,
            size,
            material,
            base,
            base_material,
            output,
        } => {
            let mut world = match base.as_str() {
                "flat" => VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
                    material: VoxelType(base_material),
                }),
                "empty" => VoxelWorld::new(),
                other => {
                    eprintln!("Unknown base type '{}'. Use 'flat' or 'empty'.", other);
                    std::process::exit(1);
                }
            };
            let min_arr = [min[0], min[1], min[2], min[3]];
            let size_arr = [size[0], size[1], size[2], size[3]];
            fill_hypercube(&mut world, min_arr, size_arr, VoxelType(material));
            save_world_to_file(&world, &output)
        }
    };

    if let Err(err) = result {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}
