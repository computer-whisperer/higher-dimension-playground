use clap::{Parser, Subcommand};
use polychora::legacy_migration;
use polychora::migration::legacy_voxel::{LegacyVoxel, RegionChunkWorld};
use polychora::migration::legacy_world_io::{load_world, save_world};
use polychora::migration::save_v3;
use polychora::save_v4;
use polychora::save_v4_migration;
use polychora::shared::entity_types::{self, EntityCategory};
use polychora::shared::voxel::{BaseWorldKind, BlockData};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 4] = b"V4DW";

#[derive(Parser)]
#[command(
    name = "worldgen-cli",
    about = "Generate legacy .v4dw worlds and migrate save formats for polychora"
)]
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
    /// Inspect world metadata and chunk bounds
    Inspect {
        /// Input .v4dw file path
        #[arg(long, short)]
        input: PathBuf,
    },
    /// Migrate an existing world by filtering persisted chunk overrides
    Migrate {
        /// Input .v4dw file path
        #[arg(long, short)]
        input: PathBuf,
        /// Output .v4dw file path (default: <input>.migrated.v4dw)
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Overwrite input file in place
        #[arg(long, default_value_t = false)]
        in_place: bool,
        /// Backup root directory (default: <input_parent>/backups)
        #[arg(long)]
        backup_dir: Option<PathBuf>,
        /// Keep-region minimum chunk coordinate (X Y Z W)
        #[arg(long, num_args = 4, value_names = ["X", "Y", "Z", "W"], allow_hyphen_values = true)]
        keep_min_chunk: Option<Vec<i32>>,
        /// Keep-region maximum chunk coordinate (X Y Z W)
        #[arg(long, num_args = 4, value_names = ["X", "Y", "Z", "W"], allow_hyphen_values = true)]
        keep_max_chunk: Option<Vec<i32>>,
        /// Drop all persisted chunk overrides outside the keep-region chunk bounds
        #[arg(long, default_value_t = false)]
        drop_outside_keep_bounds: bool,
        /// Report migration result without writing output
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Migrate legacy .v4dw (+ optional entity sidecar) into a v4 save root directory
    MigrateV4 {
        /// Input legacy .v4dw file path
        #[arg(long, short)]
        input: PathBuf,
        /// Optional legacy JSON sidecar path (<input>.entities.json)
        #[arg(long)]
        sidecar: Option<PathBuf>,
        /// Output v4 save root directory path
        #[arg(long, short)]
        output: PathBuf,
        /// World seed written to v4 global metadata
        #[arg(long, default_value_t = 1337)]
        world_seed: u64,
        /// Overwrite output directory if it already exists
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
    /// Upgrade an existing v3 save root directory into v4
    MigrateV3ToV4 {
        /// Input v3 save root directory path
        #[arg(long, short)]
        input: PathBuf,
        /// Output v4 save root directory path
        #[arg(long, short)]
        output: PathBuf,
        /// Overwrite output directory if it already exists
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
    /// Inspect v4 save root metadata and summary counts
    InspectV4 {
        /// Input v4 save root directory path
        #[arg(long, short)]
        input: PathBuf,
    },
}

#[derive(Clone, Debug)]
struct WorldStats {
    file_version: Option<u32>,
    base_kind: BaseWorldKind,
    override_chunks_total: usize,
    override_chunks_non_empty: usize,
    bounds_min: Option<[i32; 4]>,
    bounds_max: Option<[i32; 4]>,
}

fn fill_hypercube(
    world: &mut RegionChunkWorld,
    min: [i32; 4],
    size: [i32; 4],
    material: LegacyVoxel,
) {
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

fn generate_demo_cube_layout_world() -> RegionChunkWorld {
    let mut world = RegionChunkWorld::new();
    let mut texture_rot = 0u8;

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                for w in 0..2 {
                    let base = [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2];
                    let material = LegacyVoxel((texture_rot % 5) + 1);
                    fill_hypercube(&mut world, base, [2, 2, 2, 2], material);
                    texture_rot = (texture_rot + 1) % 5;
                }
            }
        }
    }

    // Central bright cube
    fill_hypercube(&mut world, [0, 0, 0, 0], [2, 2, 2, 2], LegacyVoxel(13));
    world
}

fn save_world_to_file(world: &RegionChunkWorld, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_world(world, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn save_world_in_place(world: &RegionChunkWorld, input_path: &Path) -> io::Result<()> {
    let tmp_path = PathBuf::from(format!("{}.tmp", input_path.to_string_lossy()));
    save_world_to_file(world, &tmp_path)?;
    std::fs::rename(&tmp_path, input_path)?;
    Ok(())
}

fn load_world_from_file(path: &Path) -> io::Result<RegionChunkWorld> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_world(&mut reader)
}

fn detect_v4dw_version(path: &Path) -> io::Result<Option<u32>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    if &header[0..4] != MAGIC {
        return Ok(None);
    }
    Ok(Some(u32::from_le_bytes([
        header[4], header[5], header[6], header[7],
    ])))
}

fn compute_world_stats(world: &RegionChunkWorld, file_version: Option<u32>) -> WorldStats {
    let mut bounds_min = [i32::MAX; 4];
    let mut bounds_max = [i32::MIN; 4];
    let mut have_bounds = false;
    let mut non_empty = 0usize;

    for (&pos, chunk) in &world.chunks {
        if chunk.is_empty() {
            continue;
        }
        non_empty += 1;
        have_bounds = true;
        bounds_min[0] = bounds_min[0].min(pos.x);
        bounds_min[1] = bounds_min[1].min(pos.y);
        bounds_min[2] = bounds_min[2].min(pos.z);
        bounds_min[3] = bounds_min[3].min(pos.w);
        bounds_max[0] = bounds_max[0].max(pos.x);
        bounds_max[1] = bounds_max[1].max(pos.y);
        bounds_max[2] = bounds_max[2].max(pos.z);
        bounds_max[3] = bounds_max[3].max(pos.w);
    }

    WorldStats {
        file_version,
        base_kind: world.base_kind(),
        override_chunks_total: world.chunks.len(),
        override_chunks_non_empty: non_empty,
        bounds_min: have_bounds.then_some(bounds_min),
        bounds_max: have_bounds.then_some(bounds_max),
    }
}

fn print_world_stats(label: &str, stats: &WorldStats) {
    println!("{label}:");
    if let Some(version) = stats.file_version {
        println!("  version: {version}");
    } else {
        println!("  version: unknown/non-V4DW");
    }
    match &stats.base_kind {
        BaseWorldKind::Empty => println!("  base: empty"),
        BaseWorldKind::FlatFloor { material } => {
            println!("  base: flat-floor (material {})", material.block_type)
        }
        BaseWorldKind::MassivePlatforms { material } => {
            println!("  base: massive-platforms (material {})", material.block_type)
        }
    }
    println!(
        "  overrides: {} total, {} non-empty",
        stats.override_chunks_total, stats.override_chunks_non_empty
    );
    match (stats.bounds_min, stats.bounds_max) {
        (Some(min), Some(max)) => {
            println!(
                "  non-empty override chunk bounds: min=({}, {}, {}, {}) max=({}, {}, {}, {})",
                min[0], min[1], min[2], min[3], max[0], max[1], max[2], max[3]
            );
        }
        _ => println!("  non-empty override chunk bounds: <none>"),
    }
}

fn parse_chunk_bound(arg_name: &str, raw: &[i32]) -> io::Result<[i32; 4]> {
    if raw.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{arg_name} requires exactly 4 coordinates"),
        ));
    }
    Ok([raw[0], raw[1], raw[2], raw[3]])
}

fn default_migrate_output_path(input: &Path) -> PathBuf {
    input.with_extension("migrated.v4dw")
}

fn default_backup_root(input: &Path) -> PathBuf {
    input
        .parent()
        .map(|p| p.join("backups"))
        .unwrap_or_else(|| PathBuf::from("backups"))
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

fn utc_date_timestamp_labels() -> (String, String) {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs() as i64)
        .unwrap_or(0);
    let days = now_secs.div_euclid(86_400);
    let seconds_of_day = now_secs.rem_euclid(86_400) as u32;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    (
        format!("{year:04}{month:02}{day:02}"),
        format!("{hour:02}{minute:02}{second:02}"),
    )
}

fn world_version_label(file_version: Option<u32>) -> String {
    file_version
        .map(|version| format!("v{}", version))
        .unwrap_or_else(|| "vunknown".to_string())
}

fn create_auto_backup(
    input_path: &Path,
    file_version: Option<u32>,
    backup_root_override: Option<&Path>,
) -> io::Result<PathBuf> {
    let (date, timestamp) = utc_date_timestamp_labels();

    let version_label = world_version_label(file_version);
    let backup_root = backup_root_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_backup_root(input_path));
    let backup_dir = backup_root.join(&date).join(&version_label);
    std::fs::create_dir_all(&backup_dir)?;

    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("world");
    let extension = input_path
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .unwrap_or("v4dw");

    for idx in 0..10_000u32 {
        let suffix = if idx == 0 {
            String::new()
        } else {
            format!("-{}", idx)
        };
        let file_name = format!(
            "{}-{}-{}-{}{}.{}",
            stem, date, timestamp, version_label, suffix, extension
        );
        let backup_path = backup_dir.join(file_name);
        if backup_path.exists() {
            continue;
        }
        std::fs::copy(input_path, &backup_path)?;
        return Ok(backup_path);
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "failed to allocate backup path in {} after many attempts",
            backup_dir.display()
        ),
    ))
}

fn run_migrate(
    input: PathBuf,
    output: Option<PathBuf>,
    in_place: bool,
    backup_dir: Option<PathBuf>,
    keep_min_chunk: Option<Vec<i32>>,
    keep_max_chunk: Option<Vec<i32>>,
    drop_outside_keep_bounds: bool,
    dry_run: bool,
) -> io::Result<()> {
    if in_place && output.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--output cannot be combined with --in-place",
        ));
    }

    let keep_bounds = match (keep_min_chunk, keep_max_chunk) {
        (Some(min_raw), Some(max_raw)) => {
            let min = parse_chunk_bound("--keep-min-chunk", &min_raw)?;
            let max = parse_chunk_bound("--keep-max-chunk", &max_raw)?;
            legacy_migration::validate_chunk_bounds(min, max)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
            Some((min, max))
        }
        (None, None) => None,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "both --keep-min-chunk and --keep-max-chunk must be provided together",
            ));
        }
    };

    if drop_outside_keep_bounds && keep_bounds.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--drop-outside-keep-bounds requires --keep-min-chunk and --keep-max-chunk",
        ));
    }

    let version = detect_v4dw_version(&input)?;
    let mut world = load_world_from_file(&input)?;
    let before = compute_world_stats(&world, version);
    print_world_stats("Before", &before);

    let mut dropped = 0usize;
    if drop_outside_keep_bounds {
        let (min_chunk, max_chunk) = keep_bounds.expect("checked above");
        dropped =
            legacy_migration::drop_overrides_outside_chunk_bounds(&mut world, min_chunk, max_chunk);
        println!(
            "Applied migration: dropped {} override chunks outside keep bounds",
            dropped
        );
    } else if keep_bounds.is_some() {
        println!(
            "Keep bounds provided but --drop-outside-keep-bounds is false; no changes applied."
        );
    }

    let after = compute_world_stats(&world, before.file_version);
    print_world_stats("After", &after);

    if dry_run {
        println!("Dry run enabled; no file was written.");
        return Ok(());
    }

    let backup_path = create_auto_backup(&input, before.file_version, backup_dir.as_deref())?;
    println!("Backup written: {}", backup_path.display());

    if in_place {
        save_world_in_place(&world, &input)?;
        println!("Saved migrated world in place: {}", input.display());
    } else {
        let output_path = output.unwrap_or_else(|| default_migrate_output_path(&input));
        save_world_to_file(&world, &output_path)?;
        println!("Saved migrated world to {}", output_path.display());
    }

    println!("Migration complete (dropped overrides: {}).", dropped);
    Ok(())
}

fn run_migrate_v4(
    input: PathBuf,
    sidecar: Option<PathBuf>,
    output: PathBuf,
    world_seed: u64,
    overwrite: bool,
) -> io::Result<()> {
    if output.exists() {
        if !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output path '{}' already exists (use --overwrite to replace)",
                    output.display()
                ),
            ));
        }
        if output.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output path '{}' is a file", output.display()),
            ));
        }
        std::fs::remove_dir_all(&output)?;
    }

    let save_result = save_v4_migration::migrate_legacy_world_to_v4(
        &input,
        sidecar.as_deref(),
        &output,
        world_seed,
        save_v4::now_unix_ms(),
    )?;
    println!(
        "Migrated legacy world -> v4: generation={} saved_block_regions={} saved_entity_regions={} output={}",
        save_result.generation,
        save_result.saved_block_regions,
        save_result.saved_entity_regions,
        output.display()
    );
    Ok(())
}

fn run_migrate_v3_to_v4(input: PathBuf, output: PathBuf, overwrite: bool) -> io::Result<()> {
    if !save_v3::is_v3_save_root(&input) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "input path '{}' is not a v3 save root directory",
                input.display()
            ),
        ));
    }
    let save_result = save_v4_migration::migrate_v3_save_to_v4(
        &input,
        &output,
        overwrite,
        save_v4::now_unix_ms(),
    )?;
    println!(
        "Migrated v3 save root -> v4: generation={} saved_block_regions={} saved_entity_regions={} output={}",
        save_result.generation,
        save_result.saved_block_regions,
        save_result.saved_entity_regions,
        output.display()
    );
    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn file_size_or_zero(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn count_named_files(root: &Path, prefix: &str, suffix: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.starts_with(prefix) && name.ends_with(suffix))
                .unwrap_or(false)
        })
        .count()
}

fn run_inspect_v4(input: PathBuf) -> io::Result<()> {
    let loaded = save_v4::load_state(&input)?;
    let manifest = loaded.manifest;
    let global = loaded.global;
    let players = loaded.players.players;
    let entities = loaded.entities;
    let index = loaded.index;
    let world_chunk_payloads = loaded.world_chunk_payloads;

    let index_path = input.join(&manifest.index_file);
    let global_path = input.join(&manifest.global_file);
    let players_path = input.join(&manifest.players_file);
    let data_dir = input.join("data");
    let index_dir = input.join("index");

    let mut branch_count = 0usize;
    let mut leaf_empty_count = 0usize;
    let mut leaf_uniform_count = 0usize;
    let mut leaf_chunk_array_count = 0usize;
    let mut node_min = [i32::MAX; 4];
    let mut node_max = [i32::MIN; 4];
    let mut have_node_bounds = false;
    for node in &index.nodes {
        match &node.kind {
            save_v4::IndexNodeKind::Branch { .. } => branch_count += 1,
            save_v4::IndexNodeKind::LeafEmpty => leaf_empty_count += 1,
            save_v4::IndexNodeKind::LeafUniform { .. } => leaf_uniform_count += 1,
            save_v4::IndexNodeKind::LeafChunkArray { .. } => leaf_chunk_array_count += 1,
        }
        have_node_bounds = true;
        for axis in 0..4 {
            node_min[axis] = node_min[axis].min(node.bounds_min_chunk[axis]);
            node_max[axis] = node_max[axis].max(node.bounds_max_chunk[axis]);
        }
    }

    let mut entity_class_counts = [0usize; 3];
    let mut entity_kind_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut entity_payload_bytes = 0usize;
    let mut entity_tag_count = 0usize;
    for entity in &entities {
        let category = entity_types::category_for(entity.entity.namespace, entity.entity.entity_type);
        match category {
            EntityCategory::Player => entity_class_counts[0] += 1,
            EntityCategory::Accent => entity_class_counts[1] += 1,
            EntityCategory::Mob => entity_class_counts[2] += 1,
        }
        let type_name = entity_types::lookup(entity.entity.namespace, entity.entity.entity_type)
            .map(|e| e.canonical_name)
            .unwrap_or("unknown");
        *entity_kind_counts
            .entry(type_name.to_string())
            .or_insert(0) += 1;
        entity_payload_bytes += entity.entity.data.len();
        entity_tag_count += entity.tags.len();
    }

    let mut player_ids: Vec<u64> = players.iter().map(|p| p.player_id).collect();
    player_ids.sort_unstable();

    println!("v4 save root: {}", input.display());
    println!(
        "manifest: format={} version={} generation={}",
        manifest.format, manifest.version, manifest.current_generation
    );
    println!(
        "timestamps_ms: created={} last_modified={}",
        manifest.created_ms, manifest.last_modified_ms
    );
    println!(
        "files: index='{}' ({}) global='{}' ({}) players='{}' ({})",
        manifest.index_file,
        format_size(file_size_or_zero(&index_path)),
        manifest.global_file,
        format_size(file_size_or_zero(&global_path)),
        manifest.players_file,
        format_size(file_size_or_zero(&players_path)),
    );
    println!(
        "data_files: active={} declared_count={} observed_count={}",
        manifest.active_data_file_id,
        manifest.data_file_count,
        count_named_files(&data_dir, "dt-", ".v4dt"),
    );
    println!(
        "generations: index_files={} global_files={} players_files={}",
        count_named_files(&index_dir, "ix-main.g", ".v4ix"),
        count_named_files(&input, "global.g", ".v4g"),
        count_named_files(&input, "players.g", ".v4p"),
    );
    println!(
        "limits: data_file_max_bytes={} index_soft_max_bytes={} chunk_payload_target={} chunk_payload_hard={} entity_blob_target={} entity_blob_hard={}",
        manifest.limits.data_file_max_bytes,
        manifest.limits.index_soft_max_bytes,
        manifest.limits.chunk_payload_target_bytes,
        manifest.limits.chunk_payload_hard_max_bytes,
        manifest.limits.entity_blob_target_bytes,
        manifest.limits.entity_blob_hard_max_bytes,
    );
    println!(
        "global: base_world={:?} world_seed={} procgen_manifest_hash={} next_entity_id={} next_data_file_id={} player_hints={} custom_payload_bytes={}",
        global.base_world_kind,
        global.world_seed,
        global.procgen_manifest_hash,
        global.next_entity_id,
        global.next_data_file_id,
        global.player_entity_hints.len(),
        global.custom_global_payload.len(),
    );
    println!(
        "index: generation={} nodes={} root_node_id={} entity_root_node_id={:?} branches={} leaf_empty={} leaf_uniform={} leaf_chunk_array={}",
        index.generation,
        index.nodes.len(),
        index.root_node_id,
        index.entity_root_node_id,
        branch_count,
        leaf_empty_count,
        leaf_uniform_count,
        leaf_chunk_array_count,
    );
    if have_node_bounds {
        println!(
            "index_node_bounds: min=({}, {}, {}, {}) max=({}, {}, {}, {})",
            node_min[0],
            node_min[1],
            node_min[2],
            node_min[3],
            node_max[0],
            node_max[1],
            node_max[2],
            node_max[3]
        );
    } else {
        println!("index_node_bounds: <none>");
    }
    println!(
        "world: override_chunks={} non_empty_override_chunks={}",
        world_chunk_payloads.len(),
        world_chunk_payloads.len(),
    );
    println!(
        "entities: total={} class_counts(player={}, accent={}, mob={}) payload_bytes={} tag_count={}",
        entities.len(),
        entity_class_counts[0],
        entity_class_counts[1],
        entity_class_counts[2],
        entity_payload_bytes,
        entity_tag_count,
    );
    if entity_kind_counts.is_empty() {
        println!("entity_kinds: <none>");
    } else {
        println!("entity_kinds:");
        for (kind, count) in entity_kind_counts {
            println!("  {kind}: {count}");
        }
    }
    println!("players: total={} ids={:?}", players.len(), player_ids);
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Flat { material, output } => {
            let world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, material as u32),
            });
            save_world_to_file(&world, &output).map(|_| {
                println!("Saved world to {}", output.display());
            })
        }
        Command::DemoCubes { output } => {
            let world = generate_demo_cube_layout_world();
            save_world_to_file(&world, &output).map(|_| {
                println!("Saved world to {}", output.display());
            })
        }
        Command::FilledRegion {
            min,
            size,
            material,
            base,
            base_material,
            output,
        } => {
            let world_result: io::Result<RegionChunkWorld> = match base.as_str() {
                "flat" => Ok(RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
                    material: BlockData::simple(0, base_material as u32),
                })),
                "empty" => Ok(RegionChunkWorld::new()),
                other => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown base type '{other}'. Use 'flat' or 'empty'."),
                )),
            };
            world_result.and_then(|mut world| {
                parse_chunk_bound("--min", &min).and_then(|min_arr| {
                    parse_chunk_bound("--size", &size).and_then(|size_arr| {
                        fill_hypercube(&mut world, min_arr, size_arr, LegacyVoxel(material));
                        save_world_to_file(&world, &output).map(|_| {
                            println!("Saved world to {}", output.display());
                        })
                    })
                })
            })
        }
        Command::Inspect { input } => detect_v4dw_version(&input).and_then(|version| {
            load_world_from_file(&input).map(|world| {
                let stats = compute_world_stats(&world, version);
                print_world_stats("World", &stats);
            })
        }),
        Command::Migrate {
            input,
            output,
            in_place,
            backup_dir,
            keep_min_chunk,
            keep_max_chunk,
            drop_outside_keep_bounds,
            dry_run,
        } => run_migrate(
            input,
            output,
            in_place,
            backup_dir,
            keep_min_chunk,
            keep_max_chunk,
            drop_outside_keep_bounds,
            dry_run,
        ),
        Command::MigrateV4 {
            input,
            sidecar,
            output,
            world_seed,
            overwrite,
        } => run_migrate_v4(input, sidecar, output, world_seed, overwrite),
        Command::MigrateV3ToV4 {
            input,
            output,
            overwrite,
        } => run_migrate_v3_to_v4(input, output, overwrite),
        Command::InspectV4 { input } => run_inspect_v4(input),
    };

    if let Err(err) = result {
        exit_with_error(err);
    }
}

fn exit_with_error(err: io::Error) -> ! {
    eprintln!("Error: {}", err);
    std::process::exit(1);
}
