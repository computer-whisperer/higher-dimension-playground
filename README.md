# Polychora (Higher Dimension Playground)

Polychora is a genuine 4D voxel game engine and multiplayer sandbox built on Vulkan.

It started as a rendering experiment and has evolved into a full 4D world simulation stack:
- 4D player movement and camera control
- real-time voxel world editing
- procedural 4D world generation (structures + mazes)
- server-authoritative multiplayer with mob simulation
- multiple rendering backends, including a dedicated voxel traversal path

## What "Genuine 4D" Means Here

This engine does not fake 4D as hidden 3D slices.

- World coordinates are spatial `X/Y/Z/W` (with `W` as space, not time)
- Camera/view math is 4D with explicit 4D basis vectors and 4D projection
- Rotation math supports all six spatial planes (`XY`, `XZ`, `XW`, `YZ`, `YW`, `ZW`)
- Voxel storage and traversal are native 4D chunk/cell structures
- Players can navigate and build geometry that is impossible in ordinary 3D space

## Core Capabilities

- **4D voxel world runtime**
  - Chunked 4D voxel world model (`8x8x8x8` chunks)
  - Break/place/pick gameplay editing
  - Collision and movement resolved against 4D voxel solids

- **Procedural world generation**
  - Base world modes (`FlatFloor`, `MassivePlatforms`)
  - Rich procedural structure placement from embedded blueprints
  - Procedural 4D maze generation with cached compiled topology

- **4D rendering backends**
  - **Voxel Traversal Engine (VTE)**: native 4D chunk/cell ray traversal
  - **Tetra Ray Tracing**: path-traced tetrahedron intersections in 4D
  - **Tetra Rasterization**: tetra pipeline with ZW integration
  - Runtime backend selection for comparison/debugging

- **Multiplayer + server simulation**
  - Server-authoritative world overlay + edits
  - Streaming subtree world replication
  - Mob simulation with 4D pathfinding, LOS, collision, and abilities

- **Plugin-driven content**
  - Content registry for blocks/entities/textures
  - First-party WASM content plugin (`polychora-content`)
  - Separate host/plugin ABI for steering + ability logic

## High-Level Architecture

- `common/`:
  Shared N-dimensional math/types used by CPU and GPU paths.

- `crates/polychora/`:
  Main game client + dedicated server binary.

- `crates/polychora-content/`:
  First-party WASM content plugin (block/entity declarations + mob behavior).

- `crates/polychora-plugin-api/`:
  Stable ABI/shared types for content plugins.

- `src/` + `slang-shaders/`:
  Core rendering infrastructure and shader pipelines.

## How Rendering Works (VTE View)

VTE explicitly separates 4D viewing into two stages:

1. **Stage A (4D -> 3D hyper-image)**
   Cast rays through the 4D voxel world using 4D basis vectors and chunk/cell DDA.
2. **Stage B (3D -> 2D display operator)**
   Resolve Stage A samples into final screen output (`integral`, `slice`, `thick-slice`, debug modes).

This keeps the hidden dimension as a first-class rendering quantity instead of burying it in ad-hoc projections.

## Building

### Prerequisites

1. **Rust toolchain** (stable)
2. **Vulkan SDK** with validation layers
3. **Slang compiler** and **SPIR-V tools**:

```bash
# Arch Linux
yay -S shader-slang-bin spirv-tools

# Or download Slang from:
# https://github.com/shader-slang/slang/releases
```

### Compile and Run

```bash
# Build (compiles shaders automatically)
cargo build --release

# Interactive 4D voxel explorer/game client
# (workspace default points to crates/polychora)
cargo run --release

# Dedicated multiplayer state server
cargo run -p polychora --bin polychora-server --release -- --bind 0.0.0.0:4000

# Demo with pre-set camera (supports --headless, --raytrace, --edges, etc.)
cargo run -p demo --release

# Demo headless render to PNG
cargo run -p demo --release -- --headless
```

## Configuration

### Game Controls

| Input | Action |
|-------|--------|
| W/A/S/D | Move forward/left/back/right |
| Mouse | Look around |
| Space | Jump (double-tap to toggle fly mode) |
| Shift | Descend / Crouch |
| Q/E | Move in 4D (W-axis negative/positive) |
| Hold R | Reset orientation |
| Hold F | Pull to 3D |
| G | Look at nearest block |
| Left Click | Break block |
| Right Click | Place block |
| Middle Click | Pick pointed block's material |
| Scroll Wheel | Cycle hotbar slot |
| [ / ] | Previous / Next material |
| 1-9, 0 | Select hotbar slot |
| Tab / I | Toggle inventory |
| T | Toggle teleport dialog |
| Escape | Open / close menu |
| F5 / F9 | Save/load world (`--world-file`) |
| F12 | Screenshot |
| Click | Re-grab mouse (when menu is open) |

### Game Options

```bash
cargo run --release -- [OPTIONS]
```

Core runtime options:

| Flag | Effect |
|------|--------|
| `--backend <auto\|tetra-raster\|tetra-raytrace\|voxel-traversal>` | Select renderer backend |
| `--scene <flat\|demo-cubes>` | Voxel scene preset |
| `-W, --width` / `-H, --height` | Render size |
| `--layers` | Hidden-dimension sample layers |
| `--edit-reach` | Max block remove/place/highlight reach |
| `--world-file <path>` | Save/load file path for `F5`/`F9` |
| `--load-world` | Load `--world-file` at startup |
| `--server <ip[:port]>` | Connect to multiplayer server (`:4000` default if omitted) |
| `--player-name <name>` | Multiplayer display name for `--server` |
| `--cpu-render` | CPU reference render to `frames/cpu_render.png`, then exit |

VTE-specific options:

| Flag | Effect |
|------|--------|
| `--vte-max-trace-steps` | Per-ray traversal step budget |
| `--vte-max-trace-distance` | Max ray distance before miss |
| `--vte-display-mode <integral\|slice\|thick-slice\|debug-compare\|debug-integral>` | Stage-B resolve mode |
| `--vte-slice-layer <index>` | Slice center layer (default: middle layer) |
| `--vte-thick-half-width <n>` | Thick-slice radius around center layer |

Screenshot/capture options:

| Flag | Effect |
|------|--------|
| `--gpu-screenshot` | Render one frame with screenshot camera and exit |
| `--gpu-screenshot-source <render-buffer\|framebuffer>` | Capture source for `--gpu-screenshot` |
| `--screenshot-pos X Y Z W` | Camera position override |
| `--screenshot-angles YAW PITCH XW ZW` | Camera orientation override (radians) |
| `--screenshot-angles-deg YAW PITCH XW ZW` | Camera orientation override (degrees) |
| `--screenshot-yw <rad>` | Camera YW deviation override |

The runtime `F12` screenshot path writes both `.webp` and `.png` to `frames/`,
and prints screenshot metadata (frame, camera pose, look vector, backend, and
active VTE mode details) to stdout.

### Settings Persistence

Polychora persists menu/runtime settings to a per-user config file:

- Linux: `$XDG_CONFIG_HOME/polychora/settings.json` (fallback: `~/.config/polychora/settings.json`)
- macOS: `~/Library/Application Support/polychora/settings.json`
- Windows: `%APPDATA%\polychora\settings.json`

Command-line flags still take priority for that launch.

### VTE Debug Environment Flags

These flags are intended for diagnostics and can be expensive:

| Env var | Effect |
|---------|--------|
| `R4D_VTE_REFERENCE_COMPARE=1` | Enable reference-trace comparison in VTE debug modes |
| `R4D_VTE_REFERENCE_MISMATCH_ONLY=1` | In compare mode, visualize mismatches only |
| `R4D_VTE_COMPARE_SLICE_ONLY=1` | In compare mode, compare only the selected slice layer |

### Demo Options

```bash
cargo run -p demo --release -- [OPTIONS]
```

| Flag | Effect |
|------|--------|
| `--headless` | Render single frame to PNG, no window |
| `--raytrace` | Enable path tracing |
| `--no-raster` | Disable rasterization |
| `--edges` | Render ZW wireframe edges |
| `--spin` | Animate camera rotation |
| `--floor` | Show floor plane |
| `--walls` | Show wall enclosure |
| `-W` / `-H` | Canvas width/height (default 960x540) |
| `--layers` | Depth layers for ZW supersampling (default 4) |

### Multiplayer Server (Early)

```bash
cargo run -p polychora --bin polychora-server --release -- --bind 0.0.0.0:4000
```

Server CLI options:

| Flag | Effect |
|------|--------|
| `--bind <addr:port>` | TCP listen address |
| `--world-file <path>` | Load/save world state in `.v4dw` format |
| `--tick-hz <hz>` | Player position broadcast rate |
| `--save-interval-secs <n>` | Autosave cadence (`0` disables autosave) |
| `--snapshot-on-join <bool>` | Send full world snapshot (base64 `.v4dw`) on `hello` |

Message protocol is line-delimited JSON:

- Client -> server: `hello`, `update_player`, `set_voxel`, `request_world_snapshot`, `ping`
- Server -> client: `welcome`, `player_joined`, `player_left`, `player_positions`, `world_voxel_set`, `world_snapshot`, `pong`, `error`


## Acknowledgments

- [Slang](https://shader-slang.com/) - Modern shading language with generics
- [vulkano](https://vulkano.rs/) - Safe Rust bindings for Vulkan
