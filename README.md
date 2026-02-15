# Higher Dimension Playground

A 4D rendering engine for visualizing tesseracts (hypercubes) and other four-dimensional geometry using Vulkan.

## Overview

This project renders 4D geometry with multiple backends. It supports three rendering pipelines:

- **Tetra Ray Tracing** - Monte Carlo/path-traced tetrahedron intersections in 4D
- **Tetra Rasterization** - Rasterization extended to 4D with ZW-depth integration
- **Voxel Traversal Engine (VTE)** - Native voxel chunk/cell traversal with Stage-A/Stage-B resolve

## Features

### Implemented

- **4D Math Library** (`common/`)
  - Generic N-dimensional vectors (`VecN<N>`) and matrices (`MatN<N>`)
  - 4D transformations: translation, rotation (in any plane), scaling
  - Normal calculation for 3D hyperplanes in 4D
  - Combinatorial utilities for face/simplex generation

- **Hypercube Geometry** (`src/hypercube.rs`)
  - Vertex generation for D-dimensional hypercubes
  - Edge (1-face) generation with proper connectivity
  - Cell (3-face) generation for tesseracts
  - Simplex decomposition of cells into tetrahedra

- **Rendering Engine** (`src/render.rs`)
  - Vulkan-based compute and graphics pipelines via vulkano
  - Model instancing with per-cell material assignment
  - View matrix transforms for camera positioning
  - Configurable focal lengths for XY and ZW projections
  - Runtime-selectable backend: tetra-raster, tetra-raytrace, or voxel-traversal

- **Shader System** (`slang-shaders/`)
  - Written in [Slang](https://shader-slang.com/) for stable generics support
  - Raytracing: tetrahedron preprocessing, ray-tetrahedron intersection, path tracing
  - Rasterization: tetrahedron processing, per-pixel ZW integration, edge rendering
  - VTE: chunk DDA + in-chunk voxel DDA, plus Stage-B display modes
  - Presentation: line wireframe rendering, buffer display

- **Procedural Materials** (`slang-shaders/src/materials.slang`)
  - 34 material presets (colors, floor, light source, mirrors, animated and Minecraft-style procedural blocks)
  - PBR properties: albedo, metallic, roughness, luminance

- **Output Modes**
  - Windowed display with winit
  - Headless rendering for offline frame generation
  - EXR export for high dynamic range output

### Not Yet Implemented

- Custom scene loading (currently hardcoded demo scene)
- WebGPU/WASM support (infrastructure exists in `pkg/`)

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

# Interactive FPS-style explorer (WASD + mouse look, Q/E for W-axis)
# (workspace default points to crates/polychora)
cargo run --release

# Basic multiplayer state server
cargo run -p polychora --bin polychora-server --release -- --bind 0.0.0.0:4000

# Demo with pre-set camera (supports --headless, --raytrace, --edges, etc.)
cargo run -p demo --release

# Demo headless render to PNG
cargo run -p demo --release -- --headless
```

## Project Structure

```
higher-dimension-playground/
├── src/                     # Core library
│   ├── lib.rs               # Library root
│   ├── render.rs            # Vulkan rendering context and pipelines
│   ├── hypercube.rs         # Hypercube geometry generation
│   ├── matrix_operations.rs # 4D transformation matrices
│   └── vulkan_setup.rs      # Vulkan device/instance initialization
├── crates/
│   ├── polychora/           # Interactive FPS-style 4D explorer
│   │   └── src/
│   │       ├── main.rs      # App struct, event loop, mouse grab
│   │       ├── camera.rs    # Camera4D: 5-angle orientation, auto-leveling
│   │       ├── input.rs     # InputState: keys, mouse buttons, scroll, rotation modes
│   │       ├── scene.rs     # Demo scene builder
│   │       ├── multiplayer.rs # Multiplayer client protocol + socket threads
│   │       └── bin/polychora-server.rs # Dedicated server binary entrypoint
│   ├── demo/                # CLI demo with pre-set camera angles
│   │   └── src/main.rs      # Headless/windowed demo, CLI options
│   └── exr-converter/       # Standalone EXR recompression tool
├── common/                  # Shared math library (CPU + GPU compatible)
│   └── src/
│       ├── lib.rs           # Data structures (must match Slang types)
│       ├── vec_n.rs         # Generic N-dimensional vectors
│       ├── mat_n.rs         # Generic N-dimensional matrices
│       └── layout_verify.rs # Struct layout verification tests
├── slang-shaders/           # GPU shaders in Slang
│   └── src/
│       ├── math.slang       # Generic VecN/MatN for GPU
│       ├── types.slang      # Shared data structures
│       ├── materials.slang  # Procedural material system
│       ├── raytracer.slang  # 4D path tracing
│       ├── rasterizer.slang # 4D rasterization + near-plane clipping
│       └── present.slang    # Screen presentation
└── build.rs                 # Slang shader compilation
```

## How It Works

### 4D to 2D Projection

1. **Tesseract decomposition**: The 8 cubic cells of a tesseract are each divided into 6 tetrahedra (total 48 tetrahedra per tesseract)
2. **Model transform**: Each model instance applies a 5x5 homogeneous transformation matrix
3. **4D viewing**: The view matrix positions the camera in 4D space
4. **Dual projection**: Points are projected first from 4D→3D (using ZW focal length), then 3D→2D (using XY focal length)

### Raytracing Pipeline

1. **Tetrahedron preprocessing** - Transform model tetrahedra to world space, compute normals
2. **Ray generation** - Cast rays from camera with jittered sampling
3. **Intersection** - Ray-tetrahedron intersection in 4D
4. **Shading** - Monte Carlo path tracing with material sampling
5. **Accumulation** - Progressive rendering with frame averaging

### Demo Scene

The default scene includes:
- Central light-emitting tesseract
- 16 outer colored tesseracts in a 2x2x2x2 grid
- Reflective floor plane extending in XZW
- Mirror walls for enclosure

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
