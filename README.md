# Higher Dimension Playground

A 4D rendering engine for visualizing tesseracts (hypercubes) and other four-dimensional geometry using Vulkan.

## Overview

This project renders 4D geometry by decomposing hypercubes into tetrahedra (3-simplices) and projecting them to screen space. It supports two rendering pipelines:

- **Path Tracing** - Monte Carlo path tracing in 4D space with progressive accumulation
- **Rasterization** - Traditional rasterization extended to 4D with ZW-depth integration

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

- **Shader System** (`slang-shaders/`)
  - Written in [Slang](https://shader-slang.com/) for stable generics support
  - Raytracing: tetrahedron preprocessing, ray-tetrahedron intersection, path tracing
  - Rasterization: tetrahedron processing, per-pixel ZW integration, edge rendering
  - Presentation: line wireframe rendering, buffer display

- **Procedural Materials** (`slang-shaders/src/materials.slang`)
  - 14 material presets (colors, floor, light source, mirrors)
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
cargo run -p game --release

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
│   ├── game/                # Interactive FPS-style 4D explorer
│   │   └── src/
│   │       ├── main.rs      # App struct, event loop, mouse grab
│   │       ├── camera.rs    # Camera4D: 5-angle orientation, auto-leveling
│   │       ├── input.rs     # InputState: keys, mouse buttons, scroll, rotation modes
│   │       └── scene.rs     # Demo scene builder
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
├── shaders/                 # Legacy rust-gpu shaders (deprecated)
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
| Mouse | Look around (yaw/pitch by default) |
| Space/Shift | Move up/down (Y axis) |
| Q/E | Move along W axis |
| Mouse Back | Hold for 4D rotation (XW/ZW planes) |
| Mouse Back+Forward | Hold both for double rotation (XZ yaw + YW) |
| Scroll wheel | Adjust move speed (Layers mode) or cycle rotation pair (Scroll mode) |
| Tab | Toggle control scheme (Layers / Scroll) |
| R | Reset camera orientation to defaults |
| F12 | Screenshot |
| Escape | Release mouse (press again to exit) |
| Click | Re-grab mouse |
| Double-tap Space | Toggle fly/gravity mode |

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


## Acknowledgments

- [Slang](https://shader-slang.com/) - Modern shading language with generics
- [vulkano](https://vulkano.rs/) - Safe Rust bindings for Vulkan
- rust-gpu (previous shader system, replaced due to ICE issues)
