# Slang Shader System

This directory contains the shader implementation using [Slang](https://shader-slang.com/), replacing the previous rust-gpu approach that was hitting ICE bugs.

## Why Slang?

Slang is a modern shading language that offers:
- **Generics** - We can express `VecN<N>` and `MatN<N>` patterns cleanly
- **Interfaces** - For material systems and extensibility
- **Multi-target** - Compiles to SPIR-V, GLSL, HLSL, Metal, CUDA
- **Modern syntax** - C++-like with many Rust-like safety features
- **Active development** - Backed by NVIDIA and Carnegie Mellon

## Architecture

```
slang-shaders/
├── src/
│   ├── math.slang       # Generic N-dimensional math (VecN<N>, MatN<N>)
│   ├── types.slang      # Shared data structures (must match Rust bytemuck structs)
│   ├── materials.slang  # Procedural material system
│   ├── raytracer.slang  # 4D raytracing compute shaders
│   ├── rasterizer.slang # 4D rasterization compute shaders
│   └── present.slang    # Screen presentation vertex/fragment shaders
└── README.md
```

## Shader Entry Points

### Raytracer Pipeline
- `mainRaytracerTetrahedronPreprocessor` - Transform model tetrahedra to view space (compute)
- `mainRaytracerClear` - Clear pixel buffer (compute)
- `mainRaytracerPixel` - 4D path tracing (compute)

### Present Pipeline
- `mainLineVS` / `mainLineFS` - Wireframe line rendering
- `mainBufferVS` / `mainBufferFS` - Display accumulated buffer to screen

### Rasterizer Pipeline
- `mainTetrahedronCS` - Tetrahedron preprocessing for rasterization
- `mainTetrahedronPixelCS` - Per-pixel ZW integration
- `mainEdgeCS` - Edge processing with 4D frustum clipping

## Building

### Prerequisites

Install Slang compiler:

```bash
# Arch Linux (AUR)
yay -S slang-bin

# Or download from GitHub releases:
# https://github.com/shader-slang/slang/releases
```

### Compilation

The build.rs script handles compilation. Each entry point produces a separate .spv file:

```bash
cargo build  # Compiles shaders as part of build
```

SPIR-V output goes to `$OUT_DIR/spirv/`.

## Key Differences from rust-gpu

| Aspect | rust-gpu | Slang |
|--------|----------|-------|
| Generics | Full Rust generics (with ICE issues) | Slang generics (stable) |
| Types | Shared via `common` crate | Must manually keep in sync |
| Build | spirv-builder crate | slangc CLI |
| Testing | Can run shader code as Rust tests | Separate test framework needed |
| Debugging | Rust tooling | RenderDoc + Slang debug info |

## Struct Layout Verification

The types in `types.slang` must exactly match the Rust structs in `common/src/lib.rs`. Pay attention to:
- Field ordering
- Padding bytes
- Matrix storage order (row-major vs column-major)

Layout verification tests are in `common/src/layout_verify.rs` and can be run with `cargo test`.

## Descriptor Set Layout

Shaders use explicit `[[vk::binding(n, set)]]` annotations to match the Rust descriptor layout:

| Set | Binding | Buffer | Stages |
|-----|---------|--------|--------|
| 0 | 0 | modelTetrahedrons | Compute |
| 0 | 1 | modelEdges | Compute |
| 1 | 0 | lineVertices | Compute, Vertex |
| 1 | 1 | outputTetrahedrons | Compute |
| 1 | 2 | pixelBuffer | Compute, Fragment |
| 2 | 0 | instances | Compute |
| 2 | 1 | workingData | Compute, Fragment |

## Build Notes

- Uses `-emit-spirv-via-glsl` flag for Vulkano-compatible SPIR-V output
- Uses `-fvk-use-scalar-layout` for Rust struct compatibility
- Uses `-fvk-use-entrypoint-name` to preserve entry point names for linking
