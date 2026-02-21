// Build script for Slang shaders
//
// Compiles .slang files to SPIR-V using the slangc compiler, then links them.
// Requires Slang and spirv-tools to be installed.
//
// On Arch Linux: yay -S shader-slang-bin spirv-tools

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Shader entry point definition
struct ShaderEntry {
    file: &'static str,
    entry: &'static str,
    profile: &'static str,
}

const SHADER_ENTRIES: &[ShaderEntry] = &[
    // Raytracer compute shaders
    ShaderEntry {
        file: "raytracer.slang",
        entry: "mainRaytracerTetrahedronPreprocessor",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "raytracer.slang",
        entry: "mainEntityInstanceAabbPreprocessor",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "raytracer.slang",
        entry: "mainRaytracerClear",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "raytracer.slang",
        entry: "mainRaytracerPixel",
        profile: "cs_6_5",
    },
    // BVH compute shaders
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHSceneBounds",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHMortonCodes",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHBitonicSortLocal",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHBitonicSort",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHBitonicSortLocalMerge",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHInitLeaves",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHBuildTree",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHLinkParents",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "bvh.slang",
        entry: "mainBVHPropagateAABBs",
        profile: "cs_6_5",
    },
    // Rasterizer compute shaders
    ShaderEntry {
        file: "rasterizer.slang",
        entry: "mainTetrahedronCS",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "rasterizer.slang",
        entry: "mainEdgeCS",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "rasterizer.slang",
        entry: "mainBinTetsCS",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "rasterizer.slang",
        entry: "mainTetrahedronPixelCS",
        profile: "cs_6_5",
    },
    // Voxel traversal engine compute shaders
    ShaderEntry {
        file: "voxel.slang",
        entry: "mainVoxelClear",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "voxel.slang",
        entry: "mainVoxelTraceStageAIntegralFused",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "voxel.slang",
        entry: "mainVoxelTraceStageALayered",
        profile: "cs_6_5",
    },
    ShaderEntry {
        file: "voxel.slang",
        entry: "mainVoxelDisplayStageB",
        profile: "cs_6_5",
    },
    // HUD shaders
    ShaderEntry {
        file: "hud.slang",
        entry: "mainHudVS",
        profile: "vs_6_5",
    },
    ShaderEntry {
        file: "hud.slang",
        entry: "mainHudFS",
        profile: "ps_6_5",
    },
    // Present shaders
    ShaderEntry {
        file: "present.slang",
        entry: "mainLineVS",
        profile: "vs_6_5",
    },
    ShaderEntry {
        file: "present.slang",
        entry: "mainLineFS",
        profile: "ps_6_5",
    },
    ShaderEntry {
        file: "present.slang",
        entry: "mainBufferVS",
        profile: "vs_6_5",
    },
    ShaderEntry {
        file: "present.slang",
        entry: "mainBufferFS",
        profile: "ps_6_5",
    },
];

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let shader_src_dir = Path::new("slang-shaders/src");
    let spirv_out_dir = PathBuf::from(&out_dir).join("spirv");
    let vte_diagnostics_enabled = env::var_os("CARGO_FEATURE_VTE_DIAGNOSTICS").is_some();
    let vte_diagnostics_define = if vte_diagnostics_enabled {
        "VTE_DIAGNOSTICS_ENABLED=1"
    } else {
        "VTE_DIAGNOSTICS_ENABLED=0"
    };

    // Create output directory
    std::fs::create_dir_all(&spirv_out_dir).expect("Failed to create SPIR-V output directory");

    // Track all source files for rerun-if-changed
    for entry in std::fs::read_dir(shader_src_dir).expect("Failed to read shader source directory")
    {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "slang") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VTE_DIAGNOSTICS");

    // Check if required tools are available
    check_tool(
        "slangc",
        "-v",
        "Slang compiler not found. Install: yay -S shader-slang-bin",
    );
    check_tool(
        "spirv-link",
        "--version",
        "spirv-link not found. Install: yay -S spirv-tools",
    );

    // Compile each shader entry point to a separate .spv file
    let mut spv_files = Vec::new();
    let mut failed = false;

    for shader in SHADER_ENTRIES {
        let input_path = shader_src_dir.join(shader.file);
        let output_path = spirv_out_dir.join(format!("{}.spv", shader.entry));

        print!("Compiling {} ({})... ", shader.entry, shader.file);

        let output = Command::new("slangc")
            .args([
                "-target",
                "spirv",
                "-profile",
                shader.profile,
                "-entry",
                shader.entry,
                "-I",
                shader_src_dir.to_str().unwrap(),
                // CRITICAL: Use scalar layout to match Rust struct layouts
                "-fvk-use-scalar-layout",
                // Use the actual entry point name instead of "main"
                "-fvk-use-entrypoint-name",
                // Compile via GLSL for better Vulkano compatibility (avoids SPIR-V extensions)
                "-emit-spirv-via-glsl",
                "-D",
                vte_diagnostics_define,
                "-o",
                output_path.to_str().unwrap(),
                input_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to execute slangc");

        if output.status.success() {
            println!("ok");
            spv_files.push(output_path);
        } else {
            println!("FAILED");
            eprintln!(
                "slangc stderr:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            eprintln!(
                "slangc stdout:\n{}",
                String::from_utf8_lossy(&output.stdout)
            );
            failed = true;
        }
    }

    if failed {
        panic!("Shader compilation failed");
    }

    // Validate individual SPIR-V files
    print!("Validating SPIR-V modules... ");
    let mut validation_failed = false;
    for spv_file in &spv_files {
        let output = Command::new("spirv-val")
            .args(["--scalar-block-layout", spv_file.to_str().unwrap()])
            .output()
            .expect("Failed to execute spirv-val");

        if !output.status.success() {
            if !validation_failed {
                println!("FAILED");
            }
            validation_failed = true;
            eprintln!(
                "spirv-val failed for {}: {}",
                spv_file.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    if validation_failed {
        panic!("SPIR-V validation failed");
    }
    println!("ok");

    // Export the SPIR-V output directory for the main crate
    println!("cargo:rustc-env=SPIRV_OUT_DIR={}", spirv_out_dir.display());
    println!(
        "cargo:warning=VTE diagnostics feature {}",
        if vte_diagnostics_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "cargo:warning=Slang shaders compiled to {}",
        spirv_out_dir.display()
    );
}

fn check_tool(name: &str, flag: &str, error_msg: &str) {
    let result = Command::new(name).arg(flag).output();
    if result.is_err() {
        panic!("{}", error_msg);
    }
}
