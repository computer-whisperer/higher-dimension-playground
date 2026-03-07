use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    build_wasm_plugin();
}

/// Compile the `polychora-content` WASM plugin as part of the host build.
///
/// Uses a separate `--target-dir` to avoid Cargo lock deadlock (the host build
/// already holds a lock on the workspace target directory).
fn build_wasm_plugin() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = PathBuf::from(&manifest_dir)
        .parent() // crates/
        .unwrap()
        .parent() // workspace root
        .unwrap()
        .to_path_buf();

    let content_manifest = workspace_root
        .join("crates")
        .join("polychora-content")
        .join("Cargo.toml");

    let out_dir = env::var("OUT_DIR").unwrap();
    let wasm_target_dir = PathBuf::from(&out_dir).join("wasm-target");

    // Set max WASM memory to 1GB (16384 pages). Without this, the memory section
    // is unbounded and wasmtime rejects it. Large limit accommodates procgen
    // structure/maze generation which builds large region trees and accumulates
    // allocations across many calls within a single instance.
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .arg("--manifest-path")
        .arg(&content_manifest)
        .arg("--target-dir")
        .arg(&wasm_target_dir)
        .env(
            "CARGO_ENCODED_RUSTFLAGS",
            "-Clink-args=--max-memory=1073741824",
        )
        .status()
        .expect("failed to invoke cargo for polychora-content WASM build");

    if !status.success() {
        panic!("polychora-content WASM build failed");
    }

    let wasm_path = wasm_target_dir
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("polychora_content.wasm");

    if !wasm_path.exists() {
        panic!(
            "expected WASM output at {} but file does not exist",
            wasm_path.display()
        );
    }

    println!(
        "cargo:rustc-env=POLYCHORA_CONTENT_WASM_PATH={}",
        wasm_path.display()
    );

    // Rerun if the WASM plugin sources change.
    // Watch individual files, not directories — `cargo:rerun-if-changed` on a
    // directory only fires when files are added/removed, not when existing files
    // are modified.
    println!("cargo:rerun-if-changed={}", content_manifest.display());
    let watch_dirs = [
        workspace_root.join("crates").join("polychora-content").join("src"),
        workspace_root.join("crates").join("polychora-plugin-api").join("src"),
    ];
    for dir in &watch_dirs {
        watch_dir_recursive(dir);
    }
}

fn watch_dir_recursive(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            watch_dir_recursive(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
