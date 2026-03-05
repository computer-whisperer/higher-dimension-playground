pub mod chunk_payload;
pub mod entity_types;
pub mod inventory;
pub mod item_types;
pub mod protocol;
pub mod region_tree;
pub mod render_tree;
pub mod spatial;
pub mod voxel;
pub mod wasm;

// ---------------------------------------------------------------------------
// Small utility functions shared across both the binary and library crates.
// ---------------------------------------------------------------------------

/// Returns `true` when the environment variable `name` is set to a non-empty,
/// non-falsy value.  Recognized false values: `""`, `"0"`, `"false"`, `"off"`,
/// `"no"` (case-insensitive).  An unset variable is treated as `false`.
pub fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

/// Like [`env_flag_enabled`], but returns `default_enabled` when the variable
/// is not set (instead of always returning `false`).
pub fn env_flag_enabled_or(name: &str, default_enabled: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => default_enabled,
    }
}

/// Normalize a 4-component vector, returning `fallback` when the input is
/// zero-length or contains non-finite values.
pub fn normalize4_with_fallback(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 || !len_sq.is_finite() {
        return fallback;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}
