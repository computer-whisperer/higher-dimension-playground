use serde::{Deserialize, Serialize};

/// A reference to a texture identified by its owning namespace and a
/// per-namespace unique texture ID.  Both values are stable random u32s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureRef {
    pub namespace: u32,
    pub texture_id: u32,
}
