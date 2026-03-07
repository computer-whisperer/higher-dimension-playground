use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::region_tree::{Aabb4, RegionTreeCore, TesseractOrientation};

/// A structure type declared by a plugin in its manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructureDeclaration {
    pub id: u32,
    pub name: String,
    pub spawn_weight: u32,
}

/// Input for `OP_PROCGEN_PREPARE`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcgenPrepareInput {
    pub structure_id: u32,
    pub seed: u64,
    pub orientation: TesseractOrientation,
    pub origin: [i32; 4],
}

/// Output from `OP_PROCGEN_PREPARE`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcgenPrepareOutput {
    pub bounds: Aabb4,
    pub state: Vec<u8>,
}

/// Input for `OP_PROCGEN_GENERATE`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcgenGenerateInput {
    pub structure_id: u32,
    pub seed: u64,
    pub orientation: TesseractOrientation,
    pub origin: [i32; 4],
    pub state: Vec<u8>,
}

/// Output from `OP_PROCGEN_GENERATE`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcgenGenerateOutput {
    pub tree: RegionTreeCore,
}
