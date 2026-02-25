use serde::{Deserialize, Serialize};

pub const WASM_ABI_VERSION_V1: i32 = 1;
pub const WASM_ABI_MEMORY_EXPORT: &str = "memory";
pub const WASM_ABI_EXPORT_VERSION: &str = "polychora_abi_version";
pub const WASM_ABI_EXPORT_ALLOC: &str = "polychora_alloc";
pub const WASM_ABI_EXPORT_FREE: &str = "polychora_free";
pub const WASM_ABI_EXPORT_CALL: &str = "polychora_call";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasmExecutionRole {
    ServerAuthoritative,
    ClientSpeculative,
}

impl WasmExecutionRole {
    pub fn is_authoritative(self) -> bool {
        matches!(self, Self::ServerAuthoritative)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i32)]
pub enum WasmCallOpcode {
    EntitySimulation = 1,
    ModelLogic = 2,
}

impl WasmCallOpcode {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

impl TryFrom<i32> for WasmCallOpcode {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::EntitySimulation),
            2 => Ok(Self::ModelLogic),
            _ => Err(()),
        }
    }
}
