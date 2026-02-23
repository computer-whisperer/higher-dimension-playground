/// Opcode constants for the WASM plugin ABI.
///
/// Each opcode identifies a specific host↔guest call via
/// `polychora_call(opcode, in_ptr, in_len, out_ptr, out_cap) -> i32`.
///
/// Stubbed for Phase 1; will be fleshed out in Phase 2.

pub const OP_GET_MANIFEST: u32 = 0x0010;
pub const OP_GET_TEXTURES: u32 = 0x0011;
pub const OP_MOB_STEERING: u32 = 0x0100;
pub const OP_MOB_SPECIAL_ABILITY: u32 = 0x0101;
pub const OP_BLOCK_INTERACT: u32 = 0x0200;
