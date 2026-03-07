/// Opcode constants for the WASM plugin ABI.
///
/// Each opcode identifies a specific host↔guest call via
/// `polychora_call(opcode, in_ptr, in_len, out_ptr, out_cap) -> i32`.
///
/// On success the return value is the number of bytes written to the output
/// buffer. On failure the return value is a negative `ABI_ERR_*` constant.
pub const OP_GET_MANIFEST: u32 = 0x0010;
pub const OP_GET_TEXTURES: u32 = 0x0011;
pub const OP_ENTITY_TICK: u32 = 0x0100;
pub const OP_ENTITY_ABILITY: u32 = 0x0101;
pub const OP_BLOCK_INTERACT: u32 = 0x0200;
pub const OP_BLOCK_TICK: u32 = 0x0201;
pub const OP_GUI_TICK: u32 = 0x0210;
pub const OP_GUI_ACTION: u32 = 0x0211;
pub const OP_GUI_CLOSE: u32 = 0x0212;
pub const OP_ENTITY_MODEL: u32 = 0x0300;
pub const OP_PROCGEN_PREPARE: u32 = 0x0400;
pub const OP_PROCGEN_GENERATE: u32 = 0x0401;

/// ABI error codes returned by `polychora_call`.
pub const ABI_ERR_UNKNOWN_OPCODE: i32 = -1;
pub const ABI_ERR_SERIALIZE: i32 = -2;
pub const ABI_ERR_OUTPUT_TOO_LARGE: i32 = -3;
