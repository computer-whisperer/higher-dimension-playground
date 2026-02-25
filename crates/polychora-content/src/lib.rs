#![no_std]
extern crate alloc;

mod blocks;
mod entities;
mod entity_tick;
mod math4d;
mod models;

#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

use alloc::vec::Vec;
use polychora_plugin_api::entity_tick_abi::{
    EntityAbilityCheck, EntityAbilityResult, EntityTickInput, EntityTickOutput,
};
use polychora_plugin_api::manifest::PluginManifest;
use polychora_plugin_api::model_abi::{EntityModelInput, EntityModelOutput};
use polychora_plugin_api::opcodes::{
    ABI_ERR_OUTPUT_TOO_LARGE, ABI_ERR_SERIALIZE, ABI_ERR_UNKNOWN_OPCODE, OP_ENTITY_ABILITY,
    OP_ENTITY_MODEL, OP_ENTITY_TICK, OP_GET_MANIFEST,
};

/// Namespace ID for the first-party polychora-content plugin.
pub const NAMESPACE: u32 = 0x706f6c79; // "poly" in ASCII

#[no_mangle]
pub extern "C" fn polychora_abi_version() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn polychora_alloc(len: i32) -> i32 {
    let layout = match core::alloc::Layout::from_size_align(len as usize, 1) {
        Ok(layout) => layout,
        Err(_) => return 0,
    };
    if layout.size() == 0 {
        return 1; // non-null sentinel for zero-size
    }
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        0
    } else {
        ptr as i32
    }
}

#[no_mangle]
pub extern "C" fn polychora_free(ptr: i32, len: i32) {
    if ptr <= 0 || len <= 0 {
        return;
    }
    let layout = match core::alloc::Layout::from_size_align(len as usize, 1) {
        Ok(layout) => layout,
        Err(_) => return,
    };
    if layout.size() == 0 {
        return;
    }
    unsafe {
        alloc::alloc::dealloc(ptr as *mut u8, layout);
    }
}

#[no_mangle]
pub extern "C" fn polychora_call(
    opcode: i32,
    in_ptr: i32,
    in_len: i32,
    out_ptr: i32,
    out_cap: i32,
) -> i32 {
    match opcode as u32 {
        OP_GET_MANIFEST => handle_get_manifest(out_ptr, out_cap),
        OP_ENTITY_TICK => handle_entity_tick(in_ptr, in_len, out_ptr, out_cap),
        OP_ENTITY_ABILITY => handle_entity_ability(in_ptr, in_len, out_ptr, out_cap),
        OP_ENTITY_MODEL => handle_entity_model(in_ptr, in_len, out_ptr, out_cap),
        _ => ABI_ERR_UNKNOWN_OPCODE,
    }
}

fn handle_get_manifest(out_ptr: i32, out_cap: i32) -> i32 {
    let manifest = build_manifest();
    let bytes = match postcard::to_allocvec(&manifest) {
        Ok(bytes) => bytes,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    write_output(&bytes, out_ptr, out_cap)
}

fn handle_entity_tick(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32 {
    let input_bytes = read_input(in_ptr, in_len);
    let input: EntityTickInput = match postcard::from_bytes(&input_bytes) {
        Ok(v) => v,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    let output: EntityTickOutput = entity_tick::entity_tick(&input);
    let bytes = match postcard::to_allocvec(&output) {
        Ok(bytes) => bytes,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    write_output(&bytes, out_ptr, out_cap)
}

fn handle_entity_ability(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32 {
    let input_bytes = read_input(in_ptr, in_len);
    let check: EntityAbilityCheck = match postcard::from_bytes(&input_bytes) {
        Ok(v) => v,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    let result: EntityAbilityResult = entity_tick::entity_ability_check(&check);
    let bytes = match postcard::to_allocvec(&result) {
        Ok(bytes) => bytes,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    write_output(&bytes, out_ptr, out_cap)
}

fn handle_entity_model(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32 {
    let input_bytes = read_input(in_ptr, in_len);
    let input: EntityModelInput = match postcard::from_bytes(&input_bytes) {
        Ok(v) => v,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    let output: EntityModelOutput = models::entity_model(&input);
    let bytes = match postcard::to_allocvec(&output) {
        Ok(bytes) => bytes,
        Err(_) => return ABI_ERR_SERIALIZE,
    };
    write_output(&bytes, out_ptr, out_cap)
}

fn read_input(in_ptr: i32, in_len: i32) -> Vec<u8> {
    if in_len <= 0 || in_ptr <= 0 {
        return Vec::new();
    }
    let slice = unsafe { core::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) };
    slice.to_vec()
}

fn write_output(bytes: &[u8], out_ptr: i32, out_cap: i32) -> i32 {
    let len = bytes.len();
    if len > out_cap as usize {
        return ABI_ERR_OUTPUT_TOO_LARGE;
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr as *mut u8, len);
    }
    len as i32
}

fn build_manifest() -> PluginManifest {
    PluginManifest {
        namespace_id: NAMESPACE,
        name: alloc::string::String::from("polychora-content"),
        version: [0, 1, 0],
        blocks: blocks::block_declarations(),
        entities: entities::entity_declarations(),
        items: Vec::new(),
        textures: Vec::new(),
    }
}
