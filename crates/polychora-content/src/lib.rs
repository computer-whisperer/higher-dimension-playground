#![no_std]
extern crate alloc;

mod blocks;
mod entities;

#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

use alloc::vec::Vec;
use polychora_plugin_api::manifest::PluginManifest;
use polychora_plugin_api::opcodes::OP_GET_MANIFEST;

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
    _in_ptr: i32,
    _in_len: i32,
    out_ptr: i32,
    out_cap: i32,
) -> i32 {
    match opcode as u32 {
        OP_GET_MANIFEST => handle_get_manifest(out_ptr, out_cap),
        _ => -1, // unknown opcode
    }
}

fn handle_get_manifest(out_ptr: i32, out_cap: i32) -> i32 {
    let manifest = build_manifest();
    let bytes = match postcard::to_allocvec(&manifest) {
        Ok(bytes) => bytes,
        Err(_) => return -2, // serialization error
    };
    let len = bytes.len();
    if len > out_cap as usize {
        return -3; // output too large
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
