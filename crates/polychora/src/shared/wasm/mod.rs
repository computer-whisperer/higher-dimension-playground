// Core wasm runtime/cache infrastructure lives here. Gameplay systems are not
// wired to use this yet; integration happens in later phases.
pub mod abi;
pub mod cache;
pub mod manager;
pub mod runtime;

pub use self::abi::{
    WasmCallOpcode, WasmExecutionRole, WASM_ABI_EXPORT_ALLOC, WASM_ABI_EXPORT_CALL,
    WASM_ABI_EXPORT_FREE, WASM_ABI_EXPORT_VERSION, WASM_ABI_MEMORY_EXPORT, WASM_ABI_VERSION_V1,
};
pub use self::cache::{WasmModuleCache, WasmModuleCacheError, WasmModuleHash};
pub use self::manager::{
    WasmPluginManager, WasmPluginManagerError, WasmPluginSlot, WasmSlotCallResult,
};
pub use self::runtime::{
    WasmCompiledModule, WasmExecutionLimits, WasmInvocationResult, WasmRuntime, WasmRuntimeError,
    WasmRuntimeInstance,
};
