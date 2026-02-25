use super::abi::WasmExecutionRole;
use super::cache::{WasmCachedModule, WasmModuleCache, WasmModuleCacheError, WasmModuleHash};
use super::runtime::{
    WasmExecutionLimits, WasmInvocationResult, WasmRuntime, WasmRuntimeError, WasmRuntimeInstance,
};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WasmPluginSlot {
    MobSteering,
    ModelLogic,
}

#[derive(Debug)]
pub enum WasmPluginManagerError {
    Cache(WasmModuleCacheError),
    Runtime(WasmRuntimeError),
}

impl fmt::Display for WasmPluginManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => write!(f, "{error}"),
            Self::Runtime(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for WasmPluginManagerError {}

impl From<WasmModuleCacheError> for WasmPluginManagerError {
    fn from(value: WasmModuleCacheError) -> Self {
        Self::Cache(value)
    }
}

impl From<WasmRuntimeError> for WasmPluginManagerError {
    fn from(value: WasmRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

#[derive(Debug)]
pub struct WasmSlotCallResult {
    pub module_hash: WasmModuleHash,
    pub invocation: WasmInvocationResult,
}

struct ActiveSlot {
    module_hash: WasmModuleHash,
    instance: WasmRuntimeInstance,
}

pub struct WasmPluginManager {
    role: WasmExecutionRole,
    runtime: Arc<WasmRuntime>,
    cache: Arc<WasmModuleCache>,
    active_slots: HashMap<WasmPluginSlot, ActiveSlot>,
}

impl WasmPluginManager {
    pub fn new(
        role: WasmExecutionRole,
        runtime: Arc<WasmRuntime>,
        cache: Arc<WasmModuleCache>,
    ) -> Self {
        Self {
            role,
            runtime,
            cache,
            active_slots: HashMap::new(),
        }
    }

    pub fn role(&self) -> WasmExecutionRole {
        self.role
    }

    pub fn is_authoritative(&self) -> bool {
        self.role.is_authoritative()
    }

    pub fn activate_slot_from_bytes(
        &mut self,
        slot: WasmPluginSlot,
        wasm_bytes: &[u8],
        limits: WasmExecutionLimits,
    ) -> Result<WasmModuleHash, WasmPluginManagerError> {
        let cached = self.cache.insert_bytes(wasm_bytes)?;
        self.activate_slot_from_cached(slot, cached, limits)
    }

    pub fn activate_slot_from_hash(
        &mut self,
        slot: WasmPluginSlot,
        module_hash: WasmModuleHash,
        limits: WasmExecutionLimits,
    ) -> Result<(), WasmPluginManagerError> {
        let cached = self.cache.load_or_insert_by_hash(module_hash)?;
        let _ = self.activate_slot_from_cached(slot, cached, limits)?;
        Ok(())
    }

    fn activate_slot_from_cached(
        &mut self,
        slot: WasmPluginSlot,
        cached: WasmCachedModule,
        limits: WasmExecutionLimits,
    ) -> Result<WasmModuleHash, WasmPluginManagerError> {
        let instance = self
            .runtime
            .instantiate_module(cached.module.clone(), self.role, limits)?;
        self.active_slots.insert(
            slot,
            ActiveSlot {
                module_hash: cached.hash,
                instance,
            },
        );
        Ok(cached.hash)
    }

    pub fn deactivate_slot(&mut self, slot: WasmPluginSlot) -> bool {
        self.active_slots.remove(&slot).is_some()
    }

    pub fn active_module_hash(&self, slot: WasmPluginSlot) -> Option<WasmModuleHash> {
        self.active_slots
            .get(&slot)
            .map(|active| active.module_hash)
    }

    pub fn call_slot(
        &mut self,
        slot: WasmPluginSlot,
        opcode: i32,
        input: &[u8],
    ) -> Result<Option<WasmSlotCallResult>, WasmPluginManagerError> {
        let Some(active) = self.active_slots.get_mut(&slot) else {
            return Ok(None);
        };
        let invocation = active.instance.call_bytes(opcode, input)?;
        Ok(Some(WasmSlotCallResult {
            module_hash: active.module_hash,
            invocation,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::wasm::cache::WasmModuleCache;
    use std::sync::Arc;

    fn valid_runtime_module() -> Vec<u8> {
        wat::parse_str(
            r#"
            (module
              (memory (export "memory") 4 4)
              (global $heap (mut i32) (i32.const 4096))
              (func (export "polychora_abi_version") (result i32)
                (i32.const 1))
              (func (export "polychora_alloc") (param $len i32) (result i32)
                (local $ptr i32)
                (global.get $heap)
                (local.tee $ptr)
                (local.get $len)
                (i32.add)
                (global.set $heap)
                (local.get $ptr))
              (func (export "polychora_free") (param i32 i32))
              (func (export "polychora_call")
                (param $opcode i32)
                (param i32 i32)
                (param $out_ptr i32)
                (param $out_cap i32)
                (result i32)
                (local.get $out_cap)
                (i32.const 4)
                (i32.lt_s)
                (if (result i32)
                  (then
                    (i32.const -9))
                  (else
                    (local.get $out_ptr)
                    (local.get $opcode)
                    (i32.store)
                    (i32.const 4)))))
            "#,
        )
        .expect("wat should compile")
    }

    #[test]
    fn manager_activates_slot_and_calls_module() {
        let runtime = Arc::new(WasmRuntime::new().expect("runtime should initialize"));
        let cache = Arc::new(WasmModuleCache::new(runtime.clone(), None).expect("cache init"));
        let mut manager =
            WasmPluginManager::new(WasmExecutionRole::ServerAuthoritative, runtime, cache);
        assert!(manager.is_authoritative());

        let hash = manager
            .activate_slot_from_bytes(
                WasmPluginSlot::MobSteering,
                &valid_runtime_module(),
                WasmExecutionLimits::default(),
            )
            .expect("activate should succeed");
        assert_eq!(
            manager.active_module_hash(WasmPluginSlot::MobSteering),
            Some(hash)
        );

        let called = manager
            .call_slot(WasmPluginSlot::MobSteering, 42, &[9, 8, 7])
            .expect("call should succeed")
            .expect("slot should be active");
        assert_eq!(called.module_hash, hash);
        assert_eq!(called.invocation.output, 42i32.to_le_bytes().to_vec());
    }

    #[test]
    fn manager_returns_none_for_inactive_slot() {
        let runtime = Arc::new(WasmRuntime::new().expect("runtime should initialize"));
        let cache = Arc::new(WasmModuleCache::new(runtime.clone(), None).expect("cache init"));
        let mut manager =
            WasmPluginManager::new(WasmExecutionRole::ClientSpeculative, runtime, cache);
        assert!(!manager.is_authoritative());

        let result = manager
            .call_slot(WasmPluginSlot::ModelLogic, 1, &[])
            .expect("call should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn manager_deactivates_slot() {
        let runtime = Arc::new(WasmRuntime::new().expect("runtime should initialize"));
        let cache = Arc::new(WasmModuleCache::new(runtime.clone(), None).expect("cache init"));
        let mut manager =
            WasmPluginManager::new(WasmExecutionRole::ServerAuthoritative, runtime, cache);

        let _ = manager
            .activate_slot_from_bytes(
                WasmPluginSlot::MobSteering,
                &valid_runtime_module(),
                WasmExecutionLimits::default(),
            )
            .expect("activate should succeed");
        assert!(manager.deactivate_slot(WasmPluginSlot::MobSteering));
        assert_eq!(
            manager.active_module_hash(WasmPluginSlot::MobSteering),
            None
        );
    }
}
