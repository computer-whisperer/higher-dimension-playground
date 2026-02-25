use super::runtime::{WasmCompiledModule, WasmRuntime, WasmRuntimeError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WasmModuleHash([u8; 32]);

impl WasmModuleHash {
    pub fn from_wasm_bytes(wasm_bytes: &[u8]) -> Self {
        let digest = Sha256::digest(wasm_bytes);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&digest);
        Self(hash)
    }

    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Display for WasmModuleHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum WasmModuleCacheError {
    Io(String),
    Runtime(WasmRuntimeError),
    MissingModule(WasmModuleHash),
}

impl fmt::Display for WasmModuleCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "wasm module cache io error: {message}"),
            Self::Runtime(error) => write!(f, "{error}"),
            Self::MissingModule(hash) => write!(
                f,
                "wasm module {hash} not found in memory cache or disk cache"
            ),
        }
    }
}

impl std::error::Error for WasmModuleCacheError {}

impl From<WasmRuntimeError> for WasmModuleCacheError {
    fn from(value: WasmRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

#[derive(Clone)]
pub struct WasmCachedModule {
    pub hash: WasmModuleHash,
    pub module: Arc<WasmCompiledModule>,
}

pub struct WasmModuleCache {
    runtime: Arc<WasmRuntime>,
    disk_dir: Option<PathBuf>,
    memory: RwLock<HashMap<WasmModuleHash, Arc<WasmCompiledModule>>>,
}

impl WasmModuleCache {
    pub fn new(
        runtime: Arc<WasmRuntime>,
        disk_dir: Option<PathBuf>,
    ) -> Result<Self, WasmModuleCacheError> {
        if let Some(dir) = disk_dir.as_ref() {
            fs::create_dir_all(dir).map_err(|error| {
                WasmModuleCacheError::Io(format!(
                    "failed to create cache dir {}: {error}",
                    dir.display()
                ))
            })?;
        }
        Ok(Self {
            runtime,
            disk_dir,
            memory: RwLock::new(HashMap::new()),
        })
    }

    pub fn compute_hash(&self, wasm_bytes: &[u8]) -> WasmModuleHash {
        WasmModuleHash::from_wasm_bytes(wasm_bytes)
    }

    pub fn insert_bytes(
        &self,
        wasm_bytes: &[u8],
    ) -> Result<WasmCachedModule, WasmModuleCacheError> {
        let hash = self.compute_hash(wasm_bytes);
        if let Some(cached) = self.get_from_memory(hash) {
            return Ok(WasmCachedModule {
                hash,
                module: cached,
            });
        }

        if let Some(path) = self.module_path(hash) {
            write_module_if_missing(&path, wasm_bytes)?;
        }

        let compiled = Arc::new(self.runtime.compile_module(wasm_bytes)?);
        {
            let mut guard = self.memory_write();
            guard.insert(hash, compiled.clone());
        }

        Ok(WasmCachedModule {
            hash,
            module: compiled,
        })
    }

    pub fn load_or_insert_by_hash(
        &self,
        hash: WasmModuleHash,
    ) -> Result<WasmCachedModule, WasmModuleCacheError> {
        if let Some(cached) = self.get_from_memory(hash) {
            return Ok(WasmCachedModule {
                hash,
                module: cached,
            });
        }

        let Some(path) = self.module_path(hash) else {
            return Err(WasmModuleCacheError::MissingModule(hash));
        };
        if !path.exists() {
            return Err(WasmModuleCacheError::MissingModule(hash));
        }
        let wasm_bytes = fs::read(&path).map_err(|error| {
            WasmModuleCacheError::Io(format!("failed to read {}: {error}", path.display()))
        })?;
        let check_hash = WasmModuleHash::from_wasm_bytes(&wasm_bytes);
        if check_hash != hash {
            return Err(WasmModuleCacheError::Io(format!(
                "cached wasm hash mismatch at {} (expected {hash}, found {check_hash})",
                path.display()
            )));
        }
        let compiled = Arc::new(self.runtime.compile_module(&wasm_bytes)?);
        {
            let mut guard = self.memory_write();
            guard.insert(hash, compiled.clone());
        }
        Ok(WasmCachedModule {
            hash,
            module: compiled,
        })
    }

    pub fn contains_memory(&self, hash: WasmModuleHash) -> bool {
        self.memory_read().contains_key(&hash)
    }

    pub fn remove_from_memory(&self, hash: WasmModuleHash) -> bool {
        self.memory_write().remove(&hash).is_some()
    }

    pub fn memory_entries(&self) -> usize {
        self.memory_read().len()
    }

    pub fn module_path(&self, hash: WasmModuleHash) -> Option<PathBuf> {
        self.disk_dir.as_ref().map(|dir| {
            let mut path = dir.clone();
            path.push(format!("{hash}.wasm"));
            path
        })
    }

    fn get_from_memory(&self, hash: WasmModuleHash) -> Option<Arc<WasmCompiledModule>> {
        self.memory_read().get(&hash).cloned()
    }

    fn memory_read(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<WasmModuleHash, Arc<WasmCompiledModule>>> {
        self.memory
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn memory_write(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, HashMap<WasmModuleHash, Arc<WasmCompiledModule>>> {
        self.memory
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn write_module_if_missing(path: &Path, wasm_bytes: &[u8]) -> Result<(), WasmModuleCacheError> {
    if path.exists() {
        return Ok(());
    }
    fs::write(path, wasm_bytes).map_err(|error| {
        WasmModuleCacheError::Io(format!("failed to write {}: {error}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::wasm::abi::WasmExecutionRole;
    use crate::shared::wasm::runtime::WasmExecutionLimits;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn valid_runtime_module() -> Vec<u8> {
        wat::parse_str(
            r#"
            (module
              (memory (export "memory") 4 4)
              (global $heap (mut i32) (i32.const 4096))
              (func (export "polychora_abi_version") (result i32) (i32.const 1))
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
                (param i32 i32 i32 i32 i32)
                (result i32)
                (i32.const 0)))
            "#,
        )
        .expect("wat should compile")
    }

    fn unique_test_cache_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough for tests")
            .as_nanos();
        dir.push(format!(
            "polychora-wasm-cache-test-{}-{now_nanos}",
            std::process::id()
        ));
        dir
    }

    #[test]
    fn cache_persists_wasm_and_can_reload_from_disk() {
        let runtime = Arc::new(WasmRuntime::new().expect("runtime should initialize"));
        let cache_dir = unique_test_cache_dir();
        let cache = WasmModuleCache::new(runtime.clone(), Some(cache_dir.clone()))
            .expect("cache should initialize");
        let wasm = valid_runtime_module();

        let first = cache.insert_bytes(&wasm).expect("insert should succeed");
        assert!(cache.contains_memory(first.hash));
        assert_eq!(cache.memory_entries(), 1);
        let on_disk_path = cache
            .module_path(first.hash)
            .expect("disk path should be available");
        assert!(on_disk_path.exists());

        let evicted = cache.remove_from_memory(first.hash);
        assert!(evicted);
        assert!(!cache.contains_memory(first.hash));

        let second = cache
            .load_or_insert_by_hash(first.hash)
            .expect("load by hash should succeed");
        assert!(cache.contains_memory(first.hash));

        let mut instance = runtime
            .instantiate_module(
                second.module.clone(),
                WasmExecutionRole::ClientSpeculative,
                WasmExecutionLimits::default(),
            )
            .expect("reloaded module should instantiate");
        let result = instance
            .call_bytes(1, &[])
            .expect("reloaded module should execute");
        assert_eq!(result.output.len(), 0);

        let _ = fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn hash_is_stable_for_identical_modules() {
        let runtime = Arc::new(WasmRuntime::new().expect("runtime should initialize"));
        let cache = WasmModuleCache::new(runtime, None).expect("cache should initialize");
        let wasm = valid_runtime_module();
        let hash_a = cache.compute_hash(&wasm);
        let hash_b = cache.compute_hash(&wasm);
        assert_eq!(hash_a, hash_b);
    }
}
