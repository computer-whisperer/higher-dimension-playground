use super::abi::{
    WasmExecutionRole, WASM_ABI_EXPORT_ALLOC, WASM_ABI_EXPORT_CALL, WASM_ABI_EXPORT_FREE,
    WASM_ABI_EXPORT_VERSION, WASM_ABI_MEMORY_EXPORT, WASM_ABI_VERSION_V1,
};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use wasmtime::{Config, Engine, Memory, Module, Store, TypedFunc};

#[derive(Clone, Copy, Debug)]
pub struct WasmExecutionLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_memory_pages: u32,
    pub max_fuel: u64,
}

impl Default for WasmExecutionLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 64 * 1024,
            max_output_bytes: 64 * 1024,
            max_memory_pages: 64,
            max_fuel: 200_000,
        }
    }
}

#[derive(Debug)]
pub enum WasmRuntimeError {
    EngineInit(String),
    ModuleCompile(String),
    Instantiate(String),
    MissingExport(&'static str),
    MissingMemoryExport,
    MemoryUnbounded {
        allowed_max_pages: u32,
    },
    MemoryLimitExceeded {
        required_pages: u64,
        allowed_max_pages: u32,
    },
    InvalidAbiVersion {
        expected: i32,
        found: i32,
    },
    InputTooLarge {
        actual: usize,
        max: usize,
    },
    OutputTooLarge {
        actual: usize,
        max: usize,
    },
    InvalidGuestPointer(i32),
    GuestMemoryOutOfBounds {
        ptr: i32,
        len: usize,
        memory_len: usize,
    },
    GuestFunctionError {
        function: &'static str,
        message: String,
    },
    GuestErrorCode(i32),
    IntegerOverflow(&'static str),
}

impl fmt::Display for WasmRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EngineInit(message) => write!(f, "wasm engine init failed: {message}"),
            Self::ModuleCompile(message) => write!(f, "wasm module compile failed: {message}"),
            Self::Instantiate(message) => write!(f, "wasm module instantiate failed: {message}"),
            Self::MissingExport(name) => write!(f, "wasm module missing export '{name}'"),
            Self::MissingMemoryExport => write!(
                f,
                "wasm module missing required exported memory '{WASM_ABI_MEMORY_EXPORT}'"
            ),
            Self::MemoryUnbounded { allowed_max_pages } => write!(
                f,
                "wasm memory maximum is unbounded (allowed max pages: {allowed_max_pages})"
            ),
            Self::MemoryLimitExceeded {
                required_pages,
                allowed_max_pages,
            } => write!(
                f,
                "wasm memory requires {required_pages} pages, exceeds allowed max {allowed_max_pages}"
            ),
            Self::InvalidAbiVersion { expected, found } => write!(
                f,
                "unsupported wasm abi version: expected {expected}, found {found}"
            ),
            Self::InputTooLarge { actual, max } => {
                write!(f, "wasm input too large: {actual} bytes (max {max})")
            }
            Self::OutputTooLarge { actual, max } => {
                write!(f, "wasm output too large: {actual} bytes (max {max})")
            }
            Self::InvalidGuestPointer(ptr) => write!(f, "invalid guest pointer {ptr}"),
            Self::GuestMemoryOutOfBounds {
                ptr,
                len,
                memory_len,
            } => write!(
                f,
                "guest memory out-of-bounds ptr={ptr} len={len} memory_len={memory_len}"
            ),
            Self::GuestFunctionError { function, message } => {
                write!(f, "guest function '{function}' failed: {message}")
            }
            Self::GuestErrorCode(code) => {
                write!(f, "guest returned error code {code}")
            }
            Self::IntegerOverflow(field) => {
                write!(f, "integer overflow converting {field}")
            }
        }
    }
}

impl std::error::Error for WasmRuntimeError {}

#[derive(Clone)]
pub struct WasmCompiledModule {
    module: Module,
    wasm_len: usize,
}

impl WasmCompiledModule {
    pub fn wasm_len(&self) -> usize {
        self.wasm_len
    }
}

#[derive(Clone, Copy, Debug)]
struct HostState {
    role: WasmExecutionRole,
}

pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    pub fn new() -> Result<Self, WasmRuntimeError> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|error| {
            WasmRuntimeError::EngineInit(format!("failed to create wasmtime engine: {error}"))
        })?;
        Ok(Self { engine })
    }

    pub fn compile_module(
        &self,
        wasm_bytes: &[u8],
    ) -> Result<WasmCompiledModule, WasmRuntimeError> {
        let module = Module::from_binary(&self.engine, wasm_bytes).map_err(|error| {
            WasmRuntimeError::ModuleCompile(format!("module bytes were rejected: {error}"))
        })?;
        Ok(WasmCompiledModule {
            module,
            wasm_len: wasm_bytes.len(),
        })
    }

    pub fn instantiate_module(
        &self,
        compiled: Arc<WasmCompiledModule>,
        role: WasmExecutionRole,
        limits: WasmExecutionLimits,
    ) -> Result<WasmRuntimeInstance, WasmRuntimeError> {
        let linker = wasmtime::Linker::new(&self.engine);
        let mut store = Store::new(&self.engine, HostState { role });
        store.set_fuel(limits.max_fuel).map_err(|error| {
            WasmRuntimeError::Instantiate(format!("failed to initialize fuel budget: {error}"))
        })?;
        let instance = linker
            .instantiate(&mut store, &compiled.module)
            .map_err(|error| WasmRuntimeError::Instantiate(error.to_string()))?;
        let memory = instance
            .get_memory(&mut store, WASM_ABI_MEMORY_EXPORT)
            .ok_or(WasmRuntimeError::MissingMemoryExport)?;

        let memory_ty = memory.ty(&store);
        let min_pages: u64 = memory_ty.minimum().into();
        let max_pages =
            memory_ty
                .maximum()
                .map(Into::into)
                .ok_or(WasmRuntimeError::MemoryUnbounded {
                    allowed_max_pages: limits.max_memory_pages,
                })?;
        let allowed_max_pages = u64::from(limits.max_memory_pages);
        if min_pages > allowed_max_pages {
            return Err(WasmRuntimeError::MemoryLimitExceeded {
                required_pages: min_pages,
                allowed_max_pages: limits.max_memory_pages,
            });
        }
        if max_pages > allowed_max_pages {
            return Err(WasmRuntimeError::MemoryLimitExceeded {
                required_pages: max_pages,
                allowed_max_pages: limits.max_memory_pages,
            });
        }

        let abi_version = instance
            .get_typed_func::<(), i32>(&mut store, WASM_ABI_EXPORT_VERSION)
            .map_err(|_| WasmRuntimeError::MissingExport(WASM_ABI_EXPORT_VERSION))?;
        let guest_abi_version = abi_version.call(&mut store, ()).map_err(|error| {
            WasmRuntimeError::GuestFunctionError {
                function: WASM_ABI_EXPORT_VERSION,
                message: error.to_string(),
            }
        })?;
        if guest_abi_version != WASM_ABI_VERSION_V1 {
            return Err(WasmRuntimeError::InvalidAbiVersion {
                expected: WASM_ABI_VERSION_V1,
                found: guest_abi_version,
            });
        }

        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, WASM_ABI_EXPORT_ALLOC)
            .map_err(|_| WasmRuntimeError::MissingExport(WASM_ABI_EXPORT_ALLOC))?;
        let _free = instance
            .get_typed_func::<(i32, i32), ()>(&mut store, WASM_ABI_EXPORT_FREE)
            .map_err(|_| WasmRuntimeError::MissingExport(WASM_ABI_EXPORT_FREE))?;
        let call = instance
            .get_typed_func::<(i32, i32, i32, i32, i32), i32>(&mut store, WASM_ABI_EXPORT_CALL)
            .map_err(|_| WasmRuntimeError::MissingExport(WASM_ABI_EXPORT_CALL))?;

        let io_in_cap = i32::try_from(limits.max_input_bytes)
            .map_err(|_| WasmRuntimeError::IntegerOverflow("io_in_cap"))?;
        let io_out_cap = i32::try_from(limits.max_output_bytes)
            .map_err(|_| WasmRuntimeError::IntegerOverflow("io_out_cap"))?;

        let io_in_ptr = alloc.call(&mut store, io_in_cap).map_err(|error| {
            WasmRuntimeError::GuestFunctionError {
                function: WASM_ABI_EXPORT_ALLOC,
                message: format!("failed to pre-allocate input buffer: {error}"),
            }
        })?;
        if io_in_ptr < 0 {
            return Err(WasmRuntimeError::InvalidGuestPointer(io_in_ptr));
        }

        let io_out_ptr = alloc.call(&mut store, io_out_cap).map_err(|error| {
            WasmRuntimeError::GuestFunctionError {
                function: WASM_ABI_EXPORT_ALLOC,
                message: format!("failed to pre-allocate output buffer: {error}"),
            }
        })?;
        if io_out_ptr < 0 {
            return Err(WasmRuntimeError::InvalidGuestPointer(io_out_ptr));
        }

        Ok(WasmRuntimeInstance {
            compiled,
            limits,
            store,
            memory,
            call,
            io_in_ptr,
            io_in_cap,
            io_out_ptr,
            io_out_cap,
        })
    }
}

#[derive(Debug)]
pub struct WasmInvocationResult {
    pub output: Vec<u8>,
    pub fuel_used: Option<u64>,
}

pub struct WasmRuntimeInstance {
    compiled: Arc<WasmCompiledModule>,
    limits: WasmExecutionLimits,
    store: Store<HostState>,
    memory: Memory,
    call: TypedFunc<(i32, i32, i32, i32, i32), i32>,
    io_in_ptr: i32,
    io_in_cap: i32,
    io_out_ptr: i32,
    io_out_cap: i32,
}

impl WasmRuntimeInstance {
    pub fn role(&self) -> WasmExecutionRole {
        self.store.data().role
    }

    pub fn limits(&self) -> WasmExecutionLimits {
        self.limits
    }

    pub fn module_wasm_len(&self) -> usize {
        self.compiled.wasm_len()
    }

    pub fn call_bytes(
        &mut self,
        opcode: i32,
        input: &[u8],
    ) -> Result<WasmInvocationResult, WasmRuntimeError> {
        if input.len() > self.io_in_cap as usize {
            return Err(WasmRuntimeError::InputTooLarge {
                actual: input.len(),
                max: self.io_in_cap as usize,
            });
        }

        let input_len_i32 = i32::try_from(input.len())
            .map_err(|_| WasmRuntimeError::IntegerOverflow("input_len"))?;

        self.store.set_fuel(self.limits.max_fuel).map_err(|error| {
            WasmRuntimeError::GuestFunctionError {
                function: WASM_ABI_EXPORT_CALL,
                message: format!("failed to reset fuel: {error}"),
            }
        })?;

        self.write_memory(self.io_in_ptr, input)?;

        let call_result = self.call.call(
            &mut self.store,
            (opcode, self.io_in_ptr, input_len_i32, self.io_out_ptr, self.io_out_cap),
        );

        let output = match call_result {
            Ok(return_len) => {
                if return_len < 0 {
                    Err(WasmRuntimeError::GuestErrorCode(return_len))
                } else {
                    let output_len = usize::try_from(return_len)
                        .map_err(|_| WasmRuntimeError::IntegerOverflow("guest_output_len"))?;
                    if output_len > self.io_out_cap as usize {
                        Err(WasmRuntimeError::OutputTooLarge {
                            actual: output_len,
                            max: self.io_out_cap as usize,
                        })
                    } else {
                        self.read_memory(self.io_out_ptr, output_len)
                    }
                }
            }
            Err(error) => Err(WasmRuntimeError::GuestFunctionError {
                function: WASM_ABI_EXPORT_CALL,
                message: error.to_string(),
            }),
        };

        let output = output?;
        let fuel_used = self
            .store
            .get_fuel()
            .ok()
            .map(|remaining| self.limits.max_fuel.saturating_sub(remaining));
        Ok(WasmInvocationResult { output, fuel_used })
    }

    fn write_memory(&mut self, ptr: i32, bytes: &[u8]) -> Result<(), WasmRuntimeError> {
        let data = self.memory.data_mut(&mut self.store);
        let range = checked_memory_range(ptr, bytes.len(), data.len())?;
        data[range].copy_from_slice(bytes);
        Ok(())
    }

    fn read_memory(&mut self, ptr: i32, len: usize) -> Result<Vec<u8>, WasmRuntimeError> {
        let data = self.memory.data(&self.store);
        let range = checked_memory_range(ptr, len, data.len())?;
        Ok(data[range].to_vec())
    }
}

fn checked_memory_range(
    ptr: i32,
    len: usize,
    memory_len: usize,
) -> Result<Range<usize>, WasmRuntimeError> {
    if ptr < 0 {
        return Err(WasmRuntimeError::InvalidGuestPointer(ptr));
    }
    let start = ptr as usize;
    let end = start
        .checked_add(len)
        .ok_or(WasmRuntimeError::IntegerOverflow("memory_range_end"))?;
    if end > memory_len {
        return Err(WasmRuntimeError::GuestMemoryOutOfBounds {
            ptr,
            len,
            memory_len,
        });
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                (param $in_ptr i32)
                (param $in_len i32)
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
    fn valid_module_instantiates_and_invokes() {
        let runtime = WasmRuntime::new().expect("runtime should initialize");
        let compiled = Arc::new(
            runtime
                .compile_module(&valid_runtime_module())
                .expect("module should compile"),
        );
        let mut instance = runtime
            .instantiate_module(
                compiled,
                WasmExecutionRole::ServerAuthoritative,
                WasmExecutionLimits::default(),
            )
            .expect("module should instantiate");

        let result = instance
            .call_bytes(17, &[1, 2, 3, 4])
            .expect("guest call should succeed");
        assert_eq!(result.output, 17i32.to_le_bytes().to_vec());
        assert!(result.fuel_used.is_some());
        assert!(instance.role().is_authoritative());
    }

    #[test]
    fn reject_mismatched_abi_version() {
        let runtime = WasmRuntime::new().expect("runtime should initialize");
        let wasm = wat::parse_str(
            r#"
            (module
              (memory (export "memory") 1 2)
              (func (export "polychora_abi_version") (result i32) (i32.const 2))
              (func (export "polychora_alloc") (param i32) (result i32) (i32.const 1024))
              (func (export "polychora_free") (param i32 i32))
              (func (export "polychora_call") (param i32 i32 i32 i32 i32) (result i32) (i32.const 0)))
            "#,
        )
        .expect("wat should compile");
        let compiled = Arc::new(
            runtime
                .compile_module(&wasm)
                .expect("module should compile"),
        );
        let error = match runtime.instantiate_module(
            compiled,
            WasmExecutionRole::ServerAuthoritative,
            WasmExecutionLimits::default(),
        ) {
            Ok(_) => panic!("instantiate should fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            WasmRuntimeError::InvalidAbiVersion {
                expected: WASM_ABI_VERSION_V1,
                found: 2
            }
        ));
    }

    #[test]
    fn reject_missing_required_export() {
        let runtime = WasmRuntime::new().expect("runtime should initialize");
        let wasm = wat::parse_str(
            r#"
            (module
              (memory (export "memory") 1 2)
              (func (export "polychora_abi_version") (result i32) (i32.const 1))
              (func (export "polychora_alloc") (param i32) (result i32) (i32.const 1024))
              (func (export "polychora_free") (param i32 i32)))
            "#,
        )
        .expect("wat should compile");
        let compiled = Arc::new(
            runtime
                .compile_module(&wasm)
                .expect("module should compile"),
        );
        let error = match runtime.instantiate_module(
            compiled,
            WasmExecutionRole::ServerAuthoritative,
            WasmExecutionLimits::default(),
        ) {
            Ok(_) => panic!("instantiate should fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            WasmRuntimeError::MissingExport(WASM_ABI_EXPORT_CALL)
        ));
    }

    #[test]
    fn enforce_input_limit() {
        let runtime = WasmRuntime::new().expect("runtime should initialize");
        let compiled = Arc::new(
            runtime
                .compile_module(&valid_runtime_module())
                .expect("module should compile"),
        );
        let mut limits = WasmExecutionLimits::default();
        limits.max_input_bytes = 2;
        let mut instance = runtime
            .instantiate_module(compiled, WasmExecutionRole::ServerAuthoritative, limits)
            .expect("module should instantiate");
        let error = instance
            .call_bytes(1, &[1, 2, 3])
            .expect_err("call should fail");
        assert!(matches!(
            error,
            WasmRuntimeError::InputTooLarge { actual: 3, max: 2 }
        ));
    }
}
