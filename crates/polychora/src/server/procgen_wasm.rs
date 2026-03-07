use crate::server::procgen::StructurePlacementConfig;
use crate::shared::region_tree::{region_tree_from_plugin, RegionTreeCore};
use crate::shared::wasm::{
    WasmExecutionLimits, WasmExecutionRole, WasmModuleCache, WasmPluginManager,
    WasmPluginManagerError, WasmPluginSlot, WasmRuntime,
};
use polychora_plugin_api::opcodes::{OP_PROCGEN_GENERATE, OP_PROCGEN_PREPARE};
use polychora_plugin_api::procgen_abi::{
    ProcgenGenerateInput, ProcgenGenerateOutput, ProcgenPrepareInput, ProcgenPrepareOutput,
    StructureDeclaration,
};
use polychora_plugin_api::region_tree::TesseractOrientation;
use std::fmt;
use std::sync::Arc;

/// Higher execution limits for the procgen slot: structures produce large
/// region trees that exceed the default 64 KB I/O budget.
pub const PROCGEN_EXECUTION_LIMITS: WasmExecutionLimits = WasmExecutionLimits {
    max_input_bytes: 128 * 1024,
    max_output_bytes: 2 * 1024 * 1024,
    max_memory_pages: 16384,
    max_fuel: 500_000_000,
};

const LRU_CACHE_CAPACITY: usize = 64;

#[derive(Debug)]
pub enum ProcgenWasmError {
    Manager(WasmPluginManagerError),
    SlotNotActive,
    SerializeInput(String),
    DeserializeOutput(String),
}

impl fmt::Display for ProcgenWasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manager(e) => write!(f, "wasm manager error: {e}"),
            Self::SlotNotActive => write!(f, "procgen WASM slot is not active"),
            Self::SerializeInput(e) => write!(f, "failed to serialize procgen input: {e}"),
            Self::DeserializeOutput(e) => write!(f, "failed to deserialize procgen output: {e}"),
        }
    }
}

impl std::error::Error for ProcgenWasmError {}

impl From<WasmPluginManagerError> for ProcgenWasmError {
    fn from(e: WasmPluginManagerError) -> Self {
        Self::Manager(e)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PrepareCacheKey {
    structure_id: u32,
    seed: u64,
}

struct LruEntry {
    key: PrepareCacheKey,
    value: ProcgenPrepareOutput,
}

/// Host-side caller for procgen WASM operations.
///
/// Holds structure declarations from the plugin manifest, an LRU cache of
/// prepare results, and calls the Procgen WASM slot for prepare/generate.
pub struct ProcgenWasmCaller {
    declarations: Vec<StructureDeclaration>,
    prepare_cache: Vec<LruEntry>,
}

impl ProcgenWasmCaller {
    pub fn new(declarations: Vec<StructureDeclaration>) -> Self {
        Self {
            declarations,
            prepare_cache: Vec::with_capacity(LRU_CACHE_CAPACITY),
        }
    }

    pub fn declarations(&self) -> &[StructureDeclaration] {
        &self.declarations
    }

    pub fn total_weight(&self) -> u64 {
        self.declarations
            .iter()
            .map(|d| d.spawn_weight.max(1) as u64)
            .sum::<u64>()
            .max(1)
    }

    pub fn pick_declaration_index(&self, mut roll: u64) -> usize {
        for (idx, decl) in self.declarations.iter().enumerate() {
            let weight = decl.spawn_weight.max(1) as u64;
            if roll < weight {
                return idx;
            }
            roll -= weight;
        }
        self.declarations.len().saturating_sub(1)
    }

    /// Call OP_PROCGEN_PREPARE via WASM; results are cached by (structure_id, seed).
    pub fn prepare(
        &mut self,
        manager: &mut WasmPluginManager,
        structure_id: u32,
        seed: u64,
        orientation: TesseractOrientation,
        origin: [i32; 4],
    ) -> Result<ProcgenPrepareOutput, ProcgenWasmError> {
        let cache_key = PrepareCacheKey { structure_id, seed };

        // Check cache (and promote to front on hit)
        if let Some(pos) = self.prepare_cache.iter().position(|e| e.key == cache_key) {
            let entry = self.prepare_cache.remove(pos);
            let result = entry.value.clone();
            self.prepare_cache.push(entry);
            return Ok(result);
        }

        let input = ProcgenPrepareInput {
            structure_id,
            seed,
            orientation,
            origin,
        };
        let input_bytes = postcard::to_allocvec(&input)
            .map_err(|e| ProcgenWasmError::SerializeInput(e.to_string()))?;

        let result = manager
            .call_slot(
                WasmPluginSlot::Procgen,
                OP_PROCGEN_PREPARE as i32,
                &input_bytes,
            )?
            .ok_or(ProcgenWasmError::SlotNotActive)?;

        let output: ProcgenPrepareOutput = postcard::from_bytes(&result.invocation.output)
            .map_err(|e| ProcgenWasmError::DeserializeOutput(e.to_string()))?;

        // Insert into cache, evicting oldest if at capacity
        if self.prepare_cache.len() >= LRU_CACHE_CAPACITY {
            self.prepare_cache.remove(0);
        }
        self.prepare_cache.push(LruEntry {
            key: cache_key,
            value: output.clone(),
        });

        Ok(output)
    }

    /// Call OP_PROCGEN_GENERATE via WASM and convert the returned plugin
    /// RegionTreeCore to the host's internal type.
    pub fn generate(
        &self,
        manager: &mut WasmPluginManager,
        structure_id: u32,
        seed: u64,
        orientation: TesseractOrientation,
        origin: [i32; 4],
        state: Vec<u8>,
    ) -> Result<RegionTreeCore, ProcgenWasmError> {
        let input = ProcgenGenerateInput {
            structure_id,
            seed,
            orientation,
            origin,
            state,
        };
        let input_bytes = postcard::to_allocvec(&input)
            .map_err(|e| ProcgenWasmError::SerializeInput(e.to_string()))?;

        let result = manager
            .call_slot(
                WasmPluginSlot::Procgen,
                OP_PROCGEN_GENERATE as i32,
                &input_bytes,
            )?
            .ok_or(ProcgenWasmError::SlotNotActive)?;

        let output: ProcgenGenerateOutput = postcard::from_bytes(&result.invocation.output)
            .map_err(|e| ProcgenWasmError::DeserializeOutput(e.to_string()))?;

        Ok(region_tree_from_plugin(&output.tree))
    }

    /// Convenience: prepare + generate in one call.
    pub fn prepare_and_generate(
        &mut self,
        manager: &mut WasmPluginManager,
        structure_id: u32,
        seed: u64,
        orientation: TesseractOrientation,
        origin: [i32; 4],
    ) -> Result<(ProcgenPrepareOutput, RegionTreeCore), ProcgenWasmError> {
        let prepared = self.prepare(manager, structure_id, seed, orientation, origin)?;
        let tree = self.generate(
            manager,
            structure_id,
            seed,
            orientation,
            origin,
            prepared.state.clone(),
        )?;
        Ok((prepared, tree))
    }

    pub fn clear_cache(&mut self) {
        self.prepare_cache.clear();
    }
}

/// Bundled WASM state for procgen: a dedicated WasmPluginManager (with only
/// the Procgen slot active) plus the caller with its LRU cache.
pub struct ProcgenWasmState {
    pub manager: WasmPluginManager,
    pub caller: ProcgenWasmCaller,
}

impl ProcgenWasmState {
    /// Create a procgen WASM state by sharing an existing runtime and cache.
    pub fn new(
        runtime: Arc<WasmRuntime>,
        cache: Arc<WasmModuleCache>,
        wasm_bytes: &[u8],
        declarations: Vec<StructureDeclaration>,
    ) -> Option<Self> {
        let caller = ProcgenWasmCaller::new(declarations);
        let mut manager =
            WasmPluginManager::new(WasmExecutionRole::ServerAuthoritative, runtime, cache);
        manager
            .activate_slot_from_bytes(
                WasmPluginSlot::Procgen,
                wasm_bytes,
                PROCGEN_EXECUTION_LIMITS,
            )
            .ok()?;
        Some(Self { manager, caller })
    }

    /// Create a standalone procgen WASM state with its own runtime.
    pub fn new_standalone(
        wasm_bytes: &[u8],
        declarations: Vec<StructureDeclaration>,
    ) -> Option<Self> {
        let runtime = Arc::new(WasmRuntime::new().ok()?);
        let cache = Arc::new(WasmModuleCache::new(runtime.clone(), None).ok()?);
        Self::new(runtime, cache, wasm_bytes, declarations)
    }

    /// Build a `StructurePlacementConfig` from the WASM declarations.
    pub fn structure_placement_config(&self) -> StructurePlacementConfig {
        let weights: Vec<u32> = self
            .caller
            .declarations()
            .iter()
            .map(|d| d.spawn_weight)
            .collect();
        StructurePlacementConfig::from_weights(&weights)
    }

    /// Prepare + generate a structure, returning the host RegionTreeCore.
    pub fn prepare_and_generate(
        &mut self,
        structure_id: u32,
        seed: u64,
        orientation: TesseractOrientation,
        origin: [i32; 4],
    ) -> Result<RegionTreeCore, ProcgenWasmError> {
        let (_prepared, tree) = self.caller.prepare_and_generate(
            &mut self.manager,
            structure_id,
            seed,
            orientation,
            origin,
        )?;
        Ok(tree)
    }
}

/// Convert a host-side XZW-only orientation (u8, 48 values) to a full 4D
/// TesseractOrientation that preserves the Y axis.
///
/// The old orientation encodes: `perm_idx = o % 6`, `sign_bits = (o / 6) & 7`
/// where `perm_idx` selects one of 6 permutations of {X, Z, W} and `sign_bits`
/// optionally negates each of the 3 horizontal axes.
pub fn xzw_orientation_to_tesseract(old: u8) -> TesseractOrientation {
    // Map old 3D XZW permutation indices to 4D permutation indices where Y stays at position 1.
    // Old perms: [X,Z,W], [X,W,Z], [Z,X,W], [Z,W,X], [W,X,Z], [W,Z,X]
    // 4D perms:  [0,1,2,3], [0,1,3,2], [2,1,0,3], [2,1,3,0], [3,1,0,2], [3,1,2,0]
    // 4D indices: 0,          1,          14,         15,         20,         21
    const PERM_MAP: [u16; 6] = [0, 1, 14, 15, 20, 21];
    let old_perm_idx = (old as usize) % 6;
    let old_sign = ((old as usize) / 6) & 0b111;
    let new_perm_idx = PERM_MAP[old_perm_idx];
    // Map 3 sign bits {X, Z, W} → 4 sign bits {X, Y, Z, W} with Y=0
    let new_sign = (old_sign & 1) | ((old_sign & 2) << 1) | ((old_sign & 4) << 1);
    TesseractOrientation(new_perm_idx * 16 + new_sign as u16)
}
