use crate::shared::voxel::BlockData;
use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::content_ids;
use polychora_plugin_api::entity::{
    EntityCategory, MobArchetype, MobArchetypeDefaults, MobLocomotionMode,
};
use polychora_plugin_api::texture::TextureRef;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Static material-token ↔ BlockData mapping
// ---------------------------------------------------------------------------
//
// Material tokens (u8/u16, 1-indexed) correspond to the registration order
// of blocks in the polychora-content plugin.  Token 0 = air.
// This table enables conversion without needing a ContentRegistry instance.

/// Number of legacy material tokens (1-based, excluding air).
/// Used by settings persistence for clamping saved token values.
pub const LEGACY_MATERIAL_TOKEN_COUNT: u8 = 68;

const TOKEN_TO_BLOCK_TYPE: [u32; LEGACY_MATERIAL_TOKEN_COUNT as usize] = [
    content_ids::BLOCK_RED,                // token 1
    content_ids::BLOCK_ORANGE,             // token 2
    content_ids::BLOCK_YELLOW_GREEN,       // token 3
    content_ids::BLOCK_GREEN,              // token 4
    content_ids::BLOCK_CYAN,               // token 5
    content_ids::BLOCK_BLUE,               // token 6
    content_ids::BLOCK_PURPLE,             // token 7
    content_ids::BLOCK_MAGENTA,            // token 8
    content_ids::BLOCK_RAINBOW,            // token 9
    content_ids::BLOCK_BROWN,              // token 10
    content_ids::BLOCK_GRID_FLOOR,         // token 11
    content_ids::BLOCK_WHITE,              // token 12
    content_ids::BLOCK_LIGHT,              // token 13
    content_ids::BLOCK_MIRROR,             // token 14
    content_ids::BLOCK_LAVA_VEINED_BASALT, // token 15
    content_ids::BLOCK_CRYSTAL_LATTICE,    // token 16
    content_ids::BLOCK_MARBLE,             // token 17
    content_ids::BLOCK_OXIDIZED_METAL,     // token 18
    content_ids::BLOCK_BIO_SPORE_MOSS,     // token 19
    content_ids::BLOCK_VOID_MIRROR,        // token 20
    content_ids::BLOCK_AVATAR_MARKER,      // token 21
    content_ids::BLOCK_HOLOGRAPHIC_LAMINATE, // token 22
    content_ids::BLOCK_TIDAL_GLASS,        // token 23
    content_ids::BLOCK_CIRCUIT_WEAVE,      // token 24
    content_ids::BLOCK_AURORA_STONE,       // token 25
    content_ids::BLOCK_HAZARD_CHEVRONS,    // token 26
    content_ids::BLOCK_STONE,              // token 27
    content_ids::BLOCK_COBBLESTONE,        // token 28
    content_ids::BLOCK_DIRT,               // token 29
    content_ids::BLOCK_COARSE_DIRT,        // token 30
    content_ids::BLOCK_OAK_PLANKS,         // token 31
    content_ids::BLOCK_SPRUCE_PLANKS,      // token 32
    content_ids::BLOCK_LOG_BARK,           // token 33
    content_ids::BLOCK_LOG_END_RINGS,      // token 34
    content_ids::BLOCK_SAND,               // token 35
    content_ids::BLOCK_GRAVEL,             // token 36
    content_ids::BLOCK_CLAY,               // token 37
    content_ids::BLOCK_GRASS_BLOCK,        // token 38
    content_ids::BLOCK_SNOW,               // token 39
    content_ids::BLOCK_ICE,                // token 40
    content_ids::BLOCK_COAL_ORE,           // token 41
    content_ids::BLOCK_IRON_ORE,           // token 42
    content_ids::BLOCK_GOLD_ORE,           // token 43
    content_ids::BLOCK_DIAMOND_ORE,        // token 44
    content_ids::BLOCK_REDSTONE_ORE,       // token 45
    content_ids::BLOCK_BIRCH_PLANKS,       // token 46
    content_ids::BLOCK_BRICKS,             // token 47
    content_ids::BLOCK_SANDSTONE,          // token 48
    content_ids::BLOCK_GLASS,              // token 49
    content_ids::BLOCK_GLOWSTONE,          // token 50
    content_ids::BLOCK_OBSIDIAN,           // token 51
    content_ids::BLOCK_PRISMARINE,         // token 52
    content_ids::BLOCK_TERRACOTTA,         // token 53
    content_ids::BLOCK_WOOL_WHITE,         // token 54
    content_ids::BLOCK_BASALT_TILES,       // token 55
    content_ids::BLOCK_COPPER_WEAVE,       // token 56
    content_ids::BLOCK_NEBULA_STRATA,      // token 57
    content_ids::BLOCK_STARFORGED_CORE,    // token 58
    content_ids::BLOCK_CRYO_CIRCUIT,       // token 59
    content_ids::BLOCK_SMOKED_GLASS,       // token 60
    content_ids::BLOCK_IVORY_MARBLE,       // token 61
    content_ids::BLOCK_RUNIC_ALLOY,        // token 62
    content_ids::BLOCK_HYPERPHASE_GEL,     // token 63
    content_ids::BLOCK_SINGULARITY_CORE,   // token 64
    content_ids::BLOCK_CHRONO_BLOOM,       // token 65
    content_ids::BLOCK_TESSERACT_WEAVE,    // token 66
    content_ids::BLOCK_EVENTIDE_ALLOY,     // token 67
    content_ids::BLOCK_BEACON_MATRIX,      // token 68
];

/// Convert a material token (0 = air, 1–68 = blocks) to its `BlockData`.
/// Uses the fixed polychora-content registration order.
pub fn block_data_from_material_token(token: u8) -> BlockData {
    if token == 0 {
        return BlockData::AIR;
    }
    let idx = (token as usize).wrapping_sub(1);
    TOKEN_TO_BLOCK_TYPE
        .get(idx)
        .map(|&bt| BlockData::simple(content_ids::CONTENT_NS, bt))
        .unwrap_or(BlockData::AIR)
}

/// Convert a `BlockData` back to its material token (0 if unknown/air).
pub fn material_token_from_block_data(block: &BlockData) -> u8 {
    if block.is_air() {
        return 0;
    }
    if block.namespace != content_ids::CONTENT_NS {
        return 0;
    }
    TOKEN_TO_BLOCK_TYPE
        .iter()
        .position(|&bt| bt == block.block_type)
        .map(|idx| (idx + 1) as u8)
        .unwrap_or(0)
}

/// Bit 15 of a material token: when set, bits [14:0] index into the GPU texture
/// pool instead of the procedural `sampleMaterial()` switch.
pub const MATERIAL_TOKEN_TEXTURE_POOL_FLAG: u16 = 1 << 15;

// ---------------------------------------------------------------------------
// MaterialResolver — render-owned GPU token resolution
// ---------------------------------------------------------------------------

/// Render-system-owned resolver that maps block identifiers and texture
/// references to ephemeral GPU material tokens (u16).
///
/// Built once at startup from a fully-populated `ContentRegistry`.  All
/// rendering code should use this instead of querying `ContentRegistry`
/// directly for material tokens.
#[derive(Clone, Debug)]
pub struct MaterialResolver {
    /// (namespace, block_type) → GPU u16 material token
    block_to_gpu: HashMap<(u32, u32), u16>,
    /// (namespace, texture_id) → GPU u16 material token
    texture_to_gpu: HashMap<(u32, u32), u16>,
    /// Reverse: GPU u16 → (namespace, block_type), for diagnostics
    gpu_to_block: HashMap<u16, (u32, u32)>,
}

impl MaterialResolver {
    /// Build a `MaterialResolver` from a fully-populated `ContentRegistry`.
    pub fn from_registry(registry: &ContentRegistry) -> Self {
        let mut block_to_gpu = HashMap::new();
        let mut gpu_to_block = HashMap::new();
        for (&key, entry) in &registry.blocks {
            block_to_gpu.insert(key, entry.material_token);
            gpu_to_block.insert(entry.material_token, key);
        }
        // Also register blocks reachable via legacy remaps
        for (&old_key, &new_key) in &registry.legacy_block_remap {
            if let Some(entry) = registry.blocks.get(&new_key) {
                block_to_gpu.entry(old_key).or_insert(entry.material_token);
            }
        }
        let texture_to_gpu = registry.texture_tokens.clone();
        Self {
            block_to_gpu,
            texture_to_gpu,
            gpu_to_block,
        }
    }

    /// Resolve (namespace, block_type) → GPU material token.
    /// Returns 1 (Red) as fallback for unknown blocks.
    #[inline]
    pub fn resolve_block(&self, namespace: u32, block_type: u32) -> u16 {
        self.block_to_gpu
            .get(&(namespace, block_type))
            .copied()
            .unwrap_or(1)
    }

    /// Resolve a `TextureRef` → GPU material token.
    #[inline]
    pub fn resolve_texture(&self, namespace: u32, texture_id: u32) -> Option<u16> {
        self.texture_to_gpu.get(&(namespace, texture_id)).copied()
    }

    /// Reverse lookup: GPU token → (namespace, block_type).
    pub fn block_for_gpu_token(&self, token: u16) -> Option<(u32, u32)> {
        self.gpu_to_block.get(&token).copied()
    }

    /// Build a palette mapping GPU material token indices to `BlockData`.
    ///
    /// Index 0 = Air, indices 1..=max are populated from the reverse token map.
    /// Used by diagnostic code that works with raw `ChunkPayload` u16 indices.
    pub fn build_token_palette(&self) -> Vec<BlockData> {
        use crate::shared::voxel::BlockData;
        let max_token = self.gpu_to_block.keys().copied().max().unwrap_or(0) as usize;
        let mut palette = vec![BlockData::AIR; max_token + 1];
        for (&token, &(ns, bt)) in &self.gpu_to_block {
            palette[token as usize] = BlockData::simple(ns, bt);
        }
        palette
    }
}

// ---------------------------------------------------------------------------
// Block entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BlockEntry {
    pub namespace: u32,
    pub block_type: u32,
    pub name: String,
    pub category: BlockCategory,
    pub color: [u8; 3],
    /// Stable texture reference for this block.
    pub texture: TextureRef,
    /// GPU material token (u16) assigned at registration time.
    /// Internal to render system; use MaterialResolver for GPU lookups.
    pub material_token: u16,
}

// ---------------------------------------------------------------------------
// Entity entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EntityEntry {
    pub namespace: u32,
    pub entity_type: u32,
    pub category: EntityCategory,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub default_scale: f32,
    /// Explicit texture palette for entity model parts.
    /// Rendering code resolves each TextureRef to a GPU token at draw time.
    pub model_textures: Vec<TextureRef>,
    pub mob_archetype: Option<MobArchetype>,
    pub mob_defaults: Option<MobArchetypeDefaults>,
}

impl EntityEntry {
    pub fn type_key(&self) -> (u32, u32) {
        (self.namespace, self.entity_type)
    }

    pub fn is_spawnable(&self) -> bool {
        self.category != EntityCategory::Player
    }
}

// ---------------------------------------------------------------------------
// Content Registry
// ---------------------------------------------------------------------------

/// Dynamic content registry that replaces the old static `MATERIALS` and
/// `ENTITY_TYPES` arrays.  Populated at startup by `register_builtin_content`
/// and (in future phases) by WASM plugins.
impl std::fmt::Debug for ContentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentRegistry")
            .field("blocks", &self.blocks.len())
            .field("entities", &self.entities.len())
            .finish()
    }
}

pub struct ContentRegistry {
    // Blocks keyed by (namespace, block_type)
    blocks: HashMap<(u32, u32), BlockEntry>,
    // Ordered list of block keys in registration order (for iteration)
    block_order: Vec<(u32, u32)>,
    // Reverse map: material_token → (namespace, block_type)
    token_to_block: HashMap<u16, (u32, u32)>,
    // Next procedural material token to assign (starts at 1, 0 = air)
    next_procedural_token: u16,

    // Entities keyed by (namespace, entity_type)
    entities: HashMap<(u32, u32), EntityEntry>,
    // Name lookup: normalized canonical name / alias → (namespace, entity_type)
    entity_name_index: HashMap<String, (u32, u32)>,

    // Texture registry: (namespace, texture_id) → material_token
    texture_tokens: HashMap<(u32, u32), u16>,

    // Legacy remap: old (namespace, block_type) → new (namespace, block_type)
    legacy_block_remap: HashMap<(u32, u32), (u32, u32)>,
    // Legacy remap: old (namespace, entity_type) → new (namespace, entity_type)
    legacy_entity_remap: HashMap<(u32, u32), (u32, u32)>,

    // Namespace ID → human-readable name (for WAILA display)
    namespace_names: HashMap<u32, String>,
}

impl ContentRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            blocks: HashMap::new(),
            block_order: Vec::new(),
            token_to_block: HashMap::new(),
            next_procedural_token: 1, // 0 = air
            entities: HashMap::new(),
            entity_name_index: HashMap::new(),
            texture_tokens: HashMap::new(),
            legacy_block_remap: HashMap::new(),
            legacy_entity_remap: HashMap::new(),
            namespace_names: HashMap::from([(0, "engine".into())]),
        };
        // Air is always token 0 at (0, 0)
        registry.blocks.insert(
            (0, 0),
            BlockEntry {
                namespace: 0,
                block_type: 0,
                name: "Air".into(),
                category: BlockCategory::Special,
                color: [0, 0, 0],
                texture: TextureRef { namespace: 0, texture_id: 0 },
                material_token: 0,
            },
        );
        registry.token_to_block.insert(0, (0, 0));
        registry
    }

    // -----------------------------------------------------------------------
    // Block registration
    // -----------------------------------------------------------------------

    /// Register a block and assign it the next procedural material token.
    pub fn register_block(
        &mut self,
        namespace: u32,
        block_type: u32,
        name: impl Into<String>,
        category: BlockCategory,
        color: [u8; 3],
    ) -> u16 {
        let token = self.next_procedural_token;
        self.next_procedural_token += 1;

        let entry = BlockEntry {
            namespace,
            block_type,
            name: name.into(),
            category,
            color,
            texture: TextureRef { namespace: 0, texture_id: 0 },
            material_token: token,
        };
        self.blocks.insert((namespace, block_type), entry);
        self.block_order.push((namespace, block_type));
        self.token_to_block.insert(token, (namespace, block_type));
        token
    }

    /// Register a block with a specific (forced) material token and texture reference.
    /// Used for plugin content whose textures have been resolved to GPU tokens.
    ///
    /// # Panics
    /// Panics if `forced_token` is already assigned to a different block.
    pub fn register_block_with_token(
        &mut self,
        namespace: u32,
        block_type: u32,
        name: impl Into<String>,
        category: BlockCategory,
        color: [u8; 3],
        forced_token: u16,
        texture: TextureRef,
    ) {
        let name = name.into();
        if let Some(&existing) = self.token_to_block.get(&forced_token) {
            if existing != (namespace, block_type) {
                panic!(
                    "material token {} already assigned to block ({}, {}), cannot assign to ({}, {}) \"{}\"",
                    forced_token, existing.0, existing.1, namespace, block_type, name,
                );
            }
        }
        let entry = BlockEntry {
            namespace,
            block_type,
            name,
            category,
            color,
            texture,
            material_token: forced_token,
        };
        self.blocks.insert((namespace, block_type), entry);
        self.block_order.push((namespace, block_type));
        self.token_to_block.insert(forced_token, (namespace, block_type));
        // Keep next_procedural_token ahead of any forced token
        if forced_token >= self.next_procedural_token {
            self.next_procedural_token = forced_token + 1;
        }
    }

    // -----------------------------------------------------------------------
    // Entity registration
    // -----------------------------------------------------------------------

    pub fn register_entity(&mut self, entry: EntityEntry) {
        let key = (entry.namespace, entry.entity_type);
        // Index canonical name and aliases
        let normalized_canonical = normalize_token(&entry.canonical_name);
        self.entity_name_index.insert(normalized_canonical, key);
        for alias in &entry.aliases {
            self.entity_name_index.insert(normalize_token(alias), key);
        }
        self.entities.insert(key, entry);
    }

    // -----------------------------------------------------------------------
    // Texture token registration
    // -----------------------------------------------------------------------

    /// Register a mapping from (namespace, texture_id) to a GPU material token.
    ///
    /// # Panics
    /// Panics if the same (namespace, texture_id) is already mapped to a different token.
    pub fn register_texture_token(&mut self, namespace: u32, texture_id: u32, material_token: u16) {
        if let Some(&existing) = self.texture_tokens.get(&(namespace, texture_id)) {
            if existing != material_token {
                panic!(
                    "texture ({}, {}) already mapped to token {}, cannot remap to {}",
                    namespace, texture_id, existing, material_token,
                );
            }
        }
        self.texture_tokens.insert((namespace, texture_id), material_token);
    }

    /// Resolve a `TextureRef` to its GPU material token.
    pub fn resolve_texture_token(&self, tex: &TextureRef) -> Option<u16> {
        self.texture_tokens.get(&(tex.namespace, tex.texture_id)).copied()
    }

    // -----------------------------------------------------------------------
    // Legacy remap registration
    // -----------------------------------------------------------------------

    pub fn register_legacy_block_remap(&mut self, old: (u32, u32), new: (u32, u32)) {
        self.legacy_block_remap.insert(old, new);
    }

    pub fn register_legacy_entity_remap(&mut self, old: (u32, u32), new: (u32, u32)) {
        self.legacy_entity_remap.insert(old, new);
    }

    // -----------------------------------------------------------------------
    // Block lookups
    // -----------------------------------------------------------------------

    /// Look up a block entry, transparently resolving through legacy remaps
    /// if the direct key is not found.
    fn resolve_block_entry(&self, namespace: u32, block_type: u32) -> Option<&BlockEntry> {
        self.blocks.get(&(namespace, block_type)).or_else(|| {
            self.legacy_block_remap
                .get(&(namespace, block_type))
                .and_then(|remapped| self.blocks.get(remapped))
        })
    }

    /// Get the full block entry by (namespace, block_type).
    pub fn block_entry(&self, namespace: u32, block_type: u32) -> Option<&BlockEntry> {
        self.resolve_block_entry(namespace, block_type)
    }

    /// Resolve (namespace, block_type) → GPU material token.
    /// Deprecated: use `MaterialResolver::resolve_block()` instead.
    #[deprecated(note = "use MaterialResolver::resolve_block() instead")]
    pub fn block_material_token(&self, namespace: u32, block_type: u32) -> u16 {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.material_token)
            .unwrap_or(1) // fallback to Red
    }

    /// Get block name by (namespace, block_type).
    pub fn block_name(&self, namespace: u32, block_type: u32) -> &str {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.name.as_str())
            .unwrap_or("Unknown")
    }

    /// Get block color by (namespace, block_type).
    pub fn block_color(&self, namespace: u32, block_type: u32) -> [u8; 3] {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.color)
            .unwrap_or([128, 128, 128])
    }

    /// Get block category label by (namespace, block_type).
    pub fn block_category_label(&self, namespace: u32, block_type: u32) -> &'static str {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.category.label())
            .unwrap_or("Unknown")
    }

    /// Get block category by (namespace, block_type).
    pub fn block_category(&self, namespace: u32, block_type: u32) -> Option<BlockCategory> {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.category)
    }

    // -----------------------------------------------------------------------
    // Material-token-based lookups (reverse: token → block info)
    // Deprecated: use block_name/block_color/block_entry via (namespace, block_type) instead.
    // These methods are retained for migration code and CPU render fallback.
    // -----------------------------------------------------------------------

    /// Get material name by token.
    #[deprecated(note = "use block_name(namespace, block_type) instead")]
    pub fn material_name_by_token(&self, token: u16) -> &str {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.name.as_str())
            .unwrap_or("Unknown")
    }

    /// Get material color by token.
    #[deprecated(note = "use block_color(namespace, block_type) instead")]
    pub fn material_color_by_token(&self, token: u16) -> [u8; 3] {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.color)
            .unwrap_or([128, 128, 128])
    }

    /// Get material category label by token.
    #[deprecated(note = "use block_category_label(namespace, block_type) instead")]
    pub fn material_category_label_by_token(&self, token: u16) -> &'static str {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.category.label())
            .unwrap_or("Unknown")
    }

    /// Get the block entry for a material token.
    #[deprecated(note = "use block_entry(namespace, block_type) instead")]
    pub fn block_entry_by_token(&self, token: u16) -> Option<&BlockEntry> {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
    }

    /// Convert a material token to a `BlockData` with proper namespace and block type.
    #[deprecated(note = "use block_data_from_material_token() for migration code")]
    #[allow(deprecated)]
    pub fn block_data_for_token(&self, token: u16) -> BlockData {
        self.block_entry_by_token(token)
            .map(|entry| BlockData::simple(entry.namespace, entry.block_type))
            .unwrap_or(BlockData::AIR)
    }

    // -----------------------------------------------------------------------
    // Iteration / counts
    // -----------------------------------------------------------------------

    /// Iterate all blocks in registration order (excludes air).
    /// Replacement for `MATERIALS.iter()`.
    pub fn all_blocks_ordered(&self) -> impl Iterator<Item = &BlockEntry> {
        self.block_order.iter().filter_map(|key| self.blocks.get(key))
    }

    /// Number of registered blocks (excludes air).
    /// Replacement for `MATERIALS.len()`.
    pub fn block_count(&self) -> usize {
        self.block_order.len()
    }

    /// Maximum material token assigned so far.
    #[deprecated(note = "use block_count() instead")]
    pub fn max_material_token(&self) -> u16 {
        if self.next_procedural_token > 0 {
            self.next_procedural_token - 1
        } else {
            0
        }
    }

    // -----------------------------------------------------------------------
    // Entity lookups
    // -----------------------------------------------------------------------

    /// Look up an entity entry, transparently resolving through legacy remaps
    /// if the direct key is not found.
    fn resolve_entity_entry(&self, namespace: u32, entity_type: u32) -> Option<&EntityEntry> {
        self.entities.get(&(namespace, entity_type)).or_else(|| {
            self.legacy_entity_remap
                .get(&(namespace, entity_type))
                .and_then(|remapped| self.entities.get(remapped))
        })
    }

    /// Lookup entity by (namespace, entity_type).
    pub fn entity_lookup(&self, namespace: u32, entity_type: u32) -> Option<&EntityEntry> {
        self.resolve_entity_entry(namespace, entity_type)
    }

    /// Lookup entity by name (canonical or alias, case-insensitive).
    pub fn entity_lookup_by_name(&self, name: &str) -> Option<&EntityEntry> {
        let normalized = normalize_token(name);
        self.entity_name_index
            .get(&normalized)
            .and_then(|key| self.entities.get(key))
    }

    /// Get entity category.
    pub fn entity_category(&self, namespace: u32, entity_type: u32) -> EntityCategory {
        self.resolve_entity_entry(namespace, entity_type)
            .map(|e| e.category)
            .unwrap_or(EntityCategory::Accent)
    }

    /// Get all spawnable entity canonical names.
    pub fn spawnable_entity_names(&self) -> Vec<&str> {
        self.entities
            .values()
            .filter(|e| e.is_spawnable())
            .map(|e| e.canonical_name.as_str())
            .collect()
    }

    /// Get mob archetype defaults.
    pub fn mob_archetype_defaults(&self, archetype: MobArchetype) -> MobArchetypeDefaults {
        for entry in self.entities.values() {
            if entry.mob_archetype == Some(archetype) {
                if let Some(defaults) = entry.mob_defaults {
                    return defaults;
                }
            }
        }
        // Fallback defaults
        MobArchetypeDefaults {
            move_speed: 2.0,
            preferred_distance: 3.0,
            tangent_weight: 0.5,
            locomotion: MobLocomotionMode::Walking,
        }
    }

    // -----------------------------------------------------------------------
    // Legacy compat
    // -----------------------------------------------------------------------

    /// Resolve a legacy block ID to the new (namespace, block_type).
    /// Returns the input unchanged if no remap exists.
    pub fn resolve_legacy_block(&self, namespace: u32, block_type: u32) -> (u32, u32) {
        self.legacy_block_remap
            .get(&(namespace, block_type))
            .copied()
            .unwrap_or((namespace, block_type))
    }

    /// Resolve a legacy entity ID to the new (namespace, entity_type).
    /// Returns the input unchanged if no remap exists.
    pub fn resolve_legacy_entity(&self, namespace: u32, entity_type: u32) -> (u32, u32) {
        self.legacy_entity_remap
            .get(&(namespace, entity_type))
            .copied()
            .unwrap_or((namespace, entity_type))
    }

    // -----------------------------------------------------------------------
    // Namespace metadata
    // -----------------------------------------------------------------------

    /// Register a human-readable name for a namespace ID.
    pub fn register_namespace_name(&mut self, namespace: u32, name: impl Into<String>) {
        self.namespace_names.insert(namespace, name.into());
    }

    /// Get a human-readable label for a namespace ID.
    pub fn namespace_label(&self, namespace: u32) -> &str {
        self.namespace_names
            .get(&namespace)
            .map(|s| s.as_str())
            .unwrap_or("unknown")
    }
}

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_content;
    use crate::plugin_loader;
    use crate::shared::wasm::{WasmExecutionLimits, WasmRuntime};

    /// Build a fully-populated registry: builtin (Air, Player, textures, remaps)
    /// + polychora-content WASM plugin (68 blocks, 6 entities).
    fn full_registry() -> ContentRegistry {
        let mut registry = ContentRegistry::new();
        builtin_content::register_builtin_content(&mut registry);

        let runtime = WasmRuntime::new().expect("wasm runtime");
        let plugin = plugin_loader::load_plugin(
            &runtime,
            include_bytes!(env!("POLYCHORA_CONTENT_WASM_PATH")),
            WasmExecutionLimits::default(),
        )
        .expect("load polychora-content plugin");
        plugin_loader::populate_registry_from_plugin(&mut registry, &plugin)
            .expect("populate registry from plugin");

        registry
    }

    #[test]
    fn plugin_blocks_registered_correctly() {
        let registry = full_registry();
        let resolver = MaterialResolver::from_registry(&registry);

        // Verify all 68 blocks are registered (from plugin)
        assert_eq!(registry.block_count(), 68);

        // Verify specific block lookups via MaterialResolver reverse mapping
        let check_name = |token: u16, expected_name: &str| {
            let (ns, bt) = resolver.block_for_gpu_token(token)
                .unwrap_or_else(|| panic!("token {} not in resolver", token));
            assert_eq!(registry.block_name(ns, bt), expected_name);
            assert_eq!(resolver.resolve_block(ns, bt), token);
        };

        check_name(1, "Red");
        check_name(12, "White");
        check_name(27, "Stone");
        check_name(35, "Sand");
        check_name(68, "Beacon Matrix");

        // Verify round-trip for all 68 blocks
        for token in 1u16..=68 {
            let (ns, bt) = resolver.block_for_gpu_token(token)
                .unwrap_or_else(|| panic!("token {} not in resolver", token));
            assert_eq!(resolver.resolve_block(ns, bt), token);
        }
    }

    #[test]
    fn plugin_entities_match_legacy() {
        let registry = full_registry();

        // Player avatar (engine internal, namespace 0)
        let player = registry.entity_lookup(0, 0).unwrap();
        assert_eq!(player.canonical_name.as_str(), "player");
        assert_eq!(player.category, EntityCategory::Player);

        // Seeker mob (from plugin)
        let seeker = registry.entity_lookup_by_name("seeker").unwrap();
        assert_eq!(seeker.mob_archetype, Some(MobArchetype::Seeker));

        // Spawnable names should include all non-player entities
        let spawnable = registry.spawnable_entity_names();
        assert!(spawnable.contains(&"cube"));
        assert!(spawnable.contains(&"seeker"));
        assert!(spawnable.contains(&"phase_spider"));
        assert!(!spawnable.contains(&"player"));
    }

    #[test]
    fn legacy_remap_works() {
        let registry = full_registry();
        let resolver = MaterialResolver::from_registry(&registry);

        // Legacy (0, 27) → (CONTENT_NS, BLOCK_STONE) → "Stone" with token 27
        let (ns, bt) = registry.resolve_legacy_block(0, 27);
        assert_eq!(registry.block_name(ns, bt), "Stone");
        assert_eq!(resolver.resolve_block(ns, bt), 27);

        // Legacy (0, 10) → (CONTENT_NS, ENTITY_SEEKER) → "seeker"
        let (ns, et) = registry.resolve_legacy_entity(0, 10);
        let entry = registry.entity_lookup(ns, et).unwrap();
        assert_eq!(entry.canonical_name, "seeker");
    }

    #[test]
    fn unknown_lookups_return_defaults() {
        let registry = full_registry();
        let resolver = MaterialResolver::from_registry(&registry);

        assert_eq!(resolver.resolve_block(999, 999), 1); // Red fallback
        assert_eq!(registry.block_name(999, 999), "Unknown");
        assert_eq!(registry.block_color(999, 999), [128, 128, 128]);
        assert_eq!(registry.entity_category(999, 999), EntityCategory::Accent);
    }

    #[test]
    fn texture_registry_resolves() {
        let registry = full_registry();
        use polychora_plugin_api::texture::{builtin_textures, TextureRef};

        // TEX_STONE should resolve to material token 27
        let tex = TextureRef { namespace: 0, texture_id: builtin_textures::TEX_STONE };
        assert_eq!(registry.resolve_texture_token(&tex), Some(27));

        // TEX_RED should resolve to material token 1
        let tex = TextureRef { namespace: 0, texture_id: builtin_textures::TEX_RED };
        assert_eq!(registry.resolve_texture_token(&tex), Some(1));

        // Unknown texture returns None
        let tex = TextureRef { namespace: 99, texture_id: 0xdeadbeef };
        assert_eq!(registry.resolve_texture_token(&tex), None);
    }

    #[test]
    fn entity_model_textures_declared() {
        let registry = full_registry();
        // Every non-player entity should have at least one model texture.
        let expected_entities = ["cube", "rotor", "drifter", "seeker", "creeper", "phase_spider"];
        for name in expected_entities {
            let entry = registry.entity_lookup_by_name(name)
                .unwrap_or_else(|| panic!("entity '{}' not found", name));
            assert!(
                !entry.model_textures.is_empty(),
                "entity '{}' should have model_textures declared",
                name,
            );
        }
        // Player has no model textures (rendered as avatar mesh)
        let player = registry.entity_lookup(0, 0).unwrap();
        assert!(player.model_textures.is_empty());
    }

    #[test]
    fn entity_aliases_include_legacy_names() {
        let registry = full_registry();
        // Verify that all legacy aliases still resolve correctly
        let expected: &[(&str, &str)] = &[
            ("testcube", "cube"),
            ("testrotor", "rotor"),
            ("testdrifter", "drifter"),
            ("mobseeker", "seeker"),
            ("4dcreeper", "creeper"),
            ("mobcreeper4d", "creeper"),
            ("spider", "phase_spider"),
            ("phase-spider", "phase_spider"),
            ("phasespider", "phase_spider"),
            ("mobphasespider", "phase_spider"),
        ];
        for &(alias, canonical) in expected {
            let entry = registry.entity_lookup_by_name(alias)
                .unwrap_or_else(|| panic!("alias '{}' not found", alias));
            assert_eq!(
                entry.canonical_name, canonical,
                "alias '{}': expected canonical '{}', got '{}'",
                alias, canonical, entry.canonical_name,
            );
        }
    }
}
