use crate::shared::voxel::BlockData;
use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::content_ids;
use polychora_plugin_api::entity::{EntityCategory, EntitySimConfig};
use polychora_plugin_api::manifest::{BlockTickConfig, ItemThumbnail, ItemWorldModel};
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
    content_ids::BLOCK_RED,                  // token 1
    content_ids::BLOCK_ORANGE,               // token 2
    content_ids::BLOCK_YELLOW_GREEN,         // token 3
    content_ids::BLOCK_GREEN,                // token 4
    content_ids::BLOCK_CYAN,                 // token 5
    content_ids::BLOCK_BLUE,                 // token 6
    content_ids::BLOCK_PURPLE,               // token 7
    content_ids::BLOCK_MAGENTA,              // token 8
    content_ids::BLOCK_RAINBOW,              // token 9
    content_ids::BLOCK_BROWN,                // token 10
    content_ids::BLOCK_GRID_FLOOR,           // token 11
    content_ids::BLOCK_WHITE,                // token 12
    content_ids::BLOCK_LIGHT,                // token 13
    content_ids::BLOCK_MIRROR,               // token 14
    content_ids::BLOCK_LAVA_VEINED_BASALT,   // token 15
    content_ids::BLOCK_CRYSTAL_LATTICE,      // token 16
    content_ids::BLOCK_MARBLE,               // token 17
    content_ids::BLOCK_OXIDIZED_METAL,       // token 18
    content_ids::BLOCK_BIO_SPORE_MOSS,       // token 19
    content_ids::BLOCK_VOID_MIRROR,          // token 20
    content_ids::BLOCK_AVATAR_MARKER,        // token 21
    content_ids::BLOCK_HOLOGRAPHIC_LAMINATE, // token 22
    content_ids::BLOCK_TIDAL_GLASS,          // token 23
    content_ids::BLOCK_CIRCUIT_WEAVE,        // token 24
    content_ids::BLOCK_AURORA_STONE,         // token 25
    content_ids::BLOCK_HAZARD_CHEVRONS,      // token 26
    content_ids::BLOCK_STONE,                // token 27
    content_ids::BLOCK_COBBLESTONE,          // token 28
    content_ids::BLOCK_DIRT,                 // token 29
    content_ids::BLOCK_COARSE_DIRT,          // token 30
    content_ids::BLOCK_OAK_PLANKS,           // token 31
    content_ids::BLOCK_SPRUCE_PLANKS,        // token 32
    content_ids::BLOCK_LOG_BARK,             // token 33
    content_ids::BLOCK_LOG_END_RINGS,        // token 34
    content_ids::BLOCK_SAND,                 // token 35
    content_ids::BLOCK_GRAVEL,               // token 36
    content_ids::BLOCK_CLAY,                 // token 37
    content_ids::BLOCK_GRASS_BLOCK,          // token 38
    content_ids::BLOCK_SNOW,                 // token 39
    content_ids::BLOCK_ICE,                  // token 40
    content_ids::BLOCK_COAL_ORE,             // token 41
    content_ids::BLOCK_IRON_ORE,             // token 42
    content_ids::BLOCK_GOLD_ORE,             // token 43
    content_ids::BLOCK_DIAMOND_ORE,          // token 44
    content_ids::BLOCK_REDSTONE_ORE,         // token 45
    content_ids::BLOCK_BIRCH_PLANKS,         // token 46
    content_ids::BLOCK_BRICKS,               // token 47
    content_ids::BLOCK_SANDSTONE,            // token 48
    content_ids::BLOCK_GLASS,                // token 49
    content_ids::BLOCK_GLOWSTONE,            // token 50
    content_ids::BLOCK_OBSIDIAN,             // token 51
    content_ids::BLOCK_PRISMARINE,           // token 52
    content_ids::BLOCK_TERRACOTTA,           // token 53
    content_ids::BLOCK_WOOL_WHITE,           // token 54
    content_ids::BLOCK_BASALT_TILES,         // token 55
    content_ids::BLOCK_COPPER_WEAVE,         // token 56
    content_ids::BLOCK_NEBULA_STRATA,        // token 57
    content_ids::BLOCK_STARFORGED_CORE,      // token 58
    content_ids::BLOCK_CRYO_CIRCUIT,         // token 59
    content_ids::BLOCK_SMOKED_GLASS,         // token 60
    content_ids::BLOCK_IVORY_MARBLE,         // token 61
    content_ids::BLOCK_RUNIC_ALLOY,          // token 62
    content_ids::BLOCK_HYPERPHASE_GEL,       // token 63
    content_ids::BLOCK_SINGULARITY_CORE,     // token 64
    content_ids::BLOCK_CHRONO_BLOOM,         // token 65
    content_ids::BLOCK_TESSERACT_WEAVE,      // token 66
    content_ids::BLOCK_EVENTIDE_ALLOY,       // token 67
    content_ids::BLOCK_BEACON_MATRIX,        // token 68
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
    /// If set, the server ticks instances of this block type periodically.
    pub tick_config: Option<BlockTickConfig>,
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
    /// Base material color hint (RGB), used for spawn egg icons.
    pub base_color: [u8; 3],
    /// Explicit texture palette for entity model parts.
    /// Rendering code resolves each TextureRef to a GPU token at draw time.
    pub model_textures: Vec<TextureRef>,
    /// Texture ID (namespace 0) for the spawn egg icon in the material icon sheet.
    pub spawn_egg_texture_id: u32,
    pub sim_config: Option<EntitySimConfig>,
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
// Item entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ItemEntry {
    pub namespace: u32,
    pub item_type: u32,
    pub name: String,
    pub max_stack_size: u32,
    pub color_hint: [u8; 3],
    /// In-world model texture palette for dropped item entities.
    pub world_model: ItemWorldModel,
    /// Inventory thumbnail texture.
    pub thumbnail: ItemThumbnail,
}

impl ItemEntry {
    pub fn type_key(&self) -> (u32, u32) {
        (self.namespace, self.item_type)
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
            .field("items", &self.items.len())
            .finish()
    }
}

pub struct ContentRegistry {
    // Blocks keyed by (namespace, block_type)
    blocks: HashMap<(u32, u32), BlockEntry>,
    // Ordered list of block keys in registration order (for iteration)
    block_order: Vec<(u32, u32)>,
    // Next procedural material token to assign (starts at 1, 0 = air)
    next_procedural_token: u16,

    // Entities keyed by (namespace, entity_type)
    entities: HashMap<(u32, u32), EntityEntry>,
    // Name lookup: normalized canonical name / alias → (namespace, entity_type)
    entity_name_index: HashMap<String, (u32, u32)>,

    // Items keyed by (namespace, item_type)
    items: HashMap<(u32, u32), ItemEntry>,
    // Name lookup: normalized name → (namespace, item_type)
    item_name_index: HashMap<String, (u32, u32)>,

    // Texture registry: (namespace, texture_id) → material_token
    texture_tokens: HashMap<(u32, u32), u16>,

    // Legacy remap: old (namespace, block_type) → new (namespace, block_type)
    legacy_block_remap: HashMap<(u32, u32), (u32, u32)>,
    // Legacy remap: old (namespace, entity_type) → new (namespace, entity_type)
    legacy_entity_remap: HashMap<(u32, u32), (u32, u32)>,

    // Namespace ID → human-readable name (for WAILA display)
    namespace_names: HashMap<u32, String>,
}

impl Default for ContentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            blocks: HashMap::new(),
            block_order: Vec::new(),
            next_procedural_token: 1, // 0 = air
            entities: HashMap::new(),
            entity_name_index: HashMap::new(),
            items: HashMap::new(),
            item_name_index: HashMap::new(),
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
                texture: TextureRef {
                    namespace: 0,
                    texture_id: 0,
                },
                material_token: 0,
                tick_config: None,
            },
        );
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
            texture: TextureRef {
                namespace: 0,
                texture_id: 0,
            },
            material_token: token,
            tick_config: None,
        };
        self.blocks.insert((namespace, block_type), entry);
        self.block_order.push((namespace, block_type));
        token
    }

    /// Register a block with a specific (forced) material token and texture reference.
    /// Used for plugin content whose textures have been resolved to GPU tokens.
    pub fn register_block_with_token(
        &mut self,
        namespace: u32,
        block_type: u32,
        name: impl Into<String>,
        category: BlockCategory,
        color: [u8; 3],
        forced_token: u16,
        texture: TextureRef,
        tick_config: Option<BlockTickConfig>,
    ) {
        let entry = BlockEntry {
            namespace,
            block_type,
            name: name.into(),
            category,
            color,
            texture,
            material_token: forced_token,
            tick_config,
        };
        self.blocks.insert((namespace, block_type), entry);
        self.block_order.push((namespace, block_type));
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
    // Item registration
    // -----------------------------------------------------------------------

    pub fn register_item(&mut self, entry: ItemEntry) {
        let key = (entry.namespace, entry.item_type);
        let normalized = normalize_token(&entry.name);
        self.item_name_index.insert(normalized, key);
        self.items.insert(key, entry);
    }

    // -----------------------------------------------------------------------
    // Item lookups
    // -----------------------------------------------------------------------

    /// Get an item entry by (namespace, item_type).
    pub fn item_entry(&self, namespace: u32, item_type: u32) -> Option<&ItemEntry> {
        self.items.get(&(namespace, item_type))
    }

    /// Lookup item by name (case-insensitive).
    pub fn item_lookup_by_name(&self, name: &str) -> Option<&ItemEntry> {
        let normalized = normalize_token(name);
        self.item_name_index
            .get(&normalized)
            .and_then(|key| self.items.get(key))
    }

    /// Get item name by (namespace, item_type).
    pub fn item_name(&self, namespace: u32, item_type: u32) -> &str {
        self.items
            .get(&(namespace, item_type))
            .map(|e| e.name.as_str())
            .unwrap_or("Unknown Item")
    }

    /// Get item color hint by (namespace, item_type).
    pub fn item_color(&self, namespace: u32, item_type: u32) -> [u8; 3] {
        self.items
            .get(&(namespace, item_type))
            .map(|e| e.color_hint)
            .unwrap_or([128, 128, 128])
    }

    /// Get item max stack size by (namespace, item_type).
    pub fn item_max_stack_size(&self, namespace: u32, item_type: u32) -> u32 {
        self.items
            .get(&(namespace, item_type))
            .map(|e| e.max_stack_size)
            .unwrap_or(64)
    }

    /// Iterate all registered items.
    pub fn all_items(&self) -> impl Iterator<Item = &ItemEntry> {
        self.items.values()
    }

    /// Resolve the texture palette for rendering an item in the world.
    ///
    /// For engine-internal items (ns=0):
    /// - ITEM_BLOCK: decodes BlockItemMeta → returns the block's texture
    /// - ITEM_SPAWN_EGG: decodes SpawnEggMeta → returns the entity's model_textures
    ///
    /// For plugin items: returns the declared `world_model.textures`.
    /// Fallback: empty vec (renderer will use default appearance).
    pub fn resolve_item_world_textures(
        &self,
        item: &crate::shared::protocol::Item,
    ) -> Vec<TextureRef> {
        use crate::shared::item_types::{BlockItemMeta, SpawnEggMeta, ITEM_BLOCK, ITEM_SPAWN_EGG};

        let key = (item.namespace, item.item_type);

        if key == ITEM_BLOCK {
            if let Some(meta) = BlockItemMeta::decode(&item.data) {
                if let Some(block) = self.resolve_block_entry(meta.namespace, meta.block_type) {
                    return vec![block.texture];
                }
            }
            return Vec::new();
        }

        if key == ITEM_SPAWN_EGG {
            if let Some(meta) = SpawnEggMeta::decode(&item.data) {
                if let Some(entity) =
                    self.resolve_entity_entry(meta.entity_namespace, meta.entity_type)
                {
                    return entity.model_textures.clone();
                }
            }
            return Vec::new();
        }

        // Plugin item: use declared world model textures
        self.items
            .get(&key)
            .map(|e| e.world_model.textures.clone())
            .unwrap_or_default()
    }

    /// Resolve the thumbnail texture for rendering an item in the inventory UI.
    ///
    /// For engine-internal items (ns=0):
    /// - ITEM_BLOCK: decodes BlockItemMeta → returns the block's texture
    /// - ITEM_SPAWN_EGG: decodes SpawnEggMeta → returns the entity's spawn egg texture
    ///
    /// For plugin items: returns the declared `thumbnail.texture`.
    pub fn resolve_item_thumbnail_texture(
        &self,
        item: &crate::shared::protocol::Item,
    ) -> Option<TextureRef> {
        use crate::shared::item_types::{BlockItemMeta, SpawnEggMeta, ITEM_BLOCK, ITEM_SPAWN_EGG};

        let key = (item.namespace, item.item_type);

        if key == ITEM_BLOCK {
            let meta = BlockItemMeta::decode(&item.data)?;
            let block = self.resolve_block_entry(meta.namespace, meta.block_type)?;
            return Some(block.texture);
        }

        if key == ITEM_SPAWN_EGG {
            let meta = SpawnEggMeta::decode(&item.data)?;
            let entity =
                self.resolve_entity_entry(meta.entity_namespace, meta.entity_type)?;
            if entity.spawn_egg_texture_id != 0 {
                return Some(TextureRef {
                    namespace: 0,
                    texture_id: entity.spawn_egg_texture_id,
                });
            }
            return None;
        }

        // Plugin item: use declared thumbnail texture
        self.items
            .get(&key)
            .and_then(|e| e.thumbnail.texture)
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
        self.texture_tokens
            .insert((namespace, texture_id), material_token);
    }

    /// Resolve a `TextureRef` to its GPU material token.
    pub fn resolve_texture_token(&self, tex: &TextureRef) -> Option<u16> {
        self.texture_tokens
            .get(&(tex.namespace, tex.texture_id))
            .copied()
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
    /// Prefer `MaterialResolver::resolve_block()` in render code; this method
    /// is used by save/load and server code that lacks a `MaterialResolver`.
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

    /// Get the TextureRef for a block's icon (for unified icon sheet lookups).
    pub fn block_icon_texture(&self, namespace: u32, block_type: u32) -> Option<TextureRef> {
        self.resolve_block_entry(namespace, block_type)
            .map(|e| e.texture)
    }

    /// Get the tick config for a block type, if any.
    pub fn block_tick_config(&self, namespace: u32, block_type: u32) -> Option<&BlockTickConfig> {
        self.resolve_block_entry(namespace, block_type)
            .and_then(|e| e.tick_config.as_ref())
    }

    /// Returns all block types that have a tick config, as (namespace, block_type, config).
    pub fn ticking_block_types(&self) -> Vec<(u32, u32, &BlockTickConfig)> {
        self.blocks
            .values()
            .filter_map(|entry| {
                entry
                    .tick_config
                    .as_ref()
                    .map(|cfg| (entry.namespace, entry.block_type, cfg))
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Iteration / counts
    // -----------------------------------------------------------------------

    /// Iterate all blocks in registration order (excludes air).
    /// Replacement for `MATERIALS.iter()`.
    pub fn all_blocks_ordered(&self) -> impl Iterator<Item = &BlockEntry> {
        self.block_order
            .iter()
            .filter_map(|key| self.blocks.get(key))
    }

    /// Number of registered blocks (excludes air).
    /// Replacement for `MATERIALS.len()`.
    pub fn block_count(&self) -> usize {
        self.block_order.len()
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

    /// Iterate all spawnable entities (non-player) for the creative picker.
    pub fn spawnable_entities(&self) -> impl Iterator<Item = &EntityEntry> {
        self.entities.values().filter(|e| e.is_spawnable())
    }

    /// Get simulation config for an entity type by (namespace, entity_type).
    pub fn sim_config(&self, namespace: u32, entity_type: u32) -> Option<&EntitySimConfig> {
        self.resolve_entity_entry(namespace, entity_type)
            .and_then(|e| e.sim_config.as_ref())
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
    use crate::plugin_loader;

    /// Build a fully-populated registry: builtin (Air, Player, textures, remaps)
    /// + polychora-content WASM plugin (68 blocks, 6 entities, 10 texture uploads).
    fn full_registry() -> ContentRegistry {
        let (registry, _pending) = plugin_loader::create_full_registry();
        registry
    }

    #[test]
    fn plugin_blocks_registered_correctly() {
        use crate::content_registry::MATERIAL_TOKEN_TEXTURE_POOL_FLAG;

        let registry = full_registry();
        let resolver = MaterialResolver::from_registry(&registry);

        // Verify all 68 blocks are registered (from plugin)
        assert_eq!(registry.block_count(), 68);

        // Verify specific block lookups via MaterialResolver reverse mapping
        let check_name = |token: u16, expected_name: &str| {
            let (ns, bt) = resolver
                .block_for_gpu_token(token)
                .unwrap_or_else(|| panic!("token {:#06x} not in resolver", token));
            assert_eq!(registry.block_name(ns, bt), expected_name);
            assert_eq!(resolver.resolve_block(ns, bt), token);
        };

        // Migrated blocks: texture pool tokens 0x8000..0x8009
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 0, "Red");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 1, "Orange");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 2, "Yellow-Green");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 3, "Green");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 4, "Cyan");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 5, "Blue");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 6, "Purple");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 7, "Magenta");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 8, "Brown");
        check_name(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 9, "White");

        // Non-migrated blocks still use procedural tokens
        check_name(27, "Stone");
        check_name(35, "Sand");
        check_name(68, "Beacon Matrix");

        // Verify round-trip for all non-migrated procedural tokens.
        // Procedural tokens for non-migrated blocks: 9, 11, 13, 14, 15..=68
        let procedural_tokens: Vec<u16> = [9u16, 11, 13, 14]
            .iter()
            .copied()
            .chain(15..=68)
            .collect();
        for token in procedural_tokens {
            let (ns, bt) = resolver
                .block_for_gpu_token(token)
                .unwrap_or_else(|| panic!("procedural token {} not in resolver", token));
            assert_eq!(resolver.resolve_block(ns, bt), token);
        }

        // Verify round-trip for all 10 texture pool tokens
        for i in 0u16..10 {
            let token = MATERIAL_TOKEN_TEXTURE_POOL_FLAG | i;
            let (ns, bt) = resolver
                .block_for_gpu_token(token)
                .unwrap_or_else(|| panic!("pool token {:#06x} not in resolver", token));
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
        assert!(seeker.sim_config.is_some(), "seeker should have sim_config");
        let seeker_config = seeker.sim_config.as_ref().unwrap();
        assert_eq!(seeker_config.move_speed, 3.0);
        assert_eq!(
            seeker_config.locomotion,
            polychora_plugin_api::entity::MobLocomotionMode::Walking
        );

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
        use crate::content_registry::MATERIAL_TOKEN_TEXTURE_POOL_FLAG;
        use polychora_plugin_api::texture::{builtin_textures, TextureRef};

        let registry = full_registry();
        let content_ns = 0x706f6c79u32; // polychora-content NAMESPACE

        // TEX_STONE should resolve to material token 27 via namespace 0
        let tex = TextureRef {
            namespace: 0,
            texture_id: builtin_textures::TEX_STONE,
        };
        assert_eq!(registry.resolve_texture_token(&tex), Some(27));

        // TEX_RED via namespace 0 should still resolve to procedural token 1
        let tex = TextureRef {
            namespace: 0,
            texture_id: builtin_textures::TEX_RED,
        };
        assert_eq!(registry.resolve_texture_token(&tex), Some(1));

        // TEX_RED via plugin namespace should resolve to texture pool token 0x8000
        let tex = TextureRef {
            namespace: content_ns,
            texture_id: builtin_textures::TEX_RED,
        };
        assert_eq!(
            registry.resolve_texture_token(&tex),
            Some(MATERIAL_TOKEN_TEXTURE_POOL_FLAG | 0),
        );

        // Unknown texture returns None
        let tex = TextureRef {
            namespace: 99,
            texture_id: 0xdeadbeef,
        };
        assert_eq!(registry.resolve_texture_token(&tex), None);
    }

    #[test]
    fn entity_model_textures_declared() {
        let registry = full_registry();
        // Every non-player entity should have at least one model texture.
        let expected_entities = [
            "cube",
            "rotor",
            "drifter",
            "seeker",
            "creeper",
            "phase_spider",
        ];
        for name in expected_entities {
            let entry = registry
                .entity_lookup_by_name(name)
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
            let entry = registry
                .entity_lookup_by_name(alias)
                .unwrap_or_else(|| panic!("alias '{}' not found", alias));
            assert_eq!(
                entry.canonical_name, canonical,
                "alias '{}': expected canonical '{}', got '{}'",
                alias, canonical, entry.canonical_name,
            );
        }
    }

    #[test]
    fn builtin_items_registered() {
        let registry = full_registry();
        // Block item
        let block_item = registry.item_entry(0, 1).expect("ITEM_BLOCK should be registered");
        assert_eq!(block_item.name, "Block");
        assert_eq!(block_item.max_stack_size, 64);
        // Spawn egg item
        let egg_item = registry.item_entry(0, 2).expect("ITEM_SPAWN_EGG should be registered");
        assert_eq!(egg_item.name, "Spawn Egg");
        assert_eq!(egg_item.max_stack_size, 1);
        // Name lookups
        assert!(registry.item_lookup_by_name("block").is_some());
        assert!(registry.item_lookup_by_name("SpawnEgg").is_some());
    }

    #[test]
    fn unknown_item_lookup_returns_defaults() {
        let registry = full_registry();
        assert_eq!(registry.item_name(999, 999), "Unknown Item");
        assert_eq!(registry.item_color(999, 999), [128, 128, 128]);
        assert_eq!(registry.item_max_stack_size(999, 999), 64);
        assert!(registry.item_entry(999, 999).is_none());
    }

    #[test]
    fn resolve_item_world_textures_for_block_item() {
        use crate::shared::protocol::ItemStack;

        let registry = full_registry();

        // A stone block item should resolve to Stone's texture
        let stone_stack = ItemStack::block(content_ids::CONTENT_NS, content_ids::BLOCK_STONE, 1, 0);
        let textures = registry.resolve_item_world_textures(&stone_stack.item);
        assert_eq!(textures.len(), 1, "block item should resolve to 1 texture");
        // Stone's texture should be TEX_STONE
        assert_eq!(
            textures[0].texture_id,
            polychora_plugin_api::texture::builtin_textures::TEX_STONE,
        );

        // A spawn egg item should resolve to the entity's model textures
        let seeker = registry.entity_lookup_by_name("seeker").unwrap();
        let egg_stack = ItemStack::spawn_egg(seeker.namespace, seeker.entity_type);
        let textures = registry.resolve_item_world_textures(&egg_stack.item);
        assert!(
            !textures.is_empty(),
            "spawn egg should resolve to entity model textures",
        );
        assert_eq!(
            textures.len(),
            seeker.model_textures.len(),
            "spawn egg textures should match entity model textures",
        );
    }

    #[test]
    fn resolve_item_thumbnail_texture_for_block_item() {
        use crate::shared::protocol::ItemStack;

        let registry = full_registry();

        // A stone block should resolve its thumbnail to Stone's texture
        let stone_stack = ItemStack::block(content_ids::CONTENT_NS, content_ids::BLOCK_STONE, 1, 0);
        let thumb = registry.resolve_item_thumbnail_texture(&stone_stack.item);
        assert!(thumb.is_some(), "block item should have a thumbnail texture");
        assert_eq!(
            thumb.unwrap().texture_id,
            polychora_plugin_api::texture::builtin_textures::TEX_STONE,
        );

        // Spawn egg thumbnail should resolve to the entity's spawn_egg_texture_id
        let seeker = registry.entity_lookup_by_name("seeker").unwrap();
        let egg_stack = ItemStack::spawn_egg(seeker.namespace, seeker.entity_type);
        let egg_thumb = registry.resolve_item_thumbnail_texture(&egg_stack.item);
        assert!(
            egg_thumb.is_some(),
            "spawn egg should have a thumbnail texture",
        );
        assert_eq!(
            egg_thumb.unwrap(),
            TextureRef {
                namespace: 0,
                texture_id: polychora_plugin_api::content_ids::SPAWN_EGG_TEX_SEEKER,
            },
        );
    }
}
