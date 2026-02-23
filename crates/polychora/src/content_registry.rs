use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::entity::{
    EntityCategory, MobArchetype, MobArchetypeDefaults, MobLocomotionMode,
};
use std::collections::HashMap;

/// Bit 15 of a material token: when set, bits [14:0] index into the GPU texture
/// pool instead of the procedural `sampleMaterial()` switch.
pub const MATERIAL_TOKEN_TEXTURE_POOL_FLAG: u16 = 1 << 15;

// ---------------------------------------------------------------------------
// Block entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BlockEntry {
    pub namespace: u32,
    pub block_type: u32,
    pub name: &'static str,
    pub category: BlockCategory,
    pub color: [u8; 3],
    /// GPU material token (u16) assigned at registration time.
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
    pub canonical_name: &'static str,
    pub aliases: &'static [&'static str],
    pub default_scale: f32,
    pub base_material_token: u16,
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

    // Legacy remap: old (namespace, block_type) → new (namespace, block_type)
    legacy_block_remap: HashMap<(u32, u32), (u32, u32)>,
    // Legacy remap: old (namespace, entity_type) → new (namespace, entity_type)
    legacy_entity_remap: HashMap<(u32, u32), (u32, u32)>,
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
            legacy_block_remap: HashMap::new(),
            legacy_entity_remap: HashMap::new(),
        };
        // Air is always token 0 at (0, 0)
        registry.blocks.insert(
            (0, 0),
            BlockEntry {
                namespace: 0,
                block_type: 0,
                name: "Air",
                category: BlockCategory::Special,
                color: [0, 0, 0],
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
        name: &'static str,
        category: BlockCategory,
        color: [u8; 3],
    ) -> u16 {
        let token = self.next_procedural_token;
        self.next_procedural_token += 1;

        let entry = BlockEntry {
            namespace,
            block_type,
            name,
            category,
            color,
            material_token: token,
        };
        self.blocks.insert((namespace, block_type), entry);
        self.block_order.push((namespace, block_type));
        self.token_to_block.insert(token, (namespace, block_type));
        token
    }

    /// Register a block with a specific (forced) material token.
    /// Used for builtin content that must match existing GPU shader IDs.
    ///
    /// # Panics
    /// Panics if `forced_token` is already assigned to a different block.
    pub fn register_block_with_token(
        &mut self,
        namespace: u32,
        block_type: u32,
        name: &'static str,
        category: BlockCategory,
        color: [u8; 3],
        forced_token: u16,
    ) {
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
        let normalized_canonical = normalize_token(entry.canonical_name);
        self.entity_name_index.insert(normalized_canonical, key);
        for alias in entry.aliases {
            self.entity_name_index.insert(normalize_token(alias), key);
        }
        self.entities.insert(key, entry);
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

    /// Get the full block entry by (namespace, block_type).
    pub fn block_entry(&self, namespace: u32, block_type: u32) -> Option<&BlockEntry> {
        self.blocks.get(&(namespace, block_type))
    }

    /// Resolve (namespace, block_type) → GPU material token.
    /// Replacement for `materials::block_to_material_token`.
    pub fn block_material_token(&self, namespace: u32, block_type: u32) -> u16 {
        self.blocks
            .get(&(namespace, block_type))
            .map(|e| e.material_token)
            .unwrap_or(1) // fallback to Red
    }

    /// Get block name by (namespace, block_type).
    pub fn block_name(&self, namespace: u32, block_type: u32) -> &'static str {
        self.blocks
            .get(&(namespace, block_type))
            .map(|e| e.name)
            .unwrap_or("Unknown")
    }

    /// Get block color by (namespace, block_type).
    pub fn block_color(&self, namespace: u32, block_type: u32) -> [u8; 3] {
        self.blocks
            .get(&(namespace, block_type))
            .map(|e| e.color)
            .unwrap_or([128, 128, 128])
    }

    /// Get block category label by (namespace, block_type).
    pub fn block_category_label(&self, namespace: u32, block_type: u32) -> &'static str {
        self.blocks
            .get(&(namespace, block_type))
            .map(|e| e.category.label())
            .unwrap_or("Unknown")
    }

    /// Get block category by (namespace, block_type).
    pub fn block_category(&self, namespace: u32, block_type: u32) -> Option<BlockCategory> {
        self.blocks
            .get(&(namespace, block_type))
            .map(|e| e.category)
    }

    // -----------------------------------------------------------------------
    // Material-token-based lookups (reverse: token → block info)
    // -----------------------------------------------------------------------

    /// Get material name by token. Replacement for `materials::material_name`.
    pub fn material_name_by_token(&self, token: u16) -> &'static str {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.name)
            .unwrap_or("Unknown")
    }

    /// Get material color by token. Replacement for `materials::material_color`.
    pub fn material_color_by_token(&self, token: u16) -> [u8; 3] {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.color)
            .unwrap_or([128, 128, 128])
    }

    /// Get material category label by token.
    pub fn material_category_label_by_token(&self, token: u16) -> &'static str {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
            .map(|e| e.category.label())
            .unwrap_or("Unknown")
    }

    /// Get the block entry for a material token.
    pub fn block_entry_by_token(&self, token: u16) -> Option<&BlockEntry> {
        self.token_to_block
            .get(&token)
            .and_then(|key| self.blocks.get(key))
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

    /// Lookup entity by (namespace, entity_type).
    pub fn entity_lookup(&self, namespace: u32, entity_type: u32) -> Option<&EntityEntry> {
        self.entities.get(&(namespace, entity_type))
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
        self.entities
            .get(&(namespace, entity_type))
            .map(|e| e.category)
            .unwrap_or(EntityCategory::Accent)
    }

    /// Get the base material token for an entity.
    pub fn entity_base_material_token(&self, namespace: u32, entity_type: u32) -> u16 {
        self.entities
            .get(&(namespace, entity_type))
            .map(|e| e.base_material_token)
            .unwrap_or(7) // fallback to Purple
    }

    /// Get all spawnable entity canonical names.
    pub fn spawnable_entity_names(&self) -> Vec<&'static str> {
        self.entities
            .values()
            .filter(|e| e.is_spawnable())
            .map(|e| e.canonical_name)
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

    #[test]
    fn builtin_blocks_match_legacy_materials() {
        let mut registry = ContentRegistry::new();
        builtin_content::register_builtin_content(&mut registry);

        // Verify all 68 blocks are registered
        assert_eq!(registry.block_count(), 68);

        // Verify specific lookups match the old MATERIALS data
        assert_eq!(registry.material_name_by_token(1), "Red");
        assert_eq!(registry.material_name_by_token(27), "Stone");
        assert_eq!(registry.material_name_by_token(35), "Sand");
        assert_eq!(registry.material_name_by_token(68), "Beacon Matrix");

        assert_eq!(registry.material_color_by_token(1), [255, 0, 0]);
        assert_eq!(registry.material_color_by_token(12), [255, 255, 255]);
        assert_eq!(registry.material_color_by_token(35), [237, 201, 175]);
        assert_eq!(registry.material_color_by_token(58), [255, 236, 168]);
        assert_eq!(registry.material_color_by_token(68), [255, 248, 196]);

        // Verify block_to_material_token equivalence for legacy namespace 0
        for token in 1u16..=68 {
            let (ns, bt) = registry.token_to_block[&token];
            assert_eq!(registry.block_material_token(ns, bt), token);
        }
    }

    #[test]
    fn builtin_entities_match_legacy() {
        let mut registry = ContentRegistry::new();
        builtin_content::register_builtin_content(&mut registry);

        // Player avatar
        let player = registry.entity_lookup(0, 0).unwrap();
        assert_eq!(player.canonical_name, "player");
        assert_eq!(player.category, EntityCategory::Player);

        // Seeker mob
        let seeker = registry.entity_lookup_by_name("seeker").unwrap();
        assert_eq!(seeker.entity_type, 10);
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
        let mut registry = ContentRegistry::new();
        builtin_content::register_builtin_content(&mut registry);

        // In the current implementation, legacy (0, N) maps to (0, N) since
        // we keep namespace 0 IDs identical for Phase 1 compatibility.
        // The remap tables exist for future namespace migration.
        let (ns, bt) = registry.resolve_legacy_block(0, 27);
        assert_eq!(registry.block_name(ns, bt), "Stone");

        let (ns, et) = registry.resolve_legacy_entity(0, 10);
        let entry = registry.entity_lookup(ns, et).unwrap();
        assert_eq!(entry.canonical_name, "seeker");
    }

    #[test]
    fn unknown_lookups_return_defaults() {
        let mut registry = ContentRegistry::new();
        builtin_content::register_builtin_content(&mut registry);

        assert_eq!(registry.block_material_token(999, 999), 1); // Red fallback
        assert_eq!(registry.block_name(999, 999), "Unknown");
        assert_eq!(registry.block_color(999, 999), [128, 128, 128]);
        assert_eq!(registry.entity_category(999, 999), EntityCategory::Accent);
        assert_eq!(registry.entity_base_material_token(999, 999), 7);
    }
}
