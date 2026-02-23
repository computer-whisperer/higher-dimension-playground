use crate::content_registry::{ContentRegistry, EntityEntry};
use polychora_plugin_api::entity::EntityCategory;
use polychora_plugin_api::texture::builtin_textures;

/// The namespace ID used by the polychora-content WASM plugin.
/// Must match `polychora-content/src/lib.rs::NAMESPACE`.
pub const CONTENT_PLUGIN_NAMESPACE: u32 = 0x706f6c79; // "poly" in ASCII

/// Register engine-internal content (Air block, Player entity, texture
/// mappings) and legacy remap tables.
///
/// The 68 gameplay blocks and 6 non-player entities are now declared by the
/// `polychora-content` WASM plugin.  This function only registers:
/// - Namespace 0 procedural texture mappings (tex_id → shader case)
/// - Player entity (namespace 0 internal)
/// - Legacy remap tables for save compatibility
pub fn register_builtin_content(registry: &mut ContentRegistry) {
    register_builtin_texture_mappings(registry);
    register_player_entity(registry);
    register_legacy_remaps(registry);
}

/// Namespace 0 procedural texture mappings: (0, texture_id) → shader case.
fn register_builtin_texture_mappings(registry: &mut ContentRegistry) {
    for &(tex_id, material_token) in &builtin_textures::ALL_TEXTURE_MAPPINGS {
        registry.register_texture_token(0, tex_id, material_token);
    }
}

/// Player entity stays as namespace 0 engine internal (not part of any plugin).
fn register_player_entity(registry: &mut ContentRegistry) {
    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 0,
        category: EntityCategory::Player,
        canonical_name: "player".into(),
        aliases: vec!["player".into(), "avatar".into()],
        default_scale: 1.0,
        base_material_token: 0,
        mob_archetype: None,
        mob_defaults: None,
    });
}

/// Legacy remap tables: old (0, N) IDs → new (CONTENT_NS, random_id).
///
/// Old saves store block/entity types as (namespace=0, block_type=1..68).
/// After Phase 2, these blocks live in the polychora-content namespace with
/// random type IDs.  The remap tables translate old IDs on load.
fn register_legacy_remaps(registry: &mut ContentRegistry) {
    let ns = CONTENT_PLUGIN_NAMESPACE;

    // Block remaps: (0, 1) → (ns, BLOCK_RED), (0, 2) → (ns, BLOCK_ORANGE), ...
    const LEGACY_BLOCK_REMAP: [(u32, u32); 68] = [
        (1,  0x9d5288f1), // Red
        (2,  0x5b0dbb41), // Orange
        (3,  0xe453dd32), // Yellow-Green
        (4,  0xb0ee89ae), // Green
        (5,  0xae574f7a), // Cyan
        (6,  0xf2acf72f), // Blue
        (7,  0xec98d2c1), // Purple
        (8,  0x6c941cf0), // Magenta
        (9,  0xa3cd59bf), // Rainbow
        (10, 0x4139d32c), // Brown
        (11, 0xc45ed1f0), // Grid Floor
        (12, 0x21ce5dd2), // White
        (13, 0x1bbb2599), // Light
        (14, 0xb9488d99), // Mirror
        (15, 0x4a578a8e), // Lava-Veined Basalt
        (16, 0xd5e7ce8a), // Crystal Lattice
        (17, 0x5a15544d), // Marble
        (18, 0x246d3f31), // Oxidized Metal
        (19, 0xeaf61a26), // Bio-Spore Moss
        (20, 0x4b982ef8), // Void Mirror
        (21, 0xedd1dfb2), // Avatar Marker
        (22, 0x29db3ad0), // Holographic Laminate
        (23, 0x714ff3d7), // Tidal Glass
        (24, 0x57294739), // Circuit Weave
        (25, 0x8412b293), // Aurora Stone
        (26, 0xb2bc372f), // Hazard Chevrons
        (27, 0xe58842de), // Stone
        (28, 0x6d65a441), // Cobblestone
        (29, 0x39a3b2e9), // Dirt
        (30, 0x6ec42e08), // Coarse Dirt
        (31, 0x6af30553), // Oak Planks
        (32, 0x45a240ae), // Spruce Planks
        (33, 0xbb9099a4), // Log Bark
        (34, 0x5458a885), // Log End Rings
        (35, 0xc3aa7efe), // Sand
        (36, 0xffc89849), // Gravel
        (37, 0xbefcfad8), // Clay
        (38, 0xb5e5a5ab), // Grass Block
        (39, 0x22476f57), // Snow
        (40, 0xabf00273), // Ice
        (41, 0xb28defe3), // Coal Ore
        (42, 0x3bcfbe01), // Iron Ore
        (43, 0x98bd6407), // Gold Ore
        (44, 0xcaa80dd4), // Diamond Ore
        (45, 0x4eabedcb), // Redstone Ore
        (46, 0x39d4beef), // Birch Planks
        (47, 0x8656af72), // Bricks
        (48, 0x7123fdf7), // Sandstone
        (49, 0x551b4cf3), // Glass
        (50, 0xfce66fa2), // Glowstone
        (51, 0xb3d70628), // Obsidian
        (52, 0xc02b61c4), // Prismarine
        (53, 0x9e944239), // Terracotta
        (54, 0x4838b326), // Wool (White)
        (55, 0xbf42e12f), // Basalt Tiles
        (56, 0x6304317f), // Copper Weave
        (57, 0xe7c524a5), // Nebula Strata
        (58, 0xd4b032cc), // Starforged Core
        (59, 0xd6a7ee39), // Cryo Circuit
        (60, 0x1e51f30d), // Smoked Glass
        (61, 0x4aa2e4f9), // Ivory Marble
        (62, 0xe261a7ab), // Runic Alloy
        (63, 0x1837b8a3), // Hyperphase Gel
        (64, 0xde177b4e), // Singularity Core
        (65, 0x60c187fc), // Chrono Bloom
        (66, 0x76b2bc5b), // Tesseract Weave
        (67, 0x548aaa9e), // Eventide Alloy
        (68, 0x20f1bc81), // Beacon Matrix
    ];

    for &(legacy_bt, new_bt) in &LEGACY_BLOCK_REMAP {
        registry.register_legacy_block_remap((0, legacy_bt), (ns, new_bt));
    }

    // Entity remaps: (0, 1) → (ns, ENTITY_CUBE), etc.
    const LEGACY_ENTITY_REMAP: [(u32, u32); 6] = [
        (1,  0x776b1b69), // cube
        (2,  0x71790134), // rotor
        (3,  0x433824fe), // drifter
        (10, 0xa974d75b), // seeker
        (11, 0x3dc5fd3d), // creeper
        (12, 0x4af27f80), // phase_spider
    ];

    for &(legacy_et, new_et) in &LEGACY_ENTITY_REMAP {
        registry.register_legacy_entity_remap((0, legacy_et), (ns, new_et));
    }
}
