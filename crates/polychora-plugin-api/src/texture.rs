use serde::{Deserialize, Serialize};

/// A reference to a texture identified by its owning namespace and a
/// per-namespace unique texture ID.  Both values are stable random u32s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureRef {
    pub namespace: u32,
    pub texture_id: u32,
}

/// Stable random texture IDs for namespace 0 procedural shader textures.
///
/// These are the contract between host and guest for referencing the 68
/// built-in procedural materials.  Each constant maps to a GPU shader case
/// (material_token 1..=68).  Both the host (builtin_content.rs) and WASM
/// plugins import these so block declarations can reference them.
pub mod builtin_textures {
    // Basic colored blocks (shader cases 1-8)
    pub const TEX_RED: u32                  = 0xb3b1799d;
    pub const TEX_ORANGE: u32               = 0x2c80317f;
    pub const TEX_YELLOW_GREEN: u32         = 0x16671ad1;
    pub const TEX_GREEN: u32                = 0xcdd640fb;
    pub const TEX_CYAN: u32                 = 0x56685257;
    pub const TEX_BLUE: u32                 = 0x4eb13b90;
    pub const TEX_PURPLE: u32               = 0x492456de;
    pub const TEX_MAGENTA: u32              = 0x33b8c1e9;

    // Special materials (shader cases 9-14)
    pub const TEX_RAINBOW: u32              = 0xcc8960a9;
    pub const TEX_BROWN: u32                = 0x2a3d1fa7;
    pub const TEX_GRID_FLOOR: u32           = 0xbd3c2d6d;
    pub const TEX_WHITE: u32                = 0xcd9c66b3;
    pub const TEX_LIGHT: u32                = 0xf465e150;
    pub const TEX_MIRROR: u32               = 0x9b9d2434;

    // Animated/special materials (shader cases 15-26)
    pub const TEX_LAVA_VEINED_BASALT: u32   = 0x26419f82;
    pub const TEX_CRYSTAL_LATTICE: u32      = 0xa72a8469;
    pub const TEX_MARBLE: u32               = 0x7c031199;
    pub const TEX_OXIDIZED_METAL: u32       = 0x1822e8f3;
    pub const TEX_BIO_SPORE_MOSS: u32       = 0x17a0ca6e;
    pub const TEX_VOID_MIRROR: u32          = 0x27fc695a;
    pub const TEX_AVATAR_MARKER: u32        = 0x47f8a88b;
    pub const TEX_HOLOGRAPHIC_LAMINATE: u32 = 0x4b8faa18;
    pub const TEX_TIDAL_GLASS: u32          = 0x915ef6d1;
    pub const TEX_CIRCUIT_WEAVE: u32        = 0xaa1de644;
    pub const TEX_AURORA_STONE: u32         = 0x16cb0fb3;
    pub const TEX_HAZARD_CHEVRONS: u32      = 0x9fadc1a6;

    // Natural materials (shader cases 27-30)
    pub const TEX_STONE: u32                = 0x42e70629;
    pub const TEX_COBBLESTONE: u32          = 0xc74d0fb1;
    pub const TEX_DIRT: u32                 = 0xb65ed389;
    pub const TEX_COARSE_DIRT: u32          = 0xc38a088c;

    // Wood materials (shader cases 31-34)
    pub const TEX_OAK_PLANKS: u32           = 0x9b8148f6;
    pub const TEX_SPRUCE_PLANKS: u32        = 0x7b65a6a4;
    pub const TEX_LOG_BARK: u32             = 0x486ecbe0;
    pub const TEX_LOG_END_RINGS: u32        = 0x82ff5d2a;

    // New natural materials (shader cases 35-40)
    pub const TEX_SAND: u32                 = 0xa6da1dac;
    pub const TEX_GRAVEL: u32               = 0x57378190;
    pub const TEX_CLAY: u32                 = 0xdf36d58b;
    pub const TEX_GRASS_BLOCK: u32          = 0xee8a774b;
    pub const TEX_SNOW: u32                 = 0x11a9e71f;
    pub const TEX_ICE: u32                  = 0xd241330b;

    // Ore materials (shader cases 41-45)
    pub const TEX_COAL_ORE: u32             = 0xde4a2bbd;
    pub const TEX_IRON_ORE: u32             = 0x38df6ec4;
    pub const TEX_GOLD_ORE: u32             = 0xc2b9437a;
    pub const TEX_DIAMOND_ORE: u32          = 0x7c307511;
    pub const TEX_REDSTONE_ORE: u32         = 0x671aa876;

    // Additional wood and building materials (shader cases 46-48)
    pub const TEX_BIRCH_PLANKS: u32         = 0x57229389;
    pub const TEX_BRICKS: u32               = 0x37cd8130;
    pub const TEX_SANDSTONE: u32            = 0x471ecd7b;

    // Glass and light materials (shader cases 49-51)
    pub const TEX_GLASS: u32                = 0xd37459ee;
    pub const TEX_GLOWSTONE: u32            = 0x662b0f79;
    pub const TEX_OBSIDIAN: u32             = 0x2a2a73ed;

    // Special decorative materials (shader cases 52-54)
    pub const TEX_PRISMARINE: u32           = 0x27be3111;
    pub const TEX_TERRACOTTA: u32           = 0x7142ea7d;
    pub const TEX_WOOL_WHITE: u32           = 0x28c26797;

    // Advanced structural materials (shader cases 55-62)
    pub const TEX_BASALT_TILES: u32         = 0x6be6128e;
    pub const TEX_COPPER_WEAVE: u32         = 0xe8f56413;
    pub const TEX_NEBULA_STRATA: u32        = 0x680d7b71;
    pub const TEX_STARFORGED_CORE: u32      = 0xaa8dca03;
    pub const TEX_CRYO_CIRCUIT: u32         = 0x53b7a3a6;
    pub const TEX_SMOKED_GLASS: u32         = 0xde9ff57f;
    pub const TEX_IVORY_MARBLE: u32         = 0x1b1f9163;
    pub const TEX_RUNIC_ALLOY: u32          = 0xcacfb3d0;

    // Volumetric animated materials (shader cases 63-68)
    pub const TEX_HYPERPHASE_GEL: u32       = 0x859cde66;
    pub const TEX_SINGULARITY_CORE: u32     = 0x99463e85;
    pub const TEX_CHRONO_BLOOM: u32         = 0x2ff49b78;
    pub const TEX_TESSERACT_WEAVE: u32      = 0xfc1b8ca1;
    pub const TEX_EVENTIDE_ALLOY: u32       = 0x70e7a113;
    pub const TEX_BEACON_MATRIX: u32        = 0x242c3fe8;

    /// All 68 texture IDs in order, each paired with its shader case (material_token).
    /// Index 0 = (TEX_RED, 1), Index 67 = (TEX_BEACON_MATRIX, 68).
    pub const ALL_TEXTURE_MAPPINGS: [(u32, u16); 68] = [
        (TEX_RED, 1),
        (TEX_ORANGE, 2),
        (TEX_YELLOW_GREEN, 3),
        (TEX_GREEN, 4),
        (TEX_CYAN, 5),
        (TEX_BLUE, 6),
        (TEX_PURPLE, 7),
        (TEX_MAGENTA, 8),
        (TEX_RAINBOW, 9),
        (TEX_BROWN, 10),
        (TEX_GRID_FLOOR, 11),
        (TEX_WHITE, 12),
        (TEX_LIGHT, 13),
        (TEX_MIRROR, 14),
        (TEX_LAVA_VEINED_BASALT, 15),
        (TEX_CRYSTAL_LATTICE, 16),
        (TEX_MARBLE, 17),
        (TEX_OXIDIZED_METAL, 18),
        (TEX_BIO_SPORE_MOSS, 19),
        (TEX_VOID_MIRROR, 20),
        (TEX_AVATAR_MARKER, 21),
        (TEX_HOLOGRAPHIC_LAMINATE, 22),
        (TEX_TIDAL_GLASS, 23),
        (TEX_CIRCUIT_WEAVE, 24),
        (TEX_AURORA_STONE, 25),
        (TEX_HAZARD_CHEVRONS, 26),
        (TEX_STONE, 27),
        (TEX_COBBLESTONE, 28),
        (TEX_DIRT, 29),
        (TEX_COARSE_DIRT, 30),
        (TEX_OAK_PLANKS, 31),
        (TEX_SPRUCE_PLANKS, 32),
        (TEX_LOG_BARK, 33),
        (TEX_LOG_END_RINGS, 34),
        (TEX_SAND, 35),
        (TEX_GRAVEL, 36),
        (TEX_CLAY, 37),
        (TEX_GRASS_BLOCK, 38),
        (TEX_SNOW, 39),
        (TEX_ICE, 40),
        (TEX_COAL_ORE, 41),
        (TEX_IRON_ORE, 42),
        (TEX_GOLD_ORE, 43),
        (TEX_DIAMOND_ORE, 44),
        (TEX_REDSTONE_ORE, 45),
        (TEX_BIRCH_PLANKS, 46),
        (TEX_BRICKS, 47),
        (TEX_SANDSTONE, 48),
        (TEX_GLASS, 49),
        (TEX_GLOWSTONE, 50),
        (TEX_OBSIDIAN, 51),
        (TEX_PRISMARINE, 52),
        (TEX_TERRACOTTA, 53),
        (TEX_WOOL_WHITE, 54),
        (TEX_BASALT_TILES, 55),
        (TEX_COPPER_WEAVE, 56),
        (TEX_NEBULA_STRATA, 57),
        (TEX_STARFORGED_CORE, 58),
        (TEX_CRYO_CIRCUIT, 59),
        (TEX_SMOKED_GLASS, 60),
        (TEX_IVORY_MARBLE, 61),
        (TEX_RUNIC_ALLOY, 62),
        (TEX_HYPERPHASE_GEL, 63),
        (TEX_SINGULARITY_CORE, 64),
        (TEX_CHRONO_BLOOM, 65),
        (TEX_TESSERACT_WEAVE, 66),
        (TEX_EVENTIDE_ALLOY, 67),
        (TEX_BEACON_MATRIX, 68),
    ];
}
