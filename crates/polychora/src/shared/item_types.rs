use crate::shared::protocol::{Item, ItemStack};
use crate::shared::voxel::{BlockData, TesseractOrientation};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Well-known item type constants: (namespace, item_type)
// ---------------------------------------------------------------------------

pub const ITEM_BLOCK: (u32, u32) = (0, 1);
pub const ITEM_SPAWN_EGG: (u32, u32) = (0, 2);

// ---------------------------------------------------------------------------
// BlockItemMeta — CBOR schema for block_stack item data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockItemMeta {
    pub namespace: u32,
    pub block_type: u32,
    #[serde(default)]
    pub extra_data: Vec<u8>,
    #[serde(default)]
    pub scale_exp: i8,
}

impl BlockItemMeta {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode should not fail");
        buf
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        ciborium::from_reader(data).ok()
    }
}

// ---------------------------------------------------------------------------
// SpawnEggMeta — CBOR schema for spawn_egg item data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpawnEggMeta {
    pub entity_namespace: u32,
    pub entity_type: u32,
}

impl SpawnEggMeta {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode should not fail");
        buf
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        ciborium::from_reader(data).ok()
    }
}

// ---------------------------------------------------------------------------
// ItemStack convenience constructors
// ---------------------------------------------------------------------------

impl ItemStack {
    /// Create a block_stack item for the given block type.
    pub fn block(namespace: u32, block_type: u32, count: u32, scale_exp: i8) -> Self {
        let meta = BlockItemMeta {
            namespace,
            block_type,
            extra_data: Vec::new(),
            scale_exp,
        };
        Self {
            item: Item {
                namespace: ITEM_BLOCK.0,
                item_type: ITEM_BLOCK.1,
                data: meta.encode(),
            },
            count,
        }
    }

    /// If this is a block_stack item, decode the block data (with default orientation).
    pub fn to_block_data(&self) -> Option<BlockData> {
        if (self.item.namespace, self.item.item_type) != ITEM_BLOCK {
            return None;
        }
        let meta = BlockItemMeta::decode(&self.item.data)?;
        Some(BlockData {
            namespace: meta.namespace,
            block_type: meta.block_type,
            orientation: TesseractOrientation::IDENTITY,
            extra_data: meta.extra_data,
            scale_exp: meta.scale_exp,
        })
    }

    /// If this is a block_stack item, return the (namespace, block_type).
    pub fn block_type_key(&self) -> Option<(u32, u32)> {
        if (self.item.namespace, self.item.item_type) != ITEM_BLOCK {
            return None;
        }
        let meta = BlockItemMeta::decode(&self.item.data)?;
        Some((meta.namespace, meta.block_type))
    }

    /// Create a spawn egg item for the given entity type.
    pub fn spawn_egg(entity_namespace: u32, entity_type: u32) -> Self {
        let meta = SpawnEggMeta {
            entity_namespace,
            entity_type,
        };
        Self {
            item: Item {
                namespace: ITEM_SPAWN_EGG.0,
                item_type: ITEM_SPAWN_EGG.1,
                data: meta.encode(),
            },
            count: 1,
        }
    }

    /// If this is a spawn egg item, return the (namespace, entity_type).
    pub fn spawn_egg_entity_key(&self) -> Option<(u32, u32)> {
        if (self.item.namespace, self.item.item_type) != ITEM_SPAWN_EGG {
            return None;
        }
        let meta = SpawnEggMeta::decode(&self.item.data)?;
        Some((meta.entity_namespace, meta.entity_type))
    }

    /// Encode an `ItemStack` to CBOR bytes (for entity data serialization).
    pub fn encode_to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode should not fail");
        buf
    }

    /// Decode an `ItemStack` from CBOR bytes.
    pub fn decode_from_cbor(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        ciborium::from_reader(data).ok()
    }
}
