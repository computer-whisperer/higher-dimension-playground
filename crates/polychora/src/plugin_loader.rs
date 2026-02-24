use crate::builtin_content;
use crate::content_registry::{ContentRegistry, EntityEntry};
use crate::shared::wasm::{
    WasmExecutionLimits, WasmExecutionRole, WasmRuntime, WasmRuntimeError, WasmRuntimeInstance,
};
use polychora_plugin_api::entity::{EntityCategory, MobArchetype, MobArchetypeDefaults, MobLocomotionMode};
use polychora_plugin_api::manifest::PluginManifest;
use polychora_plugin_api::opcodes::OP_GET_MANIFEST;
use std::fmt;
use std::sync::Arc;

/// A WASM plugin that has been loaded and had its manifest parsed.
pub struct LoadedPlugin {
    pub namespace_id: u32,
    pub name: String,
    pub manifest: PluginManifest,
    #[allow(dead_code)]
    instance: WasmRuntimeInstance,
}

#[derive(Debug)]
pub enum PluginLoadError {
    Wasm(WasmRuntimeError),
    ManifestDeserialize(String),
    ReservedNamespace(u32),
    TextureResolutionFailed { block_name: String, namespace: u32, texture_id: u32 },
    BlockRegistrationFailed(String),
}

impl fmt::Display for PluginLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasm(e) => write!(f, "wasm error: {e}"),
            Self::ManifestDeserialize(e) => write!(f, "manifest deserialization failed: {e}"),
            Self::ReservedNamespace(ns) => write!(f, "namespace {ns:#010x} is reserved for the engine"),
            Self::TextureResolutionFailed { block_name, namespace, texture_id } => {
                write!(f, "block '{block_name}': texture ({namespace:#010x}, {texture_id:#010x}) not found in texture registry")
            }
            Self::BlockRegistrationFailed(msg) => write!(f, "block registration failed: {msg}"),
        }
    }
}

impl std::error::Error for PluginLoadError {}

impl From<WasmRuntimeError> for PluginLoadError {
    fn from(e: WasmRuntimeError) -> Self {
        Self::Wasm(e)
    }
}

/// Load a WASM plugin, validate its ABI, and parse its manifest.
pub fn load_plugin(
    runtime: &WasmRuntime,
    wasm_bytes: &[u8],
    limits: WasmExecutionLimits,
) -> Result<LoadedPlugin, PluginLoadError> {
    let compiled = Arc::new(runtime.compile_module(wasm_bytes)?);
    let mut instance = runtime.instantiate_module(
        compiled,
        WasmExecutionRole::ServerAuthoritative,
        limits,
    )?;

    // Call OP_GET_MANIFEST with empty input
    let result = instance.call_bytes(OP_GET_MANIFEST as i32, &[])?;

    let manifest: PluginManifest = postcard::from_bytes(&result.output)
        .map_err(|e| PluginLoadError::ManifestDeserialize(e.to_string()))?;

    Ok(LoadedPlugin {
        namespace_id: manifest.namespace_id,
        name: manifest.name.clone(),
        manifest,
        instance,
    })
}

/// Populate the content registry from a loaded plugin's manifest.
///
/// For each block declaration, resolves its texture reference to a GPU material
/// token via the registry's texture mappings, then registers the block with that
/// forced token.
///
/// For each entity declaration, finds the closest matching material token by
/// color, then registers the entity.
pub fn populate_registry_from_plugin(
    registry: &mut ContentRegistry,
    plugin: &LoadedPlugin,
) -> Result<(), PluginLoadError> {
    let ns = plugin.namespace_id;

    if ns == 0 {
        return Err(PluginLoadError::ReservedNamespace(ns));
    }

    registry.register_namespace_name(ns, &plugin.name);

    // Register blocks
    for block in &plugin.manifest.blocks {
        let material_token = registry
            .resolve_texture_token(&block.texture)
            .ok_or_else(|| PluginLoadError::TextureResolutionFailed {
                block_name: block.name.clone(),
                namespace: block.texture.namespace,
                texture_id: block.texture.texture_id,
            })?;

        registry.register_block_with_token(
            ns,
            block.type_id,
            block.name.clone(),
            block.category,
            block.color_hint,
            material_token,
            block.texture.clone(),
        );
    }

    // Register entities
    for entity in &plugin.manifest.entities {
        let (mob_archetype, mob_defaults) = resolve_mob_metadata(&entity.name, entity.category);

        // Build aliases: canonical name + common variants
        let mut aliases = vec![entity.name.clone()];
        // Add underscore-stripped alias if name contains underscores
        let stripped = entity.name.replace('_', "");
        if stripped != entity.name {
            aliases.push(stripped);
        }
        // Add well-known legacy aliases for first-party entities
        for extra in legacy_aliases(&entity.name) {
            if !aliases.contains(&extra) {
                aliases.push(extra);
            }
        }

        registry.register_entity(EntityEntry {
            namespace: ns,
            entity_type: entity.type_id,
            category: entity.category,
            canonical_name: entity.name.clone(),
            aliases,
            default_scale: entity.default_scale,
            model_textures: entity.model_textures.clone(),
            mob_archetype,
            mob_defaults,
        });
    }

    Ok(())
}

/// Resolve mob archetype and defaults for known first-party entity names.
///
/// Phase 2 is declarations-only, so mob AI metadata stays hard-coded on the
/// host side. Phase 3 will move steering into WASM.
fn resolve_mob_metadata(
    name: &str,
    category: EntityCategory,
) -> (Option<MobArchetype>, Option<MobArchetypeDefaults>) {
    if category != EntityCategory::Mob {
        return (None, None);
    }

    match name {
        "seeker" => (
            Some(MobArchetype::Seeker),
            Some(MobArchetypeDefaults {
                move_speed: 3.0,
                preferred_distance: 2.6,
                tangent_weight: 0.72,
                locomotion: MobLocomotionMode::Walking,
            }),
        ),
        "creeper" => (
            Some(MobArchetype::Creeper4d),
            Some(MobArchetypeDefaults {
                move_speed: 2.55,
                preferred_distance: 3.8,
                tangent_weight: 0.92,
                locomotion: MobLocomotionMode::Walking,
            }),
        ),
        "phase_spider" => (
            Some(MobArchetype::PhaseSpider),
            Some(MobArchetypeDefaults {
                move_speed: 3.1,
                preferred_distance: 2.4,
                tangent_weight: 0.95,
                locomotion: MobLocomotionMode::Flying,
            }),
        ),
        _ => (None, None),
    }
}

/// Return additional legacy aliases for well-known first-party entity names.
/// These preserve backward compatibility with commands like `/spawn spider`.
fn legacy_aliases(name: &str) -> Vec<String> {
    match name {
        "cube" => vec!["testcube".into()],
        "rotor" => vec!["testrotor".into()],
        "drifter" => vec!["testdrifter".into()],
        "seeker" => vec!["mobseeker".into()],
        "creeper" => vec!["4dcreeper".into(), "mobcreeper4d".into()],
        "phase_spider" => vec!["phase-spider".into(), "spider".into(), "mobphasespider".into()],
        _ => vec![],
    }
}

/// Build a fully-populated content registry: engine internals (Air, Player,
/// texture mappings, legacy remaps) + embedded first-party content plugin
/// (68 blocks, 6 entities).
///
/// This is the standard way to initialize the content registry for both the
/// game client and the server.
pub fn create_full_registry() -> ContentRegistry {
    let mut registry = ContentRegistry::new();
    builtin_content::register_builtin_content(&mut registry);

    let runtime = WasmRuntime::new().expect("failed to init WASM runtime");
    let plugin = load_plugin(
        &runtime,
        include_bytes!(env!("POLYCHORA_CONTENT_WASM_PATH")),
        WasmExecutionLimits::default(),
    )
    .expect("failed to load polychora-content plugin");
    populate_registry_from_plugin(&mut registry, &plugin)
        .expect("failed to populate registry from plugin");

    registry
}
