use spirv_builder::{MetadataPrintout, SpirvBuilder, ModuleResult, Capability};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.1")
        .capability(Capability::VariablePointers)
        .capability(Capability::Int64)
        .print_metadata(MetadataPrintout::DependencyOnly)
        .build()?;
    if let ModuleResult::SingleModule(module) = result.module {
        println!("cargo:rustc-env={}={}", "SPIRV_OUT_DIR", module.parent().unwrap().display());
    }
    Ok(())
}