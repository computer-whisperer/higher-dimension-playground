mod maze;
mod structures;

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use polychora_plugin_api::procgen_abi::{
    ProcgenGenerateInput, ProcgenGenerateOutput, ProcgenPrepareInput, ProcgenPrepareOutput,
    StructureDeclaration,
};

use self::maze::MazeGenerator;
use self::structures::StructureGenerator;

const MAZE_STRUCTURE_ID: u32 = 0xFFFF_0001;

struct Generators {
    structures: StructureGenerator,
    maze: MazeGenerator,
}

struct GeneratorCell(UnsafeCell<Option<Generators>>);
unsafe impl Sync for GeneratorCell {}

static GENERATORS: GeneratorCell = GeneratorCell(UnsafeCell::new(None));

fn generators() -> &'static Generators {
    unsafe {
        let slot = &mut *GENERATORS.0.get();
        slot.get_or_insert_with(|| Generators {
            structures: StructureGenerator::new(),
            maze: MazeGenerator::new(),
        })
    }
}

pub fn structure_declarations() -> Vec<StructureDeclaration> {
    let gen = generators();
    let mut decls = gen.structures.declarations();
    decls.push(StructureDeclaration {
        id: MAZE_STRUCTURE_ID,
        name: alloc::string::String::from("maze"),
        spawn_weight: 1,
    });
    decls
}

pub fn prepare(input: &ProcgenPrepareInput) -> ProcgenPrepareOutput {
    let gen = generators();
    if input.structure_id == MAZE_STRUCTURE_ID {
        gen.maze.prepare(input)
    } else {
        gen.structures.prepare(input)
    }
}

pub fn generate(input: &ProcgenGenerateInput) -> ProcgenGenerateOutput {
    let gen = generators();
    if input.structure_id == MAZE_STRUCTURE_ID {
        gen.maze.generate(input)
    } else {
        gen.structures.generate(input)
    }
}
