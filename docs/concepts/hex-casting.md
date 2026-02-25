# Hex Casting -- Concept Report

> **What**: A Minecraft mod (Forge/Fabric) that adds a programmable spellcasting system
> where players draw patterns on a hexagonal grid to compose stack-based programs
> that manipulate the world. Created by petrak@ (FallingColors). Inspired by the
> mod Psi.
>
> **Why this matters for Polychora**: Hex Casting is one of the most successful
> examples of turning a "programming-as-gameplay" system into something that feels
> magical and physical rather than abstract. Its design decisions around embodied
> input, resource costs, spatial constraints, and progressive revelation are
> directly relevant to designing 4D interaction systems.

---

## 1. Core Loop

1. The player holds a **staff** and right-clicks to open the casting interface.
2. A **hexagonal grid** appears. The player draws a connected path across grid
   intersections using the mouse. Each path is one **pattern**.
3. Each pattern corresponds to an **action** (a function). Drawing the pattern
   executes it immediately against a **stack** of values called **iotas**.
4. Multiple patterns drawn in sequence form a **hex** -- a program.
5. When the hex includes actions that affect the world (break blocks, launch
   projectiles, teleport, etc.), those effects happen in real-time and cost
   **media** (amethyst-based energy).

The key insight: spell-writing is a *physical, gestural act* done in real-time
in the game world, not a menu or text editor. You draw with the mouse, and the
shape you draw *is* the code.

---

## 2. Patterns and the Hexagonal Grid

### How patterns work

- The grid has **six directions** from any node. A pattern is a connected path
  starting from a given direction and taking a sequence of turns.
- Each pattern is encoded as an **angle signature** -- a string of letters
  representing the sequence of turns taken:
  - `w` = continue straight
  - `e` = slight right, `q` = slight left
  - `d` = sharp right, `a` = sharp left
- Example: Mind's Reflection has the angle signature `qaq`.
- The same geometric shape drawn starting from a different direction may map to
  a different action, making starting direction significant.

### Pattern identity

A pattern is identified by its **starting direction + angle signature**. The
visual shape on the grid is the primary "name" the player learns -- experienced
players recognize patterns by sight the way a programmer recognizes syntax.

---

## 3. The Stack and Iotas

Hex Casting uses a **stack machine** (LIFO). Every hex starts with an empty
stack, and patterns push/pop values.

### Iota types

| Type | Description |
|------|-------------|
| **Number** | Double-precision float (e.g. `12`, `3.14`) |
| **Vector** | 3D coordinate (e.g. `<100, 64, -100>`) |
| **Boolean** | `true` / `false` |
| **Entity** | Reference to a living entity or player in the world |
| **List** | Ordered collection of iotas (can be nested) |
| **Pattern** | A pattern itself, treated as data rather than executed |
| **Null** | The absence of a value |
| **Garbage** | Nonsense value injected by mishaps -- a poison pill on the stack |

### Stack limits

- Maximum **1024 iotas** on the stack at once.
- Lists count as their contents + 1 (the container itself), so a list of 512
  items consumes 513 slots.

### The Ravenmind

A single-iota **register** available during casting. It starts as Null and can
be written to / read from via dedicated patterns. Useful as a temporary variable
without burning stack space.

---

## 4. Action Naming Conventions

Actions are named by a consistent taxonomy that tells the player the
input/output signature at a glance:

| Suffix | Pops | Pushes | Example |
|--------|------|--------|---------|
| **Reflection** | 0 | 1 | Mind's Reflection (pushes the caster entity) |
| **Purification** | 1 | 1 | Compass' Purification (entity -> position) |
| **Distillation** | 2 | 1 | Archer's Distillation (pos, pos -> direction) |
| **Exaltation** | 3 | 1 | (various) |
| **Gambit** | varies | varies | Jester's Gambit (swap top two stack items) |
| **Decomposition** | 1 | 2+ | Gemini Decomposition (duplicate top of stack) |
| (Spells) | varies | 0 | Explosion (consumes args, produces world effect) |

This naming convention acts as a built-in type system communicated through
flavor rather than syntax.

---

## 5. Meta-Evaluation and Control Flow

### Introspection / Retrospection

Bracket patterns that switch the interpreter between **execution mode** and
**recording mode**. Patterns drawn inside brackets are pushed to the stack as a
list of pattern iotas rather than being executed. This is how you construct
"code as data."

### Hermes' Gambit

Takes a list of patterns from the stack and **executes** them. This is the
core eval -- it turns data back into code. Enables subroutines.

### Thoth's Gambit

A **map** operation: takes a spell-list and a data-list, executes the spell
once per data element on a temporary stack, and collects results. This is the
primary looping construct.

### Recursion limit

There is a hard cap of **512 meta-evaluations** per hex. Exceeding it triggers
the "Delve Too Deep" mishap. This prevents infinite loops from freezing the
game while still allowing complex recursive spells.

---

## 6. Media -- The Resource System

### What media is

Media is the energy of thought, stored physically in **amethyst** items:
- **Amethyst Dust** (base unit)
- **Amethyst Shards**
- **Charged Amethyst** (crafted, higher density)

Players carry amethyst in their inventory. When a hex produces world effects
(moving blocks, launching projectiles, teleporting), media is consumed from the
player's inventory automatically.

### Overcasting

If a hex costs more media than the player has, the spell doesn't simply fail --
it **draws from the caster's health** instead. This is called **overcasting**.
It allows desperate or prepared players to push beyond their resources at
personal risk. Deliberately overcasting to near-death is the path to
**enlightenment** (see progression).

### Media in spell circles

Spell circles (the automation system) draw media from amethyst placed in the
impetus block, not from a player.

---

## 7. Ambit -- Spatial Constraints

Hex Casting enforces **range limits** that tie magic to physical space:

- **Personal ambit**: a 32-block radius sphere centered on the caster's eyes.
- **Greater Sentinel ambit**: a 16-block radius sphere around a placed sentinel.
- **Spell circle ambit**: the bounding box of the circle's slate blocks.

Any action that tries to affect the world outside the caster's ambit triggers a
**mishap**. This means players must physically position themselves (or their
sentinels) near their targets, keeping magic grounded in spatial gameplay.

---

## 8. Mishaps -- The Error System

Errors don't produce stack traces -- they produce **colored particle effects**
and **in-world consequences**:

| Spark Color | Cause | Consequence |
|-------------|-------|-------------|
| **Yellow** | Unknown pattern | Garbage pushed to stack |
| **Red** | Wrong iota type on stack | Garbage replaces bad iotas |
| **Magenta** | Target out of ambit (range) | Items yanked from hands, flung toward target |
| **Light Blue** | Action requires spell circle | Inventory dumped on ground |
| **Purple** | Bad Akashic Record lookup | Experience points stolen |
| **Dark Green** | Invalid mind-flay target | Subject killed |
| **Teal** | Delve Too Deep (recursion limit) | Hex terminates |

The design principle: **errors are diegetic**. They happen *in the world*, not
in a console. The particle colors teach the player to diagnose problems visually.
The consequences escalate with severity -- a typo gives you garbage, but
overreaching spatially costs you your held items.

---

## 9. Items and Storage

### Casting implements

- **Staff**: Opens the drawing grid. Required to cast freehand hexes.

### Iota storage

- **Focus**: Stores a single iota. Read/write. The basic "variable."
- **Spellbook**: Stores up to 64 iotas (one per page). Sneak-scroll to select
  the active page.
- **Abacus**: Stores a number. Scroll to increment/decrement, sneak-right-click
  to reset. Read-only during casting.

### Packaged hexes (shareable magic items)

These items store a complete hex + an internal media battery, letting anyone
use the spell without knowing how to cast:

| Item | Reusability | Media behavior |
|------|-------------|----------------|
| **Cypher** | Single-use | Destroyed when internal media runs out |
| **Trinket** | Reusable | Stops working when depleted; must recharge |
| **Artifact** | Reusable | When internal media runs out, draws from holder's amethyst inventory |

This creates a natural economy: expert casters craft items for others, and the
three tiers trade off convenience vs. power.

---

## 10. Spell Circles -- Automation

Spell circles are the **multiblock automation system**, shifting from real-time
gestural casting to persistent in-world machines.

### Components

- **Slate**: Track blocks that carry the media wave. The circuit must form a
  closed loop.
- **Impetus**: The block that starts the wave. Four variants:
  - **Toolsmith**: Activated by player right-click
  - **Fletcher**: Activated by player looking at it for 3 seconds
  - **Cleric**: Activated by redstone signal (enables **playerless casting** --
    full automation without a player present)
  - **Empty**: Passes waves through without initiating
- **Directrix**: Routing switches that send the wave down one of two paths:
  - **Mason**: Routes based on redstone signal
  - **Shepherd**: Routes based on a boolean value on the stack
  - **Empty**: Routes randomly (50/50)

### How it works

1. Build a closed loop of slate connected to an impetus.
2. Place patterns on the slate blocks (each slate holds one pattern).
3. Activate the impetus. A wave of media travels through the loop.
4. Each pattern is executed in order as the wave passes it.
5. When the wave completes the circuit, the hex is done.
6. Directrixes enable branching -- different paths for different conditions.

The circle's **ambit** is determined by the bounding box of all its slate
blocks. Larger circles = more spatial reach but slower execution (the wave
must physically travel further).

---

## 11. Progression and Enlightenment

### Early game

Players find a staff, learn basic patterns from the **Hex Book** (an in-game
Patchouli guidebook), and experiment with simple hexes -- reading positions,
breaking blocks, small explosions.

### Mid game

Players discover the meta-evaluation patterns (Hermes'/Thoth's Gambit), build
spell circles, create packaged items (cyphers/trinkets/artifacts), and set up
**Akashic Libraries** (pattern->iota databases built from bookshelves in-world).

### Enlightenment

By deliberately **overcasting to near-death**, the player achieves
**enlightenment**, which unlocks:
- **Great Spells**: Powerful actions that can only be cast with a spell circle
  and require **Ancient Scrolls** (loot-only items found in dungeons) to learn.
- Examples: **Flay Mind** (extracts villager consciousness into blocks),
  **Flight**, **Greater Teleport**, **Summon Greater Sentinel**.

### Akashic Libraries

A knowledge-storage multiblock: an **Akashic Record** block connected to
**Akashic Bookshelves** (within 32 blocks, directly connected). Each bookshelf
maps one pattern to one iota. Players can read entries during casting, creating
a shared in-world database of constants and spell fragments.

---

## 12. The Primitive Instruction Set

Understanding why Hex Casting produces emergent complexity requires seeing how
*few* actual operations it provides, and how deliberately constrained they are.

### Sensing the world (Reflections / Purifications)

The mod gives you a handful of ways to read the world state:

- **Mind's Reflection**: Push the caster entity onto the stack.
- **Compass' Purification**: Entity -> its position vector.
- **Alidade's Purification**: Entity -> its look direction (unit vector).
- **Pace Purification**: Entity -> its velocity vector.
- **Stadiometer's Purification**: Entity -> its height (number).
- **Archer's Distillation**: Position + direction -> raycast hit position.

That's essentially it for sensing. You can know *where you are*, *where you're
looking*, *how fast something is going*, and *what block is in front of you*.
Everything else must be derived. Want to know what block type is at a position?
That requires additional patterns. Want to know the positions of nearby
entities? You can get a list of entities in a zone, then map over them with
Compass' Purification.

### Math (Distillations)

Standard arithmetic that works on both numbers and vectors:

- Additive / Subtractive / Multiplicative / Division / Modulus Distillation
- Length Purification (vector -> magnitude)
- Vector construction / decomposition (number, number, number <-> vector)
- Floor / ceiling

There are no trig functions, no matrix operations. If you need to rotate a
point, you compute it from arithmetic. The polymorphism is important: adding a
number to a vector scales it, adding two vectors sums them. Fewer patterns cover
more cases.

### Stack manipulation

- **Jester's Gambit**: Swap top two.
- **Rotation Gambit**: Rotate top three.
- **Gemini Decomposition**: Duplicate top.
- **Fisherman's Gambit**: Pull the Nth item to the top.
- **Swindler's Gambit**: Arbitrary reordering of the top N items.
- **Bookkeeper's Gambit**: Selectively keep/discard items from the top of the
  stack using a bitmask drawn as the pattern shape (flat line = keep, triangle
  dip = discard). This single pattern replaces what would be dozens of
  special-purpose drop/keep operations.

### List operations

- **Flock's Gambit**: Collect top N items into a list.
- **Flock's Disintegration**: Explode a list back onto the stack.
- **Selection Distillation**: Index into a list.
- **Surgeon's Exaltation**: Replace an element at an index.
- **Abacus Purification**: Get list length.
- **Retrograde Purification**: Reverse a list.
- **Speaker's Decomposition**: Pop first element.
- **Single's Purification**: Wrap top item in a one-element list.

### Conditionals

- **Augur's Exaltation**: Takes a boolean and two iotas, pushes one based on
  the boolean. This is the *only* conditional primitive -- there is no if/else
  block, no pattern for branching. Conditional *execution* must be built from
  Augur's Exaltation + Hermes' Gambit (choose which code-list to eval).

### World actions (Spells)

The actual effects you can produce are deliberately small in number:

| Action | Effect |
|--------|--------|
| **Explosion** | Create an explosion at a position |
| **Impulse** | Apply a velocity vector to an entity |
| **Break Block** | Destroy a block at a position |
| **Place Block** | Place a block at a position (from inventory) |
| **Conjure Block** | Create a temporary ethereal solid block |
| **Conjure Light** | Create a temporary light source |
| **Ignite / Extinguish** | Set fire / put out fire |
| **Potion effects** | Apply a potion effect to an entity |
| **Blink** | Short-range teleport |
| **Create Water / Lava** | Place fluid (lava is a Great Spell) |
| **Overgrow** | Bone-meal effect on crops |
| **Craft Phial** | Create a media container item |
| **Sentinel** | Place/move your sentinel entity |
| **Recharge Item** | Push media into a casting item |

That's roughly the full set. There is no "build a wall" action. There is no
"mine a vein" action. There is no "copy-paste a structure" action. Every
complex behavior must be *composed* from these primitives plus math, stack
manipulation, and meta-evaluation.

### Number literals

Even pushing a specific number onto the stack is non-trivial. Numerical
Reflection is a *family* of patterns where the shape of the pattern encodes
the number. Small integers are simple short patterns. Larger or more precise
numbers require longer, more complex patterns -- to the point that the
community built dedicated **number literal generators** (external tools that
compute the optimal pattern sequence to push a given number). This means even
"hardcode the number 47" is a small puzzle.

---

## 13. Emergent Complexity -- Why It Gets Absurdly Powerful

The genius of Hex Casting is that the primitives listed above, combined with
meta-evaluation and list manipulation, form a system that is **Turing-complete
within its recursion limit**. The distance between "I can break one block" and
"I can build arbitrary structures" is bridged entirely by the player's growing
fluency with composition.

### The literacy curve

**Phase 1 -- Single actions**: The player learns to raycast to the block
they're looking at and break it. This is ~5 patterns: Mind's Reflection,
Compass' Purification, Mind's Reflection, Alidade's Purification, Archer's
Distillation, then Break Block. It's the "hello world" of Hex Casting.

**Phase 2 -- Parameterized actions**: The player learns vector math. Instead of
breaking one block, they compute a position offset and break a block relative
to where they're looking. They learn to push number literals and do arithmetic
on positions. A hex that breaks a 3x3 area involves computing 9 positions and
calling Break Block on each one.

**Phase 3 -- Loops via Thoth's Gambit**: Instead of manually computing 9
positions, the player builds a list of offset vectors, then uses Thoth's Gambit
to map a "break the block at this offset" spell over the list. The same hex
now handles any size. Thoth's Gambit handles time correctly -- raycasting
inside the loop re-evaluates each iteration, so breaking a line of blocks
progresses through them rather than hitting the same spot repeatedly.

**Phase 4 -- Subroutines via Hermes' Gambit**: The player starts storing
reusable spell fragments in Foci or Spellbook pages. A hex reads a spell-list
from a Focus and evals it. Hexes start calling other hexes. The Ravenmind
becomes critical for passing data between subroutines without stack pollution.

**Phase 5 -- Conditional logic**: Augur's Exaltation + Hermes' Gambit gives
if/else over code blocks. The player builds hexes that *adapt* -- check what
block is at a position, choose different actions based on the result. Error
handling emerges: check conditions before acting to avoid mishaps.

**Phase 6 -- Recursive algorithms**: Hermes' Gambit can call a spell list that
itself calls Hermes' Gambit. Players implement BFS, flood fills, recursive
descent. The 512 meta-eval limit becomes the actual design constraint, and
players learn optimization techniques to stay under it.

### The builder's-wand example

A "builder's wand" (a common modded-Minecraft item that extends a surface by
placing matching blocks along it) is a solved problem in most mods -- you
install the mod that adds the item. In Hex Casting, it doesn't exist as a
primitive. Building one from scratch requires:

1. **Raycast** to the block being looked at and determine which **face** is
   being targeted (Archer's Distillation gives position; additional patterns
   give the face normal).
2. **Read the block type** at that position.
3. **Compute adjacent positions** along the target face -- vector math to
   generate a grid of offsets relative to the hit face and position.
4. **Filter** the positions to only those where the adjacent block matches the
   target type (loop + conditional).
5. **Place blocks** at the valid positions (loop + Place Block).
6. **Package** the whole thing into a Trinket or Artifact for one-click use.

The result is ~300 patterns of carefully composed logic. It requires mastery of
raycasting, vector math, list construction, Thoth's Gambit loops, conditional
filtering, and item packaging. But once built, it works identically to a
dedicated mod item -- right-click to extend a surface. The player has
*reinvented* the tool from first principles using a magic system.

This is the core of the emergent power: anything that can be described as
"sense the world, compute, act on the world" can eventually be built. The
system provides no shortcuts for complex behaviors, but it provides all the
pieces needed to construct them.

### Community examples of composed complexity

**Veinminer**: A spell that breaks an entire connected vein of matching ore.
Implements breadth-first search using lists as queues, checking adjacent
blocks for matching types, up to 64 blocks per cast. Uses the Ravenmind to
store the BFS function itself for recursive self-invocation. Approaches the
iota stack limit and meta-eval ceiling -- the spell is optimized to stay under
both constraints. Runs in a single game tick.

**Vector fields (Kirin's Ruler)**: A utility spell for building other spells.
The player places a Sentinel as origin, then right-clicks blocks to record
their positions *relative to the sentinel*. This generates a reusable list of
offsets that can be applied at any location later. It's essentially a "record
macro" tool -- but built within the hex system itself, not provided by the mod.

**Running Stitch**: A spell-composing spell. Takes multiple hex fragments from
Spellbook pages and concatenates them into a single executable hex. This is
metaprogramming -- writing code that writes code, implemented in the game's
own magic system.

**Self-compiling hex from Akashic Records**: A hex that reads an Akashic
Library (the in-world pattern->iota database), looks up spell fragments by
pattern keys, assembles them into a complete hex, and executes it. Effectively
a build system implemented in the magic language.

### What makes this work as game design

The key observation is that the complexity ceiling is set by **player skill**,
not by mod content. The developer didn't need to add a veinminer feature, a
builder's wand feature, or a structure-copy feature. The primitives are
sufficient; the players supply the engineering.

The limitations (stack size, recursion cap, ambit range, media cost) aren't
arbitrary -- they're the design constraints that make the engineering
*interesting*. The 512 meta-eval limit means you can't just brute-force
recursion; you need to be clever about unrolling loops. The 1024 iota stack
limit means you need to manage memory. The ambit range means you need to think
about physical positioning. Media cost means bigger spells cost more resources.

The result is a system where two players with identical tools can have vastly
different capabilities based purely on their understanding and creativity. This
is the hallmark of true emergence: the system's complexity lives in the
*player's knowledge*, not in the content.

---

## 14. Design Principles Worth Noting

### Fewer primitives, more composition

The mod deliberately avoids adding high-level convenience actions. There is no
"build wall" pattern because you can compose one from "place block" + loops +
vector math. Every feature request that can be answered with "you can already
do that by combining X, Y, Z" is a feature the developer *doesn't add*. This
keeps the primitive set small and forces the combinatorial space to do the work.
The result: players who master composition feel genuinely powerful, because
their capabilities are self-made rather than unlocked from a skill tree.

### Constraints as creative fuel

The stack limit (1024), recursion limit (512 meta-evals), ambit range (32
blocks), and media cost aren't just balance knobs -- they're *design materials*
that create interesting problems. The veinminer spell is interesting *because*
it has to stay under the recursion limit. The builder's wand is a satisfying
accomplishment *because* vector math and conditional filtering aren't trivial.
Remove the constraints and you remove the engineering challenge that makes
mastery rewarding.

### Magic as programming, programming as gesture

The hexagonal grid makes code *physical*. You don't type `explode(pos)` -- you
draw the explosion pattern. Muscle memory replaces syntax memorization. The
gestural input creates a skill ceiling where experienced casters draw complex
hexes fluidly.

### Errors are consequences, not messages

Mishaps don't print error text -- they dump your inventory, steal your XP, or
fling your items. The feedback is visceral and teaches through pain, not
documentation.

### Spatial grounding

The ambit system ensures magic never becomes "press button, affect anything
anywhere." You must be physically present or extend your reach with sentinels
and circles. Magic stays connected to the world's geography.

### Progressive disclosure through naming

The action taxonomy (Reflection/Purification/Distillation/etc.) encodes
type signatures in flavorful names. New players learn the system inductively
by noticing patterns in the naming.

### The crafting-sharing loop

Packaged hexes (cyphers/trinkets/artifacts) let expert players share their
work with others as physical items. This creates social dynamics: specialists,
trade, and a natural "enchanter" role in multiplayer.

### Overcasting as risk/reward

Letting players spend health instead of media is a simple mechanic with deep
consequences -- it enables clutch plays, creates the enlightenment gate, and
gives every spell a potential cost beyond resources.

### Automation as a spatial puzzle

Spell circles require physical construction in the world. Routing, branching,
and ambit coverage become spatial design problems, not UI configuration. The
machine *is* the building.

### The power is in the player, not the content

The most important principle: two players with identical mod versions can have
wildly different effective power levels. One can break a single block; the other
can excavate entire veins, build structures with a click, and teleport across
the map. The difference is pure knowledge and craft. The mod ships no "endgame
spells" that trivialize challenges -- the endgame *is* the player's growing
library of self-authored hexes. This means the system never runs out of content,
because the content is whatever the player is clever enough to build next.
