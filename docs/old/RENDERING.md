# Intro to Higher-Dimensional Rendering

A primer for graphics programmers who understand 3D rendering and want to build
intuition for the 4D generalizations used in this engine.

This engine does full **4D path tracing** — not wireframe projection, not
cross-section slicing. It casts rays through a four-dimensional scene, bounces
them off surfaces, and accumulates light exactly the way a 3D path tracer does,
just with one more spatial dimension everywhere.

If you already understand triangles, normals, ray-triangle intersection, BVH
traversal, and Monte Carlo path tracing in 3D, you have all the prerequisites.
Every concept below is a direct generalization of something you already know.


## What Does It Mean to See in 4D?

The core pattern is a **dimension ladder**: almost every 3D rendering concept has
a 2D version below it and a 4D version above it. Once you see the pattern, the
4D case stops feeling arbitrary and starts feeling inevitable.

| Dim | Solid object | Surface element | Normal is perp to | Rendering primitive |
|-----|--------------|-----------------|--------------------|--------------------|
| 2D  | polygon      | edge (1D)       | 1 tangent vector   | line segment       |
| 3D  | polyhedron   | face (2D)       | 2 tangent vectors  | triangle           |
| 4D  | polychoron   | cell (3D)       | 3 tangent vectors  | tetrahedron        |

In 3D, a solid object's boundary is a surface made of flat **faces** (triangles).
In 4D, a solid object's boundary is a **hyper-surface** made of flat **cells**
(tetrahedra). The rendering primitive gains one vertex each time you go up a
dimension — line segments (2 vertices) become triangles (3 vertices) become
tetrahedra (4 vertices).

### How 4D becomes a 2D image

A 3D camera collapses three spatial dimensions into two. A single photograph
integrates over all depth values along each pixel's ray — you see everything in
front of the camera regardless of how far away it is.

A 4D camera must collapse **four** spatial dimensions into two. Conceptually this
happens in two stages:

1. **Integrate over ZW viewing angles** — the extra dimension W must be
   "looked through", just like depth Z is in a normal photo. The engine samples
   random ZW directions and averages them, analogous to how a 3D camera averages
   over all depths along each ray.
2. **Standard XY perspective** — once the ZW integration is accounted for, the
   remaining projection is familiar: pixels map to XY directions, focal length
   controls field of view.

**W is a spatial dimension, not time.** The scene is a static four-dimensional
space. There is no animation axis hiding inside the math — W is just a fourth
direction you can walk in, perpendicular to X, Y, and Z.


---


## Section 1: Tesseract Geometry

The primary test object in this engine is a **tesseract** (4D hypercube). Before
we can raytrace it we need to understand its structure, decompose its surface
into rendering primitives, and compute normals.

### 1.1 The Hypercube by Analogy

A D-dimensional hypercube has **2^D vertices**, each encoded as a D-bit binary
number where each bit selects 0 or 1 along one axis.

| Dimension | Name         | Vertices | Edges | Faces | Cells |
|-----------|-------------|----------|-------|-------|-------|
| 1         | line segment | 2        | 1     | —     | —     |
| 2         | square       | 4        | 4     | 1     | —     |
| 3         | cube         | 8        | 12    | 6     | —     |
| 4         | tesseract    | 16       | 32    | 24    | 8     |

A tesseract has **16 vertices**. You can label them `0000` through `1111` in
binary, where bit *k* is 1 when the vertex sits at coordinate 1 along axis *k*
(and 0 when it sits at coordinate 0).

The "surface" of a 3D cube is made of 2D faces (squares). The "surface" of a 4D
tesseract is made of **3D cells (cubes)**. A tesseract has 8 cubic cells, 24
square faces, 32 edges, and 16 vertices.

The sub-element count formula for a D-dimensional hypercube: the number of
k-dimensional sub-elements is `C(D, k) * 2^(D-k)`.

| Sub-element     | Formula for D=4   | Count |
|-----------------|-------------------|-------|
| Vertices (k=0)  | C(4,0) * 2^4 = 16 | 16    |
| Edges (k=1)     | C(4,1) * 2^3 = 32 | 32    |
| Faces (k=2)     | C(4,2) * 2^2 = 24 | 24    |
| Cells (k=3)     | C(4,3) * 2^1 = 8  | 8     |

### 1.2 Identifying Cells (3-Faces)

Each cell of a tesseract is an ordinary 3D cube embedded in 4D space. To find
a cell, **fix one of the 4 axes to either 0 or 1**. The remaining three axes
are free to vary over {0, 1}, giving 2^3 = 8 vertices — exactly a cube.

There are 4 axes to fix and 2 values to fix them at: `C(4,1) * 2 = 8` cells.

**Concrete example:** the W=0 cell contains all vertices whose W-bit is 0:

```
0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111
```

These 8 vertices form a cube in the XYZ subspace at W=0. Similarly, the X=1
cell contains all vertices whose X-bit is 1, forming a cube in the YZW subspace.

The code represents each cell as a base vertex plus three "direction" vertices
(the three axes along which the cell extends). From these four vertices, the
full 8-vertex cube can be reconstructed.

> **Code ref:** `Hypercube::<4>::generate_k_faces_3()` in `src/hypercube.rs`
> (line 53)

### 1.3 Simplex Decomposition

Triangles are the rendering primitive in 3D. **Tetrahedra** are the rendering
primitive in 4D. Just as any polygon can be triangulated, any polyhedron can
be tetrahedralized.

Each cubic cell of the tesseract is decomposed into **6 tetrahedra** using a
"permutation walk" method:

1. Start at the base vertex of the cell (e.g., `0000`).
2. The cell has three free axes. Take all **3! = 6** orderings (permutations) of
   those axes.
3. For each permutation, walk from the base vertex by toggling one axis at a
   time in that order. The 4 vertices you visit (including the start) form one
   tetrahedron.

**Concrete example:** for the W=0 cell with free axes X, Y, Z:

- Permutation X→Y→Z: `0000 → 0001 → 0011 → 0111` (toggle X, then Y, then Z)
- Permutation X→Z→Y: `0000 → 0001 → 0101 → 0111`
- Permutation Y→X→Z: `0000 → 0010 → 0011 → 0111`
- Permutation Y→Z→X: `0000 → 0010 → 0110 → 0111`
- Permutation Z→X→Y: `0000 → 0100 → 0101 → 0111`
- Permutation Z→Y→X: `0000 → 0100 → 0110 → 0111`

Each path gives 4 vertices, each set of 4 vertices forms a tetrahedron, and
the 6 tetrahedra tile the cube without overlap.

**Total for the tesseract:** 8 cells × 6 tetrahedra = **48 tetrahedra**.

> **Code ref:** `generate_simplexes_for_k_face_3()` in `src/hypercube.rs`
> (line 87)


---


## Section 2: Normals in 4D

### 2.1 Pattern Across Dimensions

A normal vector is perpendicular to a surface. As dimensions increase, the
surface gains dimensions too, so you need more tangent vectors to define it:

| Dim | Surface       | Tangent vectors needed | Normal computation         |
|-----|---------------|------------------------|----------------------------|
| 2D  | edge (1D)     | 1 vector               | rotate 90° (swap + negate) |
| 3D  | face (2D)     | 2 vectors              | cross product of 2 vectors |
| 4D  | cell (3D)     | 3 vectors              | 4D cross product of 3 vectors |

In 3D, the cross product takes two vectors and returns a third vector
perpendicular to both. In 4D, we need a **generalized cross product** that takes
**three** vectors and returns a fourth vector perpendicular to all three.

### 2.2 Computing the 4D Cross Product

The 4D cross product of vectors **a**, **b**, **c** is defined via a formal
determinant with the standard basis vectors in the first row:

```
         | e_x   e_y   e_z   e_w |
cross4D = | a_x   a_y   a_z   a_w |
         | b_x   b_y   b_z   b_w |
         | c_x   c_y   c_z   c_w |
```

Expanding along the first row (cofactor expansion), each component of the result
is a 3×3 determinant (minor) with alternating sign:

```
result.x =  det | a.y  a.z  a.w |
                | b.y  b.z  b.w |
                | c.y  c.z  c.w |

result.y = -det | a.x  a.z  a.w |
                | b.x  b.z  b.w |
                | c.x  c.z  c.w |

result.z =  det | a.x  a.y  a.w |
                | b.x  b.y  b.w |
                | c.x  c.y  c.w |

result.w = -det | a.x  a.y  a.z |
                | b.x  b.y  b.z |
                | c.x  c.y  c.z |
```

**Worked example:** if **a** = (1,0,0,0), **b** = (0,1,0,0), **c** = (0,0,1,0),
then the only nonzero minor is the w-component: `det(I₃) = 1`, with a sign flip
giving `result = (0, 0, 0, -1)`. The result is along the W-axis, perpendicular
to the XYZ subspace — exactly what you'd expect.

In the GPU shader, the implementation uses `cross()` on float3 swizzles to
compute each 3×3 determinant efficiently:

```glsl
// From math.slang, cross4D():
result.x =  determinant3x3(a.yzw, b.yzw, c.yzw);
result.y = -determinant3x3(a.xzw, b.xzw, c.xzw);
result.z =  determinant3x3(a.xyw, b.xyw, c.xyw);
result.w = -determinant3x3(a.xyz, b.xyz, c.xyz);
```

> **Code ref:** `cross4D()` in `slang-shaders/src/math.slang` (line 254)
> **Code ref:** `get_normal_4d()` in `common/src/linalg_n.rs` (line 7)

### 2.3 Normal Orientation

The 4D cross product gives a normal direction, but not necessarily the
**outward-facing** one. The code determines correct orientation by testing
the normal against the vector from the tesseract's center (0.5, 0.5, 0.5, 0.5)
to the tetrahedron's surface.

For a tesseract centered at the origin with vertices at {0, 1}⁴, the outward
direction depends on which cell the tetrahedron belongs to. The code uses a
heuristic: if the cell contains vertex `0000`, the normal should point one way;
otherwise, the other. When the normal points the wrong way, two vertices are
swapped (which reverses the winding order and flips the normal).

```rust
// From render.rs, tetrahedron generation:
let normal = get_normal(&[e1, e2, e3]);
let test_vector = Vec4::new(1.0, 1.0, 1.0, 1.0);
let is_normal_flipped = test_vector.dot(normal) < 0.0;
let should_be_flipped = cell.contains(&0); // contains vertex 0000
if should_be_flipped != is_normal_flipped {
    vertex_positions.swap(1, 2); // swap two vertices to flip winding
}
```

> **Code ref:** Normal flipping in `src/render.rs` (lines 119–136)


---


## Section 3: Homogeneous Transforms in 4D

### 3.1 Why 5×5 Matrices?

In 3D graphics, we use **4×4 homogeneous matrices** to combine rotation, scale,
and translation into a single matrix multiply. The extra row/column lets
translations be represented as linear operations on homogeneous coordinates
(x, y, z, 1).

In 4D, positions have four components, so homogeneous coordinates have five:
**(x, y, z, w, 1)**. Transforms are **5×5 matrices**.

A 4D translation matrix looks like:

```
| 1  0  0  0  tx |
| 0  1  0  0  ty |
| 0  0  1  0  tz |
| 0  0  0  1  tw |
| 0  0  0  0  1  |
```

The translation values go in the 5th column, just as 3D translations go in the
4th column of a 4×4 matrix. Positions are transformed as:

```
[x']   [1  0  0  0  tx] [x]   [x + tx]
[y'] = [0  1  0  0  ty] [y] = [y + ty]
[z']   [0  0  1  0  tz] [z]   [z + tz]
[w']   [0  0  0  1  tw] [w]   [w + tw]
[1 ]   [0  0  0  0  1 ] [1]   [  1   ]
```

> **Code ref:** `translate_matrix_4d()` in `src/matrix_operations.rs` (line 25)

### 3.2 Rotations Happen in Planes, Not Around Axes

This is the **most important conceptual shift** for 4D transforms.

In 3D, we say rotation happens "around an axis" — a line that stays fixed while
everything else rotates. But there's a more general way to think about it:
rotation happens **in a plane**. The two axes that define the rotation plane
move; everything perpendicular to the plane stays fixed.

| Dimension | Rotation plane | What stays fixed    | Count         |
|-----------|---------------|---------------------|---------------|
| 2D        | XY            | nothing (just origin) | C(2,2) = 1 |
| 3D        | XY, XZ, YZ   | 1 axis (line)       | C(3,2) = 3   |
| 4D        | XY, XZ, XW, YZ, YW, ZW | 2 axes (plane) | C(4,2) = 6 |

In 3D, the three rotation planes correspond to the three rotation axes (XY
rotation = rotation around Z). There's a one-to-one mapping between planes and
axes, so both descriptions work equally well.

In 4D, there is **no such correspondence**. There are 6 rotation planes but only
4 axes. Rotation "around the W axis" doesn't make sense — you'd have to specify
which of the three planes perpendicular to W you mean (XY? XZ? YZ? All three?).
Instead, you specify a rotation plane directly: "rotate in the XW plane."

Three of the six 4D rotation planes — **XW, YW, ZW** — have no 3D analog. They
mix a familiar axis with the W direction.

A rotation by angle θ in the XW plane looks like:

```
| cos θ   0   0   sin θ   0 |
|  0      1   0    0      0 |
|  0      0   1    0      0 |
| -sin θ  0   0   cos θ   0 |
|  0      0   0    0      1 |
```

This is identical in structure to a 3D XZ rotation matrix — just with the Z row
replaced by the W row. The Y and Z axes are untouched; only X and W participate.

The general rotation function takes an arbitrary pair of axis indices and an
angle, then fills in the four cosine/sine entries in an otherwise-identity
matrix:

```
matrix[dim_from][dim_from] =  cos θ
matrix[dim_from][dim_to]   =  sin θ
matrix[dim_to][dim_from]   = -sin θ
matrix[dim_to][dim_to]     =  cos θ
```

> **Code ref:** `rotation_matrix_one_angle()` in `src/matrix_operations.rs`
> (line 48) — works for any dimension count and any pair of axes.

### 3.3 Composing Transforms

The full model transform pipeline is the same as in 3D, just one dimension
higher:

1. **Scale** (5×5 diagonal matrix)
2. **Rotate** (compose as many plane rotations as needed)
3. **Translate** (5×5 matrix with offsets in column 5)

All three combine into a single 5×5 **model matrix**, applied as:

```
world_position = ModelMatrix × (x, y, z, w, 1)ᵀ
```

The **view transform** is the inverse of the camera's model matrix. It translates
the scene so the camera is at the origin, then rotates so the camera looks down
its forward axis.


---


## Section 4: 4D Perspective Projection

### 4.1 Dual Perspective — The Core Idea

In a 3D renderer, each pixel maps to one ray direction. Two parameters
(pixel X, pixel Y) and a focal length fully determine the ray — there's no
ambiguity about what direction the camera is looking along the depth axis.

In a 4D renderer, there are **two** "depth-like" axes: Z and W. A pixel's XY
position determines the ray's XY components, but there's a whole circle of
possible (Z, W) directions. The camera must sample this circle.

The engine uses **two focal lengths**:
- **focal_xy** controls the XY field of view (how wide the image is), just like
  a normal 3D focal length.
- **focal_zw** controls the ZW viewing range (how much of the W-direction you
  can see). A smaller focal_zw gives a wider ZW angle — you "see more" of the
  4th dimension.

### 4.2 Monte Carlo ZW Sampling

Each rendered sample picks a random ZW direction within the viewing range. The
ray construction works as follows:

```
viewAngle = (π/2) / focal_zw          // total ZW angular range
zwRand    = random(0, 1)              // uniform random sample
zwAngle   = (zwRand - 0.5) * viewAngle + π/4    // centered around 45°

rayDirection = normalize(
    px  / focal_xy,                   // X: pixel position / focal length
    -py / (focal_xy * aspect),        // Y: with aspect ratio correction
    cos(zwAngle),                     // Z: ZW angle → Z component
    sin(zwAngle)                      // W: ZW angle → W component
)
```

The ZW angle is centered around π/4 (45°), which gives equal Z and W components
by default — the camera looks equally into Z and W.

### 4.3 The Depth-Slice Buffer

The pixel buffer is not a simple 2D image. It's a **3D buffer**:
`[width × height × depth_slices]`. Each depth slice accumulates samples from
a different ZW angle.

At display time, all slices are summed together. This is the Monte Carlo
integration that collapses the W dimension into the final image, just as a
regular photograph integrates over all depth values along each ray.

The linearized buffer index is:

```
index = z_slice * width * height + y * width + x
```

The more depth slices (and the more sub-frames rendered), the more ZW angles
are sampled and the less noisy the final image becomes.

### 4.4 Intuition

Think about what a photograph does: it takes a 3D scene and collapses depth
into a 2D image. Every point along a ray contributes to one pixel, regardless of
how far away it is.

Now imagine you need to photograph a 4D scene. You must collapse **two** extra
dimensions (Z and W) into a flat image. The XY part works the same as always.
The ZW part is handled by taking many photographs at slightly different ZW
angles and averaging them. Each "photograph" is like a 3D slice through the 4D
scene, viewed from a particular W-direction. The average of all these slices is
the final image.

> **Code ref:** `mainRaytracerPixel()` in `slang-shaders/src/raytracer.slang`
> (line 490)


---


## Section 5: Ray-Tetrahedron Intersection

### 5.1 Problem Statement

Given a 4D ray `P(t) = O + t·D` and a tetrahedron with vertices
{v₀, v₁, v₂, v₃} in 4D space, determine:
1. Does the ray hit the tetrahedron?
2. If so, at what distance `t` and what barycentric coordinates?

This is the direct analog of ray-triangle intersection in 3D, one dimension
higher. A triangle defines a 2D plane in 3D space; a tetrahedron defines a 3D
**hyperplane** in 4D space.

### 5.2 Step 1 — Hit Distance via Normal

The tetrahedron lies in a hyperplane defined by its 4D normal **N** (computed
via `cross4D`) and any vertex. The hyperplane equation is:

```
dot(P - v₀, N) = 0
```

Substituting the ray equation:

```
dot(O + t·D - v₀, N) = 0
dot(O - v₀, N) + t · dot(D, N) = 0
t = -dot(O - v₀, N) / dot(D, N)
```

**Early-out conditions** (checked before the expensive barycentric computation):
- `|dot(D, N)| < 0.001` — ray is nearly parallel to the hyperplane, skip
- `t ≤ 0` — intersection is behind the ray origin, skip
- `t ≥ closestHit.distance` — farther than an already-found hit, skip

These culls are cheap (one dot product and comparisons) and reject the vast
majority of tetrahedra.

### 5.3 Step 2 — Barycentric Coordinates via 3×3 Cramer's Rule

Once we know the ray hits the hyperplane at distance `t`, we compute the hit
point and check whether it's inside the tetrahedron using barycentric
coordinates.

The hit point relative to v₀ is:

```
q = (O + t·D) - v₀
```

We want to express **q** as a linear combination of the tetrahedron's edge
vectors:

```
q = u·e₁ + v·e₂ + w·e₃
```

where `e₁ = v₁ - v₀`, `e₂ = v₂ - v₀`, `e₃ = v₃ - v₀`.

This is a system of **4 equations** (one per dimension) with **3 unknowns**
(u, v, w) — it's overdetermined. Since we know the hit point lies in the
hyperplane, any 3 of the 4 equations are sufficient. But which 3?

**Drop the axis with the largest normal component.** If the normal's biggest
component is (say) X, then the hyperplane is most "tilted away" from the X-axis,
and the remaining 3 equations (Y, Z, W) form the best-conditioned 3×3 system.

The 3×3 system is solved with **Cramer's rule** using 3D cross products:

```glsl
float3 c23 = cross(e2s, e3s);      // cross product of edges 2 and 3
float  det = dot(e1s, c23);        // determinant of 3x3 matrix
float  u   = dot(qs, c23) / det;
float  v   = dot(e1s, cross(qs, e3s)) / det;
float  w   = dot(e1s, cross(e2s, qs)) / det;
float  b0  = 1.0 - u - v - w;     // fourth barycentric coordinate
```

Where `e1s`, `e2s`, `e3s`, `qs` are the 3D projections of the 4D vectors
(with the dropped axis removed).

**Inside test:** the hit point is inside the tetrahedron if and only if all four
barycentric coordinates are non-negative:

```
u ≥ 0   AND   v ≥ 0   AND   w ≥ 0   AND   (1-u-v-w) ≥ 0
```

**Why not a 4×4 matrix inverse?** The Cramer's rule approach requires only 3D
cross products and dot products — operations that GPUs execute extremely fast.
A full 4×4 inverse would be roughly 10× more expensive and provide no accuracy
benefit, since the overdetermined system is already solved exactly by choosing
the right 3×3 sub-system.

> **Code ref:** `testTetrahedronHit()` in `slang-shaders/src/raytracer.slang`
> (line 180)


---


## Section 6: 4D BVH Acceleration

A brute-force approach would test every ray against all 48+ tetrahedra. A
**Bounding Volume Hierarchy (BVH)** reduces this to O(log N) by organizing
tetrahedra into a spatial tree of axis-aligned bounding boxes.

The 4D BVH is a direct generalization of a 3D BVH, with wider bounding boxes
and Morton codes that interleave four axes instead of three.

### 6.1 4D Axis-Aligned Bounding Boxes

A 3D AABB stores 6 values: min and max for X, Y, Z. A 4D AABB stores **8
values**: min and max for X, Y, Z, and W. In the shader, these are stored as
two `float4` vectors:

```
minBounds = (x_min, y_min, z_min, w_min)
maxBounds = (x_max, y_max, z_max, w_max)
```

### 6.2 4D Ray-AABB Slab Test

The slab test generalizes directly. For each of the 4 axes, compute the ray's
entry and exit distances for that axis's slab:

```glsl
float4 t1 = (minBounds - origin) * invDirection;
float4 t2 = (maxBounds - origin) * invDirection;
float4 tmin4 = min(t1, t2);   // handle negative direction
float4 tmax4 = max(t1, t2);
```

The ray enters the box when it has entered **all** slabs (take the maximum of
the 4 entry values) and exits when it has exited **any** slab (take the minimum
of the 4 exit values):

```glsl
float tmin = max(max(tmin4.x, tmin4.y), max(tmin4.z, tmin4.w));
float tmax = min(min(tmax4.x, tmax4.y), min(tmax4.z, tmax4.w));
```

The ray hits the box if and only if `tmin ≤ tmax` and `tmax ≥ 0`.

In 3D you take max/min over 3 slab pairs. In 4D you take max/min over 4. That's
the only difference.

> **Code ref:** `rayAABBIntersect4D()` in `slang-shaders/src/raytracer.slang`
> (line 157)

### 6.3 4D Morton Codes

Morton codes (Z-order curves) map multi-dimensional coordinates to a 1D key
while preserving spatial locality. The 3D version interleaves 3 axes into a
single integer. The 4D version interleaves **4 axes**.

Each tetrahedron's centroid is normalized to [0, 1] within the scene's bounding
box. Each axis gets **16 bits** of precision, producing a **64-bit** Morton code:

```
Bit pattern: ...w₃z₃y₃x₃ w₂z₂y₂x₂ w₁z₁y₁x₁ w₀z₀y₀x₀
```

Every group of 4 consecutive bits contains one bit from each of W, Z, Y, X.
Objects that are close in 4D space tend to have similar Morton codes, which
means they end up near each other after sorting — exactly what the BVH builder
needs.

The bit expansion is done by `expandBits16to64()`, which takes a 16-bit value
and inserts 3 zero bits after each bit, spreading it across 64 bits. Four
such expansions (one per axis) are shifted and OR'd together.

> **Code ref:** `morton4D()` in `slang-shaders/src/bvh.slang` (line 99)

### 6.4 GPU Construction Pipeline

The BVH is built entirely on the GPU in 7 phases, each a separate compute
dispatch (or series of dispatches). Between dispatches, Vulkan pipeline barriers
ensure all writes are visible to the next phase.

| Phase | Kernel                     | What it does                              |
|-------|----------------------------|-------------------------------------------|
| 1     | `raytrace_pre`             | Transform tetrahedra to view space         |
| 2a    | `mainBVHSceneBounds`       | Parallel reduction → scene AABB            |
| 2b    | `mainBVHMortonCodes`       | Compute 64-bit 4D Morton code per tetrahedron |
| 2c    | `mainBVHBitonicSort*`      | Sort Morton codes (shared-memory + global) |
| 2d    | `mainBVHInitLeaves`        | Initialize BVH leaf nodes                  |
| 2e    | `mainBVHBuildTree`         | Build internal nodes (Karras algorithm)    |
| 2f    | `mainBVHComputeLeafAABBs`  | Compute tight AABBs for leaf nodes         |
| 2g    | `mainBVHPropagateAABBs`    | Propagate AABBs from leaves to root        |

**Phase 2a** runs a single workgroup of 64 threads that strides over all
tetrahedra, computing the min/max bounds. A shared-memory reduction produces
the final scene AABB.

**Phase 2b** normalizes each tetrahedron's centroid into the scene AABB, computes
the Morton code, and writes sentinel values (`~uint64_t(0)`) for padding
elements beyond the actual tetrahedron count (required for bitonic sort).

**Phase 2c** is a bitonic sort optimized with shared memory. Local stages (where
partners are within a 64-thread workgroup) run in shared memory. Global stages
dispatch separate kernels for the wide steps, then a local-merge kernel for the
narrow steps.

**Phases 2d and 2e must be separate dispatches.** Leaf initialization writes
the sorted tetrahedron indices into leaf nodes; internal node construction reads
those leaves to build the tree structure. A single dispatch would cause race
conditions because `DeviceMemoryBarrier()` doesn't synchronize across all
threads in a dispatch — only across threads within a workgroup. Separate
dispatches with Vulkan pipeline barriers provide the needed global
synchronization.

**Phase 2g** runs `2 * ceil(log₂(N))` passes. Each pass checks whether both
children of an internal node have valid AABBs (marked by an atomic visit count
≥ 2). If both are valid, the node merges them and marks itself valid. Nodes
whose children aren't ready yet simply skip and retry on the next pass.

> **Code ref:** BVH dispatch sequence in `src/render.rs` (lines 1546–1606)

### 6.5 BVH Traversal

Traversal is a standard stack-based depth-first search:

1. Test the root AABB. If the ray misses, return immediately.
2. Push the root onto the stack.
3. Pop a node. If it's a leaf, test the tetrahedron. If it's internal, test both
   children's AABBs and push the ones that hit (farther child first so the closer
   child is popped first).
4. Repeat until the stack is empty.

The stack size is 64 entries — sufficient for a balanced binary tree of up to
2^64 leaves. In practice the tree is much shallower.

An important optimization: when pushing children, only push a child if its AABB
hit distance is closer than the current closest tetrahedron hit. This prunes
large portions of the tree once a nearby hit has been found.

> **Code ref:** `raycast()` in `slang-shaders/src/raytracer.slang` (line 304)


---


## Section 7: Path Tracing in 4D

### 7.1 Reflection and Diffuse Sampling

The reflection formula is **dimension-independent**:

```
r = d - 2(d · n)n
```

This works identically in 2D, 3D, 4D, or any dimension. The dot product and
scalar multiplication generalize trivially.

**Diffuse sampling** in 3D generates a random direction on the hemisphere above
the surface. In 4D, we generate a random direction on the **3-hemisphere**
(the upper half of the 3-sphere). The engine uses a simple approach:

```glsl
direction = normalize(randomFloat4() + hitNormal);
```

This generates a random 4D vector and biases it toward the normal. The result is
not a perfectly uniform hemisphere sample, but it's cheap and effective for a
Monte Carlo estimator — bias is corrected by the large number of samples.

### 7.2 Material Model

Materials have four properties:

| Property    | Range   | Effect                                        |
|-------------|---------|-----------------------------------------------|
| `albedo`    | RGBA    | Surface color (diffuse/specular reflectance)   |
| `metallic`  | [0, 1]  | Probability of specular vs. diffuse bounce     |
| `roughness` | [0, 1]  | Perturbation magnitude on specular reflection  |
| `luminance` | [0, ∞)  | Light emission factor (0 = non-emissive)       |

At each bounce, the engine randomly chooses between specular reflection and
diffuse scattering based on the `metallic` value:

```
if random() < metallic:
    direction = reflect(incomingDir, normal)   // mirror reflection
    direction += randomFloat4() * roughness    // roughness perturbation
else:
    direction = normalize(randomFloat4() + normal)  // diffuse scatter
```

Emissive materials (luminance > 0) add light at each bounce. The engine includes
materials ranging from pure diffuse (brown, floor) to perfect mirrors
(metallic = 1.0) to light sources (luminance = 40.0).

> **Code ref:** Material definitions in `slang-shaders/src/materials.slang`
> (line 5)

### 7.3 Bounce Accumulation

Each path traces up to **6 bounces**. At each bounce:

1. Cast a ray and find the closest tetrahedron hit.
2. Look up the material at the hit point.
3. Choose specular or diffuse direction.
4. Record the material's albedo and luminance.
5. Advance the ray origin past the surface (by a small epsilon).
6. Repeat.

If a ray escapes the scene (no hit), it samples the sky:
- A directional "sun" at `normalize(0.3, 1.0, -0.3, 0.0)` provides strong
  white light when the ray points near it.
- A blue-cyan gradient based on the ray's Y component provides ambient sky
  illumination.

The accumulated light is computed by working backwards through the bounce stack:

```
for each bounce from last to first:
    light = light * albedo + luminance * albedo
```

This is the standard recursive rendering equation unrolled into an iterative
loop: each surface modulates the incoming light by its albedo and adds its own
emission.

> **Code ref:** `raycastSample()` in `slang-shaders/src/raytracer.slang`
> (line 406)

### 7.4 Progressive Accumulation

The engine renders multiple **sub-frames** per displayed frame. Each sub-frame
uses a different random seed (producing different bounce directions and ZW
angles). Samples are accumulated in the pixel buffer and averaged at display
time.

This is standard Monte Carlo integration: more samples → less noise. The ZW
sampling is part of this same process — each sub-frame sees the 4D scene from a
slightly different ZW angle, and the average over all angles produces the final
"4D photograph."


---


## Section 8: 4D Frustum Clipping (Rasterizer)

The rasterizer projects tetrahedra from 4D view space to 2D screen space, then
integrates along the ZW axis per pixel. Before projection, tetrahedra must be
clipped to the visible region — otherwise edges that cross through depth zero
produce degenerate screen coordinates.

### 8.1 Why 4D Frustum Boundaries Are Cones

In 3D, the view frustum is bounded by flat planes: near, far, left, right, top,
bottom. The "depth" axis is a single line (Z), so the near plane is simply
`Z = near`.

In 4D, **depth has two components**: Z and W. The "depth" of a point is:

```
depth = sqrt(Z² + W²)
```

This is always non-negative, regardless of the signs of Z and W. A vertex at
(Z=1, W=0) and a vertex at (Z=-1, W=0) both have `depth = 1` — but an edge
connecting them passes through the origin (Z=0, W=0) where `depth = 0`,
producing a division-by-zero in perspective projection.

The visible region in ZW space is not a half-space — it's a **cone** defined by
an angular range `[theta_min, theta_max]` where `theta = atan2(W, Z)`. Points
at any angle have positive depth, but only points within the viewing cone should
be rendered.

### 8.2 ZW Viewing Cone Clip Boundaries

The camera samples ZW directions uniformly within an angular range centered at
`pi/4` (equal parts Z and W):

```
zwViewAngle = (pi/2) / focalLengthZW
theta_min   = pi/4 - zwViewAngle/2
theta_max   = pi/4 + zwViewAngle/2
```

For `focalLengthZW = 1.0`: `theta_min = 0`, `theta_max = pi/2`, meaning the
visible cone is the first quadrant of the ZW plane (Z >= 0 **and** W >= 0).

The key insight is that these cone boundaries are **linear hyperplanes** in 4D
view space. The boundary at angle theta is the set of points where the ZW
component is exactly on the boundary ray:

```
Lower boundary: -sin(theta_min) * Z + cos(theta_min) * W >= 0
Upper boundary:  sin(theta_max) * Z - cos(theta_max) * W >= 0
```

Because these are linear in Z and W, standard edge-clip interpolation is
**exact** — no nonlinear correction needed. This is the crucial property that
makes the clips artifact-free.

### 8.3 The FIX_DEPTH Problem

After clipping, new vertices are created by linear interpolation along edges.
For ZW cone clips, the interpolated `projectionDivisor` (= `depth / focalLength`)
is linearly interpolated, which does NOT equal the true value
`sqrt(Z² + W²) / focalLength` at the interpolated Z, W coordinates.

Since `sqrt` is a **concave function**, by Jensen's inequality:

```
sqrt(lerp(Z₁, Z₂)² + lerp(W₁, W₂)²)  ≤  lerp(sqrt(Z₁² + W₁²), sqrt(Z₂² + W₂²))
```

Recomputing `projDiv` from the interpolated Z, W (`FIX_DEPTH`) produces a
**smaller** value than the linearly interpolated one. At clip boundaries this
creates a brightness discontinuity — vertices just inside the clip have
recomputed projDiv, while nearby unclipped vertices use their original values.

**Solution**: Do NOT apply FIX_DEPTH after ZW cone clips. Instead, after both
cone clips are complete, recompute `projDiv` for ALL vertices from their actual
Z, W coordinates. This ensures consistent values across clip boundaries. Only
the near-depth clip (pass 3) uses FIX_DEPTH, since it operates on projDiv
directly and new vertices need exact values.

### 8.4 Clipping Architecture (3 Passes)

The rasterizer clips each tetrahedron in three passes:

| Pass | Clip condition | fixDepth | Purpose |
|------|---------------|----------|---------|
| 1 | `-sin(θ_min)*Z + cos(θ_min)*W >= 0` | false | ZW cone lower boundary |
| 2 | `sin(θ_max)*Z - cos(θ_max)*W >= 0` | false | ZW cone upper boundary |
| — | Recompute projDiv for all vertices | — | Sync projDiv with actual Z,W |
| 3 | `projDiv >= MIN_DEPTH_DIVISOR` | true | Near-depth safety clip |

Screen-space clips (left/right/top/bottom) are **omitted**. The pixel shader
already performs a bounding-box test per tetrahedron, which naturally skips
off-screen geometry. Adding screen clips would require FIX_DEPTH (since clipped
vertices need correct projDiv for screen-space position), reintroducing the
brightness discontinuity problem.

### 8.5 Tetrahedron Clipping Topology

The clipper handles all cases of N vertices inside the clip boundary:

| Inside | Outside | Output tetrahedra | Shape |
|--------|---------|-------------------|-------|
| 4 | 0 | 1 (passthrough) | Original |
| 3 | 1 | 3 | Truncated corner |
| 2 | 2 | 3 | Sliced in half |
| 1 | 3 | 1 | Single corner |
| 0 | 4 | 0 (culled) | — |

Each pass can multiply the tetrahedron count by up to 3x, so the buffer holds
up to 24 tetrahedra (1 input × 3^3 worst case across 3 passes, though in
practice the count stays much lower).

### 8.6 Perspective-Correct Interpolation

After projection, the pixel shader needs to interpolate ZW positions and texture
coordinates across screen-space triangular faces. Naive linear interpolation in
screen space is incorrect because perspective division is nonlinear.

The rasterizer stores `invProjectionDivisors` (1/projDiv for each of the 4
vertices) in the output tetrahedron. The pixel shader uses the standard
perspective-correct formula:

```
attr_correct = (bary₀·attr₀/d₀ + bary₁·attr₁/d₁ + bary₂·attr₂/d₂)
             / (bary₀/d₀ + bary₁/d₁ + bary₂/d₂)
```

where `bary_i` are screen-space barycentric coordinates and `d_i` are the
projection divisors. This recovers the correct view-space attribute values at
each pixel.

> **Code ref:** Clipping pipeline in `slang-shaders/src/rasterizer.slang`
> `mainTetrahedronCS()` (line 191)
> **Code ref:** Perspective-correct interpolation in `mainTetrahedronPixelCS()`
> (line 610)


---


## Appendix: Dimension Ladder Reference

A quick-reference table of how key concepts scale across dimensions.

### Geometry

| Concept              | 2D               | 3D                | 4D                    |
|----------------------|------------------|--------------------|-----------------------|
| Hypercube name       | square           | cube               | tesseract             |
| Vertices             | 4                | 8                  | 16                    |
| Edges                | 4                | 12                 | 32                    |
| Faces                | 1                | 6                  | 24                    |
| Cells                | —                | —                  | 8                     |
| Surface element      | edge             | face (triangle)    | cell (tetrahedron)    |
| Rendering primitive  | line segment     | triangle (3 verts) | tetrahedron (4 verts) |
| Simplices per cell   | —                | 2 triangles/quad   | 6 tetrahedra/cube     |

### Normals and Cross Products

| Concept              | 2D                | 3D                  | 4D                        |
|----------------------|-------------------|----------------------|---------------------------|
| Normal dimension     | 1D (scalar dir)   | 1D (vector)         | 1D (vector)               |
| Input vectors        | 1                  | 2                    | 3                         |
| Computation          | 90° rotation       | 3D cross product     | 4D cross product (cofactors) |
| Component formula    | (-y, x)           | 3×1 determinants     | 3×3 determinants          |

### Transforms

| Concept              | 2D        | 3D       | 4D        |
|----------------------|-----------|----------|-----------|
| Homogeneous coords   | (x,y,1)   | (x,y,z,1) | (x,y,z,w,1) |
| Matrix size          | 3×3       | 4×4      | 5×5       |
| Rotation planes      | 1 (XY)    | 3        | 6         |
| Novel rotation planes | —        | —        | 3 (XW, YW, ZW) |

### Bounding Volumes and Intersection

| Concept              | 2D              | 3D              | 4D                |
|----------------------|-----------------|------------------|--------------------|
| AABB values          | 4 (min/max × 2) | 6 (min/max × 3) | 8 (min/max × 4)   |
| Slab test axes       | 2               | 3                | 4                  |
| Morton code bits     | 2 interleaved   | 3 interleaved   | 4 interleaved      |
| Barycentric coords   | 2 + remainder   | 3 + remainder   | 4 + remainder      |
| Intersection system  | 1×1             | 2×2              | 3×3 (Cramer's rule) |

### Path Tracing

| Concept              | 2D             | 3D                | 4D                  |
|----------------------|----------------|--------------------|-----------------------|
| Reflection formula   | d - 2(d·n)n    | d - 2(d·n)n       | d - 2(d·n)n          |
| Hemisphere dimension | semicircle (1D) | hemisphere (2D)   | 3-hemisphere (3D)    |
| Focal lengths        | 1              | 1                  | 2 (XY + ZW)          |
| Extra integration    | —              | —                  | ZW angle sampling     |


---


## Key Source Files

| File | Contents |
|------|----------|
| `src/hypercube.rs` | Tesseract geometry: vertices, cells, simplex decomposition |
| `src/matrix_operations.rs` | 5×5 transformation matrices (translate, rotate, scale) |
| `src/render.rs` | Scene setup, tetrahedron generation, BVH dispatch orchestration |
| `common/src/linalg_n.rs` | CPU-side N-dimensional cross product and normal computation |
| `common/src/mat_n.rs` | N×N matrix type with determinant computation |
| `slang-shaders/src/math.slang` | GPU-side 4D math: `cross4D`, `VecN<N>`, `MatN<N>` |
| `slang-shaders/src/rasterizer.slang` | 4D rasterizer: frustum clipping, ZW projection, pixel shader |
| `slang-shaders/src/raytracer.slang` | Ray generation, intersection, BVH traversal, path tracing |
| `slang-shaders/src/bvh.slang` | Morton codes, bitonic sort, Karras tree, AABB propagation |
| `slang-shaders/src/materials.slang` | Procedural material definitions |
| `slang-shaders/src/hud.slang` | HUD overlay: font atlas text and panel rendering |
| `slang-shaders/src/types.slang` | Shared data structures: `Tetrahedron`, `BVHNode`, `RayHit` |
