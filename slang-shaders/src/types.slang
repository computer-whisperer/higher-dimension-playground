// Shared types matching Rust common crate
// These must be layout-compatible with bytemuck structs

import math;

// Processed tetrahedron ready for rendering
struct Tetrahedron
{
    float4 vertexPositions[4];
    float4 texturePositions[4];
    float4 normal;
    float4 invProjectionDivisors;
    uint materialId;
    uint padding[3];
}

// Model-space tetrahedron (before instance transform)
struct ModelTetrahedron
{
    float4 vertexPositions[4];
    float4 texturePositions[4];
    uint cellId;
    uint padding[3];
}

// Edge for wireframe rendering
struct ModelEdge
{
    float4 vertexPositions[2];
}

// Screen-space line vertex with per-vertex color.
// Layout must match Rust's LineVertex in src/render.rs.
struct LineVertex
{
    float2 position;
    float2 padding;
    float4 color;
}

// Instance data with 5x5 transform matrix
struct ModelInstance
{
    MatN<5> modelTransform;
    uint cellMaterialIds[8];
}

// Per-frame working data
struct WorkingData
{
    uint4 renderDimensions;   // xy = resolution, z = depth slices
    uint2 presentDimensions;
    uint2 _raytraceSeedPacked; // u64 as 2x u32
    MatN<5> viewMatrix;
    MatN<5> viewMatrixInverse;
    uint totalNumTetrahedrons;
    uint shaderFault;
    float focalLengthXY;
    float focalLengthZW;
    uint padding[2];

    property uint64_t raytraceSeed
    {
        get { return (uint64_t(_raytraceSeedPacked.y) << 32) | uint64_t(_raytraceSeedPacked.x); }
    }
}

// Material properties for shading
struct MaterialProperties
{
    float4 albedo;
    float metallic;
    float roughness;
    float luminance;
    uint padding;
}

// Ray hit result for raytracing
struct RayHit
{
    float4 hitPosition;
    float4 barycentric;
    float4 hitNormal;
    float4 textureCoordinates;
    uint tetrahedronIndex;
    float hitDistance;
    uint didHit;
    uint padding[1];

    static RayHit none()
    {
        RayHit hit;
        hit.hitPosition = float4(0);
        hit.barycentric = float4(0);
        hit.hitNormal = float4(0);
        hit.textureCoordinates = float4(0);
        hit.tetrahedronIndex = 0;
        hit.hitDistance = 0.0;
        hit.didHit = 0;
        hit.padding[0] = 0;
        return hit;
    }
}

// ZW line segment for rasterizer
struct ZWLine
{
    float2 zwPositions[2];
    float4 texturePositions[2];
    float4 normal;           // 4D surface normal (view space), flat per-tetrahedron face
    uint textureId;
    uint padding[3];

    static ZWLine empty()
    {
        ZWLine line;
        line.zwPositions[0] = float2(0);
        line.zwPositions[1] = float2(0);
        line.texturePositions[0] = float4(0);
        line.texturePositions[1] = float4(0);
        line.normal = float4(0);
        line.textureId = 0;
        line.padding[0] = 0;
        line.padding[1] = 0;
        line.padding[2] = 0;
        return line;
    }

    ZWLine getSegment(float start, float end)
    {
        ZWLine result;
        result.zwPositions[0] = zwPositions[0] * (1.0 - start) + zwPositions[1] * start;
        result.zwPositions[1] = zwPositions[0] * (1.0 - end) + zwPositions[1] * end;
        result.texturePositions[0] = texturePositions[0] * (1.0 - start) + texturePositions[1] * start;
        result.texturePositions[1] = texturePositions[0] * (1.0 - end) + texturePositions[1] * end;
        result.normal = normal;
        result.textureId = textureId;
        return result;
    }
}

// ============================================================================
// BVH types for GPU-based LBVH construction and traversal
// ============================================================================

// Morton code for spatial sorting in LBVH construction
struct MortonCode
{
    uint64_t code;
    uint tetrahedronIndex;
    uint padding;
}

// Scene bounding box for Morton code normalization
struct SceneBounds
{
    float4 minBounds;
    float4 maxBounds;
}

// BVH node for GPU-based tree traversal
// Supports both internal nodes and leaf nodes
struct BVHNode
{
    float4 minBounds;
    float4 maxBounds;
    uint leftChild;
    uint rightChild;
    uint parent;
    uint isLeaf;
    uint tetrahedronIndex;
    uint atomicVisitCount;
    uint padding[2];
}

// Constants for BVH traversal
static const uint BVH_INVALID_INDEX = 0xFFFFFFFF;
static const uint BVH_STACK_SIZE = 64;
