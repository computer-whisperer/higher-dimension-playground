// Voxel Traversal Engine (VTE) kernels.
// Stage A uses chunk-level 4D DDA with in-chunk voxel DDA traversal.
// This keeps first-hit correctness while skipping empty space aggressively.

import math;
import types;
import materials;

static const uint CHUNK_SIZE = 8;
static const uint CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096
static const uint INVALID_INDEX = 0xFFFFFFFF;
static const uint CHUNK_FLAG_FULL = 1u << 1u;
static const uint STAGE_B_MODE_INTEGRAL = 0u;
static const uint STAGE_B_MODE_SLICE = 1u;
static const uint STAGE_B_MODE_THICK_SLICE = 2u;
static const uint STAGE_B_MODE_DEBUG_COMPARE = 3u;
static const uint STAGE_B_MODE_DEBUG_INTEGRAL = 4u;
static const uint HIGHLIGHT_FLAG_HIT_VOXEL = 1u << 0u;
static const uint HIGHLIGHT_FLAG_PLACE_VOXEL = 1u << 1u;

static const uint MISS_REASON_NONE = 0u;
static const uint MISS_REASON_TOUCHED_VISIBLE_CHUNK = 1u;
static const uint MISS_REASON_VOXEL_BUDGET = 2u;
static const uint MISS_REASON_CHUNK_BUDGET = 3u;
static const uint MISS_REASON_MAX_DISTANCE = 4u;
static const uint MISS_REASON_LOOKUP_FALSE_NEGATIVE = 5u;

static const uint DEBUG_FLAG_LOOKUP_FALLBACK_USED = 1u;
static const uint META_DEBUG_FLAG_REFERENCE_COMPARE = 1u;
static const uint META_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY = 1u << 1u;
static const uint META_DEBUG_FLAG_COMPARE_SLICE_ONLY = 1u << 2u;
static const uint DEBUG_FLAG_CHUNK_TIE_STEPPED = 1u << 1u;
static const uint DEBUG_FLAG_CHUNK_ZERO_INTERVAL = 1u << 2u;

static const uint COMPARE_MISMATCH_KIND_NONE = 0u;
static const uint COMPARE_MISMATCH_KIND_HIT_STATE = 1u;
static const uint COMPARE_MISMATCH_KIND_CHUNK_MATERIAL = 2u;

static const uint COMPARE_STAT_COMPARED = 0u;
static const uint COMPARE_STAT_MATCHES = 1u;
static const uint COMPARE_STAT_MISMATCHES = 2u;
static const uint COMPARE_STAT_HIT_STATE_MISMATCHES = 3u;
static const uint COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES = 4u;
static const uint COMPARE_STAT_FAST_MISS_REF_HIT = 5u;
static const uint COMPARE_STAT_FAST_HIT_REF_MISS = 6u;
static const uint COMPARE_STAT_REASON_NONE = 7u;
static const uint COMPARE_STAT_REASON_TOUCHED_VISIBLE = 8u;
static const uint COMPARE_STAT_REASON_VOXEL_BUDGET = 9u;
static const uint COMPARE_STAT_REASON_CHUNK_BUDGET = 10u;
static const uint COMPARE_STAT_REASON_MAX_DISTANCE = 11u;
static const uint COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE = 12u;
static const uint COMPARE_STAT_ZERO_INTERVAL_FLAG = 13u;
static const uint COMPARE_STAT_TIE_STEPPED_FLAG = 14u;
static const uint COMPARE_STAT_LOOKUP_FALLBACK_FLAG = 15u;
static const uint COMPARE_STAT_COUNT = 16u;

static const uint FIRST_MISMATCH_VALID = 0u;
static const uint FIRST_MISMATCH_PIXEL_X = 1u;
static const uint FIRST_MISMATCH_PIXEL_Y = 2u;
static const uint FIRST_MISMATCH_LAYER = 3u;
static const uint FIRST_MISMATCH_KIND = 4u;
static const uint FIRST_MISMATCH_MISS_REASON = 5u;
static const uint FIRST_MISMATCH_DEBUG_FLAGS = 6u;
static const uint FIRST_MISMATCH_HIT_MASK = 7u;
static const uint FIRST_MISMATCH_FAST_CHUNK_X = 8u;
static const uint FIRST_MISMATCH_FAST_CHUNK_Y = 9u;
static const uint FIRST_MISMATCH_FAST_CHUNK_Z = 10u;
static const uint FIRST_MISMATCH_FAST_CHUNK_W = 11u;
static const uint FIRST_MISMATCH_REF_CHUNK_X = 12u;
static const uint FIRST_MISMATCH_REF_CHUNK_Y = 13u;
static const uint FIRST_MISMATCH_REF_CHUNK_Z = 14u;
static const uint FIRST_MISMATCH_REF_CHUNK_W = 15u;
static const uint FIRST_MISMATCH_FAST_MATERIAL = 16u;
static const uint FIRST_MISMATCH_REF_MATERIAL = 17u;
static const uint FIRST_MISMATCH_FAST_HIT_T = 18u;
static const uint FIRST_MISMATCH_REF_HIT_T = 19u;
static const uint FIRST_MISMATCH_CHUNK_STEPS = 20u;
static const uint FIRST_MISMATCH_REMAINING_VOXELS = 21u;
static const uint FIRST_MISMATCH_FINAL_T = 22u;
static const uint FIRST_MISMATCH_LAST_CHUNK_X = 23u;
static const uint FIRST_MISMATCH_LAST_CHUNK_Y = 24u;
static const uint FIRST_MISMATCH_LAST_CHUNK_Z = 25u;
static const uint FIRST_MISMATCH_LAST_CHUNK_W = 26u;
static const uint FIRST_MISMATCH_COUNT = 27u;

struct VoxelFrameMeta
{
    uint chunkCount;
    uint visibleChunkCount;
    uint occupancyWordCount;
    uint materialWordCount;
    uint maxTraceSteps;
    float maxTraceDistance;
    uint chunkLookupCapacity;
    uint stageBMode;
    uint stageBSliceLayer;
    uint stageBThickHalfWidth;
    uint debugFlags;
    int visibleChunkMinX;
    int visibleChunkMinY;
    int visibleChunkMinZ;
    int visibleChunkMinW;
    int visibleChunkMaxX;
    int visibleChunkMaxY;
    int visibleChunkMaxZ;
    int visibleChunkMaxW;
    uint highlightFlags;
    uint highlightPadding0;
    uint highlightPadding1;
    uint highlightPadding2;
    int4 highlightHitVoxel;
    int4 highlightPlaceVoxel;
}

struct VoxelChunkHeader
{
    int4 chunkCoord;
    uint occupancyWordOffset;
    uint materialWordOffset;
    uint flags;
    uint padding;
}

struct VoxelChunkLookupEntry
{
    int4 chunkCoord;
    uint chunkIndex;
    uint padding0;
    uint padding1;
    uint padding2;
}

// Set 1: SizedBuffers
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> pixelBuffer;

// Set 2: LiveBuffers
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;
[[vk::binding(2, 2)]] StructuredBuffer<VoxelFrameMeta> voxelFrameMetaBuffer;
[[vk::binding(3, 2)]] StructuredBuffer<VoxelChunkHeader> voxelChunkHeaders;
[[vk::binding(4, 2)]] StructuredBuffer<uint> voxelOccupancyWords;
[[vk::binding(5, 2)]] StructuredBuffer<uint> voxelMaterialWords;
[[vk::binding(6, 2)]] StructuredBuffer<uint> voxelVisibleChunkIndices;
[[vk::binding(7, 2)]] StructuredBuffer<VoxelChunkLookupEntry> voxelChunkLookup;
[[vk::binding(8, 2)]] RWStructuredBuffer<uint> vteCompareStats;
[[vk::binding(9, 2)]] RWStructuredBuffer<uint> vteFirstMismatchRecord;

void incrementCompareStat(uint statIndex)
{
    if (statIndex < COMPARE_STAT_COUNT)
    {
        InterlockedAdd(vteCompareStats[statIndex], 1u);
    }
}

void incrementCompareMissReason(uint missReason)
{
    if (missReason == MISS_REASON_NONE)
    {
        incrementCompareStat(COMPARE_STAT_REASON_NONE);
    }
    else if (missReason == MISS_REASON_TOUCHED_VISIBLE_CHUNK)
    {
        incrementCompareStat(COMPARE_STAT_REASON_TOUCHED_VISIBLE);
    }
    else if (missReason == MISS_REASON_VOXEL_BUDGET)
    {
        incrementCompareStat(COMPARE_STAT_REASON_VOXEL_BUDGET);
    }
    else if (missReason == MISS_REASON_CHUNK_BUDGET)
    {
        incrementCompareStat(COMPARE_STAT_REASON_CHUNK_BUDGET);
    }
    else if (missReason == MISS_REASON_MAX_DISTANCE)
    {
        incrementCompareStat(COMPARE_STAT_REASON_MAX_DISTANCE);
    }
    else if (missReason == MISS_REASON_LOOKUP_FALSE_NEGATIVE)
    {
        incrementCompareStat(COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE);
    }
}

void captureFirstMismatch(
    uint3 globalId,
    uint mismatchKind,
    uint missReason,
    uint debugFlags,
    bool hit,
    bool refHit,
    int4 hitChunkCoord,
    int4 refHitChunkCoord,
    uint hitMaterial,
    uint refHitMaterial,
    float4 worldOrigin,
    float4 worldDirection,
    float4 hitPos,
    float4 refHitPos,
    uint chunkStepsTaken,
    uint remainingVoxelSteps,
    float finalT,
    int4 lastChunkCoord)
{
    uint prior = 1u;
    InterlockedCompareExchange(
        vteFirstMismatchRecord[FIRST_MISMATCH_VALID],
        0u,
        1u,
        prior);
    if (prior != 0u)
    {
        return;
    }

    vteFirstMismatchRecord[FIRST_MISMATCH_PIXEL_X] = globalId.x;
    vteFirstMismatchRecord[FIRST_MISMATCH_PIXEL_Y] = globalId.y;
    vteFirstMismatchRecord[FIRST_MISMATCH_LAYER] = globalId.z;
    vteFirstMismatchRecord[FIRST_MISMATCH_KIND] = mismatchKind;
    vteFirstMismatchRecord[FIRST_MISMATCH_MISS_REASON] = missReason;
    vteFirstMismatchRecord[FIRST_MISMATCH_DEBUG_FLAGS] = debugFlags;
    vteFirstMismatchRecord[FIRST_MISMATCH_HIT_MASK] = (hit ? 1u : 0u) | (refHit ? 2u : 0u);
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_CHUNK_X] = asuint(hitChunkCoord.x);
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_CHUNK_Y] = asuint(hitChunkCoord.y);
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_CHUNK_Z] = asuint(hitChunkCoord.z);
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_CHUNK_W] = asuint(hitChunkCoord.w);
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_CHUNK_X] = asuint(refHitChunkCoord.x);
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_CHUNK_Y] = asuint(refHitChunkCoord.y);
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_CHUNK_Z] = asuint(refHitChunkCoord.z);
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_CHUNK_W] = asuint(refHitChunkCoord.w);
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_MATERIAL] = hitMaterial;
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_MATERIAL] = refHitMaterial;

    float fastHitT = hit ? dot(hitPos - worldOrigin, worldDirection) : -1.0;
    float refHitT = refHit ? dot(refHitPos - worldOrigin, worldDirection) : -1.0;
    vteFirstMismatchRecord[FIRST_MISMATCH_FAST_HIT_T] = asuint(fastHitT);
    vteFirstMismatchRecord[FIRST_MISMATCH_REF_HIT_T] = asuint(refHitT);
    vteFirstMismatchRecord[FIRST_MISMATCH_CHUNK_STEPS] = chunkStepsTaken;
    vteFirstMismatchRecord[FIRST_MISMATCH_REMAINING_VOXELS] = remainingVoxelSteps;
    vteFirstMismatchRecord[FIRST_MISMATCH_FINAL_T] = asuint(finalT);
    vteFirstMismatchRecord[FIRST_MISMATCH_LAST_CHUNK_X] = asuint(lastChunkCoord.x);
    vteFirstMismatchRecord[FIRST_MISMATCH_LAST_CHUNK_Y] = asuint(lastChunkCoord.y);
    vteFirstMismatchRecord[FIRST_MISMATCH_LAST_CHUNK_Z] = asuint(lastChunkCoord.z);
    vteFirstMismatchRecord[FIRST_MISMATCH_LAST_CHUNK_W] = asuint(lastChunkCoord.w);
}

int floorDiv(int a, int b)
{
    int q = a / b;
    int r = a % b;
    if (r < 0)
    {
        q -= 1;
    }
    return q;
}

uint computeLocalIndex(int4 local)
{
    return uint(local.w) * 512 + uint(local.z) * 64 + uint(local.y) * 8 + uint(local.x);
}

uint hashChunkCoord(int4 chunkCoord)
{
    uint x = asuint(chunkCoord.x);
    uint y = asuint(chunkCoord.y);
    uint z = asuint(chunkCoord.z);
    uint w = asuint(chunkCoord.w);
    return x * 0x8DA6B343u
         ^ y * 0xD8163841u
         ^ z * 0xCB1AB31Fu
         ^ w * 0x165667B1u;
}

float4 estimateVoxelHitNormal(float4 hitPos, float4 rayDir)
{
    // Nudge forward so boundary hits resolve to the entered voxel.
    float4 local = frac(hitPos + rayDir * 1e-4);
    float4 toLow = local;
    float4 toHigh = 1.0 - local;
    float4 edgeDist = min(toLow, toHigh);

    uint axis = argMin4(edgeDist);
    float4 normal = float4(0.0, 0.0, 0.0, 0.0);

    if (axis == 0u)
    {
        normal.x = (toLow.x <= toHigh.x) ? -1.0 : 1.0;
    }
    else if (axis == 1u)
    {
        normal.y = (toLow.y <= toHigh.y) ? -1.0 : 1.0;
    }
    else if (axis == 2u)
    {
        normal.z = (toLow.z <= toHigh.z) ? -1.0 : 1.0;
    }
    else
    {
        normal.w = (toLow.w <= toHigh.w) ? -1.0 : 1.0;
    }

    return normal;
}

float3 shadeVoxelMaterial(MaterialProperties mat, float4 hitNormal)
{
    float4 sunWorldDir = normalize(float4(0.3, 1.0, -0.3, 0.0));
    float nDotL = max(0.0, dot(hitNormal, sunWorldDir));
    float nDotLBack = max(0.0, dot(-hitNormal, sunWorldDir));
    float diffuse = max(nDotL, nDotLBack);

    float3 ambient = float3(0.08, 0.09, 0.12);
    float3 sunColor = float3(0.8, 0.76, 0.72);

    return mat.albedo.xyz * (ambient + sunColor * diffuse)
         + mat.albedo.xyz * mat.luminance;
}

int4 hitVoxelCoordFromHitPos(float4 hitPos, float4 rayDir)
{
    float4 p = hitPos + rayDir * 1e-4;
    return int4(
        int(floor(p.x)),
        int(floor(p.y)),
        int(floor(p.z)),
        int(floor(p.w)));
}

uint highlightKindForVoxel(int4 voxelCoord, VoxelFrameMeta meta)
{
    if ((meta.highlightFlags & HIGHLIGHT_FLAG_HIT_VOXEL) != 0u
        && all(voxelCoord == meta.highlightHitVoxel))
    {
        return HIGHLIGHT_FLAG_HIT_VOXEL;
    }
    if ((meta.highlightFlags & HIGHLIGHT_FLAG_PLACE_VOXEL) != 0u
        && all(voxelCoord == meta.highlightPlaceVoxel))
    {
        return HIGHLIGHT_FLAG_PLACE_VOXEL;
    }
    return 0u;
}

uint dominantAxisFromNormal(float4 n)
{
    float4 absN = abs(n);
    uint axis = 0u;
    float best = absN.x;
    if (absN.y > best)
    {
        axis = 1u;
        best = absN.y;
    }
    if (absN.z > best)
    {
        axis = 2u;
        best = absN.z;
    }
    if (absN.w > best)
    {
        axis = 3u;
    }
    return axis;
}

int normalSignOnAxis(float4 n, uint axis)
{
    if (axis == 0u) return n.x < 0.0 ? -1 : 1;
    if (axis == 1u) return n.y < 0.0 ? -1 : 1;
    if (axis == 2u) return n.z < 0.0 ? -1 : 1;
    return n.w < 0.0 ? -1 : 1;
}

bool decodePlacementFace(VoxelFrameMeta meta, out uint axis, out int sign)
{
    axis = 0u;
    sign = 0;
    uint required = HIGHLIGHT_FLAG_HIT_VOXEL | HIGHLIGHT_FLAG_PLACE_VOXEL;
    if ((meta.highlightFlags & required) != required)
    {
        return false;
    }

    int4 delta = meta.highlightPlaceVoxel - meta.highlightHitVoxel;
    uint nonZeroAxes = 0u;

    if (delta.x != 0)
    {
        if (abs(delta.x) != 1) return false;
        axis = 0u;
        sign = delta.x;
        nonZeroAxes++;
    }
    if (delta.y != 0)
    {
        if (abs(delta.y) != 1) return false;
        axis = 1u;
        sign = delta.y;
        nonZeroAxes++;
    }
    if (delta.z != 0)
    {
        if (abs(delta.z) != 1) return false;
        axis = 2u;
        sign = delta.z;
        nonZeroAxes++;
    }
    if (delta.w != 0)
    {
        if (abs(delta.w) != 1) return false;
        axis = 3u;
        sign = delta.w;
        nonZeroAxes++;
    }

    return nonZeroAxes == 1u;
}

float3 highlightBaseTint(uint highlightKind)
{
    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
    {
        return float3(0.08, 0.95, 1.0); // Cyan.
    }
    return float3(1.0, 0.55, 0.08); // Orange.
}

float3 shadeHighlightedVoxelSample(
    float3 litColor,
    int4 voxelCoord,
    VoxelFrameMeta meta,
    float4 hitPos,
    float4 rayDir,
    float4 hitNormal,
    uint highlightKind)
{
    // 4D up-convert of "black edge lines":
    // draw a contrasting band near 2D face boundaries of the 3D cell (the hit hyperface),
    // so we intentionally ignore the entered axis.
    float4 local = frac(hitPos + rayDir * 1e-4);
    float4 faceDist = min(local, 1.0 - local);
    uint guideAxis = dominantAxisFromNormal(hitNormal);
    int guideSign = normalSignOnAxis(hitNormal, guideAxis);

    // If this is the currently selected cell, prefer the explicit placement face
    // inferred from hit/place voxel relation so the guide marks "where placement goes".
    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL && all(voxelCoord == meta.highlightHitVoxel))
    {
        uint placementAxis = 0u;
        int placementSign = 0;
        if (decodePlacementFace(meta, placementAxis, placementSign))
        {
            guideAxis = placementAxis;
            guideSign = placementSign;
        }
    }

    float minCellFaceDist = 1e30;
    if (guideAxis != 0u) minCellFaceDist = min(minCellFaceDist, faceDist.x);
    if (guideAxis != 1u) minCellFaceDist = min(minCellFaceDist, faceDist.y);
    if (guideAxis != 2u) minCellFaceDist = min(minCellFaceDist, faceDist.z);
    if (guideAxis != 3u) minCellFaceDist = min(minCellFaceDist, faceDist.w);

    const float BAND_INNER = 0.07;
    const float BAND_OUTER = 0.16;
    float faceBand = 1.0 - smoothstep(BAND_INNER, BAND_OUTER, minCellFaceDist);

    float3 tint = highlightBaseTint(highlightKind);
    float3 contrasted = float3(0.03, 0.03, 0.03);
    float tintBlend = (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL) ? 0.64 : 0.50;
    float contrastBlend = (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL) ? 0.85 : 0.60;
    float3 shaded = lerp(lerp(litColor, tint, tintBlend), contrasted, faceBand * contrastBlend);

    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL && all(voxelCoord == meta.highlightHitVoxel))
    {
        float guideFaceDist = 0.0;
        if (guideAxis == 0u) guideFaceDist = (guideSign < 0) ? local.x : (1.0 - local.x);
        else if (guideAxis == 1u) guideFaceDist = (guideSign < 0) ? local.y : (1.0 - local.y);
        else if (guideAxis == 2u) guideFaceDist = (guideSign < 0) ? local.z : (1.0 - local.z);
        else guideFaceDist = (guideSign < 0) ? local.w : (1.0 - local.w);

        // Accent only the boundary of the currently looked-at 3D face.
        float faceMask = 1.0 - smoothstep(0.01, 0.06, guideFaceDist);
        float edgeMask = 1.0 - smoothstep(0.05, 0.13, minCellFaceDist);
        float accentMask = faceMask * edgeMask;
        float3 accentColor = float3(1.0, 0.20, 0.86);
        shaded = lerp(shaded, accentColor, accentMask);
    }

    return shaded;
}

uint findChunkIndex(int4 chunkCoord, VoxelFrameMeta meta)
{
    if (meta.chunkLookupCapacity == 0)
    {
        return INVALID_INDEX;
    }

    uint mask = meta.chunkLookupCapacity - 1;
    uint slot = hashChunkCoord(chunkCoord) & mask;

    for (uint probe = 0; probe < meta.chunkLookupCapacity; probe++)
    {
        VoxelChunkLookupEntry entry = voxelChunkLookup[slot];
        if (entry.chunkIndex == INVALID_INDEX)
        {
            return INVALID_INDEX;
        }
        if (entry.chunkIndex < meta.chunkCount && all(entry.chunkCoord == chunkCoord))
        {
            return entry.chunkIndex;
        }
        slot = (slot + 1) & mask;
    }
    return INVALID_INDEX;
}

uint findChunkIndexLinearVisible(int4 chunkCoord, VoxelFrameMeta meta)
{
    for (uint i = 0; i < meta.visibleChunkCount; i++)
    {
        uint chunkIndex = voxelVisibleChunkIndices[i];
        if (chunkIndex < meta.chunkCount && all(voxelChunkHeaders[chunkIndex].chunkCoord == chunkCoord))
        {
            return chunkIndex;
        }
    }
    return INVALID_INDEX;
}

float min4(float4 v)
{
    return min(min(v.x, v.y), min(v.z, v.w));
}

uint argMin4(float4 v)
{
    uint axis = 0u;
    float best = v.x;
    if (v.y < best)
    {
        best = v.y;
        axis = 1u;
    }
    if (v.z < best)
    {
        best = v.z;
        axis = 2u;
    }
    if (v.w < best)
    {
        axis = 3u;
    }
    return axis;
}

int stepSign(float d)
{
    if (d > 0.0)
    {
        return 1;
    }
    if (d < 0.0)
    {
        return -1;
    }
    return 0;
}

float safeInvAbs(float d)
{
    const float EPS = 1e-6;
    return abs(d) > EPS ? (1.0 / abs(d)) : 1e30;
}

int entryChunkCoord(float position, float direction)
{
    float chunkSizeF = float(CHUNK_SIZE);
    float q = position / chunkSizeF;
    float qFloor = floor(q);
    int chunk = int(qFloor);

    // Exact chunk-boundary ownership: negative travel owns the previous chunk.
    if (direction < 0.0)
    {
        float boundary = qFloor * chunkSizeF;
        float eps = 1e-7 * chunkSizeF;
        if (abs(position - boundary) <= eps)
        {
            chunk -= 1;
        }
    }

    return chunk;
}

bool voxelInChunk(int4 voxelCoord, int4 chunkMin, int4 chunkMax)
{
    return voxelCoord.x >= chunkMin.x && voxelCoord.x < chunkMax.x
        && voxelCoord.y >= chunkMin.y && voxelCoord.y < chunkMax.y
        && voxelCoord.z >= chunkMin.z && voxelCoord.z < chunkMax.z
        && voxelCoord.w >= chunkMin.w && voxelCoord.w < chunkMax.w;
}

bool intersectChunkAABB(
    float4 rayOrigin,
    float4 rayDir,
    int4 chunkCoord,
    out float tEnter,
    out float tExit)
{
    const float EPS = 1e-6;
    float4 bmin = float4(
        float(chunkCoord.x * int(CHUNK_SIZE)),
        float(chunkCoord.y * int(CHUNK_SIZE)),
        float(chunkCoord.z * int(CHUNK_SIZE)),
        float(chunkCoord.w * int(CHUNK_SIZE))
    );
    float4 bmax = bmin + float(CHUNK_SIZE);

    float tMin = 0.0;
    float tMax = 1e30;

    // X slab
    if (abs(rayDir.x) > EPS)
    {
        float tx0 = (bmin.x - rayOrigin.x) / rayDir.x;
        float tx1 = (bmax.x - rayOrigin.x) / rayDir.x;
        tMin = max(tMin, min(tx0, tx1));
        tMax = min(tMax, max(tx0, tx1));
    }
    else if (rayOrigin.x < bmin.x || rayOrigin.x > bmax.x)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    // Y slab
    if (abs(rayDir.y) > EPS)
    {
        float ty0 = (bmin.y - rayOrigin.y) / rayDir.y;
        float ty1 = (bmax.y - rayOrigin.y) / rayDir.y;
        tMin = max(tMin, min(ty0, ty1));
        tMax = min(tMax, max(ty0, ty1));
    }
    else if (rayOrigin.y < bmin.y || rayOrigin.y > bmax.y)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    // Z slab
    if (abs(rayDir.z) > EPS)
    {
        float tz0 = (bmin.z - rayOrigin.z) / rayDir.z;
        float tz1 = (bmax.z - rayOrigin.z) / rayDir.z;
        tMin = max(tMin, min(tz0, tz1));
        tMax = min(tMax, max(tz0, tz1));
    }
    else if (rayOrigin.z < bmin.z || rayOrigin.z > bmax.z)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    // W slab
    if (abs(rayDir.w) > EPS)
    {
        float tw0 = (bmin.w - rayOrigin.w) / rayDir.w;
        float tw1 = (bmax.w - rayOrigin.w) / rayDir.w;
        tMin = max(tMin, min(tw0, tw1));
        tMax = min(tMax, max(tw0, tw1));
    }
    else if (rayOrigin.w < bmin.w || rayOrigin.w > bmax.w)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    tEnter = tMin;
    tExit = tMax;
    return tExit >= tEnter;
}

bool intersectAABB(
    float4 rayOrigin,
    float4 rayDir,
    float4 bmin,
    float4 bmax,
    out float tEnter,
    out float tExit)
{
    const float EPS = 1e-6;
    float tMin = 0.0;
    float tMax = 1e30;

    if (abs(rayDir.x) > EPS)
    {
        float tx0 = (bmin.x - rayOrigin.x) / rayDir.x;
        float tx1 = (bmax.x - rayOrigin.x) / rayDir.x;
        tMin = max(tMin, min(tx0, tx1));
        tMax = min(tMax, max(tx0, tx1));
    }
    else if (rayOrigin.x < bmin.x || rayOrigin.x > bmax.x)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    if (abs(rayDir.y) > EPS)
    {
        float ty0 = (bmin.y - rayOrigin.y) / rayDir.y;
        float ty1 = (bmax.y - rayOrigin.y) / rayDir.y;
        tMin = max(tMin, min(ty0, ty1));
        tMax = min(tMax, max(ty0, ty1));
    }
    else if (rayOrigin.y < bmin.y || rayOrigin.y > bmax.y)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    if (abs(rayDir.z) > EPS)
    {
        float tz0 = (bmin.z - rayOrigin.z) / rayDir.z;
        float tz1 = (bmax.z - rayOrigin.z) / rayDir.z;
        tMin = max(tMin, min(tz0, tz1));
        tMax = min(tMax, max(tz0, tz1));
    }
    else if (rayOrigin.z < bmin.z || rayOrigin.z > bmax.z)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    if (abs(rayDir.w) > EPS)
    {
        float tw0 = (bmin.w - rayOrigin.w) / rayDir.w;
        float tw1 = (bmax.w - rayOrigin.w) / rayDir.w;
        tMin = max(tMin, min(tw0, tw1));
        tMax = min(tMax, max(tw0, tw1));
    }
    else if (rayOrigin.w < bmin.w || rayOrigin.w > bmax.w)
    {
        tEnter = 0.0; tExit = -1.0; return false;
    }

    tEnter = tMin;
    tExit = tMax;
    return tExit >= tEnter;
}

void advanceDDA4(inout int4 coord, int4 step, inout float4 tMax, float4 tDelta, out float nextT)
{
    // Deterministic single-axis DDA step.
    // Multi-axis tie stepping is mathematically neat, but with finite precision it can
    // promote near-ties into false ties and skip whole cells/chunks.
    uint axis = argMin4(tMax);
    if (axis == 0u)
    {
        nextT = tMax.x;
        tMax.x += tDelta.x;
        coord.x += step.x;
    }
    else if (axis == 1u)
    {
        nextT = tMax.y;
        tMax.y += tDelta.y;
        coord.y += step.y;
    }
    else if (axis == 2u)
    {
        nextT = tMax.z;
        tMax.z += tDelta.z;
        coord.z += step.z;
    }
    else
    {
        nextT = tMax.w;
        tMax.w += tDelta.w;
        coord.w += step.w;
    }
}

bool sampleVoxelInChunk(
    int4 voxelCoord,
    int4 chunkMin,
    VoxelChunkHeader h,
    VoxelFrameMeta meta,
    out uint materialId)
{
    materialId = 0;
    int4 local = voxelCoord - chunkMin;
    uint localIdx = computeLocalIndex(local);

    if ((h.flags & CHUNK_FLAG_FULL) == 0u)
    {
        uint occupancyIdx = h.occupancyWordOffset + (localIdx >> 5);
        if (occupancyIdx >= meta.occupancyWordCount)
        {
            return false;
        }
        uint occupancyWord = voxelOccupancyWords[occupancyIdx];
        if ((occupancyWord & (1u << (localIdx & 31u))) == 0)
        {
            return false;
        }
    }

    uint materialIdx = h.materialWordOffset + (localIdx >> 2);
    if (materialIdx >= meta.materialWordCount)
    {
        return false;
    }
    uint materialWord = voxelMaterialWords[materialIdx];
    materialId = (materialWord >> ((localIdx & 3u) * 8u)) & 0xFFu;
    return materialId != 0;
}

bool traceVoxelsInChunk(
    float4 rayOrigin,
    float4 rayDir,
    float tEnter,
    float tExit,
    int4 chunkCoord,
    VoxelChunkHeader h,
    VoxelFrameMeta meta,
    inout uint remainingVoxelSteps,
    out uint hitMaterial,
    out float4 hitPos,
    out int4 hitChunkCoord)
{
    const float EPS = 1e-6;
    hitMaterial = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    hitChunkCoord = int4(0, 0, 0, 0);
    float intervalEps = max(EPS, max(abs(tEnter), abs(tExit)) * 1e-7);
    if (remainingVoxelSteps == 0 || tExit <= tEnter + intervalEps)
    {
        return false;
    }

    float tStart = max(0.0, tEnter);
    float startBias = max(intervalEps, abs(tStart) * 1e-7);
    float tTraceStart = tStart + startBias;
    if (tTraceStart >= tExit)
    {
        return false;
    }
    float localTExit = tExit - tTraceStart;
    if (localTExit <= intervalEps)
    {
        // Ray only touches this chunk at a boundary point/edge/vertex.
        // Treat as no interior traversal to avoid boundary-order first-hit artifacts.
        return false;
    }

    int4 chunkMin = chunkCoord * int(CHUNK_SIZE);
    int4 chunkMax = chunkMin + int(CHUNK_SIZE);

    float4 startPos = rayOrigin + rayDir * tTraceStart;
    // tTraceStart is already biased inside the interval, so plain floor gives stable ownership.
    int4 voxelCoord = int4(
        int(floor(startPos.x)),
        int(floor(startPos.y)),
        int(floor(startPos.z)),
        int(floor(startPos.w))
    );

    if (!voxelInChunk(voxelCoord, chunkMin, chunkMax))
    {
        voxelCoord.x = clamp(voxelCoord.x, chunkMin.x, chunkMax.x - 1);
        voxelCoord.y = clamp(voxelCoord.y, chunkMin.y, chunkMax.y - 1);
        voxelCoord.z = clamp(voxelCoord.z, chunkMin.z, chunkMax.z - 1);
        voxelCoord.w = clamp(voxelCoord.w, chunkMin.w, chunkMax.w - 1);
    }

    int4 step = int4(
        stepSign(rayDir.x),
        stepSign(rayDir.y),
        stepSign(rayDir.z),
        stepSign(rayDir.w)
    );

    float4 tDelta = float4(
        safeInvAbs(rayDir.x),
        safeInvAbs(rayDir.y),
        safeInvAbs(rayDir.z),
        safeInvAbs(rayDir.w)
    );

    float4 tMaxLocal = float4(1e30, 1e30, 1e30, 1e30);
    if (step.x > 0) tMaxLocal.x = (float(voxelCoord.x + 1) - startPos.x) * tDelta.x;
    else if (step.x < 0) tMaxLocal.x = (startPos.x - float(voxelCoord.x)) * tDelta.x;
    if (step.y > 0) tMaxLocal.y = (float(voxelCoord.y + 1) - startPos.y) * tDelta.y;
    else if (step.y < 0) tMaxLocal.y = (startPos.y - float(voxelCoord.y)) * tDelta.y;
    if (step.z > 0) tMaxLocal.z = (float(voxelCoord.z + 1) - startPos.z) * tDelta.z;
    else if (step.z < 0) tMaxLocal.z = (startPos.z - float(voxelCoord.z)) * tDelta.z;
    if (step.w > 0) tMaxLocal.w = (float(voxelCoord.w + 1) - startPos.w) * tDelta.w;
    else if (step.w < 0) tMaxLocal.w = (startPos.w - float(voxelCoord.w)) * tDelta.w;

    float localT = 0.0;
    while (remainingVoxelSteps > 0)
    {
        if (localT > localTExit + intervalEps)
        {
            break;
        }

        float nextLocalT = min4(tMaxLocal);
        float segmentEnd = min(nextLocalT, localTExit);
        bool hasInterior = (segmentEnd - localT) > intervalEps;

        if (hasInterior)
        {
            remainingVoxelSteps--;
            uint materialId = 0;
            if (sampleVoxelInChunk(voxelCoord, chunkMin, h, meta, materialId))
            {
                hitMaterial = materialId;
                hitPos = rayOrigin + rayDir * (tTraceStart + localT);
                hitChunkCoord = chunkCoord;
                return true;
            }
        }

        nextLocalT = 0.0;
        advanceDDA4(voxelCoord, step, tMaxLocal, tDelta, nextLocalT);
        localT = max(localT, nextLocalT);

        if (!voxelInChunk(voxelCoord, chunkMin, chunkMax))
        {
            break;
        }
    }
    return false;
}

bool traceVoxelRayReferenceVisible(
    float4 rayOrigin,
    float4 rayDir,
    VoxelFrameMeta meta,
    out uint hitMaterial,
    out float4 hitPos,
    out int4 hitChunkCoord)
{
    const float EPS = 1e-6;
    hitMaterial = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    hitChunkCoord = int4(0, 0, 0, 0);

    if (meta.chunkCount == 0 || meta.visibleChunkCount == 0)
    {
        return false;
    }

    float MAX_DISTANCE = max(1.0, meta.maxTraceDistance);
    float bestHitT = 1e30;
    bool foundHit = false;
    uint baseVoxelBudget = max(1u, min(meta.maxTraceSteps * 8u, 32768u));

    for (uint i = 0; i < meta.visibleChunkCount; i++)
    {
        uint chunkIndex = voxelVisibleChunkIndices[i];
        if (chunkIndex >= meta.chunkCount)
        {
            continue;
        }

        VoxelChunkHeader h = voxelChunkHeaders[chunkIndex];
        float tEnter = 0.0;
        float tExit = 0.0;
        if (!intersectChunkAABB(rayOrigin, rayDir, h.chunkCoord, tEnter, tExit))
        {
            continue;
        }

        tEnter = max(0.0, tEnter);
        tExit = min(MAX_DISTANCE, tExit);
        if (tExit <= tEnter + EPS)
        {
            continue;
        }

        uint localBudget = baseVoxelBudget;
        uint chunkHitMaterial = 0;
        float4 chunkHitPos = float4(0.0, 0.0, 0.0, 0.0);
        int4 chunkHitChunkCoord = int4(0, 0, 0, 0);
        if (!traceVoxelsInChunk(
            rayOrigin,
            rayDir,
            tEnter,
            tExit,
            h.chunkCoord,
            h,
            meta,
            localBudget,
            chunkHitMaterial,
            chunkHitPos,
            chunkHitChunkCoord))
        {
            continue;
        }

        float hitT = dot(chunkHitPos - rayOrigin, rayDir);
        if (hitT < bestHitT)
        {
            bestHitT = hitT;
            hitMaterial = chunkHitMaterial;
            hitPos = chunkHitPos;
            hitChunkCoord = chunkHitChunkCoord;
            foundHit = true;
        }
    }

    return foundHit;
}

bool traceVoxelRayDDA(
    float4 rayOrigin,
    float4 rayDir,
    VoxelFrameMeta meta,
    out uint hitMaterial,
    out float4 hitPos,
    out int4 hitChunkCoord,
    out uint missReason,
    out uint debugFlags,
    out uint outChunkStepsTaken,
    out uint outRemainingVoxelSteps,
    out float outFinalT,
    out int4 outLastChunkCoord)
{
    hitMaterial = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    hitChunkCoord = int4(0, 0, 0, 0);
    missReason = MISS_REASON_NONE;
    debugFlags = 0u;
    outChunkStepsTaken = 0u;
    outRemainingVoxelSteps = 0u;
    outFinalT = 0.0;
    outLastChunkCoord = int4(0, 0, 0, 0);
    if (meta.chunkCount == 0 || meta.visibleChunkCount == 0)
    {
        return false;
    }

    float MAX_DISTANCE = max(1.0, meta.maxTraceDistance);
    uint maxChunkSteps = max(1u, min(meta.maxTraceSteps, 4096u));
    uint remainingVoxelSteps = max(1u, min(meta.maxTraceSteps * 8u, 32768u));
    bool collectDebugFlags = (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE
        || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL);
    bool useSlabChunkInterval = collectDebugFlags
        || (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_COMPARE) != 0u;
    bool forceFrequentReanchor = useSlabChunkInterval;

    int4 chunkStep = int4(
        stepSign(rayDir.x),
        stepSign(rayDir.y),
        stepSign(rayDir.z),
        stepSign(rayDir.w)
    );
    float4 tDeltaChunk = float4(
        safeInvAbs(rayDir.x) * float(CHUNK_SIZE),
        safeInvAbs(rayDir.y) * float(CHUNK_SIZE),
        safeInvAbs(rayDir.z) * float(CHUNK_SIZE),
        safeInvAbs(rayDir.w) * float(CHUNK_SIZE)
    );

    bool touchedVisibleChunk = false;
    uint chunkStepsTaken = 0u;
    float currentT = 0.0;
    float traversalMaxT = MAX_DISTANCE;
    int4 lastChunkCoord = int4(0, 0, 0, 0);
    int4 chunkCoord = int4(0, 0, 0, 0);
    float4 tMaxChunk = float4(1e30, 1e30, 1e30, 1e30);
    bool chunkStateValid = false;
    const float CHUNK_EPS = 1e-6;
    float chunkSizeF = float(CHUNK_SIZE);
    float4 visibleBMin = float4(
        float(meta.visibleChunkMinX) * chunkSizeF,
        float(meta.visibleChunkMinY) * chunkSizeF,
        float(meta.visibleChunkMinZ) * chunkSizeF,
        float(meta.visibleChunkMinW) * chunkSizeF
    );
    float4 visibleBMax = float4(
        float(meta.visibleChunkMaxX + 1) * chunkSizeF,
        float(meta.visibleChunkMaxY + 1) * chunkSizeF,
        float(meta.visibleChunkMaxZ + 1) * chunkSizeF,
        float(meta.visibleChunkMaxW + 1) * chunkSizeF
    );
    float visibleEnter = 0.0;
    float visibleExit = 0.0;
    if (!intersectAABB(rayOrigin, rayDir, visibleBMin, visibleBMax, visibleEnter, visibleExit))
    {
        return false;
    }
    currentT = max(0.0, visibleEnter);
    traversalMaxT = min(MAX_DISTANCE, visibleExit);
    if (traversalMaxT <= currentT + CHUNK_EPS)
    {
        return false;
    }
    uint chunkReanchorMask = forceFrequentReanchor ? 31u : 255u;
    for (uint chunkStepIdx = 0; chunkStepIdx < maxChunkSteps; chunkStepIdx++)
    {
        chunkStepsTaken++;
        if (currentT > traversalMaxT || remainingVoxelSteps == 0)
        {
            break;
        }

        if (!chunkStateValid || (chunkStepIdx & chunkReanchorMask) == 0u)
        {
            // Probe slightly forward so boundary points are owned by the chunk we are entering.
            float probeEps = max(CHUNK_EPS, abs(currentT) * 1e-7);
            float probeT = min(traversalMaxT, currentT + probeEps);
            float4 currentPos = rayOrigin + rayDir * probeT;
            chunkCoord = int4(
                entryChunkCoord(currentPos.x, rayDir.x),
                entryChunkCoord(currentPos.y, rayDir.y),
                entryChunkCoord(currentPos.z, rayDir.z),
                entryChunkCoord(currentPos.w, rayDir.w)
            );

            tMaxChunk = float4(1e30, 1e30, 1e30, 1e30);
            if (chunkStep.x > 0)
            {
                tMaxChunk.x = (float((chunkCoord.x + 1) * int(CHUNK_SIZE)) - rayOrigin.x) / rayDir.x;
            }
            else if (chunkStep.x < 0)
            {
                tMaxChunk.x = (float(chunkCoord.x * int(CHUNK_SIZE)) - rayOrigin.x) / rayDir.x;
            }
            if (chunkStep.y > 0)
            {
                tMaxChunk.y = (float((chunkCoord.y + 1) * int(CHUNK_SIZE)) - rayOrigin.y) / rayDir.y;
            }
            else if (chunkStep.y < 0)
            {
                tMaxChunk.y = (float(chunkCoord.y * int(CHUNK_SIZE)) - rayOrigin.y) / rayDir.y;
            }
            if (chunkStep.z > 0)
            {
                tMaxChunk.z = (float((chunkCoord.z + 1) * int(CHUNK_SIZE)) - rayOrigin.z) / rayDir.z;
            }
            else if (chunkStep.z < 0)
            {
                tMaxChunk.z = (float(chunkCoord.z * int(CHUNK_SIZE)) - rayOrigin.z) / rayDir.z;
            }
            if (chunkStep.w > 0)
            {
                tMaxChunk.w = (float((chunkCoord.w + 1) * int(CHUNK_SIZE)) - rayOrigin.w) / rayDir.w;
            }
            else if (chunkStep.w < 0)
            {
                tMaxChunk.w = (float(chunkCoord.w * int(CHUNK_SIZE)) - rayOrigin.w) / rayDir.w;
            }
            chunkStateValid = true;
        }
        lastChunkCoord = chunkCoord;

        float chunkExitT = min(traversalMaxT, min4(tMaxChunk));
        chunkExitT = max(chunkExitT, currentT);
        if (collectDebugFlags)
        {
            // Tight tie threshold: only classify true multi-plane boundary crossings.
            // Loose thresholds promote near-ties and pollute diagnostics/ownership logic.
            float tieEps = max(1e-8, abs(chunkExitT) * 1e-7);
            uint tieCount = 0u;
            if (abs(tMaxChunk.x - chunkExitT) <= tieEps) tieCount++;
            if (abs(tMaxChunk.y - chunkExitT) <= tieEps) tieCount++;
            if (abs(tMaxChunk.z - chunkExitT) <= tieEps) tieCount++;
            if (abs(tMaxChunk.w - chunkExitT) <= tieEps) tieCount++;
            if (tieCount > 1u)
            {
                debugFlags |= DEBUG_FLAG_CHUNK_TIE_STEPPED;
            }
        }

        bool inVisibleChunkBounds =
            chunkCoord.x >= meta.visibleChunkMinX && chunkCoord.x <= meta.visibleChunkMaxX
            && chunkCoord.y >= meta.visibleChunkMinY && chunkCoord.y <= meta.visibleChunkMaxY
            && chunkCoord.z >= meta.visibleChunkMinZ && chunkCoord.z <= meta.visibleChunkMaxZ
            && chunkCoord.w >= meta.visibleChunkMinW && chunkCoord.w <= meta.visibleChunkMaxW;
        if (!inVisibleChunkBounds)
        {
            bool divergingFromVisibleBounds =
                (chunkCoord.x < meta.visibleChunkMinX && chunkStep.x <= 0)
                || (chunkCoord.x > meta.visibleChunkMaxX && chunkStep.x >= 0)
                || (chunkCoord.y < meta.visibleChunkMinY && chunkStep.y <= 0)
                || (chunkCoord.y > meta.visibleChunkMaxY && chunkStep.y >= 0)
                || (chunkCoord.z < meta.visibleChunkMinZ && chunkStep.z <= 0)
                || (chunkCoord.z > meta.visibleChunkMaxZ && chunkStep.z >= 0)
                || (chunkCoord.w < meta.visibleChunkMinW && chunkStep.w <= 0)
                || (chunkCoord.w > meta.visibleChunkMaxW && chunkStep.w >= 0);
            if (divergingFromVisibleBounds)
            {
                break;
            }
        }

        uint chunkIndex = INVALID_INDEX;
        if (inVisibleChunkBounds)
        {
            chunkIndex = findChunkIndex(chunkCoord, meta);
        }
        if (chunkIndex == INVALID_INDEX && inVisibleChunkBounds
            && (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE
                || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL))
        {
            uint linearChunkIndex = findChunkIndexLinearVisible(chunkCoord, meta);
            if (linearChunkIndex != INVALID_INDEX)
            {
                // Hash lookup false negative: fall back so debug mode can keep tracing.
                debugFlags |= DEBUG_FLAG_LOOKUP_FALLBACK_USED;
                chunkIndex = linearChunkIndex;
            }
        }
        if (chunkIndex != INVALID_INDEX)
        {
            touchedVisibleChunk = true;
            VoxelChunkHeader h = voxelChunkHeaders[chunkIndex];
            uint chunkMaterial = 0;
            float4 chunkHitPos = float4(0.0, 0.0, 0.0, 0.0);
            int4 chunkHitChunkCoord = int4(0, 0, 0, 0);
            // Hot path: DDA already tracks the chunk interval [currentT, chunkExitT].
            // Slab recomputation is kept only for debug/compare paths where we want
            // extra diagnostics around boundary ownership.
            float tEnter = currentT;
            float tExit = chunkExitT;
            if (useSlabChunkInterval)
            {
                float slabEnter = 0.0;
                float slabExit = 0.0;
                if (intersectChunkAABB(rayOrigin, rayDir, chunkCoord, slabEnter, slabExit))
                {
                    tEnter = max(currentT, slabEnter);
                    tExit = min(chunkExitT, slabExit);
                }
                else
                {
                    tExit = tEnter;
                }
            }
            if (tExit > tEnter + CHUNK_EPS)
            {
                if (traceVoxelsInChunk(
                    rayOrigin,
                    rayDir,
                    tEnter,
                    tExit,
                    chunkCoord,
                    h,
                    meta,
                    remainingVoxelSteps,
                    chunkMaterial,
                    chunkHitPos,
                    chunkHitChunkCoord))
                {
                    hitMaterial = chunkMaterial;
                    hitPos = chunkHitPos;
                    hitChunkCoord = chunkHitChunkCoord;
                    missReason = MISS_REASON_NONE;
                    outChunkStepsTaken = chunkStepsTaken;
                    outRemainingVoxelSteps = remainingVoxelSteps;
                    outFinalT = currentT;
                    outLastChunkCoord = lastChunkCoord;
                    return true;
                }
            }
            else if (collectDebugFlags)
            {
                debugFlags |= DEBUG_FLAG_CHUNK_ZERO_INTERVAL;
            }
        }

        float nextChunkT = chunkExitT;
        advanceDDA4(chunkCoord, chunkStep, tMaxChunk, tDeltaChunk, nextChunkT);
        // Keep a tiny forward bias to avoid re-entering the same boundary chunk.
        float advanceEps = max(CHUNK_EPS, abs(nextChunkT) * 1e-7);
        currentT = max(currentT + CHUNK_EPS, nextChunkT + advanceEps);
    }

    if ((debugFlags & DEBUG_FLAG_LOOKUP_FALLBACK_USED) != 0u)
    {
        missReason = MISS_REASON_LOOKUP_FALSE_NEGATIVE;
    }
    else if (remainingVoxelSteps == 0)
    {
        missReason = MISS_REASON_VOXEL_BUDGET;
    }
    else if (currentT > MAX_DISTANCE)
    {
        missReason = MISS_REASON_MAX_DISTANCE;
    }
    else if (chunkStepsTaken >= maxChunkSteps)
    {
        missReason = MISS_REASON_CHUNK_BUDGET;
    }
    else if (touchedVisibleChunk)
    {
        missReason = MISS_REASON_TOUCHED_VISIBLE_CHUNK;
    }
    else
    {
        missReason = MISS_REASON_NONE;
    }
    outChunkStepsTaken = chunkStepsTaken;
    outRemainingVoxelSteps = remainingVoxelSteps;
    outFinalT = currentT;
    outLastChunkCoord = lastChunkCoord;
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainVoxelClear(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData wd = workingDataBuffer[0];
    if (globalId.x >= wd.renderDimensions.x || globalId.y >= wd.renderDimensions.y)
    {
        return;
    }

    if (globalId.x == 0u && globalId.y == 0u)
    {
        for (uint i = 0u; i < COMPARE_STAT_COUNT; i++)
        {
            vteCompareStats[i] = 0u;
        }
        for (uint i = 0u; i < FIRST_MISMATCH_COUNT; i++)
        {
            vteFirstMismatchRecord[i] = 0u;
        }
    }

    uint pixelBase = globalId.y * wd.renderDimensions.x + globalId.x;
    uint layerStride = wd.renderDimensions.x * wd.renderDimensions.y;
    for (uint z = 0; z < wd.renderDimensions.z; z++)
    {
        pixelBuffer[pixelBase + z * layerStride] = float4(0.0, 0.0, 0.0, 1.0);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainVoxelTraceStageA(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData wd = workingDataBuffer[0];
    VoxelFrameMeta meta = voxelFrameMetaBuffer[0];
    if (globalId.x >= wd.renderDimensions.x || globalId.y >= wd.renderDimensions.y || globalId.z >= wd.renderDimensions.z)
    {
        return;
    }

    uint index = globalId.z * wd.renderDimensions.x * wd.renderDimensions.y
               + globalId.y * wd.renderDimensions.x
               + globalId.x;
    uint pixelBase = globalId.y * wd.renderDimensions.x + globalId.x;
    uint layerCount = max(1u, wd.renderDimensions.z);
    uint centerLayer = min(meta.stageBSliceLayer, layerCount - 1u);
    bool compareSliceOnly = (meta.debugFlags & META_DEBUG_FLAG_COMPARE_SLICE_ONLY) != 0u
        && (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL);
    if (compareSliceOnly && globalId.z != centerLayer)
    {
        pixelBuffer[index] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Integral mode fast path: collapse across all layers in Stage A and write
    // directly to layer 0, so Stage B can be skipped for this mode.
    if (meta.stageBMode == STAGE_B_MODE_INTEGRAL && globalId.z == 0u && layerCount > 1u)
    {
        float2 pixelPos = float2(
            float(globalId.x) / float(wd.renderDimensions.x) * 2.0 - 1.0,
            float(globalId.y) / float(wd.renderDimensions.y) * 2.0 - 1.0
        );
        float aspectRatio = float(wd.presentDimensions.x) / float(wd.presentDimensions.y);
        float sx = pixelPos.x / wd.focalLengthXY;
        float sy = (-pixelPos.y / aspectRatio) / wd.focalLengthXY;

        const float PI = 3.14159265;
        float viewAngle = (PI / 2.0) / wd.focalLengthZW;
        float4 viewOrigin = float4(0.0, 0.0, 0.0, 0.0);

        VecN<5> origin5 = VecN<4>.fromFloat4(viewOrigin).extend(1.0);
        VecN<5> worldOrigin5 = wd.viewMatrixInverse * origin5;
        float4 worldOrigin = worldOrigin5.toFloat4();
        float4 worldDirX = (wd.viewMatrixInverse * VecN<4>.fromFloat4(float4(1.0, 0.0, 0.0, 0.0)).extend(0.0)).toFloat4();
        float4 worldDirY = (wd.viewMatrixInverse * VecN<4>.fromFloat4(float4(0.0, 1.0, 0.0, 0.0)).extend(0.0)).toFloat4();
        float4 worldDirZ = (wd.viewMatrixInverse * VecN<4>.fromFloat4(float4(0.0, 0.0, 1.0, 0.0)).extend(0.0)).toFloat4();
        float4 worldDirW = (wd.viewMatrixInverse * VecN<4>.fromFloat4(float4(0.0, 0.0, 0.0, 1.0)).extend(0.0)).toFloat4();
        float4 worldXY = worldDirX * sx + worldDirY * sy;

        float invLayerCount = 1.0 / float(layerCount);
        float firstZNorm = invLayerCount - 1.0;
        float angle = firstZNorm * (viewAngle * 0.5) + (PI * 0.25);
        float sinAngle = sin(angle);
        float cosAngle = cos(angle);
        float angleStep = viewAngle * invLayerCount;
        float sinStep = sin(angleStep);
        float cosStep = cos(angleStep);

        float3 accumColor = float3(0.0, 0.0, 0.0);
        float3 hitHighlightAccum = float3(0.0, 0.0, 0.0);
        float3 placeHighlightAccum = float3(0.0, 0.0, 0.0);
        float hitHighlightCount = 0.0;
        float placeHighlightCount = 0.0;
        uint hitMaterial = 0u;
        float4 hitPos = float4(0.0, 0.0, 0.0, 0.0);
        int4 hitChunkCoord = int4(0, 0, 0, 0);
        uint missReason = MISS_REASON_NONE;
        uint debugFlags = 0u;
        uint chunkStepsTaken = 0u;
        uint remainingVoxelSteps = 0u;
        float finalT = 0.0;
        int4 lastChunkCoord = int4(0, 0, 0, 0);
        for (uint sampleLayer = 0u; sampleLayer < layerCount; sampleLayer++)
        {
            float4 worldDirection = normalize(worldXY + worldDirZ * cosAngle + worldDirW * sinAngle);
            bool hit = traceVoxelRayDDA(
                worldOrigin,
                worldDirection,
                meta,
                hitMaterial,
                hitPos,
                hitChunkCoord,
                missReason,
                debugFlags,
                chunkStepsTaken,
                remainingVoxelSteps,
                finalT,
                lastChunkCoord);

            if (hit)
            {
                int4 hitVoxelCoord = hitVoxelCoordFromHitPos(hitPos, worldDirection);
                uint highlightKind = highlightKindForVoxel(hitVoxelCoord, meta);
                if (highlightKind != 0u)
                {
                    MaterialProperties mat = sampleMaterial(hitMaterial, frac(hitPos.xyz));
                    float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
                    float3 litColor = shadeVoxelMaterial(mat, hitNormal);
                    float3 highlighted = shadeHighlightedVoxelSample(
                        litColor,
                        hitVoxelCoord,
                        meta,
                        hitPos,
                        worldDirection,
                        hitNormal,
                        highlightKind);
                    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
                    {
                        hitHighlightAccum += highlighted;
                        hitHighlightCount += 1.0;
                    }
                    else
                    {
                        placeHighlightAccum += highlighted;
                        placeHighlightCount += 1.0;
                    }
                    accumColor += highlighted;
                }
                else
                {
                    MaterialProperties mat = sampleMaterial(hitMaterial, frac(hitPos.xyz));
                    float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
                    accumColor += shadeVoxelMaterial(mat, hitNormal);
                }
            }
            else
            {
                float a = 0.5 + 0.5 * worldDirection.y;
                accumColor += lerp(float3(0.02, 0.03, 0.06), float3(0.16, 0.22, 0.40), saturate(a));
            }

            float nextCos = cosAngle * cosStep - sinAngle * sinStep;
            float nextSin = sinAngle * cosStep + cosAngle * sinStep;
            cosAngle = nextCos;
            sinAngle = nextSin;
        }

        float3 collapsedColor = accumColor / float(layerCount);
        if (hitHighlightCount > 0.0)
        {
            // Keep selected-cell guides solid in integral mode when present.
            collapsedColor = hitHighlightAccum / hitHighlightCount;
        }
        else if (placeHighlightCount > 0.0)
        {
            collapsedColor = placeHighlightAccum / placeHighlightCount;
        }
        pixelBuffer[pixelBase] = float4(collapsedColor, 1.0);
        return;
    }

    float2 pixelPos = float2(
        float(globalId.x) / float(wd.renderDimensions.x) * 2.0 - 1.0,
        float(globalId.y) / float(wd.renderDimensions.y) * 2.0 - 1.0
    );
    float aspectRatio = float(wd.presentDimensions.x) / float(wd.presentDimensions.y);

    const float PI = 3.14159265;
    float viewAngle = (PI / 2.0) / wd.focalLengthZW;
    float zNorm = ((float(globalId.z) + 0.5) / float(max(1u, wd.renderDimensions.z))) * 2.0 - 1.0;
    float zwAngle = zNorm * (viewAngle * 0.5) + (PI * 0.25);

    float4 viewOrigin = float4(0.0, 0.0, 0.0, 0.0);
    float4 viewDirection = normalize(float4(
        pixelPos.x / wd.focalLengthXY,
        (-pixelPos.y / aspectRatio) / wd.focalLengthXY,
        cos(zwAngle),
        sin(zwAngle)
    ));

    VecN<5> origin5 = VecN<4>.fromFloat4(viewOrigin).extend(1.0);
    VecN<5> dir5 = VecN<4>.fromFloat4(viewOrigin + viewDirection).extend(1.0);
    VecN<5> worldOrigin5 = wd.viewMatrixInverse * origin5;
    VecN<5> worldDir5 = wd.viewMatrixInverse * dir5;

    float4 worldOrigin = worldOrigin5.toFloat4();
    float4 worldDirection = normalize(worldDir5.toFloat4() - worldOrigin);

    uint hitMaterial = 0;
    float4 hitPos = float4(0.0, 0.0, 0.0, 0.0);
    int4 hitChunkCoord = int4(0, 0, 0, 0);
    uint missReason = MISS_REASON_NONE;
    uint debugFlags = 0u;
    uint chunkStepsTaken = 0u;
    uint remainingVoxelSteps = 0u;
    float finalT = 0.0;
    int4 lastChunkCoord = int4(0, 0, 0, 0);
    bool hit = traceVoxelRayDDA(
        worldOrigin,
        worldDirection,
        meta,
        hitMaterial,
        hitPos,
        hitChunkCoord,
        missReason,
        debugFlags,
        chunkStepsTaken,
        remainingVoxelSteps,
        finalT,
        lastChunkCoord);

    float3 outColor;
    float outAlpha = 1.0;
    if (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL)
    {
        bool doReferenceCompare = (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_COMPARE) != 0u;
        bool mismatchOnly = (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY) != 0u;
        if (doReferenceCompare)
        {
            uint refHitMaterial = 0;
            float4 refHitPos = float4(0.0, 0.0, 0.0, 0.0);
            int4 refHitChunkCoord = int4(0, 0, 0, 0);
            bool refHit = traceVoxelRayReferenceVisible(
                worldOrigin,
                worldDirection,
                meta,
                refHitMaterial,
                refHitPos,
                refHitChunkCoord);

            bool hitStateMismatch = (hit != refHit);
            bool chunkMaterialMismatch = (!hitStateMismatch)
                && hit
                && refHit
                && (!all(hitChunkCoord == refHitChunkCoord) || hitMaterial != refHitMaterial);
            bool anyMismatch = hitStateMismatch || chunkMaterialMismatch;
            uint mismatchKind = COMPARE_MISMATCH_KIND_NONE;
            if (hitStateMismatch)
            {
                mismatchKind = COMPARE_MISMATCH_KIND_HIT_STATE;
            }
            else if (chunkMaterialMismatch)
            {
                mismatchKind = COMPARE_MISMATCH_KIND_CHUNK_MATERIAL;
            }

            incrementCompareStat(COMPARE_STAT_COMPARED);
            incrementCompareMissReason(missReason);
            if ((debugFlags & DEBUG_FLAG_CHUNK_ZERO_INTERVAL) != 0u)
            {
                incrementCompareStat(COMPARE_STAT_ZERO_INTERVAL_FLAG);
            }
            if ((debugFlags & DEBUG_FLAG_CHUNK_TIE_STEPPED) != 0u)
            {
                incrementCompareStat(COMPARE_STAT_TIE_STEPPED_FLAG);
            }
            if ((debugFlags & DEBUG_FLAG_LOOKUP_FALLBACK_USED) != 0u)
            {
                incrementCompareStat(COMPARE_STAT_LOOKUP_FALLBACK_FLAG);
            }

            if (anyMismatch)
            {
                incrementCompareStat(COMPARE_STAT_MISMATCHES);
                if (hitStateMismatch)
                {
                    incrementCompareStat(COMPARE_STAT_HIT_STATE_MISMATCHES);
                    if (hit)
                    {
                        incrementCompareStat(COMPARE_STAT_FAST_HIT_REF_MISS);
                    }
                    else
                    {
                        incrementCompareStat(COMPARE_STAT_FAST_MISS_REF_HIT);
                    }
                }
                if (chunkMaterialMismatch)
                {
                    incrementCompareStat(COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES);
                }
                captureFirstMismatch(
                    globalId,
                    mismatchKind,
                    missReason,
                    debugFlags,
                    hit,
                    refHit,
                    hitChunkCoord,
                    refHitChunkCoord,
                    hitMaterial,
                    refHitMaterial,
                    worldOrigin,
                    worldDirection,
                    hitPos,
                    refHitPos,
                    chunkStepsTaken,
                    remainingVoxelSteps,
                    finalT,
                    lastChunkCoord);
            }
            else
            {
                incrementCompareStat(COMPARE_STAT_MATCHES);
            }

            if (hitStateMismatch)
            {
                // Red: fast DDA reported hit, reference says miss.
                // Blue: fast DDA missed, reference says hit.
                if (hit)
                {
                    outColor = float3(1.0, 0.0, 0.0);
                }
                else if ((debugFlags & DEBUG_FLAG_CHUNK_ZERO_INTERVAL) != 0u)
                {
                    // Teal: DDA lost interior due zero-interval chunk ownership.
                    outColor = float3(0.0, 0.85, 1.0);
                }
                else if ((debugFlags & DEBUG_FLAG_CHUNK_TIE_STEPPED) != 0u)
                {
                    // Violet: DDA encountered tie crossings before miss.
                    outColor = float3(0.62, 0.25, 1.0);
                }
                else if (missReason == MISS_REASON_TOUCHED_VISIBLE_CHUNK)
                {
                    // Deep blue: stepped visible chunks but found no hit.
                    outColor = float3(0.05, 0.05, 1.0);
                }
                else if (missReason == MISS_REASON_LOOKUP_FALSE_NEGATIVE)
                {
                    // Magenta-blue: hash lookup divergence.
                    outColor = float3(0.65, 0.1, 1.0);
                }
                else if (missReason == MISS_REASON_VOXEL_BUDGET)
                {
                    // Cyan-blue: voxel budget exhausted (should not happen at high budgets).
                    outColor = float3(0.0, 0.55, 1.0);
                }
                else if (missReason == MISS_REASON_CHUNK_BUDGET)
                {
                    // Indigo: chunk budget exhausted.
                    outColor = float3(0.2, 0.0, 1.0);
                }
                else if (missReason == MISS_REASON_MAX_DISTANCE)
                {
                    // Sky-blue: max-distance clipped.
                    outColor = float3(0.25, 0.75, 1.0);
                }
                else
                {
                    // Pure blue: generic DDA miss vs reference hit.
                    outColor = float3(0.1, 0.1, 1.0);
                }
                outAlpha = 1.0;
            }
            else if (chunkMaterialMismatch)
            {
                // Orange: both hit, but chose different chunk/material.
                outColor = float3(1.0, 0.5, 0.0);
                outAlpha = 1.0;
            }
            else if (mismatchOnly)
            {
                // Mismatch-only view: neutral colors for agreement, vivid colors for disagreement.
                outColor = hit ? float3(0.12, 0.12, 0.12) : float3(0.05, 0.08, 0.12);
                outAlpha = 1.0;
            }
            else if (hit)
            {
                // Match path (hit): show normal chunk-color debug output.
                uint h = hashChunkCoord(hitChunkCoord);
                float3 chunkColor = float3(
                    float((h >> 0u) & 0xFFu),
                    float((h >> 8u) & 0xFFu),
                    float((h >> 16u) & 0xFFu)) / 255.0;

                float chunkSizeF = float(CHUNK_SIZE);
                float4 chunkLocal = hitPos - floor(hitPos / chunkSizeF) * chunkSizeF;
                float4 edgeDist = min(chunkLocal, chunkSizeF - chunkLocal);
                float seamDist = min(min(edgeDist.x, edgeDist.z), edgeDist.w);
                outColor = seamDist < 0.04
                    ? lerp(chunkColor, float3(1.0, 1.0, 1.0), 0.65)
                    : chunkColor;
                outAlpha = 1.0;
            }
            else
            {
                // Match path (miss): keep miss-reason palette.
                if (missReason == MISS_REASON_LOOKUP_FALSE_NEGATIVE)
                {
                    outColor = float3(1.0, 0.0, 1.0);
                }
                else if (missReason == MISS_REASON_TOUCHED_VISIBLE_CHUNK)
                {
                    outColor = float3(1.0, 0.0, 0.2);
                }
                else if (missReason == MISS_REASON_VOXEL_BUDGET)
                {
                    outColor = float3(1.0, 0.55, 0.0);
                }
                else if (missReason == MISS_REASON_CHUNK_BUDGET)
                {
                    outColor = float3(1.0, 0.95, 0.1);
                }
                else if (missReason == MISS_REASON_MAX_DISTANCE)
                {
                    outColor = float3(0.15, 0.65, 1.0);
                }
                else
                {
                    outColor = float3(0.03, 0.03, 0.05);
                }
                outAlpha = (meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL) ? 0.0 : 1.0;
            }
        }
        else if (hit)
        {
            if ((debugFlags & DEBUG_FLAG_LOOKUP_FALLBACK_USED) != 0u)
            {
                outColor = float3(1.0, 0.0, 1.0);
            }
            else
            {
                uint h = hashChunkCoord(hitChunkCoord);
                float3 chunkColor = float3(
                    float((h >> 0u) & 0xFFu),
                    float((h >> 8u) & 0xFFu),
                    float((h >> 16u) & 0xFFu)) / 255.0;

                float chunkSizeF = float(CHUNK_SIZE);
                float4 chunkLocal = hitPos - floor(hitPos / chunkSizeF) * chunkSizeF;
                float4 edgeDist = min(chunkLocal, chunkSizeF - chunkLocal);
                // Ignore Y seams for diagnostics here: the flat world's first hit often lies on
                // the y=0 boundary of chunk y=-1, which would otherwise paint everything as seam.
                float seamDist = min(min(edgeDist.x, edgeDist.z), edgeDist.w);
                if (seamDist < 0.04)
                {
                    outColor = lerp(chunkColor, float3(1.0, 1.0, 1.0), 0.65);
                }
                else
                {
                    outColor = chunkColor;
                }
            }
            outAlpha = 1.0;
        }
        else
        {
            if (missReason == MISS_REASON_LOOKUP_FALSE_NEGATIVE)
            {
                outColor = float3(1.0, 0.0, 1.0);
            }
            else if (missReason == MISS_REASON_TOUCHED_VISIBLE_CHUNK)
            {
                outColor = float3(1.0, 0.0, 0.2);
            }
            else if (missReason == MISS_REASON_VOXEL_BUDGET)
            {
                outColor = float3(1.0, 0.55, 0.0);
            }
            else if (missReason == MISS_REASON_CHUNK_BUDGET)
            {
                outColor = float3(1.0, 0.95, 0.1);
            }
            else if (missReason == MISS_REASON_MAX_DISTANCE)
            {
                outColor = float3(0.15, 0.65, 1.0);
            }
            else
            {
                outColor = float3(0.03, 0.03, 0.05);
            }
            outAlpha = (meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL) ? 0.0 : 1.0;
        }
    }
    else if (hit)
    {
        int4 hitVoxelCoord = hitVoxelCoordFromHitPos(hitPos, worldDirection);
        uint highlightKind = highlightKindForVoxel(hitVoxelCoord, meta);
        if (highlightKind != 0u)
        {
            MaterialProperties mat = sampleMaterial(hitMaterial, frac(hitPos.xyz));
            float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
            float3 litColor = shadeVoxelMaterial(mat, hitNormal);
            outColor = shadeHighlightedVoxelSample(
                litColor,
                hitVoxelCoord,
                meta,
                hitPos,
                worldDirection,
                hitNormal,
                highlightKind);
        }
        else
        {
            MaterialProperties mat = sampleMaterial(hitMaterial, frac(hitPos.xyz));
            float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
            outColor = shadeVoxelMaterial(mat, hitNormal);
        }
    }
    else
    {
        float a = 0.5 + 0.5 * worldDirection.y;
        outColor = lerp(float3(0.02, 0.03, 0.06), float3(0.16, 0.22, 0.40), saturate(a));
    }

    pixelBuffer[index] = float4(outColor, outAlpha);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainVoxelDisplayStageB(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData wd = workingDataBuffer[0];
    VoxelFrameMeta meta = voxelFrameMetaBuffer[0];
    if (globalId.x >= wd.renderDimensions.x || globalId.y >= wd.renderDimensions.y)
    {
        return;
    }

    uint width = wd.renderDimensions.x;
    uint height = wd.renderDimensions.y;
    uint layers = max(1u, wd.renderDimensions.z);
    uint layerStride = width * height;
    uint pixelBase = globalId.y * width + globalId.x;
    uint centerLayer = min(meta.stageBSliceLayer, layers - 1);

    float3 color = float3(0.0, 0.0, 0.0);
    if (meta.stageBMode == STAGE_B_MODE_SLICE || meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE)
    {
        color = pixelBuffer[pixelBase + centerLayer * layerStride].xyz;
    }
    else if (meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL)
    {
        float3 hitAccum = float3(0.0, 0.0, 0.0);
        float3 missAccum = float3(0.0, 0.0, 0.0);
        float hitCount = 0.0;
        float missCount = 0.0;
        for (uint z = 0; z < layers; z++)
        {
            float4 sample = pixelBuffer[pixelBase + z * layerStride];
            float hitMask = sample.w > 0.5 ? 1.0 : 0.0;
            hitAccum += sample.xyz * hitMask;
            missAccum += sample.xyz * (1.0 - hitMask);
            hitCount += hitMask;
            missCount += (1.0 - hitMask);
        }

        float3 hitColor = hitCount > 0.0 ? hitAccum / hitCount : float3(0.0, 0.0, 0.0);
        float3 missColor = missCount > 0.0 ? missAccum / missCount : float3(0.0, 0.0, 0.0);
        float hitFrac = hitCount / float(max(1u, layers));

        color = lerp(missColor, hitColor, hitFrac);
        color = lerp(color, float3(hitFrac, hitFrac, hitFrac), 0.2);
    }
    else
    {
        uint lo = 0;
        uint hi = layers - 1;
        if (meta.stageBMode == STAGE_B_MODE_THICK_SLICE)
        {
            uint half = min(meta.stageBThickHalfWidth, layers - 1);
            lo = centerLayer > half ? centerLayer - half : 0;
            hi = min(layers - 1, centerLayer + half);
        }

        float3 accum = float3(0.0, 0.0, 0.0);
        uint sampleCount = 0;
        for (uint z = lo; z <= hi; z++)
        {
            accum += pixelBuffer[pixelBase + z * layerStride].xyz;
            sampleCount++;
        }

        if (sampleCount > 0)
        {
            color = accum / float(sampleCount);
        }
    }

    pixelBuffer[pixelBase] = float4(color, 1.0);
}
