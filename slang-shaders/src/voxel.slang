// Voxel Traversal Engine (VTE) kernels.
// Stage A uses chunk-level 4D DDA with in-chunk voxel DDA traversal.
// This keeps first-hit correctness while skipping empty space aggressively.

import math;
import types;
import materials;

#ifndef VTE_DIAGNOSTICS_ENABLED
#define VTE_DIAGNOSTICS_ENABLED 1
#endif

static const uint CHUNK_SIZE = 8;
static const uint CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096
static const uint INVALID_INDEX = 0xFFFFFFFF;
static const float LOD_TINT_BLEND = 0.35;
static const uint CHUNK_FLAG_FULL = 1u << 0u;
static const uint STAGE_B_MODE_INTEGRAL = 0u;
static const uint STAGE_B_MODE_SLICE = 1u;
static const uint STAGE_B_MODE_THICK_SLICE = 2u;
static const uint HIGHLIGHT_FLAG_HIT_VOXEL = 1u << 0u;
static const uint HIGHLIGHT_FLAG_PLACE_VOXEL = 1u << 1u;
static const uint MATERIAL_ID_GROUND = 11u;
static const float INTEGRAL_HIGHLIGHT_WEIGHT_BOOST = 15.0;
// Keep shader-side traversal policy aligned with CPU-side BVH build policy:
// any non-zero entity tetra set uses BVH traversal.
static const uint ENTITY_LINEAR_THRESHOLD_TETS = 0u;

static const uint MISS_REASON_NONE = 0u;
static const uint MISS_REASON_TOUCHED_VISIBLE_CHUNK = 1u;
static const uint MISS_REASON_VOXEL_BUDGET = 2u;
static const uint MISS_REASON_CHUNK_BUDGET = 3u;
static const uint MISS_REASON_MAX_DISTANCE = 4u;
static const uint MISS_REASON_LOOKUP_FALSE_NEGATIVE = 5u;
static const uint LEAF_KIND_UNIFORM = 0u;
static const uint LEAF_KIND_VOXEL_CHUNK_ARRAY = 1u;
static const uint LEAF_CHUNK_ENTRY_EMPTY = 0u;
static const uint LEAF_CHUNK_ENTRY_UNIFORM_FLAG = 1u << 31u;

static const uint DEBUG_FLAG_LOOKUP_FALLBACK_USED = 1u;
static const uint META_DEBUG_FLAG_REFERENCE_COMPARE = 1u;
static const uint META_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY = 1u << 1u;
static const uint META_DEBUG_FLAG_COMPARE_SLICE_ONLY = 1u << 2u;
static const uint META_DEBUG_FLAG_LOD_TINT = 1u << 3u;
static const uint META_DEBUG_FLAG_ENTITY_LINEAR_ONLY = 1u << 4u;
static const uint META_DEBUG_FLAG_ENTITY_BVH_COMPARE = 1u << 5u;
static const uint META_DEBUG_FLAG_WORLD_BVH_RAY_DIAG = 1u << 6u;
static const uint META_DEBUG_FLAG_STAGE_A_BREAKDOWN = 1u << 7u;
static const float ENTITY_BVH_TOL_REL = 1e-3;
static const float ENTITY_BVH_TOL_ABS = 1e-3;
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
static const uint COMPARE_STAT_ENTITY_BVH_SAMPLE = 16u;
static const uint COMPARE_STAT_ENTITY_BVH_MISMATCH = 17u;
static const uint COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH = 18u;
static const uint COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH = 19u;
static const uint COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH = 20u;
static const uint COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT = 21u;
static const uint COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS = 22u;
static const uint COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH = 23u;
static const uint COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH = 24u;
static const uint COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH = 25u;
static const uint COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH = 26u;
static const uint COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH = 27u;
static const uint COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH = 28u;
static const uint COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH = 29u;
static const uint COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH = 30u;
static const uint COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH = 31u;
static const uint COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH = 32u;
static const uint COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH = 33u;
static const uint COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH = 34u;
static const uint COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH = 35u;
static const uint COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH = 36u;
static const uint COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH = 37u;
static const uint COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH = 38u;
static const uint COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH = 39u;
static const uint COMPARE_STAT_STAGEA_SAMPLES = 40u;
static const uint COMPARE_STAT_STAGEA_ENTITY_QUERIES = 41u;
static const uint COMPARE_STAT_STAGEA_ENTITY_HITS = 42u;
static const uint COMPARE_STAT_STAGEA_VOXEL_HITS = 43u;
static const uint COMPARE_STAT_STAGEA_SKY_MISSES = 44u;
static const uint COMPARE_STAT_STAGEA_CHUNK_STEPS_SUM = 45u;
static const uint COMPARE_STAT_STAGEA_VOXEL_STEPS_SUM = 46u;
static const uint COMPARE_STAT_STAGEA_NODE_VISITS_SUM = 47u;
static const uint COMPARE_STAT_COUNT = 48u;

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
static const uint WORLD_BVH_RAY_DIAG_CAPACITY = 128u;
static const uint WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD = 16u;
static const uint WORLD_BVH_RAY_DIAG_COUNT_WORD = 0u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_BASE_WORD = 1u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_PIXEL_X = 0u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_PIXEL_Y = 1u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_LAYER = 2u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_MASK = 3u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_MISS_REASON = 4u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_MATERIAL = 5u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_X = 6u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Y = 7u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Z = 8u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_W = 9u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_HIT_T_BITS = 10u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_CHUNK_STEPS = 11u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_REMAINING_VOXELS = 12u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_NODE_VISITS = 13u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_PATH_HASH = 14u;
static const uint WORLD_BVH_RAY_DIAG_RECORD_FLAGS = 15u;
static const uint WORLD_BVH_RAY_DIAG_WORD_COUNT =
    WORLD_BVH_RAY_DIAG_RECORD_BASE_WORD + WORLD_BVH_RAY_DIAG_CAPACITY * WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD;
static const bool VTE_DIAGNOSTICS = (VTE_DIAGNOSTICS_ENABLED != 0);

struct VoxelFrameMeta
{
    uint chunkCount;
    uint leafCount;
    uint occupancyWordCount;
    uint materialWordCount;
    uint macroWordCount;
    uint maxTraceSteps;
    float maxTraceDistance;
    uint regionBvhNodeCount;
    uint regionBvhRootIndex;
    uint leafChunkEntryCount;
    uint stageBMode;
    uint stageBSliceLayer;
    uint stageBThickHalfWidth;
    uint debugFlags;
    uint worldBvhDiagSeed;
    uint worldBvhDiagSampleCount;
    uint orientationWordCount;
    uint worldBvhDiagPadding1;
    float4 visibleWorldMin;
    float4 visibleWorldMax;
    uint highlightFlags;
    uint highlightPadding0;
    uint highlightPadding1;
    uint highlightPadding2;
    int4 highlightHitVoxel;
    int4 highlightPlaceVoxel;
    uint highlightHitScale;
    uint highlightPlaceScale;
    uint highlightScalePadding0;
    uint highlightScalePadding1;
}

struct VoxelChunkHeader
{
    uint occupancyWordOffset;
    uint materialWordOffset;
    uint flags;
    uint macroWordOffset;
    int4 solidLocalMin;
    int4 solidLocalMax;
    uint orientationWordOffset;
    uint padding1;
}

struct VoxelChunkBvhNode
{
    float4 worldMin;
    float4 worldMax;
    uint leftChild;
    uint rightChild;
    uint leafIndex;
    uint flags;
}

struct VoxelLeafHeader
{
    int4 minChunkCoord;
    int4 maxChunkCoord;
    uint leafKind;
    uint uniformMaterial;
    uint chunkEntryOffset;
    uint uniformOrientation;  // bits 0-15: orientation, bits 24-31: scale_exp as i8

    int getScaleExp() {
        int raw = int(uniformOrientation >> 24) & 0xFF;
        return raw >= 128 ? raw - 256 : raw;
    }
}

// Set 0: OneTimeBuffers
[[vk::binding(0, 0)]] StructuredBuffer<ModelTetrahedron> modelTetrahedrons;

// Set 1: SizedBuffers
[[vk::binding(1, 1)]] StructuredBuffer<Tetrahedron> outputTetrahedrons;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> pixelBuffer;
[[vk::binding(4, 1)]] StructuredBuffer<BVHNode> bvhNodes;

// Set 2: LiveBuffers
[[vk::binding(0, 2)]] StructuredBuffer<ModelInstance> instances;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;
[[vk::binding(2, 2)]] StructuredBuffer<VoxelFrameMeta> voxelFrameMetaBuffer;
[[vk::binding(3, 2)]] StructuredBuffer<VoxelChunkHeader> voxelChunkHeaders;
[[vk::binding(4, 2)]] StructuredBuffer<uint> voxelOccupancyWords;
[[vk::binding(5, 2)]] StructuredBuffer<uint> voxelMaterialWords;
[[vk::binding(6, 2)]] StructuredBuffer<VoxelLeafHeader> voxelLeafHeaders;
[[vk::binding(7, 2)]] StructuredBuffer<VoxelChunkBvhNode> voxelChunkBvhNodes;
[[vk::binding(8, 2)]] StructuredBuffer<uint> voxelLeafChunkEntries;
[[vk::binding(9, 2)]] RWStructuredBuffer<uint> vteCompareStats;
[[vk::binding(10, 2)]] RWStructuredBuffer<uint> vteFirstMismatchRecord;
[[vk::binding(11, 2)]] StructuredBuffer<uint> voxelMacroWords;
[[vk::binding(12, 2)]] RWStructuredBuffer<uint> vteWorldBvhRayDiagBuffer;
[[vk::binding(13, 2)]] StructuredBuffer<uint> voxelOrientationWords;

void incrementCompareStat(uint statIndex)
{
    if (!VTE_DIAGNOSTICS)
    {
        return;
    }
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
    if (!VTE_DIAGNOSTICS)
    {
        return;
    }
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

bool selectWorldBvhRayDiagSample(
    uint3 globalId,
    WorkingData wd,
    VoxelFrameMeta meta,
    out uint outSlot)
{
    outSlot = INVALID_INDEX;
    if (!VTE_DIAGNOSTICS)
    {
        return false;
    }
    if ((meta.debugFlags & META_DEBUG_FLAG_WORLD_BVH_RAY_DIAG) == 0u)
    {
        return false;
    }

    uint requested = min(meta.worldBvhDiagSampleCount, WORLD_BVH_RAY_DIAG_CAPACITY);
    if (requested == 0u)
    {
        return false;
    }

    uint dispatchLayers = max(1u, wd.renderDimensions.z);
    if (meta.stageBMode == STAGE_B_MODE_INTEGRAL && dispatchLayers > 1u)
    {
        dispatchLayers = 1u;
    }
    uint plane = wd.renderDimensions.x * wd.renderDimensions.y;
    uint totalDispatchThreads = plane * dispatchLayers;
    if (totalDispatchThreads == 0u)
    {
        return false;
    }

    uint stride = max(1u, totalDispatchThreads / requested);
    uint offset = meta.worldBvhDiagSeed % stride;
    uint linearDispatchIndex = globalId.z * plane + globalId.y * wd.renderDimensions.x + globalId.x;
    if ((linearDispatchIndex % stride) != offset)
    {
        return false;
    }

    outSlot = linearDispatchIndex / stride;
    return outSlot < WORLD_BVH_RAY_DIAG_CAPACITY;
}

void writeWorldBvhRayDiagRecord(
    uint slot,
    uint pixelX,
    uint pixelY,
    uint layer,
    bool hit,
    uint missReason,
    uint hitMaterial,
    int4 hitChunkCoord,
    float hitT,
    uint chunkStepsTaken,
    uint remainingVoxelSteps,
    uint nodeVisits,
    uint pathHash,
    uint flags)
{
    if (!VTE_DIAGNOSTICS)
    {
        return;
    }
    if (slot >= WORLD_BVH_RAY_DIAG_CAPACITY)
    {
        return;
    }

    uint baseWord = WORLD_BVH_RAY_DIAG_RECORD_BASE_WORD + slot * WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD;
    if ((baseWord + WORLD_BVH_RAY_DIAG_WORDS_PER_RECORD) > WORLD_BVH_RAY_DIAG_WORD_COUNT)
    {
        return;
    }

    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_PIXEL_X] = pixelX;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_PIXEL_Y] = pixelY;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_LAYER] = layer;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_MASK] = hit ? 1u : 0u;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_MISS_REASON] = missReason;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_MATERIAL] = hitMaterial;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_X] = asuint(hitChunkCoord.x);
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Y] = asuint(hitChunkCoord.y);
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_Z] = asuint(hitChunkCoord.z);
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_CHUNK_W] = asuint(hitChunkCoord.w);
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_HIT_T_BITS] = asuint(hitT);
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_CHUNK_STEPS] = chunkStepsTaken;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_REMAINING_VOXELS] = remainingVoxelSteps;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_NODE_VISITS] = nodeVisits;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_PATH_HASH] = pathHash;
    vteWorldBvhRayDiagBuffer[baseWord + WORLD_BVH_RAY_DIAG_RECORD_FLAGS] = flags;

    uint priorCount = 0u;
    InterlockedMax(vteWorldBvhRayDiagBuffer[WORLD_BVH_RAY_DIAG_COUNT_WORD], slot + 1u, priorCount);
}

bool stageABreakdownEnabled(VoxelFrameMeta meta)
{
    return VTE_DIAGNOSTICS && ((meta.debugFlags & META_DEBUG_FLAG_STAGE_A_BREAKDOWN) != 0u);
}

bool selectStageABreakdownSample(uint3 globalId, VoxelFrameMeta meta)
{
    if (!stageABreakdownEnabled(meta))
    {
        return false;
    }
    uint h =
        globalId.x * 73856093u
        ^ globalId.y * 19349663u
        ^ globalId.z * 83492791u
        ^ meta.worldBvhDiagSeed;
    // Sample 1/64 threads to keep diagnostic overhead bounded.
    return (h & 63u) == 0u;
}

void recordStageABreakdown(
    VoxelFrameMeta meta,
    bool sampled,
    bool entityQuery,
    bool entityHit,
    bool voxelHit,
    uint chunkSteps,
    uint remainingVoxelSteps,
    uint nodeVisits)
{
    if (!sampled)
    {
        return;
    }

    incrementCompareStat(COMPARE_STAT_STAGEA_SAMPLES);
    if (entityQuery)
    {
        incrementCompareStat(COMPARE_STAT_STAGEA_ENTITY_QUERIES);
    }
    if (entityHit)
    {
        incrementCompareStat(COMPARE_STAT_STAGEA_ENTITY_HITS);
    }
    else if (voxelHit)
    {
        incrementCompareStat(COMPARE_STAT_STAGEA_VOXEL_HITS);
    }
    else
    {
        incrementCompareStat(COMPARE_STAT_STAGEA_SKY_MISSES);
    }

    InterlockedAdd(vteCompareStats[COMPARE_STAT_STAGEA_CHUNK_STEPS_SUM], chunkSteps);
    uint maxVoxelSteps = max(1u, min(meta.maxTraceSteps * 8u, 32768u));
    uint voxelStepsUsed = maxVoxelSteps > remainingVoxelSteps
        ? (maxVoxelSteps - remainingVoxelSteps)
        : 0u;
    InterlockedAdd(vteCompareStats[COMPARE_STAT_STAGEA_VOXEL_STEPS_SUM], voxelStepsUsed);
    InterlockedAdd(vteCompareStats[COMPARE_STAT_STAGEA_NODE_VISITS_SUM], nodeVisits);
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

uint computeMacroIndex(int4 local)
{
    uint mx = uint(local.x >> 1);
    uint my = uint(local.y >> 1);
    uint mz = uint(local.z >> 1);
    uint mw = uint(local.w >> 1);
    return (mw * 64u) + (mz * 16u) + (my * 4u) + mx;
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

float3 applyLodTint(float3 baseColor, uint lodLevel)
{
    float3 tint = lodLevel == 0u
        ? float3(1.0, 0.66, 0.18)
        : (lodLevel == 1u ? float3(0.18, 0.72, 1.0) : float3(0.64, 0.42, 1.0));
    return lerp(baseColor, tint, LOD_TINT_BLEND);
}

float3 maybeApplyLodTint(float3 baseColor, uint lodLevel, VoxelFrameMeta meta)
{
    if ((meta.debugFlags & META_DEBUG_FLAG_LOD_TINT) == 0u)
    {
        return baseColor;
    }
    return applyLodTint(baseColor, lodLevel);
}

struct EntityTetraHit
{
    bool didHit;
    float hitDistance;
    float4 hitPosition;
    float4 hitNormal;
    float4 textureCoordinates;
    uint materialId;
    uint tetrahedronIndex;
}

EntityTetraHit entityTetraHitNone()
{
    EntityTetraHit hit;
    hit.didHit = false;
    hit.hitDistance = 0.0;
    hit.hitPosition = float4(0.0, 0.0, 0.0, 0.0);
    hit.hitNormal = float4(0.0, 0.0, 0.0, 0.0);
    hit.textureCoordinates = float4(0.0, 0.0, 0.0, 0.0);
    hit.materialId = 0u;
    hit.tetrahedronIndex = INVALID_INDEX;
    return hit;
}

bool rayAABBIntersectAxisEntity(
    float origin,
    float direction,
    float invDirection,
    float minB,
    float maxB,
    inout float tmin,
    inout float tmax)
{
    // Treat near-zero direction components as parallel to the slab and perform
    // an explicit containment test. Finite reciprocal approximations can
    // otherwise over/under-estimate t and reject valid nearer nodes.
    if (abs(direction) <= 1e-8)
    {
        float slabTolerance = max(
            ENTITY_BVH_TOL_ABS,
            ENTITY_BVH_TOL_REL * max(abs(minB), abs(maxB)));
        return origin >= (minB - slabTolerance) && origin <= (maxB + slabTolerance);
    }

    float t1 = (minB - origin) * invDirection;
    float t2 = (maxB - origin) * invDirection;
    float axisMin = min(t1, t2);
    float axisMax = max(t1, t2);
    tmin = max(tmin, axisMin);
    tmax = min(tmax, axisMax);
    return tmin <= tmax;
}

float2 rayAABBIntersect4DEntity(
    float4 origin,
    float4 direction,
    float4 invDir,
    float4 minB,
    float4 maxB)
{
    float tmin = 0.0;
    float tmax = 1e30;

    if (!rayAABBIntersectAxisEntity(
            origin.x,
            direction.x,
            invDir.x,
            minB.x,
            maxB.x,
            tmin,
            tmax)
        || !rayAABBIntersectAxisEntity(
            origin.y,
            direction.y,
            invDir.y,
            minB.y,
            maxB.y,
            tmin,
            tmax)
        || !rayAABBIntersectAxisEntity(
            origin.z,
            direction.z,
            invDir.z,
            minB.z,
            maxB.z,
            tmin,
            tmax)
        || !rayAABBIntersectAxisEntity(
            origin.w,
            direction.w,
            invDir.w,
            minB.w,
            maxB.w,
            tmin,
            tmax))
    {
        return float2(-1.0, -1.0);
    }

    if (tmax < 0.0 || tmin > tmax)
    {
        return float2(-1.0, -1.0);
    }
    return float2(max(tmin, 0.0), tmax);
}

struct EntityAffineTransform
{
    float4 col0;
    float4 col1;
    float4 col2;
    float4 col3;
    float4 translation;
};

EntityAffineTransform entityAffineFromInstance(ModelInstance instance)
{
    EntityAffineTransform affine;
    affine.col0 = float4(
        instance.modelTransform.data[0][0],
        instance.modelTransform.data[1][0],
        instance.modelTransform.data[2][0],
        instance.modelTransform.data[3][0]);
    affine.col1 = float4(
        instance.modelTransform.data[0][1],
        instance.modelTransform.data[1][1],
        instance.modelTransform.data[2][1],
        instance.modelTransform.data[3][1]);
    affine.col2 = float4(
        instance.modelTransform.data[0][2],
        instance.modelTransform.data[1][2],
        instance.modelTransform.data[2][2],
        instance.modelTransform.data[3][2]);
    affine.col3 = float4(
        instance.modelTransform.data[0][3],
        instance.modelTransform.data[1][3],
        instance.modelTransform.data[2][3],
        instance.modelTransform.data[3][3]);
    affine.translation = float4(
        instance.modelTransform.data[0][4],
        instance.modelTransform.data[1][4],
        instance.modelTransform.data[2][4],
        instance.modelTransform.data[3][4]);
    return affine;
}

float4 transformEntityModelPointToWorld(EntityAffineTransform affine, float4 modelPoint)
{
    return affine.translation
        + affine.col0 * modelPoint.x
        + affine.col1 * modelPoint.y
        + affine.col2 * modelPoint.z
        + affine.col3 * modelPoint.w;
}

bool testEntityModelTetrahedronHit(
    float4 worldStart,
    float4 worldDirection,
    ModelInstance instance,
    EntityAffineTransform affine,
    uint instanceIndex,
    uint modelTetIndex,
    uint modelTetCount,
    float maxDistance,
    inout EntityTetraHit closestHit)
{
    ModelTetrahedron tet = modelTetrahedrons[modelTetIndex];
    uint packedTetraIndex = instanceIndex * modelTetCount + modelTetIndex;
    uint cellIndex = min(tet.cellId, 7u);
    uint materialId = instance.cellMaterialIds[cellIndex];
    if (materialId == 0u)
    {
        return false;
    }

    float4 worldV0 = transformEntityModelPointToWorld(affine, tet.vertexPositions[0]);
    float4 worldV1 = transformEntityModelPointToWorld(affine, tet.vertexPositions[1]);
    float4 worldV2 = transformEntityModelPointToWorld(affine, tet.vertexPositions[2]);
    float4 worldV3 = transformEntityModelPointToWorld(affine, tet.vertexPositions[3]);

    float4 worldNormal = getNormal4D(worldV0, worldV1, worldV2, worldV3);
    float denom = dot(worldNormal, worldDirection);
    if (abs(denom) < 0.001)
    {
        return false;
    }

    float4 b = worldStart - worldV0;
    float hitDistance = -dot(b, worldNormal) / denom;
    if (hitDistance <= 0.0 || hitDistance > maxDistance)
    {
        return false;
    }
    if (closestHit.didHit)
    {
        float distTolerance = max(1e-4, 1e-4 * max(closestHit.hitDistance, hitDistance));
        if (hitDistance > closestHit.hitDistance + distTolerance)
        {
            return false;
        }
        if (abs(hitDistance - closestHit.hitDistance) <= distTolerance
            && packedTetraIndex >= closestHit.tetrahedronIndex)
        {
            return false;
        }
    }

    float4 q = b + hitDistance * worldDirection;

    float4 e1 = worldV1 - worldV0;
    float4 e2 = worldV2 - worldV0;
    float4 e3 = worldV3 - worldV0;

    float4 absN = abs(worldNormal);
    float3 e1s;
    float3 e2s;
    float3 e3s;
    float3 qs;

    if (absN.x >= absN.y && absN.x >= absN.z && absN.x >= absN.w)
    {
        e1s = e1.yzw;
        e2s = e2.yzw;
        e3s = e3.yzw;
        qs = q.yzw;
    }
    else if (absN.y >= absN.z && absN.y >= absN.w)
    {
        e1s = float3(e1.x, e1.z, e1.w);
        e2s = float3(e2.x, e2.z, e2.w);
        e3s = float3(e3.x, e3.z, e3.w);
        qs = float3(q.x, q.z, q.w);
    }
    else if (absN.z >= absN.w)
    {
        e1s = float3(e1.x, e1.y, e1.w);
        e2s = float3(e2.x, e2.y, e2.w);
        e3s = float3(e3.x, e3.y, e3.w);
        qs = float3(q.x, q.y, q.w);
    }
    else
    {
        e1s = e1.xyz;
        e2s = e2.xyz;
        e3s = e3.xyz;
        qs = q.xyz;
    }

    float3 c23 = cross(e2s, e3s);
    float det3 = dot(e1s, c23);
    if (abs(det3) < 1e-10)
    {
        return false;
    }

    float invDet = 1.0 / det3;
    float u = dot(qs, c23) * invDet;
    float v = dot(e1s, cross(qs, e3s)) * invDet;
    float w = dot(e1s, cross(e2s, qs)) * invDet;
    float bary0 = 1.0 - u - v - w;

    if (bary0 >= 0.0 && u >= 0.0 && v >= 0.0 && w >= 0.0)
    {
        closestHit.didHit = true;
        closestHit.hitDistance = hitDistance;
        closestHit.hitPosition = worldStart + hitDistance * worldDirection;
        closestHit.hitNormal = denom < 0.0 ? worldNormal : -worldNormal;
        closestHit.textureCoordinates =
            bary0 * tet.texturePositions[0]
            + u * tet.texturePositions[1]
            + v * tet.texturePositions[2]
            + w * tet.texturePositions[3];
        closestHit.materialId = materialId;
        closestHit.tetrahedronIndex = packedTetraIndex;
        return true;
    }

    return false;
}

bool testEntityLeafHit(
    float4 worldStart,
    float4 worldDirection,
    uint leafIndex,
    uint modelTetCount,
    float maxDistance,
    inout EntityTetraHit closestHit)
{
    if (modelTetCount == 0u)
    {
        return false;
    }

    ModelInstance instance = instances[leafIndex];
    EntityAffineTransform affine = entityAffineFromInstance(instance);
    bool anyHit = false;
    for (uint modelTetIndex = 0u; modelTetIndex < modelTetCount; modelTetIndex++)
    {
        anyHit = testEntityModelTetrahedronHit(
                worldStart,
                worldDirection,
                instance,
                affine,
                leafIndex,
                modelTetIndex,
                modelTetCount,
                maxDistance,
                closestHit)
            || anyHit;
    }
    return anyHit;
}

EntityTetraHit traceEntityTetraLinear(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance)
{
    EntityTetraHit closestHit = entityTetraHitNone();
    uint modelTetCount, modelTetStride;
    modelTetrahedrons.GetDimensions(modelTetCount, modelTetStride);
    for (uint i = 0u; i < numTetrahedrons; i++)
    {
        testEntityLeafHit(start, direction, i, modelTetCount, maxDistance, closestHit);
    }
    return closestHit;
}

EntityTetraHit traceEntityTetraLinearReverse(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance)
{
    EntityTetraHit closestHit = entityTetraHitNone();
    uint modelTetCount, modelTetStride;
    modelTetrahedrons.GetDimensions(modelTetCount, modelTetStride);
    for (uint i = numTetrahedrons; i > 0u; i--)
    {
        testEntityLeafHit(start, direction, i - 1u, modelTetCount, maxDistance, closestHit);
    }
    return closestHit;
}

EntityTetraHit traceEntityTetraBVH(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance,
    bool enableBoundPrune)
{
    EntityTetraHit closestHit = entityTetraHitNone();
    if (numTetrahedrons == 0u)
    {
        return closestHit;
    }

    uint modelTetCount, modelTetStride;
    modelTetrahedrons.GetDimensions(modelTetCount, modelTetStride);
    if (modelTetCount == 0u)
    {
        return closestHit;
    }

    uint totalNodes = 2u * numTetrahedrons - 1u;

    float4 invDir = float4(
        abs(direction.x) > 1e-8 ? 1.0 / direction.x : (direction.x >= 0.0 ? 1e8 : -1e8),
        abs(direction.y) > 1e-8 ? 1.0 / direction.y : (direction.y >= 0.0 ? 1e8 : -1e8),
        abs(direction.z) > 1e-8 ? 1.0 / direction.z : (direction.z >= 0.0 ? 1e8 : -1e8),
        abs(direction.w) > 1e-8 ? 1.0 / direction.w : (direction.w >= 0.0 ? 1e8 : -1e8)
    );

    float2 rootT = rayAABBIntersect4DEntity(
        start,
        direction,
        invDir,
        bvhNodes[0].minBounds,
        bvhNodes[0].maxBounds);
    float rootTolerance = max(
        ENTITY_BVH_TOL_ABS,
        ENTITY_BVH_TOL_REL * maxDistance);
    float rootMaxDistance = maxDistance + rootTolerance;
    if (rootT.x < 0.0 || rootT.x > rootMaxDistance)
    {
        return closestHit;
    }

    uint stack[BVH_STACK_SIZE];
    int stackPtr = 0;
    bool stackOverflowed = false;
    stack[stackPtr++] = 0u;

    while (stackPtr > 0)
    {
        uint nodeIdx = stack[--stackPtr];
        if (nodeIdx >= totalNodes)
        {
            stackOverflowed = true;
            break;
        }
        BVHNode node = bvhNodes[nodeIdx];

        if (node.isLeaf != 0u)
        {
            if (node.tetrahedronIndex < numTetrahedrons)
            {
                testEntityLeafHit(
                    start,
                    direction,
                    node.tetrahedronIndex,
                    modelTetCount,
                    maxDistance,
                    closestHit);
            }
        }
        else
        {
            uint left = node.leftChild;
            uint right = node.rightChild;
            if (left >= totalNodes || right >= totalNodes)
            {
                stackOverflowed = true;
                break;
            }
            float bestBound = maxDistance;
            if (enableBoundPrune && closestHit.didHit)
            {
                bestBound = min(maxDistance, closestHit.hitDistance);
            }

            float2 leftT = rayAABBIntersect4DEntity(
                start,
                direction,
                invDir,
                bvhNodes[left].minBounds,
                bvhNodes[left].maxBounds);
            float2 rightT = rayAABBIntersect4DEntity(
                start,
                direction,
                invDir,
                bvhNodes[right].minBounds,
                bvhNodes[right].maxBounds);

            float boundTolerance = max(
                ENTITY_BVH_TOL_ABS,
                ENTITY_BVH_TOL_REL * bestBound);
            bool leftHit = leftT.x >= 0.0 && leftT.x <= (bestBound + boundTolerance);
            bool rightHit = rightT.x >= 0.0 && rightT.x <= (bestBound + boundTolerance);

            if (leftHit && rightHit)
            {
                if (stackPtr + 2 > int(BVH_STACK_SIZE))
                {
                    stackOverflowed = true;
                    break;
                }
                if (leftT.x > rightT.x)
                {
                    stack[stackPtr++] = left;
                    stack[stackPtr++] = right;
                }
                else
                {
                    stack[stackPtr++] = right;
                    stack[stackPtr++] = left;
                }
            }
            else if (leftHit)
            {
                if (stackPtr + 1 > int(BVH_STACK_SIZE))
                {
                    stackOverflowed = true;
                    break;
                }
                stack[stackPtr++] = left;
            }
            else if (rightHit)
            {
                if (stackPtr + 1 > int(BVH_STACK_SIZE))
                {
                    stackOverflowed = true;
                    break;
                }
                stack[stackPtr++] = right;
            }
        }
    }

    // Preserve correctness under deep/degenerate BVHs by falling back to linear traversal
    // instead of writing past the fixed traversal stack.
    if (stackOverflowed)
    {
        for (uint i = 0u; i < numTetrahedrons; i++)
        {
            testEntityLeafHit(start, direction, i, modelTetCount, maxDistance, closestHit);
        }
    }

    return closestHit;
}

EntityTetraHit traceEntityTetra(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance)
{
    if (numTetrahedrons == 0u || maxDistance <= 0.0)
    {
        return entityTetraHitNone();
    }

    if (numTetrahedrons <= ENTITY_LINEAR_THRESHOLD_TETS)
    {
        return traceEntityTetraLinear(start, direction, numTetrahedrons, maxDistance);
    }

    return traceEntityTetraBVH(start, direction, numTetrahedrons, maxDistance, true);
}

EntityTetraHit traceEntityTetraBVHNoAabb(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance)
{
    EntityTetraHit closestHit = entityTetraHitNone();
    if (numTetrahedrons == 0u)
    {
        return closestHit;
    }

    uint modelTetCount, modelTetStride;
    modelTetrahedrons.GetDimensions(modelTetCount, modelTetStride);
    if (modelTetCount == 0u)
    {
        return closestHit;
    }

    uint totalNodes = 2u * numTetrahedrons - 1u;
    uint stack[BVH_STACK_SIZE];
    int stackPtr = 0;
    bool stackOverflowed = false;
    stack[stackPtr++] = 0u;

    while (stackPtr > 0)
    {
        uint nodeIdx = stack[--stackPtr];
        if (nodeIdx >= totalNodes)
        {
            stackOverflowed = true;
            break;
        }
        BVHNode node = bvhNodes[nodeIdx];
        if (node.isLeaf != 0u)
        {
            if (node.tetrahedronIndex < numTetrahedrons)
            {
                testEntityLeafHit(
                    start,
                    direction,
                    node.tetrahedronIndex,
                    modelTetCount,
                    maxDistance,
                    closestHit);
            }
        }
        else
        {
            uint left = node.leftChild;
            uint right = node.rightChild;
            if (left >= totalNodes || right >= totalNodes)
            {
                stackOverflowed = true;
                break;
            }
            if (stackPtr + 2 > int(BVH_STACK_SIZE))
            {
                stackOverflowed = true;
                break;
            }
            stack[stackPtr++] = right;
            stack[stackPtr++] = left;
        }
    }

    if (stackOverflowed)
    {
        for (uint i = 0u; i < numTetrahedrons; i++)
        {
            testEntityLeafHit(start, direction, i, modelTetCount, maxDistance, closestHit);
        }
    }

    return closestHit;
}

EntityTetraHit traceEntityTetraBVHLeafArray(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    float maxDistance)
{
    EntityTetraHit closestHit = entityTetraHitNone();
    if (numTetrahedrons == 0u)
    {
        return closestHit;
    }

    uint modelTetCount, modelTetStride;
    modelTetrahedrons.GetDimensions(modelTetCount, modelTetStride);
    if (modelTetCount == 0u)
    {
        return closestHit;
    }

    if (numTetrahedrons == 1u)
    {
        if (bvhNodes[0].isLeaf != 0u && bvhNodes[0].tetrahedronIndex < numTetrahedrons)
        {
            testEntityLeafHit(
                start,
                direction,
                bvhNodes[0].tetrahedronIndex,
                modelTetCount,
                maxDistance,
                closestHit);
        }
        return closestHit;
    }

    uint numInternalNodes = numTetrahedrons - 1u;
    for (uint i = 0u; i < numTetrahedrons; i++)
    {
        uint leafIdx = numInternalNodes + i;
        BVHNode leaf = bvhNodes[leafIdx];
        if (leaf.isLeaf == 0u)
        {
            continue;
        }
        uint instanceIdx = leaf.tetrahedronIndex;
        if (instanceIdx >= numTetrahedrons)
        {
            continue;
        }
        testEntityLeafHit(
            start,
            direction,
            instanceIdx,
            modelTetCount,
            maxDistance,
            closestHit);
    }
    return closestHit;
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

// 4D axis permutation lookup table — matches PERMUTATIONS from voxel.rs.
// Each uint4 gives [output_axis_0_source, ..., output_axis_3_source].
static const uint4 PERM_LUT[24] = {
    uint4(0, 1, 2, 3), uint4(0, 1, 3, 2), uint4(0, 2, 1, 3), uint4(0, 2, 3, 1),
    uint4(0, 3, 1, 2), uint4(0, 3, 2, 1), uint4(1, 0, 2, 3), uint4(1, 0, 3, 2),
    uint4(1, 2, 0, 3), uint4(1, 2, 3, 0), uint4(1, 3, 0, 2), uint4(1, 3, 2, 0),
    uint4(2, 0, 1, 3), uint4(2, 0, 3, 1), uint4(2, 1, 0, 3), uint4(2, 1, 3, 0),
    uint4(2, 3, 0, 1), uint4(2, 3, 1, 0), uint4(3, 0, 1, 2), uint4(3, 0, 2, 1),
    uint4(3, 1, 0, 2), uint4(3, 1, 2, 0), uint4(3, 2, 0, 1), uint4(3, 2, 1, 0),
};

// Apply a TesseractOrientation (encoded as permutation_index * 16 + sign_bits) to a 4D position.
float4 applyTesseractOrientation(float4 pos, uint orientation)
{
    if (orientation == 0u) return pos;
    uint permIdx = orientation / 16u;
    uint signBits = orientation % 16u;
    uint4 perm = PERM_LUT[min(permIdx, 23u)];
    float4 src = pos;
    float4 r;
    r.x = ((signBits & 1u) != 0u) ? -src[perm.x] : src[perm.x];
    r.y = ((signBits & 2u) != 0u) ? -src[perm.y] : src[perm.y];
    r.z = ((signBits & 4u) != 0u) ? -src[perm.z] : src[perm.z];
    r.w = ((signBits & 8u) != 0u) ? -src[perm.w] : src[perm.w];
    return r;
}

/// Unpack a per-voxel orientation+scale value into the u32 format used by
/// voxelMaterialSampleCoords.
///
/// Input layout (15 bits): bits 0-8 = tesseract orientation (0-383),
///                         bits 9-14 = scale_exp (6-bit signed, range [-32, +31]).
/// Output u32 layout:      bits 0-15 = tesseract orientation,
///                         bits 24-31 = scale_exp (8-bit signed, sign-extended).
uint unpackOrientationScale(uint packed)
{
    uint orientation = packed & 0x1FFu;         // 9 bits
    uint scaleBits = (packed >> 9) & 0x3Fu;     // 6 bits
    // Sign-extend 6-bit to 8-bit: if bit 5 is set, fill bits 6-7.
    if ((scaleBits & 0x20u) != 0u)
        scaleBits |= 0xC0u;
    return orientation | (scaleBits << 24);
}

float4 voxelMaterialSampleCoords(float4 hitPos, float4 rayDir, float4 hitNormal, uint orientationPacked)
{
    // Bits 0-15: tesseract orientation, bits 24-31: scale_exp as signed i8.
    uint orientation = orientationPacked & 0xFFFFu;
    int rawScale = int(orientationPacked >> 24) & 0xFF;
    int scaleExp = rawScale >= 128 ? rawScale - 256 : rawScale;
    float voxelSize = exp2(float(scaleExp));

    // Sample from the entered voxel, then map the hit hyperface to 3D local
    // coordinates by dropping the normal axis.
    // Divide by voxelSize so that frac() wraps at voxel boundaries for any scale.
    float4 local = frac(hitPos / voxelSize + rayDir * (1e-4 / voxelSize));

    // Apply block orientation to local coordinates (centered on voxel).
    local = applyTesseractOrientation(local - 0.5, orientation) + 0.5;

    float4 absN = abs(hitNormal);
    uint axis = 0u;
    float best = absN.x;
    if (absN.y > best) { axis = 1u; best = absN.y; }
    if (absN.z > best) { axis = 2u; best = absN.z; }
    if (absN.w > best) { axis = 3u; }

    int sign = 1;
    if (axis == 0u) sign = (hitNormal.x < 0.0) ? -1 : 1;
    else if (axis == 1u) sign = (hitNormal.y < 0.0) ? -1 : 1;
    else if (axis == 2u) sign = (hitNormal.z < 0.0) ? -1 : 1;
    else sign = (hitNormal.w < 0.0) ? -1 : 1;

    float3 uvw;
    if (axis == 0u)
    {
        uvw = float3(local.y, local.z, local.w);
    }
    else if (axis == 1u)
    {
        uvw = float3(local.x, local.z, local.w);
    }
    else if (axis == 2u)
    {
        uvw = float3(local.x, local.y, local.w);
    }
    else
    {
        uvw = float3(local.x, local.y, local.z);
    }

    // Basic orientation consistency for opposite faces.
    if (sign < 0)
    {
        uvw.x = 1.0 - uvw.x;
    }

    // Keep float4 API: xyz carry true 3D hyperface coordinates.
    return float4(uvw, uvw.z);
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
    if ((meta.highlightFlags & HIGHLIGHT_FLAG_HIT_VOXEL) != 0u)
    {
        int blockSize = 1 << int(meta.highlightHitScale);
        int4 rel = voxelCoord - meta.highlightHitVoxel;
        if (all(rel >= 0) && all(rel < blockSize))
            return HIGHLIGHT_FLAG_HIT_VOXEL;
    }
    if ((meta.highlightFlags & HIGHLIGHT_FLAG_PLACE_VOXEL) != 0u)
    {
        int blockSize = 1 << int(meta.highlightPlaceScale);
        int4 rel = voxelCoord - meta.highlightPlaceVoxel;
        if (all(rel >= 0) && all(rel < blockSize))
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

    int hitBlockSize = 1 << int(meta.highlightHitScale);
    int placeBlockSize = 1 << int(meta.highlightPlaceScale);
    int4 delta = meta.highlightPlaceVoxel - meta.highlightHitVoxel;
    uint faceAxes = 0u;

    // For each axis, check for scale-aware adjacency:
    //   positive face: delta == +hitBlockSize
    //   negative face: delta == -placeBlockSize
    //   other non-zero deltas: grid-alignment offset (ignored, not a face axis)
    if (delta.x == hitBlockSize) { axis = 0u; sign = 1; faceAxes++; }
    else if (delta.x == -placeBlockSize) { axis = 0u; sign = -1; faceAxes++; }

    if (delta.y == hitBlockSize) { axis = 1u; sign = 1; faceAxes++; }
    else if (delta.y == -placeBlockSize) { axis = 1u; sign = -1; faceAxes++; }

    if (delta.z == hitBlockSize) { axis = 2u; sign = 1; faceAxes++; }
    else if (delta.z == -placeBlockSize) { axis = 2u; sign = -1; faceAxes++; }

    if (delta.w == hitBlockSize) { axis = 3u; sign = 1; faceAxes++; }
    else if (delta.w == -placeBlockSize) { axis = 3u; sign = -1; faceAxes++; }

    return faceAxes == 1u;
}

float3 faceAxisSignColor(uint axis, int sign)
{
    bool isPositive = sign >= 0;
    if (axis == 0u)
    {
        return isPositive ? float3(1.00, 0.32, 0.32) : float3(0.74, 0.22, 0.22); // +/-X
    }
    if (axis == 1u)
    {
        return isPositive ? float3(0.34, 1.00, 0.52) : float3(0.22, 0.74, 0.36); // +/-Y
    }
    if (axis == 2u)
    {
        return isPositive ? float3(0.36, 0.70, 1.00) : float3(0.22, 0.44, 0.80); // +/-Z
    }
    return isPositive ? float3(1.00, 0.78, 0.28) : float3(0.80, 0.56, 0.20); // +/-W
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
    // Determine block origin and size for the highlighted block.
    int4 blockMin = (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
        ? meta.highlightHitVoxel : meta.highlightPlaceVoxel;
    int blockSize = 1 << int((highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
        ? meta.highlightHitScale : meta.highlightPlaceScale);
    float fBlockSize = float(blockSize);

    // 4D block-relative local coordinates in [0,1].
    float4 local = (hitPos + rayDir * 1e-4 - float4(blockMin)) / fBlockSize;
    float4 faceDist = min(local, 1.0 - local);
    uint guideAxis = dominantAxisFromNormal(hitNormal);
    int guideSign = normalSignOnAxis(hitNormal, guideAxis);

    // For the hit block, prefer the explicit placement face
    // inferred from hit/place voxel relation so the guide marks "where placement goes".
    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
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

    // Pure high-contrast treatment: drive edges toward black, but keep
    // center contribution unchanged.
    float edgeWeight = saturate(faceBand);
    // Keep hit and place guides visually distinct:
    // hit voxel gets near-black edge bands, placement voxel stays darker-gray.
    float blackBlend = (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL) ? 0.98 : 0.55;
    float3 shaded = lerp(litColor, float3(0.0, 0.0, 0.0), edgeWeight * blackBlend);

    if (highlightKind == HIGHLIGHT_FLAG_HIT_VOXEL)
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
        // Signed axis tint so +/− X/Y/Z/W are immediately recognizable.
        float3 axisTint = faceAxisSignColor(guideAxis, guideSign);
        shaded = lerp(shaded, axisTint, accentMask * 0.94);
    }

    return shaded;
}

uint findLeafIndexForChunkCoord(int4 chunkCoord, VoxelFrameMeta meta)
{
    if (meta.regionBvhNodeCount == 0u)
    {
        return INVALID_INDEX;
    }

    // Convert chunk coord to a world-space query point (center of the chunk at scale 0).
    float chunkSizeF = float(CHUNK_SIZE);
    float4 queryPoint = float4(chunkCoord) * chunkSizeF + chunkSizeF * 0.5;

    uint stack[64];
    int stackPtr = 0;
    if (meta.regionBvhRootIndex >= meta.regionBvhNodeCount)
    {
        return INVALID_INDEX;
    }
    stack[stackPtr++] = meta.regionBvhRootIndex;

    while (stackPtr > 0)
    {
        uint nodeIndex = stack[--stackPtr];
        if (nodeIndex >= meta.regionBvhNodeCount)
        {
            continue;
        }

        VoxelChunkBvhNode node = voxelChunkBvhNodes[nodeIndex];
        bool outside =
            queryPoint.x < node.worldMin.x || queryPoint.x >= node.worldMax.x
            || queryPoint.y < node.worldMin.y || queryPoint.y >= node.worldMax.y
            || queryPoint.z < node.worldMin.z || queryPoint.z >= node.worldMax.z
            || queryPoint.w < node.worldMin.w || queryPoint.w >= node.worldMax.w;
        if (outside)
        {
            continue;
        }

        bool isLeaf = (node.flags & 1u) != 0u;
        if (isLeaf)
        {
            if (node.leafIndex < meta.leafCount)
            {
                return node.leafIndex;
            }
            continue;
        }

        if (stackPtr + 2 > 64)
        {
            continue;
        }
        if (node.leftChild != INVALID_INDEX)
        {
            stack[stackPtr++] = node.leftChild;
        }
        if (node.rightChild != INVALID_INDEX)
        {
            stack[stackPtr++] = node.rightChild;
        }
    }
    return INVALID_INDEX;
}

uint lookupLeafChunkEntry(VoxelLeafHeader leaf, int4 chunkCoord, VoxelFrameMeta meta)
{
    int localX = chunkCoord.x - leaf.minChunkCoord.x;
    int localY = chunkCoord.y - leaf.minChunkCoord.y;
    int localZ = chunkCoord.z - leaf.minChunkCoord.z;
    int localW = chunkCoord.w - leaf.minChunkCoord.w;
    if (localX < 0 || localY < 0 || localZ < 0 || localW < 0)
    {
        return LEAF_CHUNK_ENTRY_EMPTY;
    }

    uint dimX = uint(leaf.maxChunkCoord.x - leaf.minChunkCoord.x + 1);
    uint dimY = uint(leaf.maxChunkCoord.y - leaf.minChunkCoord.y + 1);
    uint dimZ = uint(leaf.maxChunkCoord.z - leaf.minChunkCoord.z + 1);
    uint dimW = uint(leaf.maxChunkCoord.w - leaf.minChunkCoord.w + 1);
    if (uint(localX) >= dimX || uint(localY) >= dimY || uint(localZ) >= dimZ || uint(localW) >= dimW)
    {
        return LEAF_CHUNK_ENTRY_EMPTY;
    }

    uint linear = uint(localX)
        + dimX * (uint(localY) + dimY * (uint(localZ) + dimZ * uint(localW)));
    uint entryIndex = leaf.chunkEntryOffset + linear;
    if (entryIndex >= meta.leafChunkEntryCount)
    {
        return LEAF_CHUNK_ENTRY_EMPTY;
    }
    return voxelLeafChunkEntries[entryIndex];
}

uint findUniformMaterialAtChunk(int4 chunkCoord, uint lodLevel, VoxelFrameMeta meta)
{
    if (lodLevel != 0u)
    {
        return 0u;
    }
    uint leafIndex = findLeafIndexForChunkCoord(chunkCoord, meta);
    if (leafIndex == INVALID_INDEX || leafIndex >= meta.leafCount)
    {
        return 0u;
    }

    VoxelLeafHeader leaf = voxelLeafHeaders[leafIndex];
    if (leaf.leafKind == LEAF_KIND_UNIFORM)
    {
        return leaf.uniformMaterial;
    }
    if (leaf.leafKind != LEAF_KIND_VOXEL_CHUNK_ARRAY)
    {
        return 0u;
    }
    uint encoded = lookupLeafChunkEntry(leaf, chunkCoord, meta);
    if ((encoded & LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0u)
    {
        return encoded & 0xFFFFu;
    }
    return 0u;
}

uint findChunkIndex(int4 chunkCoord, uint lodLevel, VoxelFrameMeta meta)
{
    if (lodLevel != 0u)
    {
        return INVALID_INDEX;
    }
    uint leafIndex = findLeafIndexForChunkCoord(chunkCoord, meta);
    if (leafIndex == INVALID_INDEX || leafIndex >= meta.leafCount)
    {
        return INVALID_INDEX;
    }

    VoxelLeafHeader leaf = voxelLeafHeaders[leafIndex];
    if (leaf.leafKind != LEAF_KIND_VOXEL_CHUNK_ARRAY)
    {
        return INVALID_INDEX;
    }

    uint encoded = lookupLeafChunkEntry(leaf, chunkCoord, meta);
    if (encoded == LEAF_CHUNK_ENTRY_EMPTY || (encoded & LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0u)
    {
        return INVALID_INDEX;
    }
    uint chunkIndex = encoded - 1u;
    if (chunkIndex >= meta.chunkCount)
    {
        return INVALID_INDEX;
    }
    return chunkIndex;
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

float min3(float3 v)
{
    return min(v.x, min(v.y, v.z));
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

uint estimateChunkCrossings(float intervalLength, float tDeltaAxis)
{
    if (intervalLength <= 0.0 || tDeltaAxis > 1e29)
    {
        return 0u;
    }
    float ratio = intervalLength / max(tDeltaAxis, 1e-6);
    // Conservative pad for boundary ownership and numeric jitter.
    return uint(ceil(ratio)) + 3u;
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

int entryChunkCoordScaled(float position, float direction, float cellSize)
{
    float q = position / cellSize;
    float qFloor = floor(q);
    int chunk = int(qFloor);

    if (direction < 0.0)
    {
        float boundary = qFloor * cellSize;
        float eps = 1e-7 * cellSize;
        if (abs(position - boundary) <= eps)
        {
            chunk -= 1;
        }
    }

    return chunk;
}

bool intersectChunkAABBScaled(
    float4 rayOrigin,
    float4 rayDir,
    int4 chunkCoord,
    float cellSize,
    out float tEnter,
    out float tExit)
{
    float4 bmin = float4(chunkCoord) * cellSize;
    float4 bmax = bmin + cellSize;
    return intersectAABB(rayOrigin, rayDir, bmin, bmax, tEnter, tExit);
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
    out uint materialId,
    out uint orientation)
{
    materialId = 0;
    orientation = 0;
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

    uint materialIdx = h.materialWordOffset + (localIdx >> 1);
    if (materialIdx >= meta.materialWordCount)
    {
        return false;
    }
    uint materialWord = voxelMaterialWords[materialIdx];
    uint shift = (localIdx & 1u) * 16u;
    materialId = (materialWord >> shift) & 0xFFFFu;

    uint orientIdx = h.orientationWordOffset + (localIdx >> 1);
    if (orientIdx < meta.orientationWordCount)
    {
        uint orientWord = voxelOrientationWords[orientIdx];
        uint packed16 = (orientWord >> shift) & 0xFFFFu;
        orientation = unpackOrientationScale(packed16);
    }

    return materialId != 0;
}

bool macroCellMayContainSolid(
    int4 voxelCoord,
    int4 chunkMin,
    VoxelChunkHeader h,
    VoxelFrameMeta meta)
{
    if (meta.macroWordCount == 0u)
    {
        return true;
    }

    int4 local = voxelCoord - chunkMin;
    uint macroIdx = computeMacroIndex(local);
    uint macroWordIdx = h.macroWordOffset + (macroIdx >> 5);
    if (macroWordIdx >= meta.macroWordCount)
    {
        // Conservative fallback: if offsets are inconsistent, never cull.
        return true;
    }

    uint macroWord = voxelMacroWords[macroWordIdx];
    return (macroWord & (1u << (macroIdx & 31u))) != 0u;
}

int4 macroCoordForVoxel(int4 voxelCoord, int4 chunkMin)
{
    int4 local = voxelCoord - chunkMin;
    return int4(
        local.x >> 1,
        local.y >> 1,
        local.z >> 1,
        local.w >> 1
    );
}

bool traceVoxelsInChunk(
    float4 rayOrigin,
    float4 rayDir,
    float tEnter,
    float tExit,
    int4 chunkCoord,
    float cellSize,
    float voxelSize,
    VoxelChunkHeader h,
    VoxelFrameMeta meta,
    inout uint remainingVoxelSteps,
    out uint hitMaterial,
    out uint hitOrientation,
    out float4 hitPos,
    out int4 hitChunkCoord)
{
    const float EPS = 1e-6;
    hitMaterial = 0;
    hitOrientation = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    hitChunkCoord = int4(0, 0, 0, 0);
    float intervalEps = max(EPS, max(abs(tEnter), abs(tExit)) * 1e-7);
    if (remainingVoxelSteps == 0 || tExit <= tEnter + intervalEps)
    {
        return false;
    }

    float4 chunkWorldMin = float4(chunkCoord) * cellSize;
    // Conservative per-chunk solid AABB clip:
    // for sparse edited chunks this rejects rays that only traverse empty chunk space.
    int4 solidLocalMin = h.solidLocalMin;
    int4 solidLocalMax = h.solidLocalMax;
    float4 solidBMin = chunkWorldMin + float4(solidLocalMin) * voxelSize;
    float4 solidBMax = chunkWorldMin + float4(solidLocalMax + 1) * voxelSize;
    float solidEnter = 0.0;
    float solidExit = 0.0;
    if (!intersectAABB(rayOrigin, rayDir, solidBMin, solidBMax, solidEnter, solidExit))
    {
        return false;
    }
    tEnter = max(tEnter, solidEnter);
    tExit = min(tExit, solidExit);
    intervalEps = max(EPS, max(abs(tEnter), abs(tExit)) * 1e-7);
    if (tExit <= tEnter + intervalEps)
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

    float4 startPos = rayOrigin + rayDir * tTraceStart;
    // Convert to local lattice coordinates (0..CHUNK_SIZE per axis)
    float4 localPos = (startPos - chunkWorldMin) / voxelSize;
    int4 voxelCoord = int4(floor(localPos));
    // Clamp to [0, CHUNK_SIZE-1]
    voxelCoord = clamp(voxelCoord, int4(0, 0, 0, 0), int4(int(CHUNK_SIZE) - 1));

    int4 step = int4(
        stepSign(rayDir.x),
        stepSign(rayDir.y),
        stepSign(rayDir.z),
        stepSign(rayDir.w)
    );

    float4 tDelta = float4(
        safeInvAbs(rayDir.x) * voxelSize,
        safeInvAbs(rayDir.y) * voxelSize,
        safeInvAbs(rayDir.z) * voxelSize,
        safeInvAbs(rayDir.w) * voxelSize
    );

    // tMaxLocal: time from tTraceStart until the ray crosses the next voxel boundary
    float4 tMaxLocal = float4(1e30, 1e30, 1e30, 1e30);
    if (step.x > 0) tMaxLocal.x = (chunkWorldMin.x + float(voxelCoord.x + 1) * voxelSize - startPos.x) / rayDir.x;
    else if (step.x < 0) tMaxLocal.x = (startPos.x - (chunkWorldMin.x + float(voxelCoord.x) * voxelSize)) * safeInvAbs(rayDir.x);
    if (step.y > 0) tMaxLocal.y = (chunkWorldMin.y + float(voxelCoord.y + 1) * voxelSize - startPos.y) / rayDir.y;
    else if (step.y < 0) tMaxLocal.y = (startPos.y - (chunkWorldMin.y + float(voxelCoord.y) * voxelSize)) * safeInvAbs(rayDir.y);
    if (step.z > 0) tMaxLocal.z = (chunkWorldMin.z + float(voxelCoord.z + 1) * voxelSize - startPos.z) / rayDir.z;
    else if (step.z < 0) tMaxLocal.z = (startPos.z - (chunkWorldMin.z + float(voxelCoord.z) * voxelSize)) * safeInvAbs(rayDir.z);
    if (step.w > 0) tMaxLocal.w = (chunkWorldMin.w + float(voxelCoord.w + 1) * voxelSize - startPos.w) / rayDir.w;
    else if (step.w < 0) tMaxLocal.w = (startPos.w - (chunkWorldMin.w + float(voxelCoord.w) * voxelSize)) * safeInvAbs(rayDir.w);

    // Voxel coordinates are local [0..CHUNK_SIZE); use zero-origin for sampling.
    int4 localOrigin = int4(0, 0, 0, 0);
    int4 localMax = int4(int(CHUNK_SIZE), int(CHUNK_SIZE), int(CHUNK_SIZE), int(CHUNK_SIZE));

    float localT = 0.0;
    bool firstVoxelInInterval = true;
    bool canMacroSkip = ((h.flags & CHUNK_FLAG_FULL) == 0u) && (meta.macroWordCount > 0u);
    int4 cachedMacroCoord = int4(-1, -1, -1, -1);
    bool cachedMacroHasSolid = true;
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
            bool macroHasSolid = true;
            if (canMacroSkip)
            {
                int4 macroCoord = macroCoordForVoxel(voxelCoord, localOrigin);
                if (!all(macroCoord == cachedMacroCoord))
                {
                    cachedMacroCoord = macroCoord;
                    cachedMacroHasSolid = macroCellMayContainSolid(voxelCoord, localOrigin, h, meta);
                }
                macroHasSolid = cachedMacroHasSolid;
            }
            if (macroHasSolid)
            {
                uint materialId = 0;
                uint orientId = 0;
                if (sampleVoxelInChunk(voxelCoord, localOrigin, h, meta, materialId, orientId))
                {
                    float hitT = tTraceStart + localT;
                    if (firstVoxelInInterval)
                    {
                        // Match uniform-leaf behavior at interval entry so boundary-sensitive
                        // materials don't shift appearance when a chunk becomes dense.
                        hitT = tStart;
                    }
                    hitMaterial = materialId;
                    hitOrientation = orientId;
                    hitPos = rayOrigin + rayDir * hitT;
                    hitChunkCoord = chunkCoord;
                    return true;
                }
            }
        }

        nextLocalT = 0.0;
        advanceDDA4(voxelCoord, step, tMaxLocal, tDelta, nextLocalT);
        localT = max(localT, nextLocalT);
        firstVoxelInInterval = false;

        if (!voxelInChunk(voxelCoord, localOrigin, localMax))
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
    out int4 hitChunkCoord,
    out uint nodeVisitCount,
    out uint pathHash)
{
    uint hitLod = 0u;
    uint hitOrient = 0u;
    uint missReason = MISS_REASON_NONE;
    uint debugFlags = 0u;
    uint chunkSteps = 0u;
    uint remainingVoxelSteps = 0u;
    float finalT = 0.0;
    int4 lastChunk = int4(0, 0, 0, 0);
    return traceVoxelRayDDA(
        rayOrigin,
        rayDir,
        meta,
        0u,
        hitMaterial,
        hitOrient,
        hitPos,
        hitChunkCoord,
        hitLod,
        missReason,
        debugFlags,
        chunkSteps,
        remainingVoxelSteps,
        finalT,
        lastChunk,
        nodeVisitCount,
        pathHash);
}

bool traceVoxelRayDDA(
    float4 rayOrigin,
    float4 rayDir,
    VoxelFrameMeta meta,
    uint targetLodLevel,
    out uint hitMaterial,
    out uint hitOrientation,
    out float4 hitPos,
    out int4 hitChunkCoord,
    out uint hitLodLevel,
    out uint missReason,
    out uint debugFlags,
    out uint outChunkStepsTaken,
    out uint outRemainingVoxelSteps,
    out float outFinalT,
    out int4 outLastChunkCoord,
    out uint outNodeVisitCount,
    out uint outPathHash)
{
    hitMaterial = 0u;
    hitOrientation = 0u;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    hitChunkCoord = int4(0, 0, 0, 0);
    hitLodLevel = targetLodLevel;
    missReason = MISS_REASON_NONE;
    debugFlags = 0u;
    outChunkStepsTaken = 0u;
    outRemainingVoxelSteps = 0u;
    outFinalT = 0.0;
    outLastChunkCoord = int4(0, 0, 0, 0);
    outNodeVisitCount = 0u;
    outPathHash = 2166136261u;

    if (meta.leafCount == 0u || meta.regionBvhNodeCount == 0u)
    {
        return false;
    }

    const float CHUNK_EPS = 1e-6;
    float chunkSizeF = float(CHUNK_SIZE);
    float MAX_DISTANCE = max(1.0, meta.maxTraceDistance);
    uint maxChunkSteps = max(1u, min(meta.maxTraceSteps, 4096u));
    uint remainingVoxelSteps = max(1u, min(meta.maxTraceSteps * 8u, 32768u));
    bool collectDebugFlags = VTE_DIAGNOSTICS
        && (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE
        || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL);
    bool useSlabChunkInterval = collectDebugFlags
        || (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_COMPARE) != 0u;

    if (meta.regionBvhRootIndex >= meta.regionBvhNodeCount)
    {
        return false;
    }
    uint rootIndex = meta.regionBvhRootIndex;
    VoxelChunkBvhNode rootNode = voxelChunkBvhNodes[rootIndex];
    float4 rootBMin = rootNode.worldMin;
    float4 rootBMax = rootNode.worldMax;
    float rootEnter = 0.0;
    float rootExit = 0.0;
    if (!intersectAABB(rayOrigin, rayDir, rootBMin, rootBMax, rootEnter, rootExit))
    {
        return false;
    }
    float traversalMinT = max(0.0, rootEnter);
    float traversalMaxT = min(MAX_DISTANCE, rootExit);
    if (traversalMaxT <= traversalMinT + CHUNK_EPS)
    {
        return false;
    }

    const uint BVH_STACK_CAPACITY = 128u;
    uint nodeStack[BVH_STACK_CAPACITY];
    float enterStack[BVH_STACK_CAPACITY];
    float exitStack[BVH_STACK_CAPACITY];
    uint stackCount = 0u;
    nodeStack[stackCount] = rootIndex;
    enterStack[stackCount] = traversalMinT;
    exitStack[stackCount] = traversalMaxT;
    stackCount++;

    bool touchedVisibleChunk = false;
    uint chunkStepsTaken = 0u;
    int4 lastChunkCoord = int4(0, 0, 0, 0);
    bool clippedByMaxDistance = (rootExit > MAX_DISTANCE + CHUNK_EPS);

    while (stackCount > 0u && remainingVoxelSteps > 0u && chunkStepsTaken < maxChunkSteps)
    {
        // Pop nearest node interval first. This keeps first-hit correctness even
        // when sibling bounds overlap or become highly lopsided after updates.
        uint bestSlot = 0u;
        float bestEnter = enterStack[0];
        for (uint i = 1u; i < stackCount; i++)
        {
            if (enterStack[i] < bestEnter)
            {
                bestEnter = enterStack[i];
                bestSlot = i;
            }
        }
        uint nodeIndex = nodeStack[bestSlot];
        float nodeEnter = enterStack[bestSlot];
        float nodeExit = exitStack[bestSlot];
        stackCount--;
        if (bestSlot != stackCount)
        {
            nodeStack[bestSlot] = nodeStack[stackCount];
            enterStack[bestSlot] = enterStack[stackCount];
            exitStack[bestSlot] = exitStack[stackCount];
        }
        outNodeVisitCount++;
        outPathHash = (outPathHash ^ (nodeIndex + 0x9E3779B9u)) * 16777619u;

        if (nodeIndex >= meta.regionBvhNodeCount)
        {
            continue;
        }
        if (nodeExit <= nodeEnter + CHUNK_EPS)
        {
            continue;
        }

        VoxelChunkBvhNode node = voxelChunkBvhNodes[nodeIndex];
        bool isLeaf = (node.flags & 1u) != 0u;
        if (!isLeaf)
        {
            float leftEnter = 0.0;
            float leftExit = 0.0;
            float rightEnter = 0.0;
            float rightExit = 0.0;
            bool leftValid = false;
            bool rightValid = false;

            if (node.leftChild != INVALID_INDEX && node.leftChild < meta.regionBvhNodeCount)
            {
                VoxelChunkBvhNode leftNode = voxelChunkBvhNodes[node.leftChild];
                float4 leftBMin = leftNode.worldMin;
                float4 leftBMax = leftNode.worldMax;
                if (intersectAABB(rayOrigin, rayDir, leftBMin, leftBMax, leftEnter, leftExit))
                {
                    leftEnter = max(nodeEnter, leftEnter);
                    leftExit = min(nodeExit, leftExit);
                    leftValid = leftExit > leftEnter + CHUNK_EPS;
                }
            }

            if (node.rightChild != INVALID_INDEX && node.rightChild < meta.regionBvhNodeCount)
            {
                VoxelChunkBvhNode rightNode = voxelChunkBvhNodes[node.rightChild];
                float4 rightBMin = rightNode.worldMin;
                float4 rightBMax = rightNode.worldMax;
                if (intersectAABB(rayOrigin, rayDir, rightBMin, rightBMax, rightEnter, rightExit))
                {
                    rightEnter = max(nodeEnter, rightEnter);
                    rightExit = min(nodeExit, rightExit);
                    rightValid = rightExit > rightEnter + CHUNK_EPS;
                }
            }

            if (leftValid && rightValid)
            {
                if (stackCount + 2u <= BVH_STACK_CAPACITY)
                {
                    nodeStack[stackCount] = node.leftChild;
                    enterStack[stackCount] = leftEnter;
                    exitStack[stackCount] = leftExit;
                    stackCount++;
                    nodeStack[stackCount] = node.rightChild;
                    enterStack[stackCount] = rightEnter;
                    exitStack[stackCount] = rightExit;
                    stackCount++;
                }
                else
                {
                    // Keep the nearer child if stack space is tight.
                    if (leftEnter <= rightEnter)
                    {
                        nodeStack[stackCount] = node.leftChild;
                        enterStack[stackCount] = leftEnter;
                        exitStack[stackCount] = leftExit;
                        stackCount++;
                    }
                    else if (stackCount + 1u <= BVH_STACK_CAPACITY)
                    {
                        nodeStack[stackCount] = node.rightChild;
                        enterStack[stackCount] = rightEnter;
                        exitStack[stackCount] = rightExit;
                        stackCount++;
                    }
                }
            }
            else if (leftValid)
            {
                if (stackCount + 1u <= BVH_STACK_CAPACITY)
                {
                    nodeStack[stackCount] = node.leftChild;
                    enterStack[stackCount] = leftEnter;
                    exitStack[stackCount] = leftExit;
                    stackCount++;
                }
            }
            else if (rightValid)
            {
                if (stackCount + 1u <= BVH_STACK_CAPACITY)
                {
                    nodeStack[stackCount] = node.rightChild;
                    enterStack[stackCount] = rightEnter;
                    exitStack[stackCount] = rightExit;
                    stackCount++;
                }
            }
            continue;
        }

        if (node.leafIndex >= meta.leafCount)
        {
            continue;
        }
        VoxelLeafHeader leaf = voxelLeafHeaders[node.leafIndex];
        touchedVisibleChunk = true;

        int scaleExp = leaf.getScaleExp();
        float cellSize = float(CHUNK_SIZE) * exp2(float(scaleExp));
        float voxelSize = exp2(float(scaleExp));

        if (leaf.leafKind == LEAF_KIND_UNIFORM)
        {
            uint uniformMaterial = leaf.uniformMaterial;
            if (uniformMaterial != 0u)
            {
                float hitT = max(0.0, nodeEnter);
                float probeEps = max(CHUNK_EPS, abs(hitT) * 1e-7);
                float probeT = min(nodeExit, hitT + probeEps);
                float4 probePos = rayOrigin + rayDir * probeT;
                hitMaterial = uniformMaterial;
                hitOrientation = (leaf.uniformOrientation & 0xFFFFu) | (leaf.uniformOrientation & 0xFF000000u);
                hitPos = rayOrigin + rayDir * hitT;
                hitChunkCoord = int4(
                    entryChunkCoordScaled(probePos.x, rayDir.x, cellSize),
                    entryChunkCoordScaled(probePos.y, rayDir.y, cellSize),
                    entryChunkCoordScaled(probePos.z, rayDir.z, cellSize),
                    entryChunkCoordScaled(probePos.w, rayDir.w, cellSize)
                );
                hitLodLevel = targetLodLevel;
                outChunkStepsTaken = chunkStepsTaken;
                outRemainingVoxelSteps = remainingVoxelSteps;
                outFinalT = hitT;
                outLastChunkCoord = hitChunkCoord;
                missReason = MISS_REASON_NONE;
                return true;
            }
            continue;
        }

        if (leaf.leafKind != LEAF_KIND_VOXEL_CHUNK_ARRAY)
        {
            continue;
        }

        int4 chunkStep = int4(
            stepSign(rayDir.x),
            stepSign(rayDir.y),
            stepSign(rayDir.z),
            stepSign(rayDir.w)
        );
        float4 tDeltaChunk = float4(
            safeInvAbs(rayDir.x) * cellSize,
            safeInvAbs(rayDir.y) * cellSize,
            safeInvAbs(rayDir.z) * cellSize,
            safeInvAbs(rayDir.w) * cellSize
        );

        float currentT = nodeEnter;
        int4 chunkCoord = int4(0, 0, 0, 0);
        float4 tMaxChunk = float4(1e30, 1e30, 1e30, 1e30);
        bool chunkStateValid = false;

        while (remainingVoxelSteps > 0u && chunkStepsTaken < maxChunkSteps)
        {
            chunkStepsTaken++;
            if (currentT > nodeExit)
            {
                break;
            }

            if (!chunkStateValid)
            {
                float probeEps = max(CHUNK_EPS, abs(currentT) * 1e-7);
                float probeT = min(nodeExit, currentT + probeEps);
                float4 currentPos = rayOrigin + rayDir * probeT;
                chunkCoord = int4(
                    entryChunkCoordScaled(currentPos.x, rayDir.x, cellSize),
                    entryChunkCoordScaled(currentPos.y, rayDir.y, cellSize),
                    entryChunkCoordScaled(currentPos.z, rayDir.z, cellSize),
                    entryChunkCoordScaled(currentPos.w, rayDir.w, cellSize)
                );

                tMaxChunk = float4(1e30, 1e30, 1e30, 1e30);
                if (chunkStep.x > 0) tMaxChunk.x = (float(chunkCoord.x + 1) * cellSize - rayOrigin.x) / rayDir.x;
                else if (chunkStep.x < 0) tMaxChunk.x = (float(chunkCoord.x) * cellSize - rayOrigin.x) / rayDir.x;
                if (chunkStep.y > 0) tMaxChunk.y = (float(chunkCoord.y + 1) * cellSize - rayOrigin.y) / rayDir.y;
                else if (chunkStep.y < 0) tMaxChunk.y = (float(chunkCoord.y) * cellSize - rayOrigin.y) / rayDir.y;
                if (chunkStep.z > 0) tMaxChunk.z = (float(chunkCoord.z + 1) * cellSize - rayOrigin.z) / rayDir.z;
                else if (chunkStep.z < 0) tMaxChunk.z = (float(chunkCoord.z) * cellSize - rayOrigin.z) / rayDir.z;
                if (chunkStep.w > 0) tMaxChunk.w = (float(chunkCoord.w + 1) * cellSize - rayOrigin.w) / rayDir.w;
                else if (chunkStep.w < 0) tMaxChunk.w = (float(chunkCoord.w) * cellSize - rayOrigin.w) / rayDir.w;
                chunkStateValid = true;
            }
            lastChunkCoord = chunkCoord;

            float chunkExitT = min(nodeExit, min4(tMaxChunk));
            chunkExitT = max(chunkExitT, currentT);
            if (collectDebugFlags)
            {
                float tieEps = max(1e-8, abs(chunkExitT) * 1e-7);
                uint tieCount = 0u;
                if (abs(tMaxChunk.x - chunkExitT) <= tieEps) tieCount++;
                if (abs(tMaxChunk.y - chunkExitT) <= tieEps) tieCount++;
                if (abs(tMaxChunk.z - chunkExitT) <= tieEps) tieCount++;
                if (abs(tMaxChunk.w - chunkExitT) <= tieEps) tieCount++;
                if (tieCount > 1u) debugFlags |= DEBUG_FLAG_CHUNK_TIE_STEPPED;
            }

            bool inLeafBounds =
                chunkCoord.x >= leaf.minChunkCoord.x && chunkCoord.x <= leaf.maxChunkCoord.x
                && chunkCoord.y >= leaf.minChunkCoord.y && chunkCoord.y <= leaf.maxChunkCoord.y
                && chunkCoord.z >= leaf.minChunkCoord.z && chunkCoord.z <= leaf.maxChunkCoord.z
                && chunkCoord.w >= leaf.minChunkCoord.w && chunkCoord.w <= leaf.maxChunkCoord.w;
            if (!inLeafBounds)
            {
                bool divergingFromLeafBounds =
                    (chunkCoord.x < leaf.minChunkCoord.x && chunkStep.x <= 0)
                    || (chunkCoord.x > leaf.maxChunkCoord.x && chunkStep.x >= 0)
                    || (chunkCoord.y < leaf.minChunkCoord.y && chunkStep.y <= 0)
                    || (chunkCoord.y > leaf.maxChunkCoord.y && chunkStep.y >= 0)
                    || (chunkCoord.z < leaf.minChunkCoord.z && chunkStep.z <= 0)
                    || (chunkCoord.z > leaf.maxChunkCoord.z && chunkStep.z >= 0)
                    || (chunkCoord.w < leaf.minChunkCoord.w && chunkStep.w <= 0)
                    || (chunkCoord.w > leaf.maxChunkCoord.w && chunkStep.w >= 0);
                if (divergingFromLeafBounds)
                {
                    break;
                }
            }
            else
            {
                uint encoded = lookupLeafChunkEntry(leaf, chunkCoord, meta);
                if ((encoded & LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0u)
                {
                    uint uniformMaterial = encoded & 0xFFFFu;
                    uint uniformChunkOrientPacked16 = (encoded >> 16u) & 0x7FFFu;
                    if (uniformMaterial != 0u)
                    {
                        hitMaterial = uniformMaterial;
                        hitOrientation = unpackOrientationScale(uniformChunkOrientPacked16);
                        hitPos = rayOrigin + rayDir * currentT;
                        hitChunkCoord = chunkCoord;
                        hitLodLevel = targetLodLevel;
                        outChunkStepsTaken = chunkStepsTaken;
                        outRemainingVoxelSteps = remainingVoxelSteps;
                        outFinalT = currentT;
                        outLastChunkCoord = chunkCoord;
                        missReason = MISS_REASON_NONE;
                        return true;
                    }
                }
                else if (encoded != LEAF_CHUNK_ENTRY_EMPTY)
                {
                    uint chunkIndex = encoded - 1u;
                    if (chunkIndex < meta.chunkCount)
                    {
                        VoxelChunkHeader h = voxelChunkHeaders[chunkIndex];
                        uint chunkMaterial = 0u;
                        uint chunkOrientation = 0u;
                        float4 chunkHitPos = float4(0.0, 0.0, 0.0, 0.0);
                        int4 chunkHitChunkCoord = int4(0, 0, 0, 0);
                        float tEnter = currentT;
                        float tExit = chunkExitT;
                        if (useSlabChunkInterval)
                        {
                            float slabEnter = 0.0;
                            float slabExit = 0.0;
                            if (intersectChunkAABBScaled(rayOrigin, rayDir, chunkCoord, cellSize, slabEnter, slabExit))
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
                                cellSize,
                                voxelSize,
                                h,
                                meta,
                                remainingVoxelSteps,
                                chunkMaterial,
                                chunkOrientation,
                                chunkHitPos,
                                chunkHitChunkCoord))
                            {
                                hitMaterial = chunkMaterial;
                                hitOrientation = chunkOrientation;
                                hitPos = chunkHitPos;
                                hitChunkCoord = chunkHitChunkCoord;
                                hitLodLevel = targetLodLevel;
                                outChunkStepsTaken = chunkStepsTaken;
                                outRemainingVoxelSteps = remainingVoxelSteps;
                                outFinalT = currentT;
                                outLastChunkCoord = lastChunkCoord;
                                missReason = MISS_REASON_NONE;
                                return true;
                            }
                        }
                        else if (collectDebugFlags)
                        {
                            debugFlags |= DEBUG_FLAG_CHUNK_ZERO_INTERVAL;
                        }
                    }
                }
            }

            float nextChunkT = chunkExitT;
            advanceDDA4(chunkCoord, chunkStep, tMaxChunk, tDeltaChunk, nextChunkT);
            float advanceEps = max(CHUNK_EPS, abs(nextChunkT) * 1e-7);
            currentT = max(currentT + CHUNK_EPS, nextChunkT + advanceEps);
        }
    }

    if (remainingVoxelSteps == 0u)
    {
        missReason = MISS_REASON_VOXEL_BUDGET;
    }
    else if (chunkStepsTaken >= maxChunkSteps)
    {
        missReason = MISS_REASON_CHUNK_BUDGET;
    }
    else if (clippedByMaxDistance)
    {
        missReason = MISS_REASON_MAX_DISTANCE;
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
    outFinalT = traversalMaxT;
    outLastChunkCoord = lastChunkCoord;
    return false;
}

bool traceVoxelRayDDAWithLodRings(
    float4 rayOrigin,
    float4 rayDir,
    VoxelFrameMeta meta,
    out uint hitMaterial,
    out uint hitOrientation,
    out float4 hitPos,
    out int4 hitChunkCoord,
    out uint hitLodLevel,
    out uint missReason,
    out uint debugFlags,
    out uint outChunkStepsTaken,
    out uint outRemainingVoxelSteps,
    out float outFinalT,
    out int4 outLastChunkCoord,
    out uint outNodeVisitCount,
    out uint outPathHash)
{
    return traceVoxelRayDDA(
        rayOrigin,
        rayDir,
        meta,
        0u,
        hitMaterial,
        hitOrientation,
        hitPos,
        hitChunkCoord,
        hitLodLevel,
        missReason,
        debugFlags,
        outChunkStepsTaken,
        outRemainingVoxelSteps,
        outFinalT,
        outLastChunkCoord,
        outNodeVisitCount,
        outPathHash);
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

    if (VTE_DIAGNOSTICS && globalId.x == 0u && globalId.y == 0u)
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

void voxelTraceStageAImpl(uint3 globalId, bool integralFusedDispatch)
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

    uint worldBvhRayDiagSlot = INVALID_INDEX;
    bool worldBvhRayDiagSelected = selectWorldBvhRayDiagSample(globalId, wd, meta, worldBvhRayDiagSlot);
    bool worldBvhRayDiagWritten = false;
    bool stageABreakdownSampled = selectStageABreakdownSample(globalId, meta);

    // Integral mode fast path: collapse across all layers in Stage A and write
    // directly to layer 0, so Stage B can be skipped for this mode.
    if (integralFusedDispatch && globalId.z == 0u && layerCount > 1u)
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
        float4 worldOrigin = wd.worldOrigin;
        float4 worldDirX = wd.worldDirX;
        float4 worldDirY = wd.worldDirY;
        float4 worldDirZ = wd.worldDirZ;
        float4 worldDirW = wd.worldDirW;
        float4 worldXY = worldDirX * sx + worldDirY * sy;

        float invLayerCount = 1.0 / float(layerCount);
        float firstZNorm = invLayerCount - 1.0;
        float angle = firstZNorm * (viewAngle * 0.5) + (PI * 0.25);
        float sinAngle = sin(angle);
        float cosAngle = cos(angle);
        float angleStep = viewAngle * invLayerCount;
        float sinStep = sin(angleStep);
        float cosStep = cos(angleStep);

        float integralSkyScale = max(0.0, asfloat(meta.highlightPadding0));
        float integralHitEmissiveBoost = max(0.0, asfloat(meta.highlightPadding1));
        float integralLogMergeK = max(0.0, asfloat(meta.highlightPadding2));

        float3 hitAccumColor = float3(0.0, 0.0, 0.0);
        float hitAccumWeight = 0.0;
        float3 skyAccumColor = float3(0.0, 0.0, 0.0);
        float skyAccumWeight = 0.0;
        float maxEntityDistance = max(1.0, meta.maxTraceDistance);
        bool evaluateEntityTetra = wd.totalNumTetrahedrons > 0u;
        uint hitMaterial = 0u;
        uint hitOrientation = 0u;
        float4 hitPos = float4(0.0, 0.0, 0.0, 0.0);
        int4 hitChunkCoord = int4(0, 0, 0, 0);
        uint hitLodLevel = 0u;
        uint missReason = MISS_REASON_NONE;
        uint debugFlags = 0u;
        uint chunkStepsTaken = 0u;
        uint remainingVoxelSteps = 0u;
        float finalT = 0.0;
        int4 lastChunkCoord = int4(0, 0, 0, 0);
        uint nodeVisitCount = 0u;
        uint pathHash = 0u;
        for (uint sampleLayer = 0u; sampleLayer < layerCount; sampleLayer++)
        {
            float4 worldDirection = normalize(worldXY + worldDirZ * cosAngle + worldDirW * sinAngle);
            bool hit = traceVoxelRayDDAWithLodRings(
                worldOrigin,
                worldDirection,
                meta,
                hitMaterial,
                hitOrientation,
                hitPos,
                hitChunkCoord,
                hitLodLevel,
                missReason,
                debugFlags,
                chunkStepsTaken,
                remainingVoxelSteps,
                finalT,
                lastChunkCoord,
                nodeVisitCount,
                pathHash);

            if (worldBvhRayDiagSelected && !worldBvhRayDiagWritten && sampleLayer == centerLayer)
            {
                float hitT = hit ? dot(hitPos - worldOrigin, worldDirection) : -1.0;
                writeWorldBvhRayDiagRecord(
                    worldBvhRayDiagSlot,
                    globalId.x,
                    globalId.y,
                    sampleLayer,
                    hit,
                    missReason,
                    hitMaterial,
                    hitChunkCoord,
                    hitT,
                    chunkStepsTaken,
                    remainingVoxelSteps,
                    nodeVisitCount,
                    pathHash,
                    debugFlags);
                worldBvhRayDiagWritten = true;
            }

            EntityTetraHit entityHit = entityTetraHitNone();
            if (evaluateEntityTetra)
            {
                float nearestVoxelDistance = hit
                    ? max(1e-4, dot(hitPos - worldOrigin, worldDirection))
                    : maxEntityDistance;
                bool forceEntityLinear = (meta.debugFlags & META_DEBUG_FLAG_ENTITY_LINEAR_ONLY) != 0u;
                bool doEntityBvhCompare = VTE_DIAGNOSTICS
                    && (meta.debugFlags & META_DEBUG_FLAG_ENTITY_BVH_COMPARE) != 0u
                    && (((globalId.x + 3u * globalId.y + 5u * sampleLayer) & 15u) == 0u);
                if (doEntityBvhCompare)
                {
                    EntityTetraHit linearHit = traceEntityTetraLinear(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance);
                    EntityTetraHit linearReverseHit = traceEntityTetraLinearReverse(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance);
                    EntityTetraHit bvhHit = traceEntityTetraBVH(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance,
                        true);
                    EntityTetraHit bvhNoPruneHit = traceEntityTetraBVH(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance,
                        false);
                    EntityTetraHit bvhNoAabbHit = traceEntityTetraBVHNoAabb(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance);
                    EntityTetraHit bvhLeafArrayHit = traceEntityTetraBVHLeafArray(
                        worldOrigin,
                        worldDirection,
                        wd.totalNumTetrahedrons,
                        nearestVoxelDistance);
                    incrementCompareStat(COMPARE_STAT_ENTITY_BVH_SAMPLE);

                    bool hitStateMismatch = (linearHit.didHit != bvhHit.didHit);
                    bool materialMismatch = false;
                    bool distanceMismatch = false;
                    bool tetraMismatch = false;
                    if (!hitStateMismatch && linearHit.didHit && bvhHit.didHit)
                    {
                        materialMismatch = linearHit.materialId != bvhHit.materialId;
                        float distDelta = abs(linearHit.hitDistance - bvhHit.hitDistance);
                        float distTolerance = max(1e-3, 1e-3 * max(linearHit.hitDistance, bvhHit.hitDistance));
                        distanceMismatch = distDelta > distTolerance;
                        tetraMismatch = linearHit.tetrahedronIndex != bvhHit.tetrahedronIndex;
                    }
                    if (hitStateMismatch || materialMismatch || distanceMismatch || tetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MISMATCH);
                    }
                    if (hitStateMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH);
                        if (!bvhHit.didHit && linearHit.didHit)
                        {
                            incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT);
                        }
                        else if (bvhHit.didHit && !linearHit.didHit)
                        {
                            incrementCompareStat(COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS);
                        }
                    }
                    if (materialMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH);
                    }
                    if (distanceMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH);
                    }
                    if (tetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH);
                    }
                    bool noPruneHitStateMismatch = (linearHit.didHit != bvhNoPruneHit.didHit);
                    bool noPruneDistanceMismatch = false;
                    bool noPruneTetraMismatch = false;
                    if (!noPruneHitStateMismatch && linearHit.didHit && bvhNoPruneHit.didHit)
                    {
                        float noPruneDistDelta = abs(linearHit.hitDistance - bvhNoPruneHit.hitDistance);
                        float noPruneDistTolerance = max(
                            1e-3,
                            1e-3 * max(linearHit.hitDistance, bvhNoPruneHit.hitDistance));
                        noPruneDistanceMismatch = noPruneDistDelta > noPruneDistTolerance;
                        noPruneTetraMismatch = linearHit.tetrahedronIndex != bvhNoPruneHit.tetrahedronIndex;
                    }
                    if (noPruneHitStateMismatch || noPruneDistanceMismatch || noPruneTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH);
                    }
                    if (noPruneHitStateMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH);
                    }
                    if (noPruneDistanceMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH);
                    }
                    if (noPruneTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH);
                    }
                    bool noAabbHitStateMismatch = (linearHit.didHit != bvhNoAabbHit.didHit);
                    bool noAabbDistanceMismatch = false;
                    bool noAabbTetraMismatch = false;
                    if (!noAabbHitStateMismatch && linearHit.didHit && bvhNoAabbHit.didHit)
                    {
                        float noAabbDistDelta = abs(linearHit.hitDistance - bvhNoAabbHit.hitDistance);
                        float noAabbDistTolerance = max(
                            1e-3,
                            1e-3 * max(linearHit.hitDistance, bvhNoAabbHit.hitDistance));
                        noAabbDistanceMismatch = noAabbDistDelta > noAabbDistTolerance;
                        noAabbTetraMismatch = linearHit.tetrahedronIndex != bvhNoAabbHit.tetrahedronIndex;
                    }
                    if (noAabbHitStateMismatch || noAabbDistanceMismatch || noAabbTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH);
                    }
                    if (noAabbHitStateMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH);
                    }
                    if (noAabbDistanceMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH);
                    }
                    if (noAabbTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH);
                    }
                    bool linearOrderHitStateMismatch = (linearHit.didHit != linearReverseHit.didHit);
                    bool linearOrderDistanceMismatch = false;
                    bool linearOrderTetraMismatch = false;
                    if (!linearOrderHitStateMismatch && linearHit.didHit && linearReverseHit.didHit)
                    {
                        float linearOrderDistDelta = abs(linearHit.hitDistance - linearReverseHit.hitDistance);
                        float linearOrderDistTolerance = max(
                            1e-3,
                            1e-3 * max(linearHit.hitDistance, linearReverseHit.hitDistance));
                        linearOrderDistanceMismatch = linearOrderDistDelta > linearOrderDistTolerance;
                        linearOrderTetraMismatch =
                            linearHit.tetrahedronIndex != linearReverseHit.tetrahedronIndex;
                    }
                    if (linearOrderHitStateMismatch
                        || linearOrderDistanceMismatch
                        || linearOrderTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH);
                    }
                    if (linearOrderHitStateMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH);
                    }
                    if (linearOrderDistanceMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH);
                    }
                    if (linearOrderTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH);
                    }
                    bool leafArrayHitStateMismatch = (linearHit.didHit != bvhLeafArrayHit.didHit);
                    bool leafArrayDistanceMismatch = false;
                    bool leafArrayTetraMismatch = false;
                    if (!leafArrayHitStateMismatch && linearHit.didHit && bvhLeafArrayHit.didHit)
                    {
                        float leafArrayDistDelta = abs(linearHit.hitDistance - bvhLeafArrayHit.hitDistance);
                        float leafArrayDistTolerance = max(
                            1e-3,
                            1e-3 * max(linearHit.hitDistance, bvhLeafArrayHit.hitDistance));
                        leafArrayDistanceMismatch = leafArrayDistDelta > leafArrayDistTolerance;
                        leafArrayTetraMismatch = linearHit.tetrahedronIndex != bvhLeafArrayHit.tetrahedronIndex;
                    }
                    if (leafArrayHitStateMismatch || leafArrayDistanceMismatch || leafArrayTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH);
                    }
                    if (leafArrayHitStateMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH);
                    }
                    if (leafArrayDistanceMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH);
                    }
                    if (leafArrayTetraMismatch)
                    {
                        incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH);
                    }
                    entityHit = forceEntityLinear ? linearHit : bvhHit;
                }
                else
                {
                    entityHit = forceEntityLinear
                        ? traceEntityTetraLinear(
                            worldOrigin,
                            worldDirection,
                            wd.totalNumTetrahedrons,
                            nearestVoxelDistance)
                        : traceEntityTetra(
                            worldOrigin,
                            worldDirection,
                            wd.totalNumTetrahedrons,
                            nearestVoxelDistance);
                }
            }

            recordStageABreakdown(
                meta,
                stageABreakdownSampled,
                evaluateEntityTetra,
                entityHit.didHit,
                hit,
                chunkStepsTaken,
                remainingVoxelSteps,
                nodeVisitCount);

            if (entityHit.didHit)
            {
                MaterialProperties mat = sampleMaterial(
                    entityHit.materialId,
                    entityHit.textureCoordinates,
                    wd.timeTicksMs);
                float w = max(1e-4, mat.integrationWeight);
                float hitBrightnessScale = (entityHit.materialId == MATERIAL_ID_GROUND)
                    ? integralSkyScale
                    : 1.0;
                float3 hitColor = shadeVoxelMaterial(mat, entityHit.hitNormal)
                    + mat.albedo.xyz * integralHitEmissiveBoost;
                hitColor *= hitBrightnessScale;
                hitColor = maybeApplyZwAngleColorShift(
                    hitColor,
                    sampleLayer,
                    layerCount,
                    wd,
                    meta);
                hitAccumColor += hitColor * w;
                hitAccumWeight += w;
            }
            else if (hit)
            {
                int4 hitVoxelCoord = hitVoxelCoordFromHitPos(hitPos, worldDirection);
                uint highlightKind = highlightKindForVoxel(hitVoxelCoord, meta);
                if (highlightKind != 0u)
                {
                    float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
                    MaterialProperties mat = sampleMaterial(
                        hitMaterial,
                        voxelMaterialSampleCoords(hitPos, worldDirection, hitNormal, hitOrientation),
                        wd.timeTicksMs);
                    float3 litColor = shadeVoxelMaterial(mat, hitNormal);
                    float3 highlighted = shadeHighlightedVoxelSample(
                        litColor,
                        hitVoxelCoord,
                        meta,
                        hitPos,
                        worldDirection,
                        hitNormal,
                        highlightKind);
                    highlighted = maybeApplyLodTint(highlighted, hitLodLevel, meta);
                    float w = max(1e-4, mat.integrationWeight * INTEGRAL_HIGHLIGHT_WEIGHT_BOOST);
                    float hitBrightnessScale = (hitMaterial == MATERIAL_ID_GROUND)
                        ? integralSkyScale
                        : 1.0;
                    float3 sampleColor =
                        (highlighted + mat.albedo.xyz * integralHitEmissiveBoost)
                        * hitBrightnessScale;
                    sampleColor = maybeApplyZwAngleColorShift(
                        sampleColor,
                        sampleLayer,
                        layerCount,
                        wd,
                        meta);
                    hitAccumColor += sampleColor * w;
                    hitAccumWeight += w;
                }
                else
                {
                    float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
                    MaterialProperties mat = sampleMaterial(
                        hitMaterial,
                        voxelMaterialSampleCoords(hitPos, worldDirection, hitNormal, hitOrientation),
                        wd.timeTicksMs);
                    float w = max(1e-4, mat.integrationWeight);
                    float hitBrightnessScale = (hitMaterial == MATERIAL_ID_GROUND)
                        ? integralSkyScale
                        : 1.0;
                    float3 hitColor = shadeVoxelMaterial(mat, hitNormal)
                        + mat.albedo.xyz * integralHitEmissiveBoost;
                    hitColor = maybeApplyLodTint(hitColor, hitLodLevel, meta);
                    hitColor *= hitBrightnessScale;
                    hitColor = maybeApplyZwAngleColorShift(
                        hitColor,
                        sampleLayer,
                        layerCount,
                        wd,
                        meta);
                    hitAccumColor += hitColor * w;
                    hitAccumWeight += w;
                }
            }
            else
            {
                float a = 0.5 + 0.5 * worldDirection.y;
                float3 sky = lerp(float3(0.02, 0.03, 0.06), float3(0.16, 0.22, 0.40), saturate(a));
                sky *= integralSkyScale;
                sky = maybeApplyZwAngleColorShift(
                    sky,
                    sampleLayer,
                    layerCount,
                    wd,
                    meta);
                skyAccumColor += sky;
                skyAccumWeight += 1.0;
            }

            float nextCos = cosAngle * cosStep - sinAngle * sinStep;
            float nextSin = sinAngle * cosStep + cosAngle * sinStep;
            cosAngle = nextCos;
            sinAngle = nextSin;
        }

        float totalWeight = hitAccumWeight + skyAccumWeight;
        float3 collapsedColor = totalWeight > 1e-6
            ? (hitAccumColor + skyAccumColor) / totalWeight
            : float3(0.0, 0.0, 0.0);
        if (integralLogMergeK > 1e-6 && totalWeight > 1e-6)
        {
            float3 hitMean = hitAccumWeight > 1e-6
                ? hitAccumColor / hitAccumWeight
                : float3(0.0, 0.0, 0.0);
            float3 skyMean = skyAccumWeight > 1e-6
                ? skyAccumColor / skyAccumWeight
                : float3(0.0, 0.0, 0.0);
            float hitFraction = hitAccumWeight / totalWeight;
            float blend = log(1.0 + integralLogMergeK * hitFraction) / log(1.0 + integralLogMergeK);
            collapsedColor = lerp(skyMean, hitMean, saturate(blend));
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

    float sx = pixelPos.x / wd.focalLengthXY;
    float sy = (-pixelPos.y / aspectRatio) / wd.focalLengthXY;
    float4 worldOrigin = wd.worldOrigin;
    float4 worldDirection = normalize(
        wd.worldDirX * sx
        + wd.worldDirY * sy
        + wd.worldDirZ * cos(zwAngle)
        + wd.worldDirW * sin(zwAngle)
    );

    uint hitMaterial = 0;
    uint hitOrientation = 0;
    float4 hitPos = float4(0.0, 0.0, 0.0, 0.0);
    int4 hitChunkCoord = int4(0, 0, 0, 0);
    uint hitLodLevel = 0u;
    uint missReason = MISS_REASON_NONE;
    uint debugFlags = 0u;
    uint chunkStepsTaken = 0u;
    uint remainingVoxelSteps = 0u;
    float finalT = 0.0;
    int4 lastChunkCoord = int4(0, 0, 0, 0);
    uint nodeVisitCount = 0u;
    uint pathHash = 0u;
    bool hit = traceVoxelRayDDAWithLodRings(
        worldOrigin,
        worldDirection,
        meta,
        hitMaterial,
        hitOrientation,
        hitPos,
        hitChunkCoord,
        hitLodLevel,
        missReason,
        debugFlags,
        chunkStepsTaken,
        remainingVoxelSteps,
        finalT,
        lastChunkCoord,
        nodeVisitCount,
        pathHash);

    if (worldBvhRayDiagSelected)
    {
        float hitT = hit ? dot(hitPos - worldOrigin, worldDirection) : -1.0;
        writeWorldBvhRayDiagRecord(
            worldBvhRayDiagSlot,
            globalId.x,
            globalId.y,
            globalId.z,
            hit,
            missReason,
            hitMaterial,
            hitChunkCoord,
            hitT,
            chunkStepsTaken,
            remainingVoxelSteps,
            nodeVisitCount,
            pathHash,
            debugFlags);
    }

    bool evaluateEntityTetra = wd.totalNumTetrahedrons > 0u;
    EntityTetraHit entityHit = entityTetraHitNone();
    if (evaluateEntityTetra)
    {
        float maxEntityDistance = max(1.0, meta.maxTraceDistance);
        if (hit)
        {
            maxEntityDistance = max(1e-4, dot(hitPos - worldOrigin, worldDirection));
        }
        bool forceEntityLinear = (meta.debugFlags & META_DEBUG_FLAG_ENTITY_LINEAR_ONLY) != 0u;
        bool doEntityBvhCompare = VTE_DIAGNOSTICS
            && (meta.debugFlags & META_DEBUG_FLAG_ENTITY_BVH_COMPARE) != 0u
            && (((globalId.x + 3u * globalId.y) & 15u) == 0u);
        if (doEntityBvhCompare)
        {
            EntityTetraHit linearHit = traceEntityTetraLinear(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance);
            EntityTetraHit linearReverseHit = traceEntityTetraLinearReverse(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance);
            EntityTetraHit bvhHit = traceEntityTetraBVH(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance,
                true);
            EntityTetraHit bvhNoPruneHit = traceEntityTetraBVH(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance,
                false);
            EntityTetraHit bvhNoAabbHit = traceEntityTetraBVHNoAabb(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance);
            EntityTetraHit bvhLeafArrayHit = traceEntityTetraBVHLeafArray(
                worldOrigin,
                worldDirection,
                wd.totalNumTetrahedrons,
                maxEntityDistance);
            incrementCompareStat(COMPARE_STAT_ENTITY_BVH_SAMPLE);

            bool hitStateMismatch = (linearHit.didHit != bvhHit.didHit);
            bool materialMismatch = false;
            bool distanceMismatch = false;
            bool tetraMismatch = false;
            if (!hitStateMismatch && linearHit.didHit && bvhHit.didHit)
            {
                materialMismatch = linearHit.materialId != bvhHit.materialId;
                float distDelta = abs(linearHit.hitDistance - bvhHit.hitDistance);
                float distTolerance = max(1e-3, 1e-3 * max(linearHit.hitDistance, bvhHit.hitDistance));
                distanceMismatch = distDelta > distTolerance;
                tetraMismatch = linearHit.tetrahedronIndex != bvhHit.tetrahedronIndex;
            }
            if (hitStateMismatch || materialMismatch || distanceMismatch || tetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MISMATCH);
            }
            if (hitStateMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH);
                if (!bvhHit.didHit && linearHit.didHit)
                {
                    incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT);
                }
                else if (bvhHit.didHit && !linearHit.didHit)
                {
                    incrementCompareStat(COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS);
                }
            }
            if (materialMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH);
            }
            if (distanceMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH);
            }
            if (tetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH);
            }
            bool noPruneHitStateMismatch = (linearHit.didHit != bvhNoPruneHit.didHit);
            bool noPruneDistanceMismatch = false;
            bool noPruneTetraMismatch = false;
            if (!noPruneHitStateMismatch && linearHit.didHit && bvhNoPruneHit.didHit)
            {
                float noPruneDistDelta = abs(linearHit.hitDistance - bvhNoPruneHit.hitDistance);
                float noPruneDistTolerance = max(
                    1e-3,
                    1e-3 * max(linearHit.hitDistance, bvhNoPruneHit.hitDistance));
                noPruneDistanceMismatch = noPruneDistDelta > noPruneDistTolerance;
                noPruneTetraMismatch = linearHit.tetrahedronIndex != bvhNoPruneHit.tetrahedronIndex;
            }
            if (noPruneHitStateMismatch || noPruneDistanceMismatch || noPruneTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH);
            }
            if (noPruneHitStateMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH);
            }
            if (noPruneDistanceMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH);
            }
            if (noPruneTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH);
            }
            bool noAabbHitStateMismatch = (linearHit.didHit != bvhNoAabbHit.didHit);
            bool noAabbDistanceMismatch = false;
            bool noAabbTetraMismatch = false;
            if (!noAabbHitStateMismatch && linearHit.didHit && bvhNoAabbHit.didHit)
            {
                float noAabbDistDelta = abs(linearHit.hitDistance - bvhNoAabbHit.hitDistance);
                float noAabbDistTolerance = max(
                    1e-3,
                    1e-3 * max(linearHit.hitDistance, bvhNoAabbHit.hitDistance));
                noAabbDistanceMismatch = noAabbDistDelta > noAabbDistTolerance;
                noAabbTetraMismatch = linearHit.tetrahedronIndex != bvhNoAabbHit.tetrahedronIndex;
            }
            if (noAabbHitStateMismatch || noAabbDistanceMismatch || noAabbTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH);
            }
            if (noAabbHitStateMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH);
            }
            if (noAabbDistanceMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH);
            }
            if (noAabbTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH);
            }
            bool linearOrderHitStateMismatch = (linearHit.didHit != linearReverseHit.didHit);
            bool linearOrderDistanceMismatch = false;
            bool linearOrderTetraMismatch = false;
            if (!linearOrderHitStateMismatch && linearHit.didHit && linearReverseHit.didHit)
            {
                float linearOrderDistDelta = abs(linearHit.hitDistance - linearReverseHit.hitDistance);
                float linearOrderDistTolerance = max(
                    1e-3,
                    1e-3 * max(linearHit.hitDistance, linearReverseHit.hitDistance));
                linearOrderDistanceMismatch = linearOrderDistDelta > linearOrderDistTolerance;
                linearOrderTetraMismatch = linearHit.tetrahedronIndex != linearReverseHit.tetrahedronIndex;
            }
            if (linearOrderHitStateMismatch || linearOrderDistanceMismatch || linearOrderTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH);
            }
            if (linearOrderHitStateMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH);
            }
            if (linearOrderDistanceMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH);
            }
            if (linearOrderTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH);
            }
            bool leafArrayHitStateMismatch = (linearHit.didHit != bvhLeafArrayHit.didHit);
            bool leafArrayDistanceMismatch = false;
            bool leafArrayTetraMismatch = false;
            if (!leafArrayHitStateMismatch && linearHit.didHit && bvhLeafArrayHit.didHit)
            {
                float leafArrayDistDelta = abs(linearHit.hitDistance - bvhLeafArrayHit.hitDistance);
                float leafArrayDistTolerance = max(
                    1e-3,
                    1e-3 * max(linearHit.hitDistance, bvhLeafArrayHit.hitDistance));
                leafArrayDistanceMismatch = leafArrayDistDelta > leafArrayDistTolerance;
                leafArrayTetraMismatch = linearHit.tetrahedronIndex != bvhLeafArrayHit.tetrahedronIndex;
            }
            if (leafArrayHitStateMismatch || leafArrayDistanceMismatch || leafArrayTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH);
            }
            if (leafArrayHitStateMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH);
            }
            if (leafArrayDistanceMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH);
            }
            if (leafArrayTetraMismatch)
            {
                incrementCompareStat(COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH);
            }
            entityHit = forceEntityLinear ? linearHit : bvhHit;
        }
        else
        {
            entityHit = forceEntityLinear
                ? traceEntityTetraLinear(
                    worldOrigin,
                    worldDirection,
                    wd.totalNumTetrahedrons,
                    maxEntityDistance)
                : traceEntityTetra(
                    worldOrigin,
                    worldDirection,
                    wd.totalNumTetrahedrons,
                    maxEntityDistance);
        }
    }

    recordStageABreakdown(
        meta,
        stageABreakdownSampled,
        evaluateEntityTetra,
        entityHit.didHit,
        hit,
        chunkStepsTaken,
        remainingVoxelSteps,
        nodeVisitCount);

    float3 outColor;
    float outAlpha = 1.0;
    if (VTE_DIAGNOSTICS
        && (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL))
    {
        bool doReferenceCompare = (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_COMPARE) != 0u;
        bool mismatchOnly = (meta.debugFlags & META_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY) != 0u;
        if (doReferenceCompare)
        {
            uint refHitMaterial = 0;
            float4 refHitPos = float4(0.0, 0.0, 0.0, 0.0);
            int4 refHitChunkCoord = int4(0, 0, 0, 0);
            uint refNodeVisitCount = 0u;
            uint refPathHash = 0u;
            bool refHit = traceVoxelRayReferenceVisible(
                worldOrigin,
                worldDirection,
                meta,
                refHitMaterial,
                refHitPos,
                refHitChunkCoord,
                refNodeVisitCount,
                refPathHash);

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
                outColor = maybeApplyLodTint(outColor, hitLodLevel, meta);
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
                outColor = maybeApplyLodTint(outColor, hitLodLevel, meta);
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
    else if (entityHit.didHit)
    {
        MaterialProperties mat = sampleMaterial(
            entityHit.materialId,
            entityHit.textureCoordinates,
            wd.timeTicksMs);
        outColor = shadeVoxelMaterial(mat, entityHit.hitNormal);
    }
    else if (hit)
    {
        int4 hitVoxelCoord = hitVoxelCoordFromHitPos(hitPos, worldDirection);
        uint highlightKind = highlightKindForVoxel(hitVoxelCoord, meta);
        if (highlightKind != 0u)
        {
            float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
            MaterialProperties mat = sampleMaterial(
                hitMaterial,
                voxelMaterialSampleCoords(hitPos, worldDirection, hitNormal, hitOrientation),
                wd.timeTicksMs);
            float3 litColor = shadeVoxelMaterial(mat, hitNormal);
            outColor = shadeHighlightedVoxelSample(
                litColor,
                hitVoxelCoord,
                meta,
                hitPos,
                worldDirection,
                hitNormal,
                highlightKind);
            outColor = maybeApplyLodTint(outColor, hitLodLevel, meta);
        }
        else
        {
            float4 hitNormal = estimateVoxelHitNormal(hitPos, worldDirection);
            MaterialProperties mat = sampleMaterial(
                hitMaterial,
                voxelMaterialSampleCoords(hitPos, worldDirection, hitNormal, hitOrientation),
                wd.timeTicksMs);
            outColor = shadeVoxelMaterial(mat, hitNormal);
            outColor = maybeApplyLodTint(outColor, hitLodLevel, meta);
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
void mainVoxelTraceStageAIntegralFused(uint3 globalId : SV_DispatchThreadID)
{
    voxelTraceStageAImpl(globalId, true);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainVoxelTraceStageALayered(uint3 globalId : SV_DispatchThreadID)
{
    voxelTraceStageAImpl(globalId, false);
}

float decodeZwAngleColorShiftStrength(WorkingData wd)
{
    uint q = (wd.padding[0] >> WORKING_ZW_SHIFT_STRENGTH_SHIFT) & 0xFFu;
    return float(q) / 255.0;
}

float zwLayerCoord(uint layer, uint layerCount)
{
    if (layerCount <= 1u)
    {
        return 0.0;
    }
    float t = float(layer) / float(layerCount - 1u);
    return t * 2.0 - 1.0;
}

float3 maybeApplyZwAngleColorShift(
    float3 color,
    uint layer,
    uint layerCount,
    WorkingData wd,
    VoxelFrameMeta meta)
{
    if ((wd.padding[0] & WORKING_FLAG_ZW_ANGLE_COLOR_SHIFT) == 0u)
    {
        return color;
    }
    if (meta.stageBMode == STAGE_B_MODE_DEBUG_COMPARE || meta.stageBMode == STAGE_B_MODE_DEBUG_INTEGRAL)
    {
        return color;
    }

    float strength = decodeZwAngleColorShiftStrength(wd);
    if (strength <= 0.0)
    {
        return color;
    }

    float coord = zwLayerCoord(layer, layerCount);
    float redBoost = saturate(-coord) * strength;
    float blueBoost = saturate(coord) * strength;
    float neutralFade = 1.0 - 0.12 * abs(coord) * strength;
    return float3(
        color.x * (1.0 + redBoost),
        color.y * neutralFade,
        color.z * (1.0 + blueBoost)
    );
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
        color = maybeApplyZwAngleColorShift(color, centerLayer, layers, wd, meta);
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
            float3 sample = pixelBuffer[pixelBase + z * layerStride].xyz;
            sample = maybeApplyZwAngleColorShift(sample, z, layers, wd, meta);
            accum += sample;
            sampleCount++;
        }

        if (sampleCount > 0)
        {
            color = accum / float(sampleCount);
        }
    }

    pixelBuffer[pixelBase] = float4(color, 1.0);
}
