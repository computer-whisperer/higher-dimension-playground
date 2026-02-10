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

struct VoxelFrameMeta
{
    uint chunkCount;
    uint visibleChunkCount;
    uint occupancyWordCount;
    uint materialWordCount;
    uint maxTraceSteps;
    uint chunkLookupCapacity;
    uint padding0;
    uint padding1;
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

void advanceDDA4(inout int4 coord, int4 step, inout float4 tMax, float4 tDelta, out float nextT)
{
    // Step all tied axes at the same boundary crossing distance.
    // This avoids traversal instability when rays pass through voxel/chunk edges/corners.
    nextT = min4(tMax);
    float eps = max(1e-5, abs(nextT) * 1e-5);

    if (abs(tMax.x - nextT) <= eps)
    {
        tMax.x += tDelta.x;
        coord.x += step.x;
    }
    if (abs(tMax.y - nextT) <= eps)
    {
        tMax.y += tDelta.y;
        coord.y += step.y;
    }
    if (abs(tMax.z - nextT) <= eps)
    {
        tMax.z += tDelta.z;
        coord.z += step.z;
    }
    if (abs(tMax.w - nextT) <= eps)
    {
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
    out float4 hitPos)
{
    hitMaterial = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    if (remainingVoxelSteps == 0 || tExit < tEnter)
    {
        return false;
    }

    const float START_EPS = 1e-4;
    float tStart = max(0.0, tEnter + START_EPS);
    if (tStart > tExit)
    {
        return false;
    }

    int4 chunkMin = chunkCoord * int(CHUNK_SIZE);
    int4 chunkMax = chunkMin + int(CHUNK_SIZE);

    float4 startPos = rayOrigin + rayDir * tStart;
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
        float worldT = tStart + localT;
        if (worldT > tExit)
        {
            break;
        }

        remainingVoxelSteps--;
        uint materialId = 0;
        if (sampleVoxelInChunk(voxelCoord, chunkMin, h, meta, materialId))
        {
            hitMaterial = materialId;
            hitPos = rayOrigin + rayDir * worldT;
            return true;
        }

        float nextLocalT = 0.0;
        advanceDDA4(voxelCoord, step, tMaxLocal, tDelta, nextLocalT);
        localT = max(localT, nextLocalT);

        if (!voxelInChunk(voxelCoord, chunkMin, chunkMax))
        {
            break;
        }
    }
    return false;
}

bool traceVoxelRayDDA(
    float4 rayOrigin,
    float4 rayDir,
    VoxelFrameMeta meta,
    out uint hitMaterial,
    out float4 hitPos)
{
    hitMaterial = 0;
    hitPos = float4(0.0, 0.0, 0.0, 0.0);
    if (meta.chunkCount == 0 || meta.visibleChunkCount == 0)
    {
        return false;
    }

    const float MAX_DISTANCE = 160.0;
    uint maxChunkSteps = max(1u, min(meta.maxTraceSteps, 4096u));
    uint remainingVoxelSteps = max(1u, min(meta.maxTraceSteps * 8u, 32768u));

    int4 startVoxel = int4(
        int(floor(rayOrigin.x)),
        int(floor(rayOrigin.y)),
        int(floor(rayOrigin.z)),
        int(floor(rayOrigin.w))
    );
    int4 chunkCoord = int4(
        floorDiv(startVoxel.x, int(CHUNK_SIZE)),
        floorDiv(startVoxel.y, int(CHUNK_SIZE)),
        floorDiv(startVoxel.z, int(CHUNK_SIZE)),
        floorDiv(startVoxel.w, int(CHUNK_SIZE))
    );
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
    float4 tMaxChunk = float4(1e30, 1e30, 1e30, 1e30);
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

    float currentT = 0.0;
    for (uint chunkStepIdx = 0; chunkStepIdx < maxChunkSteps; chunkStepIdx++)
    {
        if (currentT > MAX_DISTANCE || remainingVoxelSteps == 0)
        {
            break;
        }

        float chunkExitT = min(MAX_DISTANCE, min4(tMaxChunk));
        chunkExitT = max(chunkExitT, currentT);

        uint chunkIndex = findChunkIndex(chunkCoord, meta);
        if (chunkIndex != INVALID_INDEX)
        {
            VoxelChunkHeader h = voxelChunkHeaders[chunkIndex];
            uint chunkMaterial = 0;
            float4 chunkHitPos = float4(0.0, 0.0, 0.0, 0.0);
            float aabbEnter = 0.0;
            float aabbExit = -1.0;
            if (intersectChunkAABB(rayOrigin, rayDir, chunkCoord, aabbEnter, aabbExit))
            {
                float tEnter = max(currentT, aabbEnter);
                float tExit = min(chunkExitT, aabbExit);
                if (tExit >= tEnter)
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
                        chunkHitPos))
                    {
                        hitMaterial = chunkMaterial;
                        hitPos = chunkHitPos;
                        return true;
                    }
                }
            }
        }

        float nextChunkT = 0.0;
        advanceDDA4(chunkCoord, chunkStep, tMaxChunk, tDeltaChunk, nextChunkT);
        currentT = max(currentT, nextChunkT);
    }
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
    bool hit = traceVoxelRayDDA(worldOrigin, worldDirection, meta, hitMaterial, hitPos);

    float3 outColor;
    if (hit)
    {
        MaterialProperties mat = sampleMaterial(hitMaterial, frac(hitPos.xyz));
        outColor = mat.albedo.xyz * 0.75 + mat.albedo.xyz * mat.luminance;
    }
    else
    {
        float a = 0.5 + 0.5 * worldDirection.y;
        outColor = lerp(float3(0.02, 0.03, 0.06), float3(0.16, 0.22, 0.40), saturate(a));
    }

    uint index = globalId.z * wd.renderDimensions.x * wd.renderDimensions.y
               + globalId.y * wd.renderDimensions.x
               + globalId.x;
    pixelBuffer[index] = float4(outColor, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainVoxelDisplayStageB(uint3 globalId : SV_DispatchThreadID)
{
    // Stage B currently relies on existing present shader accumulation across Z slices.
}
