// GPU BVH (Linear Bounding Volume Hierarchy) construction and traversal
// Implements LBVH with Karras 2012 parallel tree construction

import types;

// Explicit bindings to match Rust descriptor layout:
// Set 0: OneTimeBuffers (not needed for BVH)

// Set 1: SizedBuffers
[[vk::binding(1, 1)]] RWStructuredBuffer<Tetrahedron> outputTetrahedrons;
[[vk::binding(3, 1)]] RWStructuredBuffer<MortonCode> mortonCodes;
[[vk::binding(4, 1)]] RWStructuredBuffer<BVHNode> bvhNodes;
[[vk::binding(5, 1)]] RWStructuredBuffer<SceneBounds> sceneBoundsBuffer;

// Set 2: LiveBuffers
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

// Shared memory for parallel reduction
groupshared float4 sharedMin[64];
groupshared float4 sharedMax[64];

// ============================================================================
// Phase 1: Compute Scene Bounds (Parallel Reduction)
// ============================================================================

// First pass: reduce within workgroups and write partial results
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHSceneBounds(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numTetrahedrons = workingData.totalNumTetrahedrons;
    uint tid = localId.x;
    uint gid = globalId.x;

    // Initialize with extreme values
    float4 localMin = float4(1e30, 1e30, 1e30, 1e30);
    float4 localMax = float4(-1e30, -1e30, -1e30, -1e30);

    // Each thread processes multiple tetrahedrons (stride = workgroup size)
    for (uint i = gid; i < numTetrahedrons; i += 64)
    {
        Tetrahedron tet = outputTetrahedrons[i];

        // Check all 4 vertices
        [ForceUnroll]
        for (int v = 0; v < 4; v++)
        {
            float4 pos = tet.vertexPositions[v];
            localMin = min(localMin, pos);
            localMax = max(localMax, pos);
        }
    }

    // Store in shared memory
    sharedMin[tid] = localMin;
    sharedMax[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction in shared memory
    [ForceUnroll]
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedMin[tid] = min(sharedMin[tid], sharedMin[tid + s]);
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // First thread writes result
    if (tid == 0)
    {
        // Atomic min/max operations - use atomicMin/Max on uint reinterpretation
        // For simplicity, we use a single workgroup approach here
        // The dispatch should use only 1 workgroup, or chain multiple passes
        sceneBoundsBuffer[0].minBounds = sharedMin[0];
        sceneBoundsBuffer[0].maxBounds = sharedMax[0];
    }
}

// ============================================================================
// Phase 2: Compute Morton Codes
// ============================================================================

// Expand a 16-bit integer to 64 bits by inserting 3 zeros after each bit
uint64_t expandBits16to64(uint v)
{
    uint64_t x = uint64_t(v) & 0xFFFF;
    x = (x | (x << 24)) & 0x000000FF000000FF;
    x = (x | (x << 12)) & 0x000F000F000F000F;
    x = (x | (x <<  6)) & 0x0303030303030303;
    x = (x | (x <<  3)) & 0x1111111111111111;
    return x;
}

// Compute 64-bit 4D Morton code from normalized coordinates
uint64_t morton4D(float4 normalized)
{
    // Convert to 16-bit integers (0-65535 range)
    uint x = uint(clamp(normalized.x, 0.0, 1.0) * 65535.0);
    uint y = uint(clamp(normalized.y, 0.0, 1.0) * 65535.0);
    uint z = uint(clamp(normalized.z, 0.0, 1.0) * 65535.0);
    uint w = uint(clamp(normalized.w, 0.0, 1.0) * 65535.0);

    // Interleave bits: w,z,y,x pattern for each bit position
    uint64_t mx = expandBits16to64(x);
    uint64_t my = expandBits16to64(y) << 1;
    uint64_t mz = expandBits16to64(z) << 2;
    uint64_t mw = expandBits16to64(w) << 3;

    return mx | my | mz | mw;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHMortonCodes(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numTetrahedrons = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    if (idx >= numTetrahedrons)
    {
        // Write sentinel values for bitonic sort padding (positions N..N_pow2-1)
        // Use ~uint64_t(0) to ensure a true 64-bit max value
        MortonCode sentinel;
        sentinel.code = ~uint64_t(0);
        sentinel.tetrahedronIndex = BVH_INVALID_INDEX;
        sentinel.padding = 0;
        mortonCodes[idx] = sentinel;
        return;
    }

    Tetrahedron tet = outputTetrahedrons[idx];
    SceneBounds bounds = sceneBoundsBuffer[0];

    // Compute tetrahedron centroid
    float4 centroid = (tet.vertexPositions[0] + tet.vertexPositions[1] +
                       tet.vertexPositions[2] + tet.vertexPositions[3]) * 0.25;

    // Normalize to [0,1] range based on scene bounds
    float4 extent = bounds.maxBounds - bounds.minBounds;
    float4 normalized = (centroid - bounds.minBounds) / max(extent, float4(1e-10));

    // Compute Morton code
    MortonCode mc;
    mc.code = morton4D(normalized);
    mc.tetrahedronIndex = idx;
    mc.padding = 0;

    mortonCodes[idx] = mc;
}

// ============================================================================
// Phase 3: Bitonic Sort
// ============================================================================

// Push constants for bitonic sort parameters
struct BitonicSortParams
{
    uint stage;     // Current stage of bitonic sort
    uint step;      // Current step within stage
    uint count;     // Total number of elements (must be power of 2)
    uint padding;
}

[[vk::push_constant]]
BitonicSortParams sortParams;

// Shared memory for local sorting (workgroup size = 64)
groupshared MortonCode sharedMorton[64];

// Phase 3a: Local sort - sort each 64-element block entirely in shared memory
// Performs stages 0-5 (all steps) without any global memory barriers
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHBitonicSortLocal(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    uint gid = globalId.x;
    uint tid = localId.x;

    // Load from global memory into shared memory
    sharedMorton[tid] = mortonCodes[gid];
    GroupMemoryBarrierWithGroupSync();

    // Full bitonic sort for 64 elements (stages 0-5, 21 total steps)
    // Direction uses global index so blocks alternate ascending/descending
    for (uint stage = 0; stage <= 5; stage++)
    {
        for (int step = int(stage); step >= 0; step--)
        {
            uint stepSize = 1u << uint(step);
            uint partner = tid ^ stepSize;

            if (tid < partner)
            {
                bool ascending = ((gid >> (stage + 1)) & 1) == 0;
                bool needSwap = ascending
                    ? (sharedMorton[tid].code > sharedMorton[partner].code)
                    : (sharedMorton[tid].code < sharedMorton[partner].code);

                if (needSwap)
                {
                    MortonCode tmp = sharedMorton[tid];
                    sharedMorton[tid] = sharedMorton[partner];
                    sharedMorton[partner] = tmp;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write back to global memory
    mortonCodes[gid] = sharedMorton[tid];
}

// Phase 3b: Global step - one compare-exchange step via global memory
// Used for steps where stepSize >= 64 (cross-workgroup comparisons)
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHBitonicSort(uint3 globalId : SV_DispatchThreadID)
{
    uint idx = globalId.x;
    uint count = sortParams.count;

    if (idx >= count)
        return;

    uint stage = sortParams.stage;
    uint step = sortParams.step;

    // Compute partner index using XOR (standard bitonic sort comparator network)
    uint stepSize = 1u << step;
    uint partner = idx ^ stepSize;

    // Only process if partner is valid and we're the "lower" index
    if (partner >= count || idx > partner)
        return;

    // Determine if we're in ascending or descending part of the bitonic sequence
    // Each block of size 2^(stage+1) alternates direction
    bool ascending = ((idx >> (stage + 1)) & 1) == 0;

    // Compare-exchange
    MortonCode a = mortonCodes[idx];
    MortonCode b = mortonCodes[partner];

    bool needSwap = ascending ? (a.code > b.code) : (a.code < b.code);

    if (needSwap)
    {
        mortonCodes[idx] = b;
        mortonCodes[partner] = a;
    }
}

// Phase 3c: Local merge - perform steps 5-0 of a global stage in shared memory
// After the global steps (stepSize >= 64) of a stage, the remaining steps
// (stepSize < 64) can be done in shared memory since both elements of each
// compare-exchange pair are in the same 64-element block.
// All elements in a workgroup share the same ascending/descending direction
// because the direction block size 2^(stage+1) >= 128 > 64 for stage >= 6.
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHBitonicSortLocalMerge(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    uint gid = globalId.x;
    uint tid = localId.x;
    uint stage = sortParams.stage;

    // Load from global memory
    sharedMorton[tid] = mortonCodes[gid];
    GroupMemoryBarrierWithGroupSync();

    // Direction is uniform across the workgroup for stage >= 6
    bool ascending = ((gid >> (stage + 1)) & 1) == 0;

    // Perform steps 5 down to 0
    for (int step = 5; step >= 0; step--)
    {
        uint stepSize = 1u << uint(step);
        uint partner = tid ^ stepSize;

        if (tid < partner)
        {
            bool needSwap = ascending
                ? (sharedMorton[tid].code > sharedMorton[partner].code)
                : (sharedMorton[tid].code < sharedMorton[partner].code);

            if (needSwap)
            {
                MortonCode tmp = sharedMorton[tid];
                sharedMorton[tid] = sharedMorton[partner];
                sharedMorton[partner] = tmp;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write back to global memory
    mortonCodes[gid] = sharedMorton[tid];
}

// ============================================================================
// Phase 4: Build Tree Structure (Karras 2012 Algorithm)
// ============================================================================

// Compute longest common prefix between Morton codes at indices i and j
int delta(uint i, uint j, uint numLeaves)
{
    if (j >= numLeaves)
        return -1;

    uint64_t codeI = mortonCodes[i].code;
    uint64_t codeJ = mortonCodes[j].code;

    if (codeI == codeJ)
    {
        // If codes are equal, use index as tiebreaker
        // Count leading zeros of XOR of indices
        uint xorIdx = i ^ j;
        return 64 + int(31 - firstbithigh(xorIdx));
    }

    // Count leading zeros of XOR
    uint64_t xorVal = codeI ^ codeJ;

    // Count leading zeros in 64-bit value
    uint high = uint(xorVal >> 32);
    uint low = uint(xorVal & 0xFFFFFFFF);

    if (high != 0)
        return int(31 - firstbithigh(high));
    else if (low != 0)
        return int(63 - firstbithigh(low));
    else
        return 64;
}

// Initialize leaf nodes - separate dispatch to avoid race conditions with internal node construction
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHInitLeaves(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    // Handle edge cases
    if (numLeaves == 0)
        return;

    if (numLeaves == 1)
    {
        // Single tetrahedron: root is the only leaf
        if (idx == 0)
        {
            BVHNode node;
            node.minBounds = float4(0);
            node.maxBounds = float4(0);
            node.leftChild = BVH_INVALID_INDEX;
            node.rightChild = BVH_INVALID_INDEX;
            node.parent = BVH_INVALID_INDEX;
            node.isLeaf = 1;
            node.tetrahedronIndex = mortonCodes[0].tetrahedronIndex;
            node.atomicVisitCount = 0;
            node.padding[0] = 0;
            node.padding[1] = 0;
            bvhNodes[0] = node;
        }
        return;
    }

    if (idx >= numLeaves)
        return;

    uint numInternalNodes = numLeaves - 1;
    uint leafIdx = numInternalNodes + idx;

    // Initialize leaf node.
    bvhNodes[leafIdx].minBounds = float4(0);
    bvhNodes[leafIdx].maxBounds = float4(0);
    bvhNodes[leafIdx].leftChild = BVH_INVALID_INDEX;
    bvhNodes[leafIdx].rightChild = BVH_INVALID_INDEX;
    bvhNodes[leafIdx].parent = BVH_INVALID_INDEX;
    bvhNodes[leafIdx].isLeaf = 1;
    bvhNodes[leafIdx].tetrahedronIndex = mortonCodes[idx].tetrahedronIndex;
    bvhNodes[leafIdx].atomicVisitCount = 0;
    bvhNodes[leafIdx].padding[0] = 0;
    bvhNodes[leafIdx].padding[1] = 0;
}

// Build internal nodes - runs after leaves are initialized
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHBuildTree(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    // Handle edge cases - single leaf is already handled in leaf init
    if (numLeaves <= 1)
        return;

    uint numInternalNodes = numLeaves - 1;

    // Build internal nodes (indices 0 to numInternalNodes - 1)
    if (idx >= numInternalNodes)
        return;

    // Determine direction of the range
    int d = (delta(idx, idx + 1, numLeaves) > delta(idx, idx - 1, numLeaves)) ? 1 : -1;

    // Compute upper bound for the length of the range
    int deltaMin = delta(idx, idx - d, numLeaves);
    int lMax = 2;
    while (delta(idx, idx + lMax * d, numLeaves) > deltaMin)
    {
        lMax *= 2;
    }

    // Binary search for the actual length
    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
    {
        if (delta(idx, idx + (l + t) * d, numLeaves) > deltaMin)
        {
            l += t;
        }
    }

    int j = int(idx) + l * d;

    // Find the split position using Karras' binary search refinement.
    // The (s + t) probe naturally clamps because delta(idx, j) == deltaNode,
    // so the final step cannot cross the range endpoint.
    int deltaNode = delta(idx, j, numLeaves);
    int s = 0;
    int t2 = l;
    while (t2 > 1)
    {
        t2 = (t2 + 1) / 2;
        int splitIdx = int(idx) + (s + t2) * d;
        if (delta(idx, splitIdx, numLeaves) > deltaNode)
        {
            s += t2;
        }
    }

    int gamma = int(idx) + s * d + min(d, 0);

    // Determine child indices
    uint leftChild, rightChild;

    if (min(int(idx), j) == gamma)
    {
        // Left child is a leaf
        leftChild = numInternalNodes + uint(gamma);
    }
    else
    {
        // Left child is internal
        leftChild = uint(gamma);
    }

    if (max(int(idx), j) == gamma + 1)
    {
        // Right child is a leaf
        rightChild = numInternalNodes + uint(gamma + 1);
    }
    else
    {
        // Right child is internal
        rightChild = uint(gamma + 1);
    }

    // Create internal node
    BVHNode node;
    node.minBounds = float4(0);
    node.maxBounds = float4(0);
    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.parent = BVH_INVALID_INDEX;
    node.isLeaf = 0;
    node.atomicVisitCount = 0;
    node.padding[0] = 0;
    node.padding[1] = 0;

    node.tetrahedronIndex = 0;

    bvhNodes[idx] = node;

    // Explicitly reset atomicVisitCount with atomic operation to ensure it takes effect
    uint dummy;
    InterlockedExchange(bvhNodes[idx].atomicVisitCount, 0, dummy);

    // Parent links are populated in a dedicated pass (`mainBVHLinkParents`).
    // Keep this pass write-local to avoid cross-thread child-node races.
}

void computeTetrahedronAabb(uint tetIdx, out float4 minB, out float4 maxB)
{
    Tetrahedron tet = outputTetrahedrons[tetIdx];
    minB = min(min(tet.vertexPositions[0], tet.vertexPositions[1]),
               min(tet.vertexPositions[2], tet.vertexPositions[3]));
    maxB = max(max(tet.vertexPositions[0], tet.vertexPositions[1]),
               max(tet.vertexPositions[2], tet.vertexPositions[3]));
}

// ============================================================================
// Phase 5: Link parent pointers
// ============================================================================

[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHLinkParents(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    if (numLeaves == 0)
        return;

    if (numLeaves == 1)
    {
        if (idx == 0)
        {
            bvhNodes[0].parent = BVH_INVALID_INDEX;
            uint dummy;
            InterlockedExchange(bvhNodes[0].atomicVisitCount, 0, dummy);
        }
        return;
    }

    uint numInternalNodes = numLeaves - 1;
    if (idx >= numInternalNodes)
        return;

    uint totalNodes = numInternalNodes + numLeaves;

    uint dummy;
    InterlockedExchange(bvhNodes[idx].atomicVisitCount, 0, dummy);
    if (idx == 0)
    {
        bvhNodes[idx].parent = BVH_INVALID_INDEX;
    }

    uint left = bvhNodes[idx].leftChild;
    uint right = bvhNodes[idx].rightChild;

    if (left < totalNodes)
    {
        bvhNodes[left].parent = idx;
    }
    if (right < totalNodes)
    {
        bvhNodes[right].parent = idx;
    }
}

// ============================================================================
// Phase 6: Compute leaf AABBs and propagate to root in one dispatch
// ============================================================================

// Legacy leaf-AABB kernel kept for diagnostics/back-compat; the runtime path now
// uses `mainBVHPropagateAABBs` directly.
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHComputeLeafAABBs(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    if (numLeaves == 0 || idx >= numLeaves)
        return;

    if (numLeaves == 1)
    {
        if (idx == 0)
        {
            float4 minB, maxB;
            computeTetrahedronAabb(bvhNodes[0].tetrahedronIndex, minB, maxB);
            bvhNodes[0].minBounds = minB;
            bvhNodes[0].maxBounds = maxB;
        }
        return;
    }

    uint numInternalNodes = numLeaves - 1;
    uint leafIdx = numInternalNodes + idx;

    float4 minB, maxB;
    computeTetrahedronAabb(bvhNodes[leafIdx].tetrahedronIndex, minB, maxB);
    bvhNodes[leafIdx].minBounds = minB;
    bvhNodes[leafIdx].maxBounds = maxB;
}

// Each leaf thread computes its own AABB, then climbs parent links.
// The second child to arrive at each parent computes that internal-node AABB and continues.
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHPropagateAABBs(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    if (numLeaves == 0 || idx >= numLeaves)
        return;

    if (numLeaves == 1)
    {
        if (idx == 0)
        {
            float4 minB, maxB;
            computeTetrahedronAabb(bvhNodes[0].tetrahedronIndex, minB, maxB);
            bvhNodes[0].minBounds = minB;
            bvhNodes[0].maxBounds = maxB;
            uint dummy;
            InterlockedExchange(bvhNodes[0].atomicVisitCount, 2, dummy);
        }
        return;
    }

    uint numInternalNodes = numLeaves - 1;
    uint leafIdx = numInternalNodes + idx;
    uint tetIdx = bvhNodes[leafIdx].tetrahedronIndex;
    if (tetIdx >= numLeaves)
        return;

    float4 leafMin, leafMax;
    computeTetrahedronAabb(tetIdx, leafMin, leafMax);
    bvhNodes[leafIdx].minBounds = leafMin;
    bvhNodes[leafIdx].maxBounds = leafMax;
    DeviceMemoryBarrier();

    uint nodeIdx = leafIdx;
    while (true)
    {
        uint parentIdx = bvhNodes[nodeIdx].parent;
        if (parentIdx == BVH_INVALID_INDEX)
            break;

        uint priorCount;
        InterlockedAdd(bvhNodes[parentIdx].atomicVisitCount, 1u, priorCount);
        if (priorCount != 1u)
        {
            break; // First child arrived (or already processed) - sibling will continue upward.
        }

        uint left = bvhNodes[parentIdx].leftChild;
        uint right = bvhNodes[parentIdx].rightChild;
        if (left == BVH_INVALID_INDEX || right == BVH_INVALID_INDEX)
        {
            break;
        }

        DeviceMemoryBarrier();
        bvhNodes[parentIdx].minBounds = min(bvhNodes[left].minBounds, bvhNodes[right].minBounds);
        bvhNodes[parentIdx].maxBounds = max(bvhNodes[left].maxBounds, bvhNodes[right].maxBounds);
        DeviceMemoryBarrier();

        nodeIdx = parentIdx;
    }
}
