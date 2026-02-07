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
    uint count;     // Total number of elements
    uint padding;
}

[[vk::push_constant]]
BitonicSortParams sortParams;

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

    // Initialize leaf node - parent will be set by internal node construction
    bvhNodes[leafIdx].minBounds = float4(0);
    bvhNodes[leafIdx].maxBounds = float4(0);
    bvhNodes[leafIdx].leftChild = BVH_INVALID_INDEX;
    bvhNodes[leafIdx].rightChild = BVH_INVALID_INDEX;
    bvhNodes[leafIdx].parent = BVH_INVALID_INDEX; // Will be overwritten by internal node construction
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

    // Find the split position
    // NOTE: gamma must be in range [min(idx,j), max(idx,j) - 1] to leave room for right subtree
    // This means s must be in range [0, l - 1], so we use maxS = l - 1
    int deltaNode = delta(idx, j, numLeaves);
    int s = 0;
    int maxS = max(l - 1, 0);  // Prevent s from reaching l (which would put gamma out of range)

    for (int t2 = (maxS + 1) / 2; t2 >= 1; t2 = (t2 + 1) / 2)
    {
        int testS = s + t2;
        if (testS <= maxS)
        {
            int splitIdx = int(idx) + testS * d;
            if (delta(idx, splitIdx, numLeaves) > deltaNode)
            {
                s = testS;
            }
        }
        if (t2 == 1) break;
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

    // Set parent pointers
    bvhNodes[leftChild].parent = idx;
    bvhNodes[rightChild].parent = idx;
}

// ============================================================================
// Phase 5: Compute AABBs Bottom-Up (Split into two dispatches for proper sync)
// ============================================================================

// Phase 5a: Compute leaf AABBs only (separate dispatch)
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
        // Single node case - leaf is at index 0
        if (idx == 0)
        {
            uint tetIdx = bvhNodes[0].tetrahedronIndex;
            Tetrahedron tet = outputTetrahedrons[tetIdx];

            float4 minB = min(min(tet.vertexPositions[0], tet.vertexPositions[1]),
                              min(tet.vertexPositions[2], tet.vertexPositions[3]));
            float4 maxB = max(max(tet.vertexPositions[0], tet.vertexPositions[1]),
                              max(tet.vertexPositions[2], tet.vertexPositions[3]));

            bvhNodes[0].minBounds = minB;
            bvhNodes[0].maxBounds = maxB;
        }
        return;
    }

    uint numInternalNodes = numLeaves - 1;
    uint leafIdx = numInternalNodes + idx;

    // Compute leaf AABB from tetrahedron
    uint tetIdx = bvhNodes[leafIdx].tetrahedronIndex;
    Tetrahedron tet = outputTetrahedrons[tetIdx];

    float4 minB = min(min(tet.vertexPositions[0], tet.vertexPositions[1]),
                      min(tet.vertexPositions[2], tet.vertexPositions[3]));
    float4 maxB = max(max(tet.vertexPositions[0], tet.vertexPositions[1]),
                      max(tet.vertexPositions[2], tet.vertexPositions[3]));

    bvhNodes[leafIdx].minBounds = minB;
    bvhNodes[leafIdx].maxBounds = maxB;
}

// Phase 5b: Propagate AABBs from leaves to root
// Uses atomicVisitCount as a validity marker (2 = both children processed, AABB is valid)
// Must be called log2(N) times to fully propagate to root
// Each dispatch processes all internal nodes whose children have both been marked as valid
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBVHPropagateAABBs(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    uint numLeaves = workingData.totalNumTetrahedrons;
    uint idx = globalId.x;

    // Edge cases
    if (numLeaves <= 1)
        return;

    uint numInternalNodes = numLeaves - 1;

    // Each thread handles one internal node
    if (idx >= numInternalNodes)
        return;

    // Check if this node already has a valid AABB (atomicVisitCount >= 2)
    // Use atomic load to ensure we see the latest value
    uint myCount;
    InterlockedAdd(bvhNodes[idx].atomicVisitCount, 0, myCount);
    if (myCount >= 2)
        return; // Already processed

    // Check if both children have valid AABBs
    uint left = bvhNodes[idx].leftChild;
    uint right = bvhNodes[idx].rightChild;

    if (left == BVH_INVALID_INDEX || right == BVH_INVALID_INDEX)
        return;

    // Use the isLeaf field directly instead of index comparison
    // This is more reliable as it's explicitly set during tree construction
    bool leftValid;
    if (bvhNodes[left].isLeaf != 0) {
        leftValid = true; // Leaf - always valid (AABB computed in leaf dispatch)
    } else {
        uint leftCount;
        InterlockedAdd(bvhNodes[left].atomicVisitCount, 0, leftCount);
        leftValid = (leftCount >= 2);
    }

    bool rightValid;
    if (bvhNodes[right].isLeaf != 0) {
        rightValid = true; // Leaf - always valid
    } else {
        uint rightCount;
        InterlockedAdd(bvhNodes[right].atomicVisitCount, 0, rightCount);
        rightValid = (rightCount >= 2);
    }

    if (!leftValid || !rightValid)
        return; // Children not ready yet, will try again in next dispatch

    // Both children are valid - compute this node's AABB
    float4 leftMin = bvhNodes[left].minBounds;
    float4 leftMax = bvhNodes[left].maxBounds;
    float4 rightMin = bvhNodes[right].minBounds;
    float4 rightMax = bvhNodes[right].maxBounds;

    bvhNodes[idx].minBounds = min(leftMin, rightMin);
    bvhNodes[idx].maxBounds = max(leftMax, rightMax);

    // Mark this node as valid using atomic exchange to ensure visibility
    uint oldVal;
    InterlockedExchange(bvhNodes[idx].atomicVisitCount, 2, oldVal);
}

