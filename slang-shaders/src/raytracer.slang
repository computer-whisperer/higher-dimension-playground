// 4D Raytracer compute shaders

import math;
import types;
import materials;

// Explicit bindings to match Rust descriptor layout:
// Set 0: OneTimeBuffers
[[vk::binding(0, 0)]] StructuredBuffer<ModelTetrahedron> modelTetrahedrons;

// Set 1: SizedBuffers
[[vk::binding(1, 1)]] RWStructuredBuffer<Tetrahedron> outputTetrahedrons;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> pixelBuffer;
[[vk::binding(4, 1)]] StructuredBuffer<BVHNode> bvhNodes;

// Set 2: LiveBuffers
[[vk::binding(0, 2)]] StructuredBuffer<ModelInstance> instances;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

// Simple RNG for monte-carlo sampling
struct BasicRNG
{
    uint64_t state;

    __init(uint64_t seed)
    {
        state = seed;
    }

    [mutating]
    uint64_t rand()
    {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    [mutating]
    float randFloat()
    {
        return float(rand() & 0xFFFFFF) / float(0xFFFFFF);
    }

    [mutating]
    float4 randFloat4()
    {
        return float4(
            randFloat() * 2.0 - 1.0,
            randFloat() * 2.0 - 1.0,
            randFloat() * 2.0 - 1.0,
            randFloat() * 2.0 - 1.0
        );
    }
}

// Vertex shader stage for compute preprocessing
struct CSVertexOutput
{
    float4 vertexPosition;
    float4 texturePosition;
}

CSVertexOutput csVertexShader(
    float4 vertexPosition,
    float4 texturePosition,
    ModelInstance instance,
    WorkingData workingData)
{
    // Extend to 5D homogeneous coordinates
    VecN<5> pos;
    pos[0] = vertexPosition.x;
    pos[1] = vertexPosition.y;
    pos[2] = vertexPosition.z;
    pos[3] = vertexPosition.w;
    pos[4] = 1.0;

    // Apply model transform
    VecN<5> viewPos = instance.modelTransform * pos;

    CSVertexOutput output;
    output.vertexPosition = float4(viewPos[0], viewPos[1], viewPos[2], viewPos[3]);
    output.texturePosition = texturePosition;
    return output;
}

// Tetrahedron preprocessor - transforms model tetrahedra to view space
[shader("compute")]
[numthreads(64, 1, 1)]
void mainRaytracerTetrahedronPreprocessor(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint outputIndex = globalId.x;
    uint modelCount, modelStride;
    modelTetrahedrons.GetDimensions(modelCount, modelStride);
    uint modelIndex = outputIndex % modelCount;
    uint instanceIndex = outputIndex / modelCount;

    ModelInstance instance = instances[instanceIndex];
    ModelTetrahedron inputTet = modelTetrahedrons[modelIndex];

    float4 outputVertexPositions[4];
    float4 outputTexturePositions[4];

    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        CSVertexOutput result = csVertexShader(
            inputTet.vertexPositions[i],
            inputTet.texturePositions[i],
            instance,
            workingData);
        outputVertexPositions[i] = result.vertexPosition;
        outputTexturePositions[i] = result.texturePosition;
    }

    // Compute 4D normal
    float4 normal = getNormal4D(
        outputVertexPositions[0],
        outputVertexPositions[1],
        outputVertexPositions[2],
        outputVertexPositions[3]);

    Tetrahedron outTet;
    outTet.vertexPositions = outputVertexPositions;
    outTet.texturePositions = outputTexturePositions;
    outTet.materialId = instance.cellMaterialIds[inputTet.cellId];
    outTet.normal = normal;
    outTet.padding[0] = 0;
    outTet.padding[1] = 0;
    outTet.padding[2] = 0;

    outputTetrahedrons[outputIndex] = outTet;
}

// Clear the pixel buffer
[shader("compute")]
[numthreads(8, 8, 1)]
void mainRaytracerClear(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint2 pixelPos = globalId.xy;
    for (uint z = 0; z < workingData.renderDimensions.z; z++)
    {
        uint index = z * workingData.renderDimensions.x * workingData.renderDimensions.y
                   + pixelPos.y * workingData.renderDimensions.x
                   + pixelPos.x;
        pixelBuffer[index] = float4(0);
    }
}

// 4D ray-AABB intersection test
// Returns (tMin, tMax) where tMin >= 0 for valid hit, or (-1, -1) if no intersection
float2 rayAABBIntersect4D(float4 origin, float4 invDir, float4 minB, float4 maxB)
{
    float4 t1 = (minB - origin) * invDir;
    float4 t2 = (maxB - origin) * invDir;

    float4 tmin4 = min(t1, t2);
    float4 tmax4 = max(t1, t2);

    float tmin = max(max(tmin4.x, tmin4.y), max(tmin4.z, tmin4.w));
    float tmax = min(min(tmax4.x, tmax4.y), min(tmax4.z, tmax4.w));

    // No intersection if ray exits before entering or exits behind origin
    if (tmax < 0.0 || tmin > tmax)
        return float2(-1.0, -1.0);

    // If tmin < 0, origin is inside box - use 0 as entry point
    return float2(max(tmin, 0.0), tmax);
}

// Test a single tetrahedron for ray intersection
bool testTetrahedronHit(
    float4 start,
    float4 direction,
    uint tetIndex,
    inout RayHit closestHit)
{
    Tetrahedron tet = outputTetrahedrons[tetIndex];

    if (tet.materialId == 0)
        return false;

    // Check if ray is parallel to tetrahedron hyperplane
    float value = dot(tet.normal, direction);
    if (abs(value) < 0.001)
        return false;

    // Construct matrix for intersection calculation
    // [d, v1-v0, v2-v0, v3-v0] * [t, u, v, w]^T = start - v0
    float4x4 mat = float4x4(
        direction,
        tet.vertexPositions[1] - tet.vertexPositions[0],
        tet.vertexPositions[2] - tet.vertexPositions[0],
        tet.vertexPositions[3] - tet.vertexPositions[0]
    );

    float4x4 invMat = inverse4x4(mat);
    float4 result = mul(invMat, start - tet.vertexPositions[0]);

    // Barycentric coordinates
    float4 barycentric = float4(
        1.0 - (result[1] + result[2] + result[3]),
        result[1],
        result[2],
        result[3]
    );

    float hitDistance = -result[0];

    // Check if hit is valid (in front of ray, inside tetrahedron)
    if (hitDistance > 0.0 &&
        barycentric.x >= 0.0 && barycentric.y >= 0.0 &&
        barycentric.z >= 0.0 && barycentric.w >= 0.0)
    {
        if (closestHit.didHit == 0 || closestHit.hitDistance > hitDistance)
        {
            float4 hitPosition = hitDistance * direction + start;
            float4 hitNormal = dot(tet.normal, direction) < 0.0
                ? tet.normal
                : -tet.normal;

            closestHit.tetrahedronIndex = tetIndex;
            closestHit.hitPosition = hitPosition;
            closestHit.hitDistance = hitDistance;
            closestHit.barycentric = barycentric;
            closestHit.hitNormal = hitNormal;
            closestHit.didHit = 1;
            return true;
        }
    }
    return false;
}

// Set to true to use linear O(n) traversal instead of BVH (for debugging)
static const bool USE_LINEAR_TRAVERSAL = false;

// Linear traversal (original O(n) algorithm)
RayHit raycastLinear(
    float4 start,
    float4 direction,
    uint numTetrahedrons)
{
    direction = normalize(direction);
    RayHit closestHit = RayHit.none();

    for (uint i = 0; i < numTetrahedrons; i++)
    {
        testTetrahedronHit(start, direction, i, closestHit);
    }

    // Interpolate texture coordinates for closest hit
    if (closestHit.didHit != 0)
    {
        Tetrahedron tet = outputTetrahedrons[closestHit.tetrahedronIndex];
        closestHit.textureCoordinates =
            closestHit.barycentric.x * tet.texturePositions[0] +
            closestHit.barycentric.y * tet.texturePositions[1] +
            closestHit.barycentric.z * tet.texturePositions[2] +
            closestHit.barycentric.w * tet.texturePositions[3];
    }

    return closestHit;
}

// 4D ray-tetrahedron intersection using BVH traversal
RayHit raycast(
    float4 start,
    float4 direction,
    uint numTetrahedrons)
{
    direction = normalize(direction);
    RayHit closestHit = RayHit.none();

    // Handle edge cases
    if (numTetrahedrons == 0)
        return closestHit;

    // Debug mode: use linear traversal
    if (USE_LINEAR_TRAVERSAL)
    {
        return raycastLinear(start, direction, numTetrahedrons);
    }

    // Precompute inverse direction for AABB tests (preserve sign for near-zero values)
    float4 invDir = float4(
        abs(direction.x) > 1e-8 ? 1.0 / direction.x : (direction.x >= 0.0 ? 1e8 : -1e8),
        abs(direction.y) > 1e-8 ? 1.0 / direction.y : (direction.y >= 0.0 ? 1e8 : -1e8),
        abs(direction.z) > 1e-8 ? 1.0 / direction.z : (direction.z >= 0.0 ? 1e8 : -1e8),
        abs(direction.w) > 1e-8 ? 1.0 / direction.w : (direction.w >= 0.0 ? 1e8 : -1e8)
    );

    // DEBUG: Check root node structure
    uint numInternalNodes = numTetrahedrons - 1;
    uint totalNodes = 2 * numTetrahedrons - 1;
    BVHNode root = bvhNodes[0];

    // Check if root children are valid
    if (root.leftChild >= totalNodes || root.rightChild >= totalNodes)
    {
        // Invalid children - return red with child index encoded
        closestHit.didHit = 1;
        closestHit.hitDistance = 1.0;
        closestHit.tetrahedronIndex = 0xFF000000 | ((root.leftChild & 0xFFF) << 12) | (root.rightChild & 0xFFF);
        return closestHit;
    }

    // Check if root AABB is valid (not all zeros)
    float rootAABBSize = length(root.maxBounds - root.minBounds);
    if (rootAABBSize < 0.001)
    {
        // Root AABB is too small - return green
        closestHit.didHit = 1;
        closestHit.hitDistance = 1.0;
        closestHit.tetrahedronIndex = 0xFF00FF00;
        return closestHit;
    }

    // DEBUG: Check if ray hits root AABB
    float2 rootHitT = rayAABBIntersect4D(start, invDir, root.minBounds, root.maxBounds);
    if (rootHitT.x < 0.0)
    {
        // Ray doesn't hit root AABB - return blue with size indicator
        // This means either ray is outside scene or AABB is wrong
        closestHit.didHit = 1;
        closestHit.hitDistance = 1.0;
        uint sizeIndicator = uint(clamp(rootAABBSize, 0.0, 255.0));
        closestHit.tetrahedronIndex = 0xFF0000FF | (sizeIndicator << 8);
        return closestHit;
    }

    // BVH traversal

    // Stack-based BVH traversal
    uint stack[BVH_STACK_SIZE];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // Start at root

    while (stackPtr > 0)
    {
        uint nodeIdx = stack[--stackPtr];
        BVHNode node = bvhNodes[nodeIdx];

        // Test ray against node AABB
        float2 hitT = rayAABBIntersect4D(start, invDir, node.minBounds, node.maxBounds);

        // Skip if no intersection or if intersection is behind closest hit
        if (hitT.x < 0.0)
            continue;

        if (closestHit.didHit != 0 && hitT.x > closestHit.hitDistance)
            continue;

        if (node.isLeaf != 0)
        {
            // Leaf node - test tetrahedron
            testTetrahedronHit(start, direction, node.tetrahedronIndex, closestHit);
        }
        else
        {
            // Internal node - push children to stack
            // Push farther child first so closer child is processed first
            uint left = node.leftChild;
            uint right = node.rightChild;

            if (left != BVH_INVALID_INDEX && right != BVH_INVALID_INDEX)
            {
                BVHNode leftNode = bvhNodes[left];
                BVHNode rightNode = bvhNodes[right];

                float2 leftT = rayAABBIntersect4D(start, invDir, leftNode.minBounds, leftNode.maxBounds);
                float2 rightT = rayAABBIntersect4D(start, invDir, rightNode.minBounds, rightNode.maxBounds);

                // Push in order so closer child is popped first
                if (leftT.x > rightT.x)
                {
                    if (leftT.x >= 0.0 && stackPtr < BVH_STACK_SIZE)
                        stack[stackPtr++] = left;
                    if (rightT.x >= 0.0 && stackPtr < BVH_STACK_SIZE)
                        stack[stackPtr++] = right;
                }
                else
                {
                    if (rightT.x >= 0.0 && stackPtr < BVH_STACK_SIZE)
                        stack[stackPtr++] = right;
                    if (leftT.x >= 0.0 && stackPtr < BVH_STACK_SIZE)
                        stack[stackPtr++] = left;
                }
            }
            else if (left != BVH_INVALID_INDEX && stackPtr < BVH_STACK_SIZE)
            {
                stack[stackPtr++] = left;
            }
            else if (right != BVH_INVALID_INDEX && stackPtr < BVH_STACK_SIZE)
            {
                stack[stackPtr++] = right;
            }
        }
    }

    // Interpolate texture coordinates for closest hit
    if (closestHit.didHit != 0)
    {
        Tetrahedron tet = outputTetrahedrons[closestHit.tetrahedronIndex];
        closestHit.textureCoordinates =
            closestHit.barycentric.x * tet.texturePositions[0] +
            closestHit.barycentric.y * tet.texturePositions[1] +
            closestHit.barycentric.z * tet.texturePositions[2] +
            closestHit.barycentric.w * tet.texturePositions[3];
    }

    return closestHit;
}

// Path trace with bounces
float4 raycastSample(
    float4 start,
    float4 direction,
    uint numTetrahedrons,
    inout BasicRNG rng)
{
    direction = normalize(direction);

    const int MAX_BOUNCES = 6;
    RayHit hitStack[MAX_BOUNCES];
    int numHits = 0;

    while (numHits < MAX_BOUNCES)
    {
        RayHit hit = raycast(start, direction, numTetrahedrons);

        // DEBUG: Check for debug hit indicating BVH issues
        // All debug values are >= 0xFF000000 (high bit set)
        if (hit.didHit != 0 && hit.tetrahedronIndex >= 0xFF000000)
        {
            if (hit.tetrahedronIndex == 0xFFFFFFFE)
                return float4(0.0, 1.0, 0.0, 1.0); // Green = leaf parent not set
            else if (hit.tetrahedronIndex == 0xFFFFFFFD)
                return float4(1.0, 1.0, 0.0, 1.0); // Yellow = leaf AABB zero
            else if (hit.tetrahedronIndex == 0xFFFFFFFC)
                return float4(0.0, 1.0, 1.0, 1.0); // Cyan = root has no children
            else if (hit.tetrahedronIndex == 0xFFFFFFFB)
                return float4(1.0, 0.0, 1.0, 1.0); // Magenta = L's children valid but L not marked
            else if (hit.tetrahedronIndex == 0xFFFFFFFA)
                return float4(1.0, 0.5, 0.0, 1.0); // Orange = right child not valid
            else if (hit.tetrahedronIndex == 0xFFFFFFF9)
                return float4(0.5, 0.0, 1.0, 1.0); // Purple = left child AABB zero
            else if (hit.tetrahedronIndex == 0xFFFFFFF8)
                return float4(0.0, 0.5, 0.5, 1.0); // Dark cyan = right child AABB zero
            else if (hit.tetrahedronIndex == 0xFFFFFFF7)
                return float4(1.0, 1.0, 1.0, 1.0); // White = child index out of bounds
            else if (hit.tetrahedronIndex == 0xFFFFFFF6)
                return float4(1.0, 0.5, 0.5, 1.0); // Pink = LL has internal children (problem deeper)
            else if (hit.tetrahedronIndex == 0xFFFFFFF5)
                return float4(0.5, 0.5, 1.0, 1.0); // Light blue = LR internal and not valid
            else if (hit.tetrahedronIndex == 0xFFFFFFF4)
                return float4(0.5, 0.5, 0.5, 1.0); // Gray = L has BVH_INVALID_INDEX children
            else if (hit.tetrahedronIndex == 0xFFFFFFF3)
                return float4(1.0, 0.0, 0.5, 1.0); // Bright magenta = L's children are leaves but L not valid
            else if (hit.tetrahedronIndex == 0xFFFFFFF2)
                return float4(0.8, 0.4, 0.0, 1.0); // Brown = LL's children are leaves but LL not valid
            else if (hit.tetrahedronIndex == 0xFFFFFFF1)
                return float4(0.0, 1.0, 0.5, 1.0); // Bright green = isLeaf NEVER set (critical bug)
            else if (hit.tetrahedronIndex == 0xFFFFFFF0)
                return float4(0.0, 0.8, 0.8, 1.0); // Teal = first leaf missing isLeaf
            else if (hit.tetrahedronIndex == 0xFFFFFFE2)
                return float4(0.0, 0.0, 0.5, 1.0); // Dark blue = child indices out of range
            else if (hit.tetrahedronIndex == 0xFFFFFFE1)
                return float4(0.5, 0.0, 0.0, 1.0); // Dark red = last internal has 2 internal children
            else if (hit.tetrahedronIndex == 0xFFFFFFE0)
                return float4(0.0, 0.5, 0.0, 1.0); // Dark green = last internal 2 leaves but not propagated
            else if (hit.tetrahedronIndex == 0xFFFFFFD0)
                return float4(1.0, 1.0, 0.5, 1.0); // Light yellow = propagation shader never ran
            else if (hit.tetrahedronIndex == 0xFF00FFFF)
                return float4(0.0, 1.0, 1.0, 1.0); // Bright cyan = absolute debug marker
            else
            {
                // Generic debug: decode RGB from the index value
                float r = float((hit.tetrahedronIndex >> 16) & 0xFF) / 255.0;
                float g = float((hit.tetrahedronIndex >> 8) & 0xFF) / 255.0;
                float b = float(hit.tetrahedronIndex & 0xFF) / 255.0;
                return float4(r, g, b, 1.0);
            }
        }

        if (hit.didHit != 0)
        {
            Tetrahedron tet = outputTetrahedrons[hit.tetrahedronIndex];
            MaterialProperties mat = sampleMaterial(tet.materialId, hit.textureCoordinates.xyz);

            // Compute reflection
            float4 reflection = direction - 2.0 * dot(direction, hit.hitNormal) * hit.hitNormal;

            // Choose between specular and diffuse based on metallic
            if (rng.randFloat() < mat.metallic)
            {
                direction = reflection;
            }
            else
            {
                direction = normalize(rng.randFloat4() + hit.hitNormal);
            }

            // Add roughness
            direction = direction + rng.randFloat4() * mat.roughness * 0.01;
            direction = normalize(direction);

            // Move origin past hit point
            start = hit.hitPosition + 0.001 * direction;

            hitStack[numHits] = hit;
            numHits++;
        }
        else
        {
            break;
        }
    }

    // Compute lighting from hits
    float3 lightValue = float3(0);
    float4 backgroundLightDir = normalize(float4(0.3, 1.0, -0.3, 0.0));

    if (numHits < MAX_BOUNCES)
    {
        // Hit sky - compute sky color
        if (dot(direction, backgroundLightDir) > 0.95)
        {
            lightValue = float3(1.0) * 5.0; // Sun
        }
        else
        {
            float a = 0.5 * (direction.y + 1.0);
            lightValue = ((1.0 - a) * float3(0.8, 0.8, 1.0) + a * float3(0.2, 0.4, 1.0)) * 0.1;
        }
    }

    // Accumulate light through hit stack (back to front)
    for (int i = 0; i < numHits; i++)
    {
        RayHit hit = hitStack[numHits - i - 1];
        Tetrahedron tet = outputTetrahedrons[hit.tetrahedronIndex];
        MaterialProperties mat = sampleMaterial(tet.materialId, hit.textureCoordinates.xyz);

        lightValue = lightValue * mat.albedo.xyz + mat.luminance * mat.albedo.xyz;
    }

    return float4(lightValue, 1.0);
}

// Main raytracer pixel shader
[shader("compute")]
[numthreads(8, 8, 1)]
void mainRaytracerPixel(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint2 uPixelPos = globalId.xy;
    float2 pixelPos = float2(
        float(uPixelPos.x) / float(workingData.renderDimensions.x) * 2.0 - 1.0,
        float(uPixelPos.y) / float(workingData.renderDimensions.y) * 2.0 - 1.0
    );

    // Initialize RNG
    uint64_t seed = workingData.raytraceSeed
        * uint64_t(workingData.renderDimensions.x)
        * uint64_t(workingData.renderDimensions.y)
        + uint64_t(globalId.x)
        + uint64_t(globalId.y) * uint64_t(workingData.renderDimensions.x);

    BasicRNG rng = BasicRNG(seed);

    // Prime RNG
    for (int i = 0; i < 30; i++) rng.rand();
    for (int i = 0; i < 30; i++)
    {
        if (rng.rand() % 3 == 0) break;
    }

    float aspectRatio = float(workingData.presentDimensions.x) / float(workingData.presentDimensions.y);

    const float PI = 3.14159;
    float viewAngle = (PI / 2.0) / workingData.focalLengthZW;
    float zwRand = rng.randFloat();
    uint zwCoord = uint(zwRand * float(workingData.renderDimensions.z));
    float zwAngle = (zwRand - 0.5) * viewAngle + (PI / 4.0);

    // Construct ray in view space
    float4 viewOrigin = float4(0);
    float4 viewDirection = normalize(float4(
        pixelPos.x / workingData.focalLengthXY,
        (-pixelPos.y / aspectRatio) / workingData.focalLengthXY,
        cos(zwAngle),
        sin(zwAngle)
    ));

    // Transform to world space using 5x5 matrix
    VecN<5> origin5 = VecN<4>.fromFloat4(viewOrigin).extend(1.0);
    VecN<5> dir5 = VecN<4>.fromFloat4(viewOrigin + viewDirection).extend(1.0);

    VecN<5> newOrigin = workingData.viewMatrixInverse * origin5;
    VecN<5> newDir = workingData.viewMatrixInverse * dir5;

    float4 worldOrigin = newOrigin.toFloat4();
    float4 worldDirection = newDir.toFloat4() - worldOrigin;

    // Trace ray
    float4 outputPixel = raycastSample(
        worldOrigin,
        worldDirection,
        workingData.totalNumTetrahedrons,
        rng);

    // Write to buffer
    if (uPixelPos.x < workingData.renderDimensions.x && uPixelPos.y < workingData.renderDimensions.y)
    {
        uint index = zwCoord * workingData.renderDimensions.x * workingData.renderDimensions.y
                   + uPixelPos.y * workingData.renderDimensions.x
                   + uPixelPos.x;
        pixelBuffer[index] += outputPixel;
    }
}
