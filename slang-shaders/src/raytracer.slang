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

// Shared push constants (offset/length controls for preprocessing dispatches).
// If tetrahedronCount is 0, preprocess falls back to workingData.totalNumTetrahedrons.
struct DispatchParams
{
    uint instanceBase;
    uint tetrahedronCount;
    uint outputBase;
    uint padding;
}

[[vk::push_constant]]
DispatchParams dispatchParams;

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
    uint tetrahedronCount = dispatchParams.tetrahedronCount != 0u
        ? dispatchParams.tetrahedronCount
        : workingData.totalNumTetrahedrons;
    if (outputIndex >= tetrahedronCount)
    {
        return;
    }

    uint modelCount, modelStride;
    modelTetrahedrons.GetDimensions(modelCount, modelStride);
    uint modelIndex = outputIndex % modelCount;
    uint instanceIndex = dispatchParams.instanceBase + (outputIndex / modelCount);
    uint outTetIndex = dispatchParams.outputBase + outputIndex;

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

    outputTetrahedrons[outTetIndex] = outTet;
}

// Clear the pixel buffer
[shader("compute")]
[numthreads(8, 8, 1)]
void mainRaytracerClear(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint2 pixelPos = globalId.xy;
    if (pixelPos.x >= workingData.renderDimensions.x || pixelPos.y >= workingData.renderDimensions.y)
    {
        return;
    }
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
// Uses the precomputed normal for fast t computation (2 dot products),
// early rejection on t, then 3x3 Cramer's rule for barycentric coordinates.
// This avoids the expensive 4x4 matrix inverse.
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
    float denom = dot(tet.normal, direction);
    if (abs(denom) < 0.001)
        return false;

    // Compute t using the normal (perpendicular to all edge vectors).
    // Since b = start - v0 decomposes as b = -t*d + u*e1 + v*e2 + w*e3,
    // and N is perpendicular to e1,e2,e3: dot(b,N) = -t*dot(d,N)
    float4 b = start - tet.vertexPositions[0];
    float hitDistance = -dot(b, tet.normal) / denom;

    // Early rejection: behind ray or farther than closest hit
    if (hitDistance <= 0.0)
        return false;
    if (closestHit.didHit != 0 && closestHit.hitDistance <= hitDistance)
        return false;

    // Compute hit point relative to v0
    float4 q = b + hitDistance * direction;

    // Edge vectors
    float4 e1 = tet.vertexPositions[1] - tet.vertexPositions[0];
    float4 e2 = tet.vertexPositions[2] - tet.vertexPositions[0];
    float4 e3 = tet.vertexPositions[3] - tet.vertexPositions[0];

    // Project 4D system to 3D by dropping the axis with largest normal component.
    // The 4x3 system q = u*e1 + v*e2 + w*e3 is overdetermined but consistent;
    // dropping the most-constrained row gives the best-conditioned 3x3 system.
    float4 absN = abs(tet.normal);
    float3 e1s, e2s, e3s, qs;

    if (absN.x >= absN.y && absN.x >= absN.z && absN.x >= absN.w)
    {
        e1s = e1.yzw; e2s = e2.yzw; e3s = e3.yzw; qs = q.yzw;
    }
    else if (absN.y >= absN.z && absN.y >= absN.w)
    {
        e1s = float3(e1.x, e1.z, e1.w); e2s = float3(e2.x, e2.z, e2.w);
        e3s = float3(e3.x, e3.z, e3.w); qs = float3(q.x, q.z, q.w);
    }
    else if (absN.z >= absN.w)
    {
        e1s = float3(e1.x, e1.y, e1.w); e2s = float3(e2.x, e2.y, e2.w);
        e3s = float3(e3.x, e3.y, e3.w); qs = float3(q.x, q.y, q.w);
    }
    else
    {
        e1s = e1.xyz; e2s = e2.xyz; e3s = e3.xyz; qs = q.xyz;
    }

    // Solve 3x3 system using Cramer's rule:
    // [e1s | e2s | e3s] * [u, v, w]^T = qs
    float3 c23 = cross(e2s, e3s);
    float det3 = dot(e1s, c23);
    if (abs(det3) < 1e-10)
        return false;

    float invDet = 1.0 / det3;
    float u = dot(qs, c23) * invDet;
    float v = dot(e1s, cross(qs, e3s)) * invDet;
    float w = dot(e1s, cross(e2s, qs)) * invDet;
    float bary0 = 1.0 - u - v - w;

    // Check if inside tetrahedron
    if (bary0 >= 0.0 && u >= 0.0 && v >= 0.0 && w >= 0.0)
    {
        float4 hitPosition = hitDistance * direction + start;
        float4 hitNormal = denom < 0.0 ? tet.normal : -tet.normal;

        closestHit.tetrahedronIndex = tetIndex;
        closestHit.hitPosition = hitPosition;
        closestHit.hitDistance = hitDistance;
        closestHit.barycentric = float4(bary0, u, v, w);
        closestHit.hitNormal = hitNormal;
        closestHit.didHit = 1;
        return true;
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
// direction must be normalized before calling
RayHit raycast(
    float4 start,
    float4 direction,
    uint numTetrahedrons)
{
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

    // Stack-based BVH traversal
    // Uses test-before-push: children are AABB-tested when pushing,
    // so popped nodes are known to intersect the ray.
    uint stack[BVH_STACK_SIZE];
    int stackPtr = 0;

    // Test root AABB before entering the loop
    BVHNode root = bvhNodes[0];
    float2 rootT = rayAABBIntersect4D(start, invDir, root.minBounds, root.maxBounds);
    if (rootT.x < 0.0)
        return closestHit;

    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        uint nodeIdx = stack[--stackPtr];
        BVHNode node = bvhNodes[nodeIdx];

        if (node.isLeaf != 0)
        {
            // Leaf node - test tetrahedron
            testTetrahedronHit(start, direction, node.tetrahedronIndex, closestHit);
        }
        else
        {
            // Internal node - test children and push those that intersect
            uint left = node.leftChild;
            uint right = node.rightChild;

            float2 leftT = rayAABBIntersect4D(start, invDir, bvhNodes[left].minBounds, bvhNodes[left].maxBounds);
            float2 rightT = rayAABBIntersect4D(start, invDir, bvhNodes[right].minBounds, bvhNodes[right].maxBounds);

            // Filter by closest hit distance
            bool leftHit = leftT.x >= 0.0 && (closestHit.didHit == 0 || leftT.x <= closestHit.hitDistance);
            bool rightHit = rightT.x >= 0.0 && (closestHit.didHit == 0 || rightT.x <= closestHit.hitDistance);

            // Push farther child first so closer child is popped first
            if (leftHit && rightHit)
            {
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
                stack[stackPtr++] = left;
            }
            else if (rightHit)
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

        if (hit.didHit != 0)
        {
            Tetrahedron tet = outputTetrahedrons[hit.tetrahedronIndex];
            MaterialProperties mat = sampleMaterial(tet.materialId, hit.textureCoordinates);

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
        MaterialProperties mat = sampleMaterial(tet.materialId, hit.textureCoordinates);

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

    float aspectRatio = float(workingData.presentDimensions.x) / float(workingData.presentDimensions.y);

    const float PI = 3.14159;
    uint layerCount = max(1u, workingData.renderDimensions.z);
    float viewAngle = (PI / 2.0) / workingData.focalLengthZW;
    float zwRand = rng.randFloat();
    uint zwCoord = min(uint(zwRand * float(layerCount)), layerCount - 1u);
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
