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

// 4D ray-tetrahedron intersection using extended Moller-Trumbore
RayHit raycast(
    float4 start,
    float4 direction,
    uint numTetrahedrons)
{
    direction = normalize(direction);
    RayHit closestHit = RayHit.none();

    for (uint i = 0; i < numTetrahedrons; i++)
    {
        Tetrahedron tet = outputTetrahedrons[i];

        if (tet.materialId == 0)
            continue;

        // Check if ray is parallel to tetrahedron hyperplane
        float value = dot(tet.normal, direction);
        if (abs(value) < 0.001)
            continue;

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
        float4 hitPosition = hitDistance * direction + start;

        // Check if hit is valid (in front of ray, inside tetrahedron)
        if (hitDistance > 0.0 &&
            barycentric.x >= 0.0 && barycentric.y >= 0.0 &&
            barycentric.z >= 0.0 && barycentric.w >= 0.0)
        {
            if (closestHit.didHit == 0 || closestHit.hitDistance > hitDistance)
            {
                float4 hitNormal = dot(tet.normal, direction) < 0.0
                    ? tet.normal
                    : -tet.normal;

                closestHit.tetrahedronIndex = i;
                closestHit.hitPosition = hitPosition;
                closestHit.hitDistance = hitDistance;
                closestHit.barycentric = barycentric;
                closestHit.hitNormal = hitNormal;
                closestHit.didHit = 1;
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
