// 4D Rasterizer compute shaders
// Alternative to raytracing - projects tetrahedra and integrates along ZW axis

import math;
import types;
import materials;

// Explicit bindings to match Rust descriptor layout:
// Set 0: OneTimeBuffers
[[vk::binding(0, 0)]] StructuredBuffer<ModelTetrahedron> modelTetrahedrons;
[[vk::binding(1, 0)]] StructuredBuffer<ModelEdge> modelEdges;

// Set 1: SizedBuffers
[[vk::binding(0, 1)]] RWStructuredBuffer<float2> outputLineVertices;
[[vk::binding(1, 1)]] RWStructuredBuffer<Tetrahedron> outputTetrahedrons;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> pixelBuffer;

// Set 2: LiveBuffers
[[vk::binding(0, 2)]] StructuredBuffer<ModelInstance> instances;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

// Vertex shader for rasterization pipeline
struct RasterVertexOutput
{
    VecN<5> vertexPosition;
    float4 texturePosition;
}

RasterVertexOutput rasterVertexShader(
    float4 vertexPosition,
    float4 texturePosition,
    ModelInstance instance,
    WorkingData workingData)
{
    // Extend to 5D homogeneous
    VecN<5> pos;
    pos[0] = vertexPosition.x;
    pos[1] = vertexPosition.y;
    pos[2] = vertexPosition.z;
    pos[3] = vertexPosition.w;
    pos[4] = 1.0;

    // Full transform: view * model
    VecN<5> viewPos = workingData.viewMatrix * instance.modelTransform * pos;

    // Compute depth for perspective division
    float aspectRatio = float(workingData.presentDimensions.x) / float(workingData.presentDimensions.y);
    float depth = sqrt(viewPos[2] * viewPos[2] + viewPos[3] * viewPos[3]);
    float projectionDivisor = depth / workingData.focalLengthXY;

    RasterVertexOutput output;
    output.vertexPosition[0] = viewPos[0];
    output.vertexPosition[1] = aspectRatio * (-viewPos[1]);
    output.vertexPosition[2] = viewPos[2];
    output.vertexPosition[3] = viewPos[3];
    output.vertexPosition[4] = projectionDivisor;
    output.texturePosition = texturePosition;
    return output;
}

// Project 4D vertex to screen space
float4 project4DVertex(VecN<5> v)
{
    return float4(v[0] / v[4], v[1] / v[4], v[2], v[3]);
}

// Tetrahedron preprocessing compute shader
[shader("compute")]
[numthreads(64, 1, 1)]
void mainTetrahedronCS(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint outputIndex = globalId.x;
    uint modelCount, modelStride;
    modelTetrahedrons.GetDimensions(modelCount, modelStride);
    uint modelIndex = outputIndex % modelCount;
    uint instanceIndex = outputIndex / modelCount;

    ModelInstance instance = instances[instanceIndex];
    ModelTetrahedron inputTet = modelTetrahedrons[modelIndex];
    uint textureId = instance.cellMaterialIds[inputTet.cellId];

    VecN<5> vertexPositions[4];
    float4 texturePositions[4];

    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        RasterVertexOutput result = rasterVertexShader(
            inputTet.vertexPositions[i],
            inputTet.texturePositions[i],
            instance,
            workingData);
        vertexPositions[i] = result.vertexPosition;
        texturePositions[i] = result.texturePosition;
    }

    // Project to screen space
    float4 projected[4];
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        projected[i] = project4DVertex(vertexPositions[i]);
    }

    // Compute 4D normal from projected positions
    float4 normal = getNormal4D(projected[0], projected[1], projected[2], projected[3]);

    // Write output tetrahedron
    Tetrahedron outTet;
    outTet.vertexPositions = projected;
    outTet.texturePositions = texturePositions;
    outTet.materialId = textureId;
    outTet.normal = normal;
    outputTetrahedrons[outputIndex] = outTet;

    // Generate wireframe edges (6 edges per tetrahedron, 2 vertices each = 12 vertices)
    uint baseIdx = outputIndex * 12;
    outputLineVertices[baseIdx + 0] = projected[0].xy;
    outputLineVertices[baseIdx + 1] = projected[1].xy;
    outputLineVertices[baseIdx + 2] = projected[0].xy;
    outputLineVertices[baseIdx + 3] = projected[2].xy;
    outputLineVertices[baseIdx + 4] = projected[0].xy;
    outputLineVertices[baseIdx + 5] = projected[3].xy;
    outputLineVertices[baseIdx + 6] = projected[1].xy;
    outputLineVertices[baseIdx + 7] = projected[2].xy;
    outputLineVertices[baseIdx + 8] = projected[1].xy;
    outputLineVertices[baseIdx + 9] = projected[3].xy;
    outputLineVertices[baseIdx + 10] = projected[2].xy;
    outputLineVertices[baseIdx + 11] = projected[3].xy;
}

// Edge processing with 4D frustum clipping
[shader("compute")]
[numthreads(64, 1, 1)]
void mainEdgeCS(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint outputIndex = globalId.x;
    uint modelCount, modelStride;
    modelEdges.GetDimensions(modelCount, modelStride);
    uint modelIndex = outputIndex % modelCount;
    uint instanceIndex = outputIndex / modelCount;

    ModelInstance instance = instances[instanceIndex];
    ModelEdge inputEdge = modelEdges[modelIndex];

    VecN<5> vertexPositions[2];

    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        RasterVertexOutput result = rasterVertexShader(
            inputEdge.vertexPositions[i],
            float4(0),
            instance,
            workingData);
        vertexPositions[i] = result.vertexPosition;
    }

    // TODO: Implement 4D frustum clipping (see clip_line_to_frustum in original)
    // For now, just project directly

    float depths[2];
    VecN<4> clipped[2];
    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        clipped[i][0] = vertexPositions[i][0];
        clipped[i][1] = vertexPositions[i][1];
        clipped[i][2] = vertexPositions[i][2];
        clipped[i][3] = vertexPositions[i][3];
        depths[i] = sqrt(clipped[i][2] * clipped[i][2] + clipped[i][3] * clipped[i][3]);
    }

    float2 projected[2];
    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        float divisor = depths[i] / workingData.focalLengthXY;
        projected[i] = float2(clipped[i][0] / divisor, clipped[i][1] / divisor);
    }

    outputLineVertices[outputIndex * 2 + 0] = projected[0];
    outputLineVertices[outputIndex * 2 + 1] = projected[1];
}

// Barycentric coordinates for 2D triangle
float3 getBarycentric(float2 vertices[3], float2 point)
{
    float3 u = cross(
        float3(vertices[2].x - vertices[0].x, vertices[1].x - vertices[0].x, vertices[0].x - point.x),
        float3(vertices[2].y - vertices[0].y, vertices[1].y - vertices[0].y, vertices[0].y - point.y)
    );

    if (abs(u.z) < 0.000001)
        return float3(-1.0, 1.0, 1.0);

    return float3(
        1.0 - (u.x + u.y) / u.z,
        u.y / u.z,
        u.x / u.z
    );
}

// Line-line intersection
float2 getIntersection(float2 lineA[2], float2 lineB[2])
{
    float rNum = (lineA[0].y - lineB[0].y) * (lineB[1].x - lineB[0].x) - (lineA[0].x - lineB[0].x) * (lineB[1].y - lineB[0].y);
    float rDen = (lineA[1].x - lineA[0].x) * (lineB[1].y - lineB[0].y) - (lineA[1].y - lineA[0].y) * (lineB[1].x - lineB[0].x);
    float sNum = (lineA[0].y - lineB[0].y) * (lineA[1].x - lineA[0].x) - (lineA[0].x - lineB[0].x) * (lineA[1].y - lineA[0].y);
    return float2(rNum / rDen, sNum / rDen);
}

// Get ZW angle range from focal length
float2 getZWAngleRange(WorkingData workingData)
{
    const float PI = 3.14159;
    float viewAngle = (PI / 2.0) / workingData.focalLengthZW;
    return float2(PI / 4.0 - viewAngle / 2.0, PI / 4.0 + viewAngle / 2.0);
}

// Simple ZW line rendering - samples along the ZW angle range
float4 renderZWLinesSimple(ZWLine lines[96], uint numLines, WorkingData workingData)
{
    const uint DEPTH_FACTOR = 256;
    float2 angleRange = getZWAngleRange(workingData);

    float3 outputAccum = float3(0);

    for (uint i = 0; i < DEPTH_FACTOR; i++)
    {
        float angleRatio = float(i) / float(DEPTH_FACTOR);
        float theta = angleRatio * (angleRange.y - angleRange.x) + angleRange.x;
        float2 sampleRay = float2(cos(theta), sin(theta));

        int closestLine = -1;
        float closestDist = 1e10;
        float closestVal = 0;

        for (uint j = 0; j < numLines; j++)
        {
            if (lines[j].textureId == 0)
                continue;

            float2 lineA[2] = { float2(0), sampleRay };
            float2 lineB[2] = { lines[j].zwPositions[0], lines[j].zwPositions[1] };

            float2 intersection = getIntersection(lineA, lineB);
            float dist = intersection.x;

            if (intersection.y < 0.0 || intersection.y > 1.0)
                continue;
            if (dist < 0.0)
                continue;

            if (closestLine < 0 || dist < closestDist)
            {
                closestLine = int(j);
                closestDist = dist;
                closestVal = intersection.y;
            }
        }

        if (closestLine >= 0)
        {
            float val = closestVal;
            float4 texPos = lines[closestLine].texturePositions[0] * val +
                           lines[closestLine].texturePositions[1] * (1.0 - val);
            MaterialProperties mat = sampleMaterial(lines[closestLine].textureId, texPos.xyz);
            outputAccum += mat.albedo.xyz / float(DEPTH_FACTOR);
        }
    }

    return float4(outputAccum, 1.0);
}

// Per-pixel tetrahedron rasterization
[shader("compute")]
[numthreads(8, 8, 1)]
void mainTetrahedronPixelCS(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];
    // Combinations for iterating over tetrahedron faces
    static const uint3 COMBINATIONS_4_3[4] = {
        uint3(0, 1, 2), uint3(0, 1, 3), uint3(0, 2, 3), uint3(1, 2, 3)
    };

    ZWLine currentZWLines[96];
    uint numZWLines = 0;

    // Initialize lines array
    for (uint i = 0; i < 96; i++)
        currentZWLines[i] = ZWLine.empty();

    uint2 uPixelPos = globalId.xy;
    float2 pixelPos = float2(
        float(uPixelPos.x) / float(workingData.renderDimensions.x) * 2.0 - 1.0,
        float(uPixelPos.y) / float(workingData.renderDimensions.y) * 2.0 - 1.0
    );

    // Test each tetrahedron
    for (uint i = 0; i < workingData.totalNumTetrahedrons; i++)
    {
        Tetrahedron tet = outputTetrahedrons[i];

        if (tet.materialId == 0)
            continue;

        // Compute 2D bounding box
        float4 minBound = min(min(tet.vertexPositions[0], tet.vertexPositions[1]),
                              min(tet.vertexPositions[2], tet.vertexPositions[3]));
        float4 maxBound = max(max(tet.vertexPositions[0], tet.vertexPositions[1]),
                              max(tet.vertexPositions[2], tet.vertexPositions[3]));

        if (pixelPos.x < minBound.x || pixelPos.x > maxBound.x ||
            pixelPos.y < minBound.y || pixelPos.y > maxBound.y)
            continue;

        // Test each face of the tetrahedron
        float3 barycentrics[4];
        bool inTriangle[4];
        bool anyIn = false;

        for (uint j = 0; j < 4; j++)
        {
            float2 vertices[3] = {
                tet.vertexPositions[COMBINATIONS_4_3[j][0]].xy,
                tet.vertexPositions[COMBINATIONS_4_3[j][1]].xy,
                tet.vertexPositions[COMBINATIONS_4_3[j][2]].xy
            };

            barycentrics[j] = getBarycentric(vertices, pixelPos);
            inTriangle[j] = (barycentrics[j].x >= 0.0) && (barycentrics[j].y >= 0.0) && (barycentrics[j].z >= 0.0);
            anyIn = anyIn || inTriangle[j];
        }

        if (!anyIn)
            continue;

        // Find which two faces we intersect
        uint triIndexes[2] = { 0, 0 };
        if (inTriangle[0]) {
            triIndexes[0] = 0;
            triIndexes[1] = inTriangle[1] ? 1 : (inTriangle[2] ? 2 : 3);
        } else if (inTriangle[1]) {
            triIndexes[0] = 1;
            triIndexes[1] = inTriangle[2] ? 2 : 3;
        } else {
            triIndexes[0] = 2;
            triIndexes[1] = 3;
        }

        if (numZWLines >= 96)
            continue;

        // Compute ZW line from the two face intersections
        currentZWLines[numZWLines].textureId = tet.materialId;

        for (uint j = 0; j < 2; j++)
        {
            float3 bary = barycentrics[triIndexes[j]];
            uint3 combo = COMBINATIONS_4_3[triIndexes[j]];

            float2 zwVertices[3] = {
                tet.vertexPositions[combo[0]].zw,
                tet.vertexPositions[combo[1]].zw,
                tet.vertexPositions[combo[2]].zw
            };

            float4 texVertices[3] = {
                tet.texturePositions[combo[0]],
                tet.texturePositions[combo[1]],
                tet.texturePositions[combo[2]]
            };

            float2 zwValue = bary.x * zwVertices[0] + bary.y * zwVertices[1] + bary.z * zwVertices[2];
            float4 texValue = bary.x * texVertices[0] + bary.y * texVertices[1] + bary.z * texVertices[2];

            currentZWLines[numZWLines].zwPositions[j] = zwValue;
            currentZWLines[numZWLines].texturePositions[j] = texValue;
        }

        numZWLines++;
    }

    // Render the collected ZW lines
    float4 output = renderZWLinesSimple(currentZWLines, numZWLines, workingData);

    // Write output
    if (uPixelPos.x < workingData.renderDimensions.x && uPixelPos.y < workingData.renderDimensions.y)
    {
        pixelBuffer[uPixelPos.y * workingData.renderDimensions.x + uPixelPos.x] = output;
    }

    // Clear other Z slices
    for (uint z = 1; z < workingData.renderDimensions.z; z++)
    {
        uint index = z * workingData.renderDimensions.x * workingData.renderDimensions.y
                   + uPixelPos.y * workingData.renderDimensions.x
                   + uPixelPos.x;
        pixelBuffer[index] = float4(0);
    }
}
