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
[[vk::binding(0, 1)]] RWStructuredBuffer<LineVertex> outputLineVertices;
[[vk::binding(1, 1)]] RWStructuredBuffer<Tetrahedron> outputTetrahedrons;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> pixelBuffer;
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> atomicCounter;
[[vk::binding(7, 1)]] RWStructuredBuffer<uint> tileTetCounts;
[[vk::binding(8, 1)]] RWStructuredBuffer<uint> tileTetIndices;

static const uint MAX_TETS_PER_TILE = 8192;

// Set 2: LiveBuffers
[[vk::binding(0, 2)]] StructuredBuffer<ModelInstance> instances;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

// Shared push constants (instance-base/length controls for preprocessing dispatches).
// If tetrahedronCount is 0, preprocessing falls back to workingData.totalNumTetrahedrons.
struct DispatchParams
{
    uint instanceBase;
    uint tetrahedronCount;
    uint outputBase;
    uint padding;
}

[[vk::push_constant]]
DispatchParams dispatchParams;

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

// Working tetrahedron for clipping (pre-projection space)
struct ClipTet
{
    VecN<5> verts[4];
    float4 texCoords[4];
    uint materialId;
}

// Interpolate a point on an edge between two vertices, given signed distances
void clipEdge(VecN<5> vA, float4 tA, float dA, VecN<5> vB, float4 tB, float dB, out VecN<5> vOut, out float4 tOut)
{
    float t = dA / (dA - dB);
    vOut = vA * (1.0 - t) + vB * t;
    tOut = tA * (1.0 - t) + tB * t;
}

// Generic tetrahedron clipper: clips a tetrahedron given pre-computed signed distances
// per vertex. Vertices with dist >= 0 are "inside" (kept). Returns 0-3 output tetrahedra.
// When fixDepth is true, new vertices have verts[4] recomputed from Z,W (needed for
// screen-space clips). When false, verts[4] is linearly interpolated (for 4D-space clips).
uint clipTetrahedronGeneric(ClipTet tet, float dist[4], float focalLengthXY, bool fixDepth, out ClipTet results[3])
{
    bool inside[4];
    uint numInside = 0;

    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        inside[i] = dist[i] >= 0.0;
        if (inside[i]) numInside++;
    }

    if (numInside == 4) { results[0] = tet; return 1; }
    if (numInside == 0) return 0;

    uint insideIdx[4], outsideIdx[4];
    uint nIn = 0, nOut = 0;
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        if (inside[i]) insideIdx[nIn++] = i;
        else           outsideIdx[nOut++] = i;
    }

    // Recompute projectionDivisor from actual Z,W after interpolation
    // Only applied when fixDepth is true (for screen-space clips where projDiv must be exact).
    // For 4D-space clips (like ZW cone), linear interpolation of projDiv is kept.
    #define FIX_DEPTH(v) if (fixDepth) { v[4] = sqrt(v[2] * v[2] + v[3] * v[3]) / focalLengthXY; }

    if (numInside == 1)
    {
        uint a = insideIdx[0];
        uint b = outsideIdx[0], c = outsideIdx[1], d = outsideIdx[2];
        VecN<5> vAB, vAC, vAD; float4 tAB, tAC, tAD;
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[b], tet.texCoords[b], dist[b], vAB, tAB);
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[c], tet.texCoords[c], dist[c], vAC, tAC);
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[d], tet.texCoords[d], dist[d], vAD, tAD);
        FIX_DEPTH(vAB); FIX_DEPTH(vAC); FIX_DEPTH(vAD);
        ClipTet r; r.materialId = tet.materialId;
        r.verts[0] = tet.verts[a]; r.texCoords[0] = tet.texCoords[a];
        r.verts[1] = vAB; r.texCoords[1] = tAB;
        r.verts[2] = vAC; r.texCoords[2] = tAC;
        r.verts[3] = vAD; r.texCoords[3] = tAD;
        results[0] = r; return 1;
    }

    if (numInside == 2)
    {
        uint a = insideIdx[0], b = insideIdx[1];
        uint c = outsideIdx[0], d = outsideIdx[1];
        VecN<5> vAC, vAD, vBC, vBD; float4 tAC, tAD, tBC, tBD;
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[c], tet.texCoords[c], dist[c], vAC, tAC);
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[d], tet.texCoords[d], dist[d], vAD, tAD);
        clipEdge(tet.verts[b], tet.texCoords[b], dist[b], tet.verts[c], tet.texCoords[c], dist[c], vBC, tBC);
        clipEdge(tet.verts[b], tet.texCoords[b], dist[b], tet.verts[d], tet.texCoords[d], dist[d], vBD, tBD);
        FIX_DEPTH(vAC); FIX_DEPTH(vAD); FIX_DEPTH(vBC); FIX_DEPTH(vBD);
        ClipTet r0, r1, r2;
        r0.materialId = tet.materialId; r1.materialId = tet.materialId; r2.materialId = tet.materialId;
        r0.verts[0] = tet.verts[a]; r0.texCoords[0] = tet.texCoords[a];
        r0.verts[1] = tet.verts[b]; r0.texCoords[1] = tet.texCoords[b];
        r0.verts[2] = vAC; r0.texCoords[2] = tAC; r0.verts[3] = vAD; r0.texCoords[3] = tAD;
        r1.verts[0] = tet.verts[b]; r1.texCoords[0] = tet.texCoords[b];
        r1.verts[1] = vAC; r1.texCoords[1] = tAC; r1.verts[2] = vAD; r1.texCoords[2] = tAD;
        r1.verts[3] = vBC; r1.texCoords[3] = tBC;
        r2.verts[0] = tet.verts[b]; r2.texCoords[0] = tet.texCoords[b];
        r2.verts[1] = vAD; r2.texCoords[1] = tAD; r2.verts[2] = vBC; r2.texCoords[2] = tBC;
        r2.verts[3] = vBD; r2.texCoords[3] = tBD;
        results[0] = r0; results[1] = r1; results[2] = r2; return 3;
    }

    // numInside == 3
    {
        uint d = outsideIdx[0];
        uint a = insideIdx[0], b = insideIdx[1], c = insideIdx[2];
        VecN<5> vAD, vBD, vCD; float4 tAD, tBD, tCD;
        clipEdge(tet.verts[a], tet.texCoords[a], dist[a], tet.verts[d], tet.texCoords[d], dist[d], vAD, tAD);
        clipEdge(tet.verts[b], tet.texCoords[b], dist[b], tet.verts[d], tet.texCoords[d], dist[d], vBD, tBD);
        clipEdge(tet.verts[c], tet.texCoords[c], dist[c], tet.verts[d], tet.texCoords[d], dist[d], vCD, tCD);
        FIX_DEPTH(vAD); FIX_DEPTH(vBD); FIX_DEPTH(vCD);
        ClipTet r0, r1, r2;
        r0.materialId = tet.materialId; r1.materialId = tet.materialId; r2.materialId = tet.materialId;
        r0.verts[0] = tet.verts[a]; r0.texCoords[0] = tet.texCoords[a];
        r0.verts[1] = tet.verts[b]; r0.texCoords[1] = tet.texCoords[b];
        r0.verts[2] = tet.verts[c]; r0.texCoords[2] = tet.texCoords[c];
        r0.verts[3] = vAD; r0.texCoords[3] = tAD;
        r1.verts[0] = tet.verts[b]; r1.texCoords[0] = tet.texCoords[b];
        r1.verts[1] = tet.verts[c]; r1.texCoords[1] = tet.texCoords[c];
        r1.verts[2] = vAD; r1.texCoords[2] = tAD; r1.verts[3] = vBD; r1.texCoords[3] = tBD;
        r2.verts[0] = tet.verts[c]; r2.texCoords[0] = tet.texCoords[c];
        r2.verts[1] = vAD; r2.texCoords[1] = tAD; r2.verts[2] = vBD; r2.texCoords[2] = tBD;
        r2.verts[3] = vCD; r2.texCoords[3] = tCD;
        results[0] = r0; results[1] = r1; results[2] = r2; return 3;
    }

    #undef FIX_DEPTH
}

// Near-depth clipping threshold.
static const float MIN_DEPTH_DIVISOR = 0.15;

// Tetrahedron preprocessing compute shader with frustum clipping
[shader("compute")]
[numthreads(64, 1, 1)]
void mainTetrahedronCS(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint outputIndex = globalId.x;
    uint tetrahedronCount = dispatchParams.tetrahedronCount != 0u
        ? dispatchParams.tetrahedronCount
        : workingData.totalNumTetrahedrons;
    if (outputIndex >= tetrahedronCount)
        return;

    uint modelCount, modelStride;
    modelTetrahedrons.GetDimensions(modelCount, modelStride);
    uint modelIndex = outputIndex % modelCount;
    uint instanceIndex = dispatchParams.instanceBase + (outputIndex / modelCount);

    ModelInstance instance = instances[instanceIndex];
    ModelTetrahedron inputTet = modelTetrahedrons[modelIndex];
    uint textureId = instance.cellMaterialIds[inputTet.cellId];

    if (textureId == 0)
        return;

    // Transform vertices to view space
    ClipTet initialTet;
    initialTet.materialId = textureId;
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        RasterVertexOutput result = rasterVertexShader(
            inputTet.vertexPositions[i],
            inputTet.texturePositions[i],
            instance,
            workingData);
        initialTet.verts[i] = result.vertexPosition;
        initialTet.texCoords[i] = result.texturePosition;
    }

    // Frustum clipping in 3 passes:
    // 1-2. ZW viewing cone: clip to the ZW angle range [theta_min, theta_max].
    //      This is the critical fix — without it, edges crossing the ZW origin (depth=0)
    //      produce degenerate screen coordinates. The clip boundaries are LINEAR hyperplanes
    //      in 4D view space (Z,W), so linear edge interpolation is exact.
    // 3.   Near depth: projectionDivisor >= MIN_DEPTH_DIVISOR.
    //      Safety clip to prevent division by near-zero depth.
    //
    // Screen-space clips are omitted — the pixel shader handles off-screen geometry via
    // bounding box tests. Screen clips caused artifacts because FIX_DEPTH (needed to correct
    // projDiv after interpolation through nonlinear sqrt) creates brightness discontinuities.
    float fxy = workingData.focalLengthXY;

    ClipTet bufA[24];
    ClipTet bufB[24];
    uint countA = 1;
    bufA[0] = initialTet;

    // ZW viewing cone clips: the ZW angle range is [theta_min, theta_max].
    // For focalLengthZW=1.0: theta_min=0, theta_max=PI/2, giving Z>=0 and W>=0.
    // Clip conditions are linear in view-space Z,W:
    //   lower boundary: -sin(theta_min)*Z + cos(theta_min)*W >= 0
    //   upper boundary:  sin(theta_max)*Z - cos(theta_max)*W >= 0
    float zwViewAngle = (3.14159265 / 2.0) / workingData.focalLengthZW;
    float thetaMin = 3.14159265 / 4.0 - zwViewAngle / 2.0;
    float thetaMax = 3.14159265 / 4.0 + zwViewAngle / 2.0;
    float sinMin = sin(thetaMin), cosMin = cos(thetaMin);
    float sinMax = sin(thetaMax), cosMax = cos(thetaMax);

    // Pass 1: ZW cone lower boundary (theta_min) — 4D clip, no FIX_DEPTH
    {
        uint countB = 0;
        for (uint t = 0; t < countA; t++)
        {
            float dist[4];
            [ForceUnroll] for (int i = 0; i < 4; i++)
                dist[i] = -sinMin * bufA[t].verts[i][2] + cosMin * bufA[t].verts[i][3];
            ClipTet clipResults[3];
            uint n = clipTetrahedronGeneric(bufA[t], dist, fxy, false, clipResults);
            for (uint r = 0; r < n && countB < 24; r++)
                bufB[countB++] = clipResults[r];
        }
        countA = countB;
        for (uint t = 0; t < countA; t++) bufA[t] = bufB[t];
    }

    // Pass 2: ZW cone upper boundary (theta_max) — 4D clip, no FIX_DEPTH
    {
        uint countB = 0;
        for (uint t = 0; t < countA; t++)
        {
            float dist[4];
            [ForceUnroll] for (int i = 0; i < 4; i++)
                dist[i] = sinMax * bufA[t].verts[i][2] - cosMax * bufA[t].verts[i][3];
            ClipTet clipResults[3];
            uint n = clipTetrahedronGeneric(bufA[t], dist, fxy, false, clipResults);
            for (uint r = 0; r < n && countB < 24; r++)
                bufB[countB++] = clipResults[r];
        }
        countA = countB;
        for (uint t = 0; t < countA; t++) bufA[t] = bufB[t];
    }

    // After ZW cone clips, recompute projDiv from actual Z,W for all vertices.
    // This ensures subsequent depth/screen clips use correct projDiv values.
    for (uint t = 0; t < countA; t++)
    {
        [ForceUnroll]
        for (int i = 0; i < 4; i++)
            bufA[t].verts[i][4] = sqrt(bufA[t].verts[i][2] * bufA[t].verts[i][2] +
                                       bufA[t].verts[i][3] * bufA[t].verts[i][3]) / fxy;
    }

    // Pass 3: Near depth clip (projDivisor >= MIN_DEPTH_DIVISOR) — uses FIX_DEPTH
    {
        uint countB = 0;
        for (uint t = 0; t < countA; t++)
        {
            float dist[4];
            [ForceUnroll] for (int i = 0; i < 4; i++)
                dist[i] = bufA[t].verts[i][4] - MIN_DEPTH_DIVISOR;
            ClipTet clipResults[3];
            uint n = clipTetrahedronGeneric(bufA[t], dist, fxy, true, clipResults);
            for (uint r = 0; r < n && countB < 24; r++)
                bufB[countB++] = clipResults[r];
        }
        countA = countB;
        for (uint t = 0; t < countA; t++) bufA[t] = bufB[t];
    }

    // Write surviving tetrahedra via atomic counter
    uint bufSize, bufStride;
    outputTetrahedrons.GetDimensions(bufSize, bufStride);

    for (uint t = 0; t < countA; t++)
    {
        uint slot;
        InterlockedAdd(atomicCounter[0], 1, slot);

        if (slot >= bufSize)
            continue; // overflow protection

        // Compute normal from pre-projection view-space vertices
        float4 preProj[4];
        [ForceUnroll]
        for (int i = 0; i < 4; i++)
            preProj[i] = bufA[t].verts[i].toFloat4();
        float4 normal = getNormal4D(preProj[0], preProj[1], preProj[2], preProj[3]);

        // Project to screen space and store inverse projection divisors
        float4 projected[4];
        float4 invProjDivisors;
        [ForceUnroll]
        for (int i = 0; i < 4; i++)
        {
            projected[i] = project4DVertex(bufA[t].verts[i]);
            invProjDivisors[i] = 1.0 / bufA[t].verts[i][4];
        }

        Tetrahedron outTet;
        outTet.vertexPositions = projected;
        outTet.texturePositions = bufA[t].texCoords;
        outTet.materialId = bufA[t].materialId;
        outTet.normal = normal;
        outTet.invProjectionDivisors = invProjDivisors;
        outputTetrahedrons[slot] = outTet;
    }
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

    // 4D frustum clipping using hyperplanes (operates in view space before perspective division)
    // Near plane: prevents depth ≈ 0 (div-by-zero in perspective)
    HyperPlane4D nearPlane = HyperPlane4D.from4Points(
        float4(1.0, 1.0, 0.1, 0.1),
        float4(1.0, -1.0, 0.1, 0.1),
        float4(-1.0, 1.0, 0.1, 0.1),
        float4(1.0, 1.0, 1.1, -0.9)
    );
    nearPlane.clipLineWithData(vertexPositions[0], vertexPositions[1]);

    // Top plane
    HyperPlane4D topPlane = HyperPlane4D.from4Points(
        float4(0.0, 0.0, 0.0, 0.0),
        float4(1.0, 1.0, 1.0, 1.0),
        float4(-1.0, 1.0, 1.0, 1.0),
        float4(0.0, 0.0, -1.0, 1.0)
    );
    topPlane.clipLineWithData(vertexPositions[0], vertexPositions[1]);

    // Bottom plane
    HyperPlane4D bottomPlane = HyperPlane4D.from4Points(
        float4(0.0, 0.0, 0.0, 0.0),
        float4(1.0, -1.0, 1.0, 1.0),
        float4(-1.0, -1.0, 1.0, 1.0),
        float4(0.0, 0.0, 1.0, -1.0)
    );
    bottomPlane.clipLineWithData(vertexPositions[0], vertexPositions[1]);

    // Left plane
    HyperPlane4D leftPlane = HyperPlane4D.from4Points(
        float4(0.0, 0.0, 0.0, 0.0),
        float4(1.0, 1.0, 1.0, 1.0),
        float4(1.0, -1.0, 1.0, 1.0),
        float4(0.0, 0.0, 1.0, -1.0)
    );
    leftPlane.clipLineWithData(vertexPositions[0], vertexPositions[1]);

    // Right plane
    HyperPlane4D rightPlane = HyperPlane4D.from4Points(
        float4(0.0, 0.0, 0.0, 0.0),
        float4(-1.0, 1.0, 1.0, 1.0),
        float4(-1.0, -1.0, 1.0, 1.0),
        float4(0.0, 0.0, -1.0, 1.0)
    );
    rightPlane.clipLineWithData(vertexPositions[0], vertexPositions[1]);

    // Project clipped vertices
    float2 projected[2];
    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        projected[i] = float2(vertexPositions[i][0] / vertexPositions[i][4],
                              vertexPositions[i][1] / vertexPositions[i][4]);
    }

    LineVertex v0;
    v0.position = projected[0];
    v0.padding = float2(0.0, 0.0);
    v0.color = float4(0.30, 0.70, 1.00, 1.0);

    LineVertex v1;
    v1.position = projected[1];
    v1.padding = float2(0.0, 0.0);
    v1.color = float4(0.30, 0.70, 1.00, 1.0);

    outputLineVertices[outputIndex * 2 + 0] = v0;
    outputLineVertices[outputIndex * 2 + 1] = v1;
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

// Maximum ZW lines per pixel. With tile-based binning, each pixel only processes
// tets overlapping its tile (~30-50), so 64 is rarely hit.
static const uint MAX_ZW_LINES = 64;

// Remove dead lines (textureId == 0) and compact the array
uint condenseZWLines(inout ZWLine lines[MAX_ZW_LINES], uint numLines)
{
    uint count = 0;
    for (uint i = 0; i < numLines; i++)
    {
        if (lines[i].textureId != 0)
        {
            lines[count] = lines[i];
            count++;
        }
    }
    return count;
}

// Simple ZW line rendering - samples along the ZW angle range
// Uses incremental sin/cos rotation to avoid per-sample transcendentals.
// Ray-from-origin intersection is specialized: for ray P(r)=r*(rayX,rayY)
// and segment Q(s)=Q0+s*(dx,dy), the intersection parameters are:
//   rNum = Q0.x*dy - Q0.y*dx  (constant per line — precomputed)
//   rDen = rayX*dy - rayY*dx
//   sNum = Q0.x*rayY - Q0.y*rayX
//   r = rNum/rDen, s = sNum/rDen
float4 renderZWLinesSimple(
    ZWLine lines[MAX_ZW_LINES],
    uint numLines,
    WorkingData workingData,
    uint depthFactor)
{
    uint sampleCount = max(1u, depthFactor);
    float2 angleRange = getZWAngleRange(workingData);
    float angleStep = (angleRange.y - angleRange.x) / float(sampleCount);

    // Transform sun direction from world space to view space (direction only, w5=0)
    float4 sunDirWorld = normalize(float4(0.3, 1.0, -0.3, 0.0));
    VecN<5> sunDir5 = VecN<4>.fromFloat4(sunDirWorld).extend(0.0);
    VecN<5> sunViewDir5 = workingData.viewMatrix * sunDir5;
    float4 sunViewDir = normalize(sunViewDir5.toFloat4());

    // Precompute per-line constants for ray-from-origin intersection
    float lineDx[MAX_ZW_LINES];
    float lineDy[MAX_ZW_LINES];
    float lineRNum[MAX_ZW_LINES];
    float lineQ0x[MAX_ZW_LINES];
    float lineQ0y[MAX_ZW_LINES];

    for (uint j = 0; j < numLines; j++)
    {
        float2 q0 = lines[j].zwPositions[0];
        float2 q1 = lines[j].zwPositions[1];
        float dx = q1.x - q0.x;
        float dy = q1.y - q0.y;
        lineDx[j] = dx;
        lineDy[j] = dy;
        lineRNum[j] = q0.x * dy - q0.y * dx;
        lineQ0x[j] = q0.x;
        lineQ0y[j] = q0.y;
    }

    // Initialize sin/cos for incremental rotation
    float theta0 = angleRange.x;
    float rayX = cos(theta0);
    float rayY = sin(theta0);
    float cosStep = cos(angleStep);
    float sinStep = sin(angleStep);

    float3 outputAccum = float3(0);
    float alphaAccum = 0.0;

    for (uint i = 0; i < sampleCount; i++)
    {
        int closestLine = -1;
        float closestDist = 1e10;
        float closestVal = 0;

        for (uint j = 0; j < numLines; j++)
        {
            float rDen = rayX * lineDy[j] - rayY * lineDx[j];
            if (abs(rDen) < 1e-12)
                continue;
            float invRDen = 1.0 / rDen;
            float r = lineRNum[j] * invRDen;
            float s = (lineQ0x[j] * rayY - lineQ0y[j] * rayX) * invRDen;

            if (s < 0.0 || s > 1.0)
                continue;
            if (r < 0.0)
                continue;

            if (closestLine < 0 || r < closestDist)
            {
                closestLine = int(j);
                closestDist = r;
                closestVal = s;
            }
        }

        if (closestLine >= 0)
        {
            float val = closestVal;
            float4 texPos = lines[closestLine].texturePositions[0] * (1.0 - val) +
                           lines[closestLine].texturePositions[1] * val;
            MaterialProperties mat = sampleMaterial(
                lines[closestLine].textureId,
                texPos,
                workingData.timeTicksMs);
            float4 normal = lines[closestLine].normal;

            // Two-sided Lambert diffuse
            float NdotL = max(0.0, dot(normal, sunViewDir));
            float NdotLBack = max(0.0, dot(-normal, sunViewDir));
            float diffuse = max(NdotL, NdotLBack);

            float3 ambient = float3(0.08, 0.09, 0.12);
            float3 sunColor = float3(1.0, 0.95, 0.9) * 0.8;

            float3 litColor = mat.albedo.xyz * (ambient + sunColor * diffuse)
                            + mat.luminance * mat.albedo.xyz;

            outputAccum += litColor / float(sampleCount);
            alphaAccum += 1.0 / float(sampleCount);
        }

        // Incremental rotation: (cos(θ+Δ), sin(θ+Δ)) = rotate(cos(θ), sin(θ))
        float newRayX = rayX * cosStep - rayY * sinStep;
        float newRayY = rayX * sinStep + rayY * cosStep;
        rayX = newRayX;
        rayY = newRayY;
    }

    return float4(outputAccum, alphaAccum);
}

// ZW line rendering with occlusion culling
// First removes fully occluded lines, then uses the simple closest-line-per-angle algorithm
// on the reduced set for correct depth ordering.
float4 renderZWLines2(inout ZWLine lines[MAX_ZW_LINES], uint numLines, WorkingData workingData)
{
    // VTE overlay mode only draws small post-raster previews. Use a cheaper path:
    // skip O(n^2) occlusion prepass and reduce hidden-dimension sampling.
    if (workingData.padding[0] != 0u)
    {
        return renderZWLinesSimple(lines, numLines, workingData, 64u);
    }

    // Stage 1: Remove completely occluded lines.
    for (uint i = 0; i < numLines; i++)
    {
        if (lines[i].textureId == 0)
            continue;

        bool completeOcclusion = false;
        for (uint j = 0; j < numLines; j++)
        {
            if (j == i || lines[j].textureId == 0)
                continue;

            bool localFullOcclusion = true;
            for (uint k = 0; k < 2; k++)
            {
                float2 lineA[2] = { float2(0), lines[i].zwPositions[k] };
                float2 lineB[2] = { lines[j].zwPositions[0], lines[j].zwPositions[1] };
                float2 intersection = getIntersection(lineA, lineB);
                bool doesOcclude = intersection.x >= 0.0 && intersection.x <= 1.0
                    && intersection.y >= 0.0 && intersection.y <= 1.0;
                if (!doesOcclude)
                {
                    localFullOcclusion = false;
                    break;
                }
            }
            if (localFullOcclusion)
            {
                completeOcclusion = true;
                break;
            }
        }
        if (completeOcclusion)
            lines[i].textureId = 0;
    }

    numLines = condenseZWLines(lines, numLines);

    // After culling fully occluded lines, use the simple closest-line-per-angle
    // algorithm on the reduced set. Stages 2-3 (intersection detection, trimming,
    // and per-line integration) can miss partial overlap configurations where
    // neither line fully occludes the other, causing double-counted contributions.
    return renderZWLinesSimple(lines, numLines, workingData, 256u);
}

// Bin clipped tetrahedra into screen-space tiles
[shader("compute")]
[numthreads(64, 1, 1)]
void mainBinTetsCS(uint3 globalId : SV_DispatchThreadID)
{
    WorkingData workingData = workingDataBuffer[0];

    uint tetIndex = globalId.x;
    uint actualTetCount = atomicCounter[0];
    if (tetIndex >= actualTetCount)
        return;

    Tetrahedron tet = outputTetrahedrons[tetIndex];
    if (tet.materialId == 0)
        return;

    // Compute 2D bounding box in NDC [-1,1]
    float2 bboxMin = min(min(tet.vertexPositions[0].xy, tet.vertexPositions[1].xy),
                         min(tet.vertexPositions[2].xy, tet.vertexPositions[3].xy));
    float2 bboxMax = max(max(tet.vertexPositions[0].xy, tet.vertexPositions[1].xy),
                         max(tet.vertexPositions[2].xy, tet.vertexPositions[3].xy));

    // Convert NDC to pixel coordinates
    float2 pixMin = (bboxMin + 1.0) * 0.5 * float2(workingData.renderDimensions.xy);
    float2 pixMax = (bboxMax + 1.0) * 0.5 * float2(workingData.renderDimensions.xy);

    // Convert to tile coordinates (8x8 tiles)
    uint tilesX = (workingData.renderDimensions.x + 7) / 8;
    uint tilesY = (workingData.renderDimensions.y + 7) / 8;

    int tileMinX = max(int(pixMin.x) / 8, 0);
    int tileMinY = max(int(pixMin.y) / 8, 0);
    int tileMaxX = min(int(pixMax.x) / 8, int(tilesX) - 1);
    int tileMaxY = min(int(pixMax.y) / 8, int(tilesY) - 1);

    for (int ty = tileMinY; ty <= tileMaxY; ty++)
    {
        for (int tx = tileMinX; tx <= tileMaxX; tx++)
        {
            uint tileIdx = uint(ty) * tilesX + uint(tx);
            uint slot;
            InterlockedAdd(tileTetCounts[tileIdx], 1, slot);
            if (slot < MAX_TETS_PER_TILE)
                tileTetIndices[tileIdx * MAX_TETS_PER_TILE + slot] = tetIndex;
        }
    }
}

// Shared memory for tile tet list — cooperatively loaded by all 64 threads in workgroup
groupshared uint sharedTetCount;
groupshared uint sharedTetIndices[MAX_TETS_PER_TILE];

// Per-pixel tetrahedron rasterization
[shader("compute")]
[numthreads(8, 8, 1)]
void mainTetrahedronPixelCS(uint3 globalId : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    WorkingData workingData = workingDataBuffer[0];
    // Combinations for iterating over tetrahedron faces
    static const uint3 COMBINATIONS_4_3[4] = {
        uint3(0, 1, 2), uint3(0, 1, 3), uint3(0, 2, 3), uint3(1, 2, 3)
    };

    // Load tile tet list into shared memory
    uint tilesX = (workingData.renderDimensions.x + 7) / 8;
    uint tileIdx = groupId.y * tilesX + groupId.x;

    if (localIndex == 0)
        sharedTetCount = min(tileTetCounts[tileIdx], MAX_TETS_PER_TILE);
    GroupMemoryBarrierWithGroupSync();

    uint tetCount = sharedTetCount;
    for (uint loadIdx = localIndex; loadIdx < tetCount; loadIdx += 64)
        sharedTetIndices[loadIdx] = tileTetIndices[tileIdx * MAX_TETS_PER_TILE + loadIdx];
    GroupMemoryBarrierWithGroupSync();

    ZWLine currentZWLines[MAX_ZW_LINES];
    uint numZWLines = 0;

    uint2 uPixelPos = globalId.xy;
    if (uPixelPos.x >= workingData.renderDimensions.x || uPixelPos.y >= workingData.renderDimensions.y)
        return;

    float2 pixelPos = float2(
        float(uPixelPos.x) / float(workingData.renderDimensions.x) * 2.0 - 1.0,
        float(uPixelPos.y) / float(workingData.renderDimensions.y) * 2.0 - 1.0
    );

    // Test each tetrahedron in tile
    for (uint i = 0; i < tetCount; i++)
    {
        Tetrahedron tet = outputTetrahedrons[sharedTetIndices[i]];

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

        // Compute ZW line from the two face intersections
        // Uses perspective-correct interpolation: interpolate attr/d and 1/d
        // in screen space, then divide to recover correct view-space values.
        ZWLine newLine;
        newLine.textureId = tet.materialId;
        newLine.normal = tet.normal;

        for (uint j = 0; j < 2; j++)
        {
            float3 bary = barycentrics[triIndexes[j]];
            uint3 combo = COMBINATIONS_4_3[triIndexes[j]];

            float3 invD = float3(
                tet.invProjectionDivisors[combo[0]],
                tet.invProjectionDivisors[combo[1]],
                tet.invProjectionDivisors[combo[2]]
            );

            // Interpolate 1/d in screen space
            float interpInvD = bary.x * invD.x + bary.y * invD.y + bary.z * invD.z;

            // Interpolate (zw * invD) in screen space, then divide by interpInvD
            float2 zwVertices[3] = {
                tet.vertexPositions[combo[0]].zw,
                tet.vertexPositions[combo[1]].zw,
                tet.vertexPositions[combo[2]].zw
            };
            float2 zwOverD = bary.x * zwVertices[0] * invD.x
                           + bary.y * zwVertices[1] * invD.y
                           + bary.z * zwVertices[2] * invD.z;
            float2 zwValue = zwOverD / interpInvD;

            // Interpolate (tex * invD) in screen space, then divide by interpInvD
            float4 texVertices[3] = {
                tet.texturePositions[combo[0]],
                tet.texturePositions[combo[1]],
                tet.texturePositions[combo[2]]
            };
            float4 texOverD = bary.x * texVertices[0] * invD.x
                            + bary.y * texVertices[1] * invD.y
                            + bary.z * texVertices[2] * invD.z;
            float4 texValue = texOverD / interpInvD;

            newLine.zwPositions[j] = zwValue;
            newLine.texturePositions[j] = texValue;
        }

        if (numZWLines < MAX_ZW_LINES)
        {
            currentZWLines[numZWLines] = newLine;
            numZWLines++;
        }
        else
        {
            // Array full: replace the farthest ZW line if new one is closer.
            // ZW depth = distance from origin; smaller = closer to camera.
            float newDepth = length(newLine.zwPositions[0]) + length(newLine.zwPositions[1]);
            float maxDepth = 0.0;
            uint maxIdx = 0;
            for (uint k = 0; k < MAX_ZW_LINES; k++)
            {
                float d = length(currentZWLines[k].zwPositions[0]) + length(currentZWLines[k].zwPositions[1]);
                if (d > maxDepth) { maxDepth = d; maxIdx = k; }
            }
            if (newDepth < maxDepth)
                currentZWLines[maxIdx] = newLine;
        }
    }

    uint pixelIndex = uPixelPos.y * workingData.renderDimensions.x + uPixelPos.x;

    if (numZWLines > 0)
    {
        // Normal raster output: opaque color where tetrahedra project.
        pixelBuffer[pixelIndex] = renderZWLines2(currentZWLines, numZWLines, workingData);
    }
    else if (workingData.padding[0] == 0u)
    {
        // Legacy raster/ray paths expect uncovered pixels to be cleared each frame.
        pixelBuffer[pixelIndex] = float4(0, 0, 0, 1.0);
    }
    // In VTE mode (padding[0] != 0), leave uncovered pixels untouched so
    // voxel Stage A/B output remains visible behind the overlay raster.

    // Only clear other Z slices for non-VTE paths. VTE uses layer 0 as the
    // collapsed display target, and clearing here can clobber Stage A data.
    if (workingData.padding[0] == 0u)
    {
        for (uint z = 1; z < workingData.renderDimensions.z; z++)
        {
            uint index = z * workingData.renderDimensions.x * workingData.renderDimensions.y
                       + uPixelPos.y * workingData.renderDimensions.x
                       + uPixelPos.x;
            pixelBuffer[index] = float4(0);
        }
    }
}
