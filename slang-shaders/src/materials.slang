// Material sampling - procedural materials by ID

import types;

MaterialProperties sampleMaterial(uint textureId, float4 texturePos, uint timeTicksMs)
{
    const float basicLuminance = 0.001;
    const float tau = 6.2831853;
    const uint previewMaterialFlag = 0x80000000u;
    bool isPreviewMaterial = (textureId & previewMaterialFlag) != 0u;
    uint baseTextureId = textureId & (~previewMaterialFlag);
    float4 p4 = texturePos;
    float3 p = p4.xyz;
    float t = float(timeTicksMs) * 0.001;

    MaterialProperties mat;
    mat.metallic = 0.0;
    mat.roughness = 0.0;
    mat.luminance = basicLuminance;
    mat.integrationWeight = 1.0;

    switch (baseTextureId)
    {
    case 1:
        mat.albedo = float4(1.0, 0.0, 0.0, 1.0); // Red
        break;
    case 2:
        mat.albedo = float4(1.0, 0.8, 0.0, 1.0); // Orange
        break;
    case 3:
        mat.albedo = float4(0.5, 1.0, 0.0, 1.0); // Yellow-green
        break;
    case 4:
        mat.albedo = float4(0.0, 1.0, 0.2, 1.0); // Green
        break;
    case 5:
        mat.albedo = float4(0.0, 1.0, 1.0, 1.0); // Cyan
        break;
    case 6:
        mat.albedo = float4(0.0, 0.2, 1.0, 1.0); // Blue
        break;
    case 7:
        mat.albedo = float4(0.5, 0.0, 1.0, 1.0); // Purple
        break;
    case 8:
        mat.albedo = float4(1.0, 0.0, 0.8, 1.0); // Magenta
        break;
    case 9:
        // Texture-coordinate based color
        mat.albedo = float4((p + float3(1.0)) / 2.0, 1.0);
        mat.luminance = 0.4;
        break;
    case 10:
        mat.albedo = float4(39.0/256.0, 69.0/256.0, 19.8/256.0, 1.0); // Brown
        mat.luminance = 0.0;
        break;
    case 11:
        // Neutral ground with 4D grid references:
        // - major lines every 4 units
        // - minor lines every 1 unit
        {
            float4 minorDist = abs(frac(p4 + 0.5) - 0.5);
            float minorAxis = min(minorDist.x, min(minorDist.y, min(minorDist.z, minorDist.w)));
            float minorLine = 1.0 - saturate(minorAxis / 0.055);

            float4 majorDist = abs(frac(p4 * 0.25 + 0.5) - 0.5);
            float majorAxis = min(majorDist.x, min(majorDist.y, min(majorDist.z, majorDist.w)));
            float majorLine = 1.0 - saturate(majorAxis / 0.085);

            float3 baseColor = float3(0.45, 0.47, 0.49);
            float3 minorColor = float3(0.36, 0.38, 0.40);
            float3 majorColor = float3(0.66, 0.68, 0.71);
            float3 color = lerp(baseColor, minorColor, minorLine * 0.6);
            color = lerp(color, majorColor, majorLine);

            mat.albedo = float4(color, 1.0);
        }
        mat.metallic = 0.05;
        mat.roughness = 0.75;
        mat.luminance = 0.0;
        break;
    case 12:
        mat.albedo = float4(1.0, 1.0, 1.0, 1.0); // White
        mat.luminance = 0.0;
        break;
    case 13:
        // Light source
        mat.albedo = float4(1.0, 1.0, 1.0, 1.0);
        mat.luminance = 40.0;
        break;
    case 14:
        // Mirror walls
        mat.albedo = float4(1.0, 1.0, 1.0, 1.0);
        mat.metallic = 1.0;
        mat.luminance = 0.0;
        break;
    case 15:
        // Lava-veined basalt
        {
            p = frac(p);
            float veinField = 0.5 + 0.5 * sin(
                (p.x * 11.0 + p.y * 13.0 + p.z * 17.0 + p4.w * 7.0) * tau + t * 1.8);
            float veins = pow(veinField, 6.0);
            float thermal = 0.65 + 0.35 * sin((p.x * 4.0 + p4.w * 3.0 + t * 0.7) * tau);
            float3 rockColor = float3(0.08, 0.05, 0.04);
            float3 lavaColor = float3(0.95, 0.36, 0.08);
            mat.albedo = float4(lerp(rockColor, lavaColor, veins), 1.0);
            mat.roughness = 0.8;
            mat.luminance = (0.08 + 6.0 * veins) * thermal;
        }
        break;
    case 16:
        // Crystal lattice
        {
            p = frac(p);
            float3 cell = abs(frac(p * 8.0) - 0.5);
            float line = 1.0 - saturate(min(cell.x, min(cell.y, cell.z)) * 36.0);
            float linePulse = 0.65 + 0.35 * sin(
                (p.x * 5.0 + p.y * 7.0 + p.z * 9.0 + p4.w * 11.0 + t * 2.2) * tau);
            float3 baseColor = float3(0.02, 0.06, 0.09);
            float3 lineColor = float3(0.25, 0.9, 1.0);
            mat.albedo = float4(lerp(baseColor, lineColor, line), 1.0);
            mat.roughness = 0.1;
            mat.luminance = 0.16 + line * (1.6 + 1.2 * linePulse);
        }
        break;
    case 17:
        // Marble swirl
        {
            p = frac(p);
            float swirl = sin((p.x * 14.0 + p.z * 9.0) * tau + sin(p.y * tau * 5.0) * 1.5);
            float veins = smoothstep(0.62, 0.88, 0.5 + 0.5 * swirl);
            float3 stoneBase = float3(0.82, 0.84, 0.88);
            float3 veinColor = float3(0.96, 0.97, 1.0);
            mat.albedo = float4(lerp(stoneBase, veinColor, veins), 1.0);
            mat.roughness = 0.25;
            mat.luminance = 0.0;
        }
        break;
    case 18:
        // Oxidized metal
        {
            p = frac(p);
            float noise = frac(sin(dot(p * 16.0, float3(12.9898, 78.233, 37.719))) * 43758.5453);
            float rust = smoothstep(0.42, 0.8, noise);
            float3 steel = float3(0.55, 0.58, 0.60);
            float3 oxide = float3(0.48, 0.18, 0.07);
            mat.albedo = float4(lerp(steel, oxide, rust), 1.0);
            mat.metallic = 0.65 * (1.0 - rust);
            mat.roughness = lerp(0.2, 0.85, rust);
            mat.luminance = 0.0;
        }
        break;
    case 19:
        // Bio-spore moss
        {
            p = frac(p);
            float3 voxelCell = floor(p * 10.0);
            float seed = frac(sin(dot(voxelCell, float3(17.13, 3.71, 29.97))) * 15731.743);
            float spores = smoothstep(0.93, 1.0, seed);
            float sporePulse = 0.5 + 0.5 * sin((seed * 31.0 + t * 2.6) * tau);
            float3 mossBase = float3(0.06, 0.22, 0.09);
            float3 sporeGlow = float3(0.25, 0.75, 0.35);
            mat.albedo = float4(lerp(mossBase, sporeGlow, spores), 1.0);
            mat.roughness = 0.9;
            mat.luminance = spores * (1.4 + 3.1 * sporePulse);
        }
        break;
    case 20:
        // Void mirror
        {
            p = frac(p);
            float3 q = p - 0.5;
            float centerGlow = pow(saturate(1.0 - length(q) * 1.8), 2.0);
            float centerPulse = 0.5 + 0.5 * sin(
                (p4.x * 3.0 + p4.z * 4.0 + p4.w * 5.0 + t * 0.9) * tau);
            mat.albedo = float4(
                0.03 + 0.20 * centerGlow,
                0.05 + 0.10 * centerGlow,
                0.12 + 0.20 * centerGlow,
                1.0);
            mat.metallic = 1.0;
            mat.roughness = lerp(0.01, 0.05, 1.0 - centerPulse);
            mat.luminance = 0.08 + 0.28 * centerGlow * centerPulse;
        }
        break;
    case 21:
        // Avatar marker: dark edge bands with signed-axis accent lines.
        {
            float4 local = frac(p4 + 1e-4);
            float4 faceDist = min(local, 1.0 - local);
            float edgeDist = min(min(faceDist.x, faceDist.y), min(faceDist.z, faceDist.w));
            float edgeBand = 1.0 - smoothstep(0.05, 0.16, edgeDist);

            float4 axisLineDist = abs(local - 0.5);
            float xLine = 1.0 - smoothstep(0.03, 0.12, axisLineDist.x);
            float yLine = 1.0 - smoothstep(0.03, 0.12, axisLineDist.y);
            float zLine = 1.0 - smoothstep(0.03, 0.12, axisLineDist.z);
            float wLine = 1.0 - smoothstep(0.03, 0.12, axisLineDist.w);
            float3 axisTint =
                xLine * float3(0.95, 0.32, 0.32) +
                yLine * float3(0.35, 0.95, 0.52) +
                zLine * float3(0.34, 0.64, 0.98) +
                wLine * float3(0.95, 0.74, 0.28);
            float tintWeight = saturate((xLine + yLine + zLine + wLine) * 0.35);

            float pulse = 0.5 + 0.5 * sin((p4.x * 9.0 + p4.y * 11.0 + p4.z * 7.0 + p4.w * 13.0) * tau);

            float3 baseColor = float3(0.17, 0.20, 0.30);
            float3 edgeColor = float3(0.05, 0.05, 0.07);
            float3 color = lerp(baseColor, edgeColor, edgeBand * 0.88);
            color = lerp(color, axisTint, tintWeight * (0.28 + 0.72 * edgeBand));
            float responseBoost = 1.45;
            color *= responseBoost;

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.08;
            mat.roughness = 0.18;
            mat.luminance = 0.16 + 0.70 * edgeBand + 0.45 * tintWeight * pulse;
            mat.integrationWeight = 10.0;
        }
        break;
    case 22:
        // Holographic laminate
        {
            float4 local = frac(p4 + float4(t * 0.20, t * 0.11, -t * 0.17, t * 0.07));
            float bands = 0.5 + 0.5 * sin(
                (local.x * 10.0 + local.z * 14.0 + local.w * 9.0 + t * 1.4) * tau);
            float grain = frac(sin(dot(local * 19.0, float4(11.1, 21.7, 7.3, 15.9))) * 43758.5453);
            float glint = smoothstep(0.92, 1.0, grain);
            float3 baseColor = float3(0.08, 0.16, 0.28);
            float3 sweepColor = float3(0.20, 0.70, 1.00);
            float3 color = lerp(baseColor, sweepColor, bands);
            color += glint * 0.35;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.55;
            mat.roughness = 0.16;
            mat.luminance = 0.12 + 0.55 * bands + 0.40 * glint;
        }
        break;
    case 23:
        // Tidal glass
        {
            float4 q4 = p4 + float4(t * 0.30, -t * 0.20, t * 0.18, t * 0.24);
            float waveA = sin((q4.x * 5.0 + q4.z * 6.0 + q4.w * 4.0) * tau);
            float waveB = sin((q4.y * 7.0 - q4.z * 3.0 + q4.w * 5.0) * tau);
            float foam = smoothstep(0.55, 0.92, 0.5 + 0.25 * waveA + 0.25 * waveB);
            float3 deep = float3(0.03, 0.12, 0.32);
            float3 shallow = float3(0.10, 0.46, 0.78);
            mat.albedo = float4(lerp(deep, shallow, foam), 1.0);
            mat.metallic = 0.10;
            mat.roughness = 0.07;
            mat.luminance = 0.04 + 0.22 * foam;
        }
        break;
    case 24:
        // Circuit weave
        {
            float4 grid = abs(frac((p4 + float4(t * 0.12, 0.0, -t * 0.08, 0.0)) * 6.0) - 0.5);
            float traces = 1.0 - saturate(min(min(grid.x, grid.z), grid.w) * 18.0);
            float4 cell = floor((p4 + t * 0.10) * 6.0);
            float seed = frac(sin(dot(cell, float4(9.3, 19.7, 13.1, 5.9))) * 15731.743);
            float nodes = smoothstep(0.95, 1.0, seed);
            float3 baseColor = float3(0.04, 0.07, 0.05);
            float3 traceColor = float3(0.40, 0.95, 0.62);
            float3 nodeColor = float3(0.95, 0.98, 0.70);
            float3 color = lerp(baseColor, traceColor, traces * 0.75);
            color = lerp(color, nodeColor, nodes);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.20;
            mat.roughness = 0.42;
            mat.luminance = 0.06 + 0.65 * traces + 1.40 * nodes;
        }
        break;
    case 25:
        // Aurora stone
        {
            float4 q4 = p4 * 0.65;
            float flow = sin((q4.x * 4.0 + q4.y * 3.0 - q4.z * 5.0 + q4.w * 6.0 + t * 0.9) * tau)
                + 0.5 * sin((q4.x * 9.0 - q4.y * 7.0 + q4.z * 8.0 + t * 1.7) * tau);
            float band = smoothstep(0.15, 0.9, 0.5 + 0.25 * flow);
            float streak = smoothstep(0.74, 0.96, 0.5 + 0.5 * sin((q4.y * 12.0 + q4.w * 8.0 + t * 1.3) * tau));
            float3 low = float3(0.07, 0.04, 0.12);
            float3 high = float3(0.45, 0.16, 0.68);
            float3 streakColor = float3(0.17, 0.76, 0.48);
            float3 color = lerp(low, high, band);
            color = lerp(color, streakColor, streak * 0.6);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.08;
            mat.roughness = 0.60;
            mat.luminance = 0.03 + 0.22 * band + 0.20 * streak;
        }
        break;
    case 26:
        // Hazard chevrons
        {
            float4 local = frac(p4 + float4(t * 0.35, 0.0, 0.0, t * 0.35));
            float diag = frac(local.x * 0.9 + local.z * 1.4 + local.w * 1.2);
            float stripe = smoothstep(0.42, 0.58, diag);
            float pulse = 0.5 + 0.5 * sin((local.x * 4.0 + local.w * 6.0 + t * 3.0) * tau);
            float edge = 1.0 - smoothstep(
                0.05,
                0.20,
                min(min(local.x, 1.0 - local.x), min(local.z, 1.0 - local.z)));
            float3 dark = float3(0.07, 0.07, 0.06);
            float3 warning = float3(0.92, 0.66, 0.09);
            float3 color = lerp(dark, warning, stripe);
            color = lerp(color, float3(0.02, 0.02, 0.02), edge * 0.6);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.12;
            mat.roughness = 0.34;
            mat.luminance = 0.04 + 0.50 * stripe * pulse;
        }
        break;
    case 27:
        // Stone (Minecraft-style variant)
        {
            float4 cell = floor((p4 + 64.0) * 8.0);
            float n0 = frac(sin(dot(cell, float4(12.9898, 78.233, 37.719, 53.157))) * 43758.5453);
            float n1 = frac(sin(dot(cell + 19.0, float4(93.9898, 67.345, 12.731, 45.332))) * 24634.6345);
            float n2 = frac(sin(dot(cell + 7.0, float4(41.123, 13.173, 91.777, 27.909))) * 32145.8796);
            float speck = smoothstep(0.86, 1.0, n0);
            float fracture = smoothstep(0.42, 0.70, abs(n1 - 0.5));
            float3 base = float3(0.49, 0.50, 0.52) + (n2 - 0.5) * 0.06;
            float3 speckColor = float3(0.36, 0.37, 0.39);
            float3 color = lerp(base, speckColor, speck * 0.65);
            color = lerp(color, float3(0.58, 0.59, 0.62), fracture * 0.2);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.86;
            mat.luminance = 0.0;
        }
        break;
    case 28:
        // Cobblestone (Minecraft-style variant)
        {
            float4 blockUv = frac(p4 * 2.8);
            float4 blockDist = min(blockUv, 1.0 - blockUv);
            float mortarMask = 1.0 - smoothstep(
                0.06,
                0.18,
                min(min(blockDist.x, blockDist.y), min(blockDist.z, blockDist.w)));
            float4 blockCell = floor(p4 * 2.8);
            float jitter = frac(sin(dot(blockCell, float4(17.17, 31.41, 47.73, 11.31))) * 19753.381);
            float hue = frac(sin(dot(blockCell + 13.0, float4(67.1, 5.7, 23.3, 101.9))) * 15731.743);
            float lichen = smoothstep(0.95, 1.0, hue);
            float3 stoneA = float3(0.43, 0.44, 0.45);
            float3 stoneB = float3(0.53, 0.54, 0.56);
            float3 cobble = lerp(stoneA, stoneB, jitter);
            cobble = lerp(cobble, float3(0.40, 0.47, 0.37), lichen * 0.45);
            float3 mortar = float3(0.26, 0.27, 0.29);
            float3 color = lerp(cobble, mortar, mortarMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.92;
            mat.luminance = 0.0;
        }
        break;
    case 29:
        // Dirt (Minecraft-style variant)
        {
            float4 soilCell = floor((p4 + 32.0) * 10.0);
            float grain0 = frac(sin(dot(soilCell, float4(13.731, 57.173, 21.917, 83.113))) * 18731.531);
            float grain1 = frac(sin(dot(soilCell + 9.0, float4(71.177, 9.711, 45.317, 29.219))) * 42731.911);
            float clumps = smoothstep(0.62, 0.9, grain0);
            float pebbles = smoothstep(0.94, 1.0, grain1);
            float3 base = float3(0.43, 0.30, 0.18);
            float3 clumpColor = float3(0.35, 0.23, 0.13);
            float3 pebbleColor = float3(0.23, 0.22, 0.21);
            float3 color = lerp(base, clumpColor, clumps * 0.55);
            color = lerp(color, pebbleColor, pebbles * 0.9);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.95;
            mat.luminance = 0.0;
        }
        break;
    case 30:
        // Coarse dirt (Minecraft-style variant)
        {
            float4 soilCell = floor((p4 + 48.0) * 7.0);
            float dirt = frac(sin(dot(soilCell, float4(9.21, 61.73, 37.19, 15.91))) * 31731.173);
            float gravel = frac(sin(dot(soilCell + 17.0, float4(51.37, 19.19, 77.71, 33.17))) * 51731.713);
            float grit = smoothstep(0.58, 0.88, dirt);
            float stones = smoothstep(0.88, 0.985, gravel);
            float3 base = float3(0.41, 0.29, 0.17);
            float3 gritColor = float3(0.30, 0.22, 0.12);
            float3 stoneColor = float3(0.47, 0.44, 0.40);
            float3 color = lerp(base, gritColor, grit * 0.6);
            color = lerp(color, stoneColor, stones * 0.8);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.96;
            mat.luminance = 0.0;
        }
        break;
    case 31:
        // Oak planks (Minecraft-style variant)
        {
            float plankBand = frac(p4.y * 4.0);
            float seam = 1.0 - smoothstep(0.02, 0.09, min(plankBand, 1.0 - plankBand));
            float grain = 0.5 + 0.5 * sin((p4.x * 8.0 + p4.z * 5.0 + p4.w * 3.0) * tau);
            float knotField = 0.5 + 0.5 * sin((p4.x * 15.0 + p4.z * 13.0 + p4.w * 4.0) * tau);
            float knot = smoothstep(0.83, 0.97, knotField);
            float3 plankBase = lerp(float3(0.60, 0.46, 0.24), float3(0.53, 0.39, 0.19), grain);
            plankBase = lerp(plankBase, float3(0.37, 0.26, 0.12), knot * 0.45);
            plankBase = lerp(plankBase, float3(0.32, 0.22, 0.10), seam * 0.8);
            mat.albedo = float4(plankBase, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.88;
            mat.luminance = 0.0;
        }
        break;
    case 32:
        // Spruce planks (Minecraft-style variant)
        {
            float plankBand = frac(p4.y * 5.0);
            float seam = 1.0 - smoothstep(0.02, 0.08, min(plankBand, 1.0 - plankBand));
            float grain = 0.5 + 0.5 * sin((p4.x * 10.0 + p4.z * 4.0 + p4.w * 6.0) * tau);
            float knotField = 0.5 + 0.5 * sin((p4.x * 18.0 + p4.z * 9.0 + p4.w * 11.0) * tau);
            float knot = smoothstep(0.80, 0.96, knotField);
            float3 plankBase = lerp(float3(0.42, 0.30, 0.16), float3(0.34, 0.23, 0.12), grain);
            plankBase = lerp(plankBase, float3(0.25, 0.17, 0.09), knot * 0.5);
            plankBase = lerp(plankBase, float3(0.18, 0.12, 0.06), seam * 0.85);
            mat.albedo = float4(plankBase, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.90;
            mat.luminance = 0.0;
        }
        break;
    case 33:
        // Log bark (Minecraft-style variant)
        {
            float ridge = 0.5 + 0.5 * sin((p4.x * 11.0 + p4.z * 9.0 + p4.w * 7.0) * tau);
            float crack = 0.5 + 0.5 * sin((p4.x * 23.0 - p4.z * 17.0 + p4.w * 13.0) * tau);
            float crackMask = smoothstep(0.80, 0.96, crack);
            float growthBand = frac(p4.y * 3.0);
            float ringSeam = 1.0 - smoothstep(0.03, 0.12, min(growthBand, 1.0 - growthBand));
            float3 barkA = float3(0.34, 0.24, 0.13);
            float3 barkB = float3(0.26, 0.17, 0.09);
            float3 color = lerp(barkA, barkB, ridge);
            color = lerp(color, float3(0.15, 0.10, 0.06), crackMask * 0.8);
            color = lerp(color, float3(0.44, 0.32, 0.18), ringSeam * 0.25);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.93;
            mat.luminance = 0.0;
        }
        break;
    case 34:
        // Log end rings (Minecraft-style variant)
        {
            float3 endLocal = frac(float3(p4.x, p4.z, p4.w)) - 0.5;
            float radius = length(endLocal) * 2.2;
            float ringWave = 0.5 + 0.5 * sin(radius * 22.0);
            float core = smoothstep(0.0, 0.35, 1.0 - radius);
            float3 ringCell = floor((endLocal + 0.5) * 12.0);
            float pores = frac(sin(dot(ringCell, float3(12.91, 78.23, 45.17))) * 41358.123);
            float poreMask = smoothstep(0.94, 1.0, pores);
            float3 sapwood = float3(0.67, 0.52, 0.29);
            float3 heartwood = float3(0.50, 0.34, 0.17);
            float3 ringDark = float3(0.34, 0.22, 0.11);
            float3 color = lerp(sapwood, heartwood, core);
            color = lerp(color, ringDark, ringWave * 0.35);
            color = lerp(color, float3(0.25, 0.18, 0.10), poreMask * 0.6);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.89;
            mat.luminance = 0.0;
        }
        break;
    case 35:
        // Sand
        {
            float4 cell = floor((p4 + 128.0) * 12.0);
            float n0 = frac(sin(dot(cell, float4(17.91, 43.17, 67.23, 29.71))) * 31731.941);
            float n1 = frac(sin(dot(cell + 11.0, float4(53.17, 97.11, 13.91, 71.23))) * 47831.173);
            float grains = 0.5 + 0.5 * sin((p4.x * 25.0 + p4.z * 23.0) * tau);
            float3 base = float3(0.89, 0.83, 0.64);
            float3 dark = float3(0.81, 0.74, 0.56);
            float3 color = lerp(base, dark, n0 * 0.2);
            color = lerp(color, float3(0.93, 0.88, 0.70), n1 * grains * 0.15);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.94;
            mat.luminance = 0.0;
        }
        break;
    case 36:
        // Gravel
        {
            float4 cell = floor((p4 + 128.0) * 5.0);
            float n0 = frac(sin(dot(cell, float4(19.31, 61.17, 47.91, 83.23))) * 21731.531);
            float n1 = frac(sin(dot(cell + 7.0, float4(71.93, 13.47, 89.11, 37.67))) * 39731.817);
            float n2 = frac(sin(dot(cell + 23.0, float4(41.71, 97.13, 17.91, 53.29))) * 51731.293);
            float stones = smoothstep(0.3, 0.7, n0);
            float3 gray1 = float3(0.52, 0.52, 0.54);
            float3 gray2 = float3(0.44, 0.43, 0.46);
            float3 gray3 = float3(0.36, 0.37, 0.39);
            float3 color = lerp(gray1, gray2, stones);
            color = lerp(color, gray3, n1 * 0.4);
            color *= 0.96 + n2 * 0.08;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.93;
            mat.luminance = 0.0;
        }
        break;
    case 37:
        // Clay
        {
            float4 cell = floor((p4 + 128.0) * 9.0);
            float n0 = frac(sin(dot(cell, float4(23.17, 59.91, 37.23, 71.47))) * 27731.391);
            float n1 = frac(sin(dot(cell + 13.0, float4(83.19, 17.71, 91.13, 41.97))) * 43731.671);
            float smooth = 0.5 + 0.5 * sin((p4.x * 7.0 + p4.y * 5.0 + p4.w * 6.0) * tau);
            float3 base = float3(0.63, 0.61, 0.59);
            float3 warm = float3(0.67, 0.59, 0.52);
            float3 color = lerp(base, warm, smooth * 0.4);
            color *= 0.94 + (n0 + n1) * 0.06;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.88;
            mat.luminance = 0.0;
        }
        break;
    case 38:
        // Grass Block (normal-dependent: green top, brown sides/bottom)
        {
            // Use normal vector to determine if this is a "top" face
            // In 4D, we check if any component is strongly positive
            float4 approxNormal = normalize(frac(p4 * 100.0) - 0.5);
            float isTop = saturate((approxNormal.y - 0.3) * 5.0);

            float4 cell = floor((p4 + 128.0) * 11.0);
            float n0 = frac(sin(dot(cell, float4(31.17, 67.91, 19.23, 53.71))) * 29731.471);
            float n1 = frac(sin(dot(cell + 17.0, float4(47.93, 23.17, 79.11, 61.37))) * 37731.923);

            // Green grass top
            float3 grassBase = float3(0.35, 0.62, 0.28);
            float3 grassDark = float3(0.28, 0.52, 0.22);
            float3 grassColor = lerp(grassBase, grassDark, n0 * 0.4);

            // Brown dirt sides
            float3 dirtBase = float3(0.43, 0.30, 0.18);
            float3 dirtDark = float3(0.35, 0.23, 0.13);
            float3 dirtColor = lerp(dirtBase, dirtDark, n1 * 0.5);

            float3 color = lerp(dirtColor, grassColor, isTop);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.90;
            mat.luminance = 0.0;
        }
        break;
    case 39:
        // Snow
        {
            float4 cell = floor((p4 + 128.0) * 16.0);
            float n0 = frac(sin(dot(cell, float4(37.91, 73.17, 11.23, 59.47))) * 33731.591);
            float n1 = frac(sin(dot(cell + 19.0, float4(61.17, 29.93, 87.11, 43.71))) * 41731.293);
            float crystals = smoothstep(0.88, 0.98, n0);
            float3 base = float3(0.94, 0.96, 0.98);
            float3 shadow = float3(0.88, 0.91, 0.94);
            float3 sparkle = float3(0.98, 0.99, 1.00);
            float3 color = lerp(base, shadow, n1 * 0.15);
            color = lerp(color, sparkle, crystals * 0.3);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.85;
            mat.luminance = 0.0;
        }
        break;
    case 40:
        // Ice (translucent)
        {
            float4 cell = floor((p4 + 128.0) * 8.0);
            float n0 = frac(sin(dot(cell, float4(41.17, 79.91, 23.71, 67.13))) * 35731.691);
            float cracks = 0.5 + 0.5 * sin((p4.x * 13.0 + p4.z * 11.0 + p4.w * 9.0) * tau);
            float crackMask = smoothstep(0.82, 0.96, cracks);
            float3 base = float3(0.72, 0.88, 0.96);
            float3 crackColor = float3(0.85, 0.94, 0.98);
            float3 color = lerp(base, crackColor, crackMask * 0.3);
            color *= 0.94 + n0 * 0.12;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.1;
            mat.roughness = 0.05;
            mat.luminance = 0.0;
            mat.integrationWeight = 0.3;
        }
        break;
    case 41:
        // Coal Ore
        {
            float4 stoneCell = floor((p4 + 128.0) * 8.0);
            float stoneNoise = frac(sin(dot(stoneCell, float4(43.91, 81.17, 29.23, 73.47))) * 37731.791);
            float3 stoneBase = float3(0.49, 0.50, 0.52) + (stoneNoise - 0.5) * 0.06;

            float4 oreCell = floor((p4 + 128.0) * 5.0);
            float oreNoise = frac(sin(dot(oreCell, float4(67.17, 19.91, 97.11, 31.73))) * 43731.391);
            float oreMask = smoothstep(0.65, 0.82, oreNoise);

            float3 coalColor = float3(0.09, 0.09, 0.10);
            float3 color = lerp(stoneBase, coalColor, oreMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.86;
            mat.luminance = 0.0;
        }
        break;
    case 42:
        // Iron Ore
        {
            float4 stoneCell = floor((p4 + 128.0) * 8.0);
            float stoneNoise = frac(sin(dot(stoneCell, float4(47.17, 83.91, 31.71, 79.23))) * 39731.891);
            float3 stoneBase = float3(0.49, 0.50, 0.52) + (stoneNoise - 0.5) * 0.06;

            float4 oreCell = floor((p4 + 128.0) * 6.0);
            float oreNoise = frac(sin(dot(oreCell, float4(71.93, 23.17, 89.71, 37.91))) * 45731.491);
            float oreMask = smoothstep(0.62, 0.80, oreNoise);

            float3 ironColor = float3(0.72, 0.62, 0.53);
            float3 color = lerp(stoneBase, ironColor, oreMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.86;
            mat.luminance = 0.0;
        }
        break;
    case 43:
        // Gold Ore
        {
            float4 stoneCell = floor((p4 + 128.0) * 8.0);
            float stoneNoise = frac(sin(dot(stoneCell, float4(51.91, 87.17, 37.23, 83.71))) * 41731.991);
            float3 stoneBase = float3(0.49, 0.50, 0.52) + (stoneNoise - 0.5) * 0.06;

            float4 oreCell = floor((p4 + 128.0) * 5.5);
            float oreNoise = frac(sin(dot(oreCell, float4(79.17, 29.93, 93.11, 41.67))) * 47731.591);
            float oreMask = smoothstep(0.64, 0.84, oreNoise);

            float3 goldColor = float3(0.95, 0.78, 0.18);
            float3 color = lerp(stoneBase, goldColor, oreMask);
            mat.metallic = oreMask * 0.7;
            mat.albedo = float4(color, 1.0);
            mat.roughness = 0.86;
            mat.luminance = 0.0;
        }
        break;
    case 44:
        // Diamond Ore
        {
            float4 stoneCell = floor((p4 + 128.0) * 8.0);
            float stoneNoise = frac(sin(dot(stoneCell, float4(53.17, 89.91, 39.71, 87.23))) * 43731.091);
            float3 stoneBase = float3(0.49, 0.50, 0.52) + (stoneNoise - 0.5) * 0.06;

            float4 oreCell = floor((p4 + 128.0) * 7.0);
            float oreNoise = frac(sin(dot(oreCell, float4(83.93, 31.17, 97.71, 47.91))) * 49731.691);
            float oreMask = smoothstep(0.70, 0.88, oreNoise);

            float sparkle = 0.5 + 0.5 * sin((p4.x * 40.0 + p4.w * 35.0 + t * 3.0) * tau);
            float3 diamondColor = float3(0.42, 0.78, 0.88);
            float3 color = lerp(stoneBase, diamondColor, oreMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.86;
            mat.luminance = oreMask * sparkle * 0.08;
        }
        break;
    case 45:
        // Redstone Ore
        {
            float4 stoneCell = floor((p4 + 128.0) * 8.0);
            float stoneNoise = frac(sin(dot(stoneCell, float4(59.91, 91.17, 41.23, 89.47))) * 45731.191);
            float3 stoneBase = float3(0.49, 0.50, 0.52) + (stoneNoise - 0.5) * 0.06;

            float4 oreCell = floor((p4 + 128.0) * 6.5);
            float oreNoise = frac(sin(dot(oreCell, float4(87.17, 37.93, 99.11, 53.71))) * 51731.791);
            float oreMask = smoothstep(0.63, 0.83, oreNoise);

            float pulse = 0.5 + 0.5 * sin(t * 1.5);
            float3 redstoneColor = float3(0.82, 0.08, 0.08);
            float3 color = lerp(stoneBase, redstoneColor, oreMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.86;
            mat.luminance = oreMask * (0.3 + pulse * 0.4);
        }
        break;
    case 46:
        // Birch Planks
        {
            float plankBand = frac(p4.y * 4.5);
            float seam = 1.0 - smoothstep(0.02, 0.08, min(plankBand, 1.0 - plankBand));
            float grain = 0.5 + 0.5 * sin((p4.x * 9.0 + p4.z * 6.0 + p4.w * 4.0) * tau);
            float knotField = 0.5 + 0.5 * sin((p4.x * 16.0 + p4.z * 12.0 + p4.w * 7.0) * tau);
            float knot = smoothstep(0.85, 0.98, knotField);
            float3 plankBase = lerp(float3(0.88, 0.84, 0.72), float3(0.82, 0.77, 0.64), grain);
            plankBase = lerp(plankBase, float3(0.62, 0.55, 0.42), knot * 0.35);
            plankBase = lerp(plankBase, float3(0.54, 0.48, 0.36), seam * 0.7);
            mat.albedo = float4(plankBase, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.82;
            mat.luminance = 0.0;
        }
        break;
    case 47:
        // Bricks
        {
            float4 brickUv = frac(p4 * float4(3.0, 6.0, 3.0, 3.0) + float4(0.0, floor(p4.x * 3.0) * 0.5, 0.0, 0.0));
            float4 brickDist = min(brickUv, 1.0 - brickUv);
            float mortarMask = 1.0 - smoothstep(0.04, 0.10, min(min(brickDist.x, brickDist.y), min(brickDist.z, brickDist.w)));

            float4 brickCell = floor(p4 * 3.0);
            float variation = frac(sin(dot(brickCell, float4(61.17, 93.91, 43.23, 91.47))) * 47731.291);
            float3 brickBase = float3(0.58, 0.27, 0.17);
            float3 brickVar = float3(0.52, 0.23, 0.14);
            float3 mortar = float3(0.64, 0.62, 0.60);
            float3 color = lerp(brickBase, brickVar, variation);
            color = lerp(color, mortar, mortarMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.87;
            mat.luminance = 0.0;
        }
        break;
    case 48:
        // Sandstone (layered)
        {
            float layers = frac(p4.y * 8.0);
            float band = smoothstep(0.45, 0.55, layers);
            float4 cell = floor((p4 + 128.0) * 10.0);
            float grain = frac(sin(dot(cell, float4(67.91, 97.17, 47.71, 93.23))) * 49731.391);
            float3 light = float3(0.87, 0.80, 0.62);
            float3 dark = float3(0.78, 0.71, 0.54);
            float3 color = lerp(light, dark, band * 0.3);
            color *= 0.94 + grain * 0.12;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.82;
            mat.luminance = 0.0;
        }
        break;
    case 49:
        // Glass (transparent)
        {
            float4 cell = floor((p4 + 128.0) * 20.0);
            float imperfections = frac(sin(dot(cell, float4(71.17, 99.93, 51.11, 97.71))) * 51731.891);
            float specs = smoothstep(0.97, 1.0, imperfections);
            float3 base = float3(0.88, 0.93, 0.96);
            float3 color = lerp(base, float3(0.95, 0.98, 1.00), specs);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.02;
            mat.luminance = 0.0;
            mat.integrationWeight = 0.05;
        }
        break;
    case 50:
        // Glowstone
        {
            float4 cell = floor((p4 + 128.0) * 6.0);
            float cracks0 = frac(sin(dot(cell, float4(73.93, 101.17, 53.71, 99.91))) * 53731.991);
            float cracks1 = frac(sin(dot(cell + 13.0, float4(89.17, 43.93, 107.11, 59.67))) * 57731.491);
            float crackMask = smoothstep(0.35, 0.65, cracks0 * cracks1);
            float pulse = 0.5 + 0.5 * sin(t * 1.2);
            float3 bright = float3(0.98, 0.93, 0.62);
            float3 dim = float3(0.88, 0.78, 0.45);
            float3 cracks = float3(0.72, 0.64, 0.38);
            float3 color = lerp(bright, dim, pulse * 0.15);
            color = lerp(color, cracks, crackMask);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.65;
            mat.luminance = 15.0 * (0.85 + pulse * 0.15);
        }
        break;
    case 51:
        // Obsidian
        {
            float4 cell = floor((p4 + 128.0) * 5.0);
            float flow = frac(sin(dot(cell, float4(79.17, 103.91, 59.23, 101.47))) * 55731.091);
            float swirl = 0.5 + 0.5 * sin((p4.x * 8.0 + p4.z * 6.0 + p4.w * 7.0) * tau);
            float3 deep = float3(0.06, 0.04, 0.08);
            float3 purple = float3(0.14, 0.08, 0.18);
            float3 color = lerp(deep, purple, flow * swirl * 0.5);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.2;
            mat.roughness = 0.15;
            mat.luminance = 0.0;
        }
        break;
    case 52:
        // Prismarine
        {
            float4 q4 = p4 * 0.8 + float4(t * 0.15, -t * 0.10, t * 0.12, t * 0.08);
            float pattern = 0.5 + 0.5 * sin((q4.x * 7.0 + q4.y * 5.0 + q4.z * 6.0 + q4.w * 8.0) * tau);
            float pattern2 = 0.5 + 0.5 * sin((q4.x * 11.0 - q4.z * 9.0 + q4.w * 10.0) * tau);
            float scales = smoothstep(0.35, 0.70, pattern * pattern2);
            float3 base = float3(0.25, 0.64, 0.62);
            float3 light = float3(0.38, 0.78, 0.74);
            float3 color = lerp(base, light, scales);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.1;
            mat.roughness = 0.6;
            mat.luminance = 0.02;
        }
        break;
    case 53:
        // Terracotta
        {
            float4 cell = floor((p4 + 128.0) * 8.0);
            float grain = frac(sin(dot(cell, float4(83.91, 107.17, 61.71, 103.23))) * 59731.191);
            float bands = 0.5 + 0.5 * sin(p4.y * 12.0 * tau);
            float3 base = float3(0.68, 0.42, 0.28);
            float3 light = float3(0.76, 0.48, 0.32);
            float3 color = lerp(base, light, bands * 0.2);
            color *= 0.94 + grain * 0.12;
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.85;
            mat.luminance = 0.0;
        }
        break;
    case 54:
        // Wool (White)
        {
            float4 cell = floor((p4 + 128.0) * 14.0);
            float fiber0 = frac(sin(dot(cell, float4(87.17, 109.93, 67.11, 107.71))) * 61731.291);
            float fiber1 = frac(sin(dot(cell + 17.0, float4(91.93, 47.17, 113.71, 71.91))) * 63731.391);
            float texture = fiber0 * fiber1;
            float3 base = float3(0.93, 0.93, 0.93);
            float3 shadow = float3(0.86, 0.86, 0.86);
            float3 color = lerp(base, shadow, texture * 0.25);
            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.0;
            mat.roughness = 0.96;
            mat.luminance = 0.0;
        }
        break;
    case 55:
        // Basalt tiles
        {
            float4 tileUv = frac(p4 * 2.4);
            float4 tileDist = min(tileUv, 1.0 - tileUv);
            float seam = 1.0 - smoothstep(
                0.04,
                0.10,
                min(min(tileDist.x, tileDist.y), min(tileDist.z, tileDist.w)));
            float4 tileCell = floor(p4 * 2.4);
            float grain = frac(sin(dot(tileCell, float4(97.17, 59.31, 83.71, 41.93))) * 65731.193);
            float crack = 0.5 + 0.5 * sin((p4.x * 9.0 + p4.z * 11.0 + p4.w * 7.0) * tau);
            float crackMask = smoothstep(0.78, 0.96, crack);

            float3 base = float3(0.20, 0.21, 0.24);
            float3 high = float3(0.31, 0.32, 0.36);
            float3 color = lerp(base, high, grain * 0.45);
            color = lerp(color, float3(0.08, 0.08, 0.09), seam * 0.85);
            color = lerp(color, float3(0.11, 0.11, 0.12), crackMask * 0.35);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.05;
            mat.roughness = 0.88;
            mat.luminance = 0.0;
        }
        break;
    case 56:
        // Copper weave
        {
            float4 weave = abs(frac((p4 + float4(t * 0.08, 0.0, -t * 0.04, t * 0.06)) * 8.0) - 0.5);
            float strands = 1.0 - saturate(min(min(weave.x, weave.z), weave.w) * 22.0);
            float cross = 1.0 - saturate(abs(frac((p4.y + p4.w) * 4.0) - 0.5) * 8.0);
            float4 cell = floor((p4 + 64.0) * 10.0);
            float patinaNoise = frac(sin(dot(cell, float4(101.17, 67.93, 43.71, 89.31))) * 67731.391);
            float patina = smoothstep(0.72, 0.94, patinaNoise);

            float3 copper = float3(0.67, 0.39, 0.20);
            float3 bright = float3(0.86, 0.55, 0.28);
            float3 oxide = float3(0.20, 0.42, 0.36);
            float3 color = lerp(copper, bright, strands * 0.45 + cross * 0.2);
            color = lerp(color, oxide, patina * 0.55);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.82;
            mat.roughness = lerp(0.20, 0.55, patina);
            mat.luminance = 0.02 + 0.08 * strands;
        }
        break;
    case 57:
        // Nebula strata
        {
            float4 q4 = p4 * 0.55 + float4(t * 0.05, -t * 0.03, t * 0.07, -t * 0.04);
            float flowA = sin((q4.x * 5.0 + q4.z * 4.0 + q4.w * 6.0) * tau);
            float flowB = sin((q4.y * 7.0 - q4.z * 3.0 + q4.w * 8.0) * tau);
            float cloud = smoothstep(0.25, 0.85, 0.5 + 0.3 * flowA + 0.2 * flowB);
            float4 cell = floor((q4 + 32.0) * 14.0);
            float star = smoothstep(
                0.97,
                1.0,
                frac(sin(dot(cell, float4(109.17, 71.93, 53.71, 97.31))) * 69731.731));

            float3 deep = float3(0.06, 0.07, 0.16);
            float3 mid = float3(0.28, 0.18, 0.42);
            float3 high = float3(0.46, 0.44, 0.78);
            float3 color = lerp(deep, mid, cloud);
            color = lerp(color, high, cloud * cloud * 0.5);
            color = lerp(color, float3(0.92, 0.88, 1.0), star);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.05;
            mat.roughness = 0.35;
            mat.luminance = 0.05 + 0.35 * cloud + 1.2 * star;
        }
        break;
    case 58:
        // Starforged core
        {
            float4 local = frac(p4) - 0.5;
            float radius = length(local.xyz) + abs(local.w) * 0.6;
            float core = pow(saturate(1.0 - radius * 1.9), 3.0);
            float corona = 0.5 + 0.5 * sin((radius * 14.0 - t * 3.2) * tau);
            float filaments = 0.5 + 0.5 * sin(
                (p4.x * 9.0 + p4.y * 7.0 + p4.z * 5.0 + p4.w * 11.0 + t * 2.8) * tau);

            float3 shell = float3(0.38, 0.30, 0.14);
            float3 hot = float3(1.0, 0.84, 0.42);
            float3 color = lerp(shell, hot, core);
            color = lerp(color, float3(1.0, 0.95, 0.75), filaments * core * 0.5);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.10;
            mat.roughness = 0.18;
            mat.luminance = 6.0 + 24.0 * core * (0.7 + 0.3 * corona);
        }
        break;
    case 59:
        // Cryo circuit
        {
            float4 grid = abs(frac((p4 + float4(0.0, t * 0.10, 0.0, -t * 0.10)) * 7.0) - 0.5);
            float traces = 1.0 - saturate(min(min(grid.x, grid.z), grid.w) * 18.0);
            float frost = 0.5 + 0.5 * sin((p4.x * 12.0 + p4.y * 9.0 + p4.w * 8.0) * tau);
            float crystal = smoothstep(0.82, 0.97, frost);

            float3 base = float3(0.08, 0.16, 0.20);
            float3 traceColor = float3(0.36, 0.78, 0.86);
            float3 frostColor = float3(0.74, 0.92, 0.97);
            float3 color = lerp(base, traceColor, traces * 0.7);
            color = lerp(color, frostColor, crystal * 0.45);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.15;
            mat.roughness = 0.20;
            mat.luminance = 0.08 + 0.7 * traces + 0.3 * crystal;
        }
        break;
    case 60:
        // Smoked glass
        {
            float4 cell = floor((p4 + 256.0) * 18.0);
            float impurity = frac(sin(dot(cell, float4(113.17, 79.93, 59.71, 101.31))) * 71731.193);
            float streak = 0.5 + 0.5 * sin((p4.x * 10.0 + p4.z * 9.0 + p4.w * 7.0) * tau);
            float haze = smoothstep(0.65, 0.93, impurity * streak);

            float3 base = float3(0.21, 0.28, 0.34);
            float3 tint = float3(0.32, 0.39, 0.46);
            float3 color = lerp(base, tint, haze * 0.45);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.02;
            mat.roughness = 0.04;
            mat.luminance = 0.0;
            mat.integrationWeight = 0.10;
        }
        break;
    case 61:
        // Ivory marble
        {
            float swirl = sin(
                (p4.x * 9.0 + p4.z * 7.0 + p4.w * 5.0) * tau
                + sin(p4.y * tau * 4.0) * 1.4);
            float veins = smoothstep(0.68, 0.91, 0.5 + 0.5 * swirl);
            float warm = 0.5 + 0.5 * sin((p4.y * 3.0 + p4.w * 2.0 + t * 0.2) * tau);

            float3 base = float3(0.87, 0.88, 0.90);
            float3 veinColor = float3(0.96, 0.96, 0.98);
            float3 warmTint = float3(0.91, 0.89, 0.84);
            float3 color = lerp(base, veinColor, veins * 0.8);
            color = lerp(color, warmTint, warm * 0.18);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.02;
            mat.roughness = 0.24;
            mat.luminance = 0.01 + veins * 0.04;
        }
        break;
    case 62:
        // Runic alloy
        {
            float4 local = frac((p4 + float4(t * 0.03, 0.0, -t * 0.02, t * 0.04)) * 4.0);
            float4 dist = abs(local - 0.5);
            float strokes = 1.0 - smoothstep(0.05, 0.16, min(min(dist.x, dist.z), dist.w));
            float glyph = 1.0 - smoothstep(
                0.03,
                0.10,
                abs(frac((local.x + local.z * 0.5 + local.w * 0.75) * 3.0) - 0.5));
            float pulse = 0.5 + 0.5 * sin((p4.x * 2.0 + p4.w * 3.0 + t * 1.8) * tau);

            float3 base = float3(0.42, 0.45, 0.50);
            float3 plate = float3(0.30, 0.33, 0.37);
            float3 rune = float3(0.72, 0.84, 0.95);
            float3 color = lerp(base, plate, strokes * 0.55);
            color = lerp(color, rune, glyph * (0.25 + 0.75 * pulse));

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.72;
            mat.roughness = 0.28;
            mat.luminance = glyph * (0.08 + 0.42 * pulse);
        }
        break;
    case 63:
        // Hyperphase gel
        {
            float4 q4 = p4 * 0.72 + float4(t * 0.28, -t * 0.22, t * 0.19, t * 0.31);
            float waveA = sin((q4.x * 4.0 + q4.y * 6.0 + q4.w * 5.0) * tau);
            float waveB = sin((q4.z * 7.0 - q4.x * 3.0 + q4.w * 4.0) * tau);
            float foam = smoothstep(0.50, 0.92, 0.5 + 0.3 * waveA + 0.2 * waveB);
            float bubble = smoothstep(
                0.95,
                1.0,
                frac(sin(dot(floor((q4 + 32.0) * 11.0), float4(121.17, 67.93, 83.71, 59.11))) * 73731.193));

            float3 deep = float3(0.08, 0.21, 0.33);
            float3 bright = float3(0.34, 0.82, 0.98);
            float3 color = lerp(deep, bright, foam);
            color = lerp(color, float3(0.78, 0.97, 1.0), bubble * 0.6);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.02;
            mat.roughness = 0.07;
            mat.luminance = 0.10 + 0.30 * foam + 0.35 * bubble;
            mat.integrationWeight = 0.22;
        }
        break;
    case 64:
        // Singularity core
        {
            float4 local = frac(p4) - 0.5;
            float radial = length(local.xyz) + abs(local.w) * 0.85;
            float core = pow(saturate(1.0 - radial * 2.2), 4.0);
            float accretion = 0.5 + 0.5 * sin(
                (local.x * 6.0 - local.z * 7.0 + local.w * 6.0 - t * 3.2) * tau);
            float jets = smoothstep(
                0.75,
                0.97,
                0.5 + 0.5 * sin((local.y * 9.0 + local.w * 11.0 + t * 4.0) * tau));

            float3 horizon = float3(0.09, 0.07, 0.10);
            float3 ring = float3(0.98, 0.54, 0.22);
            float3 flare = float3(1.0, 0.89, 0.62);
            float3 color = lerp(horizon, ring, accretion * (0.35 + 0.65 * core));
            color = lerp(color, flare, jets * core);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.18;
            mat.roughness = 0.10;
            mat.luminance = 1.4 + 16.0 * core + 4.0 * jets * core;
        }
        break;
    case 65:
        // Chrono bloom
        {
            float4 q4 = p4 + float4(t * 0.14, -t * 0.09, t * 0.12, t * 0.17);
            float4 fold = abs(frac(q4 * 2.2) - 0.5);
            float petals = 1.0 - saturate(min(min(fold.x, fold.y), min(fold.z, fold.w)) * 7.5);
            float rings = 0.5 + 0.5 * sin((length(q4 * 0.75) * 9.0 - t * 1.6) * tau);
            float pollen = smoothstep(
                0.97,
                1.0,
                frac(sin(dot(floor((q4 + 48.0) * 13.0), float4(131.17, 73.93, 97.71, 61.31))) * 75731.391));

            float3 stem = float3(0.10, 0.22, 0.14);
            float3 bloom = float3(0.38, 0.89, 0.56);
            float3 flare = float3(0.92, 1.0, 0.84);
            float3 color = lerp(stem, bloom, petals * (0.5 + 0.5 * rings));
            color = lerp(color, flare, pollen);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.04;
            mat.roughness = 0.42;
            mat.luminance = 0.10 + 0.55 * petals * rings + 0.70 * pollen;
        }
        break;
    case 66:
        // Tesseract weave
        {
            float4 weave = abs(frac((p4 + float4(t * 0.16, -t * 0.11, t * 0.08, t * 0.20)) * 9.0) - 0.5);
            float xzwBands = 1.0 - saturate(min(min(weave.x, weave.z), weave.w) * 20.0);
            float yBands = 1.0 - saturate(weave.y * 14.0);
            float knots = smoothstep(0.84, 0.97, xzwBands * yBands);
            float drift = 0.5 + 0.5 * sin((p4.x * 5.0 + p4.z * 7.0 + p4.w * 9.0 + t * 2.6) * tau);

            float3 dark = float3(0.11, 0.09, 0.18);
            float3 thread = float3(0.47, 0.30, 0.88);
            float3 knotColor = float3(0.84, 0.68, 1.0);
            float3 color = lerp(dark, thread, xzwBands * (0.6 + 0.4 * drift));
            color = lerp(color, knotColor, knots * 0.9);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.36;
            mat.roughness = 0.18;
            mat.luminance = 0.08 + 0.45 * xzwBands + 0.90 * knots;
        }
        break;
    case 67:
        // Eventide alloy
        {
            float4 q4 = p4 * 0.6 + float4(t * 0.05, t * 0.08, -t * 0.07, t * 0.09);
            float sheen = 0.5 + 0.5 * sin((q4.x * 8.0 + q4.y * 6.0 - q4.z * 5.0 + q4.w * 7.0) * tau);
            float strata = 0.5 + 0.5 * sin((q4.y * 17.0 + q4.w * 13.0) * tau);
            float inclusions = smoothstep(
                0.94,
                1.0,
                frac(sin(dot(floor((q4 + 80.0) * 9.0), float4(139.17, 79.93, 109.71, 71.31))) * 77731.193));

            float3 steel = float3(0.26, 0.30, 0.38);
            float3 violet = float3(0.36, 0.30, 0.52);
            float3 cyan = float3(0.34, 0.54, 0.62);
            float3 color = lerp(steel, violet, sheen * 0.7);
            color = lerp(color, cyan, strata * 0.35);
            color = lerp(color, float3(0.92, 0.96, 1.0), inclusions * 0.4);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.84;
            mat.roughness = 0.24 + 0.20 * (1.0 - sheen);
            mat.luminance = 0.02 + 0.12 * inclusions;
        }
        break;
    case 68:
        // Beacon matrix
        {
            float4 local = frac((p4 + float4(0.0, t * 0.20, 0.0, t * 0.14)) * 6.0);
            float4 dist = abs(local - 0.5);
            float rails = 1.0 - smoothstep(0.05, 0.17, min(min(dist.x, dist.z), dist.w));
            float nodes = smoothstep(
                0.92,
                1.0,
                frac(sin(dot(floor((p4 + 96.0) * 6.0), float4(149.17, 83.93, 113.71, 73.31))) * 79731.193));
            float pulse = 0.5 + 0.5 * sin((p4.y * 2.0 + t * 3.5) * tau);

            float3 base = float3(0.24, 0.24, 0.20);
            float3 rail = float3(0.92, 0.86, 0.54);
            float3 node = float3(1.0, 0.98, 0.82);
            float3 color = lerp(base, rail, rails);
            color = lerp(color, node, nodes * pulse);

            mat.albedo = float4(color, 1.0);
            mat.metallic = 0.22;
            mat.roughness = 0.26;
            mat.luminance = 0.35 + 2.4 * rails * pulse + 3.2 * nodes * pulse;
        }
        break;
    default:
        mat.albedo = float4(0.0, 0.0, 0.0, 1.0);
        mat.luminance = 0.0;
        break;
    }

    if (isPreviewMaterial)
    {
        // Held-block preview should over-contribute enough to read clearly in
        // integral mode, while preserving material hue (avoid whitening).
        mat.integrationWeight *= 5.0;
        mat.luminance += 0.02;
    }

    return mat;
}
