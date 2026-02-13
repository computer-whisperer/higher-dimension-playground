// Material sampling - procedural materials by ID

import types;

MaterialProperties sampleMaterial(uint textureId, float4 texturePos, uint timeTicksMs)
{
    const float basicLuminance = 0.001;
    const float tau = 6.2831853;
    float4 p4 = texturePos;
    float3 p = p4.xyz;
    float t = float(timeTicksMs) * 0.001;

    MaterialProperties mat;
    mat.metallic = 0.0;
    mat.roughness = 0.0;
    mat.luminance = basicLuminance;
    mat.integrationWeight = 1.0;

    switch (textureId)
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
    default:
        mat.albedo = float4(0.0, 0.0, 0.0, 1.0);
        mat.luminance = 0.0;
        break;
    }

    return mat;
}
