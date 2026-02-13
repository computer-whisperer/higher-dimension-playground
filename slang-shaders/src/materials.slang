// Material sampling - procedural materials by ID

import types;

MaterialProperties sampleMaterial(uint textureId, float4 texturePos)
{
    const float basicLuminance = 0.001;
    const float tau = 6.2831853;
    float4 p4 = texturePos;
    float3 p = p4.xyz;

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
            float veinField = 0.5 + 0.5 * sin((p.x * 11.0 + p.y * 13.0 + p.z * 17.0) * tau);
            float veins = pow(veinField, 6.0);
            float3 rockColor = float3(0.08, 0.05, 0.04);
            float3 lavaColor = float3(0.95, 0.36, 0.08);
            mat.albedo = float4(lerp(rockColor, lavaColor, veins), 1.0);
            mat.roughness = 0.8;
            mat.luminance = 0.08 + 6.0 * veins;
        }
        break;
    case 16:
        // Crystal lattice
        {
            p = frac(p);
            float3 cell = abs(frac(p * 8.0) - 0.5);
            float line = 1.0 - saturate(min(cell.x, min(cell.y, cell.z)) * 36.0);
            float3 baseColor = float3(0.02, 0.06, 0.09);
            float3 lineColor = float3(0.25, 0.9, 1.0);
            mat.albedo = float4(lerp(baseColor, lineColor, line), 1.0);
            mat.roughness = 0.1;
            mat.luminance = 0.2 + 2.2 * line;
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
            float3 mossBase = float3(0.06, 0.22, 0.09);
            float3 sporeGlow = float3(0.25, 0.75, 0.35);
            mat.albedo = float4(lerp(mossBase, sporeGlow, spores), 1.0);
            mat.roughness = 0.9;
            mat.luminance = spores * 3.5;
        }
        break;
    case 20:
        // Void mirror
        {
            p = frac(p);
            float3 q = p - 0.5;
            float centerGlow = pow(saturate(1.0 - length(q) * 1.8), 2.0);
            mat.albedo = float4(
                0.03 + 0.20 * centerGlow,
                0.05 + 0.10 * centerGlow,
                0.12 + 0.20 * centerGlow,
                1.0);
            mat.metallic = 1.0;
            mat.roughness = 0.02;
            mat.luminance = 0.15 * centerGlow;
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
    default:
        mat.albedo = float4(0.0, 0.0, 0.0, 1.0);
        mat.luminance = 0.0;
        break;
    }

    return mat;
}
