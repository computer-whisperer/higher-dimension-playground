// Material sampling - procedural materials by ID

import types;

MaterialProperties sampleMaterial(uint textureId, float3 texturePos)
{
    const float basicLuminance = 0.001;

    MaterialProperties mat;
    mat.metallic = 0.0;
    mat.roughness = 0.0;
    mat.luminance = basicLuminance;
    mat.padding = 0;

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
        mat.albedo = float4((texturePos + float3(1.0)) / 2.0, 1.0);
        mat.luminance = 0.4;
        break;
    case 10:
        mat.albedo = float4(39.0/256.0, 69.0/256.0, 19.8/256.0, 1.0); // Brown
        mat.luminance = 0.0;
        break;
    case 11:
        // Floor
        mat.albedo = float4(float3(0.1) * 0.4, 1.0);
        mat.metallic = 0.2;
        mat.roughness = 0.3;
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
    default:
        mat.albedo = float4(0.0, 0.0, 0.0, 1.0);
        mat.luminance = 0.0;
        break;
    }

    return mat;
}
