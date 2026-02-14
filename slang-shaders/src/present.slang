// Presentation shaders - display computed buffer to screen

import types;

// Shared resources with explicit bindings matching Rust descriptor layout:
// Set 0: OneTimeBuffers (model data)
// Set 1: SizedBuffers (line vertices, output tetrahedrons, pixel buffer)
// Set 2: LiveBuffers (model instances, working data)

[[vk::binding(0, 1)]] StructuredBuffer<LineVertex> lineVertices;
[[vk::binding(2, 1)]] StructuredBuffer<float4> pixelBuffer;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

static const uint STAGE_B_MODE_DEBUG_COMPARE = 3u;
static const uint STAGE_B_MODE_DEBUG_INTEGRAL = 4u;

// Vertex shader for line rendering
struct LineVSOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    float style : TEXCOORD0;
}

[shader("vertex")]
LineVSOutput mainLineVS(uint vertexId : SV_VertexID)
{
    LineVSOutput output;
    LineVertex vertex = lineVertices[vertexId];
    output.position = float4(vertex.position.x, vertex.position.y, 0.0, 1.0);
    output.color = vertex.color;
    output.style = vertex.padding.x;
    return output;
}

// Vertex shader for fullscreen quad (buffer display)
struct BufferVSOutput
{
    float4 position : SV_Position;
}

[shader("vertex")]
BufferVSOutput mainBufferVS(uint vertexId : SV_VertexID)
{
    // Fullscreen triangle pair
    static const float4 vertices[6] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0),
        float4(-1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0),
    };

    BufferVSOutput output;
    output.position = vertices[vertexId];
    return output;
}

// ACES filmic tone mapping
float3 acesToneMap(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Gamma correction
float linearToGamma(float value)
{
    const float gamma = 2.2;
    return pow(value, 1.0 / gamma);
}

float4 samplePresentedColor(float2 fragCoord)
{
    WorkingData wd = workingDataBuffer[0];

    // Map fragment coordinate to buffer coordinate
    float2 normalPos = float2(
        fragCoord.x / float(wd.presentDimensions.x),
        fragCoord.y / float(wd.presentDimensions.y)
    );

    float2 pixelPosF = float2(
        float(wd.renderDimensions.x) * normalPos.x,
        float(wd.renderDimensions.y) * normalPos.y
    );

    uint px = min(uint(pixelPosF.x), wd.renderDimensions.x - 1);
    uint py = min(uint(pixelPosF.y), wd.renderDimensions.y - 1);

    float4 csResult = float4(0.0, 0.0, 0.0, 0.0);
    // For VTE, Stage B collapses to layer 0 in compute.
    // Legacy raster/raytrace paths still accumulate all Z slices here.
    if (wd.padding[0] != 0u)
    {
        uint index = py * wd.renderDimensions.x + px;
        csResult = pixelBuffer[index];
    }
    else
    {
        for (uint z = 0; z < wd.renderDimensions.z; z++)
        {
            uint index = z * wd.renderDimensions.x * wd.renderDimensions.y
                       + py * wd.renderDimensions.x
                       + px;
            csResult += pixelBuffer[index];
        }
    }

    // Tone map and gamma correct
    float alpha = csResult.w;
    if (alpha > 0.0)
    {
        float3 color = csResult.xyz / alpha;
        // Debug mode relies on exact false-color output; skip post-tonemap transforms.
        if (wd.padding[0] != 0u
            && (wd.padding[1] == STAGE_B_MODE_DEBUG_COMPARE
                || wd.padding[1] == STAGE_B_MODE_DEBUG_INTEGRAL))
        {
            return float4(saturate(color), alpha);
        }
        color = acesToneMap(color);
        return float4(
            linearToGamma(color.x),
            linearToGamma(color.y),
            linearToGamma(color.z),
            alpha
        );
    }

    return float4(0.0, 0.0, 0.0, 0.0);
}

float3 adaptiveCrosshairColor(float3 baseColor, float tint)
{
    // Inversion-like behavior with a black/white anchor to hold contrast.
    float3 inverted = 1.0 - baseColor;
    float luma = dot(baseColor, float3(0.2126, 0.7152, 0.0722));
    float3 bw = (luma > 0.5) ? float3(0.0, 0.0, 0.0) : float3(1.0, 1.0, 1.0);
    float3 adaptive = lerp(inverted, bw, 0.45);

    // White-authored lines use adaptive directly; dark-authored lines use
    // the inverse, preserving a clear outline/inner relationship.
    return lerp(1.0 - adaptive, adaptive, saturate(tint));
}

// Fragment shader for line rendering
[shader("fragment")]
float4 mainLineFS(LineVSOutput input, float4 fragCoord : SV_Position) : SV_Target
{
    if (input.style > 0.5)
    {
        float3 base = samplePresentedColor(fragCoord.xy).xyz;
        float tint = dot(input.color.rgb, float3(0.33333334, 0.33333334, 0.33333334));
        return float4(adaptiveCrosshairColor(base, tint), 1.0);
    }

    return input.color;
}

// Fragment shader to display accumulated buffer
[shader("fragment")]
float4 mainBufferFS(float4 fragCoord : SV_Position) : SV_Target
{
    return samplePresentedColor(fragCoord.xy);
}
