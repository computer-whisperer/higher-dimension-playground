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
}

[shader("vertex")]
LineVSOutput mainLineVS(uint vertexId : SV_VertexID)
{
    LineVSOutput output;
    LineVertex vertex = lineVertices[vertexId];
    output.position = float4(vertex.position.x, vertex.position.y, 0.0, 1.0);
    output.color = vertex.color;
    return output;
}

// Fragment shader for line rendering
[shader("fragment")]
float4 mainLineFS(LineVSOutput input) : SV_Target
{
    return input.color;
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

// Fragment shader to display accumulated buffer
[shader("fragment")]
float4 mainBufferFS(float4 fragCoord : SV_Position) : SV_Target
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

    uint2 uPixelPos = uint2(uint(pixelPosF.x), uint(pixelPosF.y));

    float4 csResult = float4(0);
    // For VTE, Stage B collapses to layer 0 in compute.
    // Legacy raster/raytrace paths still accumulate all Z slices here.
    if (wd.padding[0] != 0u)
    {
        uint index = uPixelPos.y * wd.renderDimensions.x + uPixelPos.x;
        csResult = pixelBuffer[index];
    }
    else
    {
        for (uint z = 0; z < wd.renderDimensions.z; z++)
        {
            uint index = z * wd.renderDimensions.x * wd.renderDimensions.y
                       + uPixelPos.y * wd.renderDimensions.x
                       + uPixelPos.x;
            csResult += pixelBuffer[index];
        }
    }

    // Tone map and gamma correct
    if (csResult.w > 0.0)
    {
        float3 color = csResult.xyz / csResult.w;
        // Debug mode relies on exact false-color output; skip post-tonemap transforms.
        if (wd.padding[0] != 0u
            && (wd.padding[1] == STAGE_B_MODE_DEBUG_COMPARE
                || wd.padding[1] == STAGE_B_MODE_DEBUG_INTEGRAL))
        {
            return float4(saturate(color), 1.0);
        }
        color = acesToneMap(color);
        float4 adjusted = float4(
            linearToGamma(color.x),
            linearToGamma(color.y),
            linearToGamma(color.z),
            1.0
        );
        return adjusted;
    }

    return float4(0);
}
