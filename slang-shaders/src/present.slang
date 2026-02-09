// Presentation shaders - display computed buffer to screen

import types;

// Shared resources with explicit bindings matching Rust descriptor layout:
// Set 0: OneTimeBuffers (model data)
// Set 1: SizedBuffers (line vertices, output tetrahedrons, pixel buffer)
// Set 2: LiveBuffers (model instances, working data)

[[vk::binding(0, 1)]] StructuredBuffer<LineVertex> lineVertices;
[[vk::binding(2, 1)]] StructuredBuffer<float4> pixelBuffer;
[[vk::binding(1, 2)]] StructuredBuffer<WorkingData> workingDataBuffer;

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

    // Accumulate across Z slices
    float4 csResult = float4(0);
    for (uint z = 0; z < wd.renderDimensions.z; z++)
    {
        uint index = z * wd.renderDimensions.x * wd.renderDimensions.y
                   + uPixelPos.y * wd.renderDimensions.x
                   + uPixelPos.x;
        csResult += pixelBuffer[index];
    }

    // Gamma correct and normalize
    if (csResult.w > 0.0)
    {
        float4 adjusted = float4(
            linearToGamma(csResult.x / csResult.w),
            linearToGamma(csResult.y / csResult.w),
            linearToGamma(csResult.z / csResult.w),
            1.0
        );
        return adjusted;
    }

    return float4(0);
}
