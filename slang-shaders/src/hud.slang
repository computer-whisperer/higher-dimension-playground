// HUD overlay shaders - textured quad rendering for text and panel backgrounds
// Self-contained: does NOT import types.slang to avoid type duplication issues

struct HudVertex
{
    float2 position;
    float2 texcoord;
    float4 color;
}

[[vk::binding(0, 0)]] StructuredBuffer<HudVertex> hudVertices;
[[vk::binding(1, 0)]] Sampler2D<float4> fontAtlas;

// Per-draw base vertex into the shared HUD/egui vertex buffer.
struct HudDrawParams
{
    uint vertexBase;
}

[[vk::push_constant]]
HudDrawParams hudDrawParams;

struct HudVSOutput
{
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
    float4 color    : COLOR0;
}

[shader("vertex")]
HudVSOutput mainHudVS(uint vertexId : SV_VertexID)
{
    HudVSOutput output;
    HudVertex v = hudVertices[hudDrawParams.vertexBase + vertexId];
    output.position = float4(v.position.x, v.position.y, 0.0, 1.0);
    output.texcoord = v.texcoord;
    output.color = v.color;
    return output;
}

[shader("fragment")]
float4 mainHudFS(HudVSOutput input) : SV_Target
{
    float4 texColor = fontAtlas.Sample(input.texcoord);
    return input.color * texColor;
}
