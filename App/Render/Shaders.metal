#include <metal_stdlib>
using namespace metal;

struct PreviewUniforms {
    float2 uvScale;    // aspect-fill scale around center
    float2 uvOffset;   // centering offset
    uint   mirror;     // 1 flips horizontally
};

struct VOut {
    float4 position [[position]];
    float2 uv;
};

// full-screen triangle pair generated from vertexID, no vertex buffer needed
vertex VOut preview_vertex(uint vid [[vertex_id]],
                           constant PreviewUniforms& u [[buffer(0)]]) {
    float2 positions[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    float2 uvs[6] = {
        float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0),
        float2(0.0, 0.0), float2(1.0, 1.0), float2(1.0, 0.0)
    };
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    float2 uv = uvs[vid];
    if (u.mirror == 1) { uv.x = 1.0 - uv.x; }
    uv = (uv - 0.5) * u.uvScale + 0.5 + u.uvOffset;
    out.uv = uv;
    return out;
}

fragment float4 preview_fragment(VOut in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return tex.sample(s, in.uv);
}
