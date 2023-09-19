
bool triangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise)
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;
    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, xy - v1);
    float d1 = dot(k1, xy - v2);
    float d2 = dot(k2, xy - v3);
    return bool(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0);
}

float2 screen_to_world(float2 uv, float2 frameDim, float2 o, float2 scale)
{
    // Convert screen coordinates to world coordinates
    float2 xy = uv / frameDim;
    float2 ndc = xy * 2.0f - 1.0f;
    return ndc * scale + o;
}

float3 render_pixel(float2 pixel, float2 frameDim, float2 o, float2 scale, 
                   float2 v1, float2 v2, float2 v3, float3 color)
{
    // Create 4 samples around the pixel center
    float2 samples[4];
    samples[0] = pixel + float2(-1, -1) * 0.25 + 0.5; // +0.5 to center the sample
    samples[1] = pixel + float2(-1, +1) * 0.25 + 0.5;
    samples[2] = pixel + float2(+1, -1) * 0.25 + 0.5;
    samples[3] = pixel + float2(+1, +1) * 0.25 + 0.5;

    float3 result = 0.0f;

    for (int i = 0; i < 4; ++i)
    {
        float2 screen_sample = samples[i];
        float2 world_sample = screen_to_world(screen_sample, frameDim, o, scale);
        bool hit = triangle(world_sample, v1, v2, v3);
        result += hit ? color : float3(1.f);
    }

    return result / 4.0f;
}

[AutoPyBindCUDA]
[CudaKernel]
void rasterize(
    TensorView<float> vertices,
    TensorView<float> color,
    TensorView<float3> output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x > output.size(0) || globalIdx.y > output.size(1))
        return;

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    // 
    float2 v1 = float2(vertices[uint2(0, 0)], vertices[uint2(0, 1)]);
    float2 v2 = float2(vertices[uint2(1, 0)], vertices[uint2(1, 1)]);
    float2 v3 = float2(vertices[uint2(2, 0)], vertices[uint2(2, 1)]);
    float3 c = float3(color[0], color[1], color[2]);
    
    // Setup default image parameters..

    // Image size (size of the output tensor)
    float2 frameDim = float2(output.size(0), output.size(1));

    // Center point of the image
    float2 o = float2(0.0, 0.0);

    // Scale of the image
    float2 scale = float2(1.0, 1.0);

    // Compute value for each pixel.
    output[globalIdx.xy] = render_pixel(globalIdx.xy, frameDim, o, scale, v1, v2, v3, c);
}