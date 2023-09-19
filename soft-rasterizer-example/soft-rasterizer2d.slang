
[PreferRecompute]
int _sign(float x)
{
    return sign(x);
}

[Differentiable]
[PreferRecompute]
float distance_to_edge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distance_to_vertex(pt, u) * (_sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distance_to_vertex(pt, v) * (_sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

[Differentiable]
[PreferRecompute]
float distance_to_vertex(float2 xy2, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy2 - v);
}

[Differentiable]
[PreferRecompute]
float distance_to_triangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distance_to_edge(v2, v1, xy);
    float d2 = distance_to_edge(v3, v2, xy);
    float d3 = distance_to_edge(v1, v3, xy);

    int sd1 = _sign(d1);
    int sd2 = _sign(d2);
    int sd3 = _sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distance_to_vertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distance_to_vertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distance_to_vertex(xy, v3); // v3 is the closest vertex
    else 
        return max(max(d1, d2), d3);

}

[Differentiable]
[PreferRecompute]
float sigmoid(float x, float sigma)
{
    // Sigmoid function
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
[PreferRecompute]
float soft_triangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distance_to_triangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}

[Differentiable]
[PreferRecompute]
float2 screen_to_world(float2 uv, float2 frameDim, float2 o, float2 scale)
{
    // Convert screen coordinates to world coordinates
    float2 xy = uv / frameDim;
    float2 ndc = xy * 2.0f - 1.0f;
    return ndc * scale + o;
}

[Differentiable]
[PreferRecompute]
float3 render_pixel(no_diff float2 pixel, no_diff float2 frameDim, no_diff float2 o, no_diff float2 scale, 
                   float2 v1, float2 v2, float2 v3, float3 color, no_diff float sigma)
{
    // Create 4 samples around the pixel center
    float2 samples[4];
    samples[0] = pixel + float2(-1, -1) * 0.25 + 0.5; // +0.5 to center the sample
    samples[1] = pixel + float2(-1, +1) * 0.25 + 0.5;
    samples[2] = pixel + float2(+1, -1) * 0.25 + 0.5;
    samples[3] = pixel + float2(+1, +1) * 0.25 + 0.5;

    float3 result = 0.0f;

    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float2 screen_sample = samples[i];
        float2 world_sample = screen_to_world(screen_sample, frameDim, o, scale);
        float hit = soft_triangle(world_sample, v1, v2, v3, sigma);
        result += hit * color + (1 - hit) * float3(1.f);
    }

    return result / 4.0f;
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void rasterize(
    DiffTensorView vertices,
    DiffTensorView color,
    DiffTensorView output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x > output.size(0) || globalIdx.y > output.size(1))
        return;

    // Load vertices of our triangle.
    // Assume our input tensor is of the form (3, 2) where 3 is the number of vertices
    // and 2 is the number of coordinates per vertex.
    //
    float2 v1 = float2(vertices.load(uint2(0, 0)), vertices.load(uint2(0, 1)));
    float2 v2 = float2(vertices.load(uint2(1, 0)), vertices.load(uint2(1, 1)));
    float2 v3 = float2(vertices.load(uint2(2, 0)), vertices.load(uint2(2, 1)));
    float3 c = float3(color.load(0), color.load(1), color.load(2));
    
    // Setup default image parameters..

    // Image size (size of the output tensor)
    float2 frameDim = float2(output.size(0), output.size(1));

    // Center point of the image
    float2 o = float2(0.0, 0.0);

    // Scale of the image
    float2 scale = float2(1.0, 1.0);

    // Sigma for the sigmoid function
    float sigma = 0.02;

    // Compute value for each pixel.
    float3 color = render_pixel(globalIdx.xy, frameDim, o, scale, v1, v2, v3, c, sigma);
    output.store(uint3(globalIdx.xy, 0), color.x);
    output.store(uint3(globalIdx.xy, 1), color.y);
    output.store(uint3(globalIdx.xy, 2), color.z);
}