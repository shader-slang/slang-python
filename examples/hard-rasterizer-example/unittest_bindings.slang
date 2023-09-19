import hard_rasterizer2d;

[CudaKernel]
void aabb_intersection_kernel(
    TensorView<float2> aabb,
    TensorView<float2> segment,
    TensorView<bool> outValid,
    TensorView<float2> outNear,
    TensorView<float2> outFar)
{
    // Call the function from hard_rasterizer2d.
    AABB _aabb = { aabb[0], aabb[1] };
    float2 _a = segment[0];
    float2 _b = segment[1];
    Maybe<Intersection> intersection = intersectSegmentAABB(_aabb, _a, _b);

    // Write the results.
    outValid[0] = intersection.is_valid;
    outNear[0] = intersection.value.isect_near;
    outFar[0] = intersection.value.isect_far;
}

[TorchEntryPoint]
void aabb_intersection(
    TorchTensor<float2> aabb,
    TorchTensor<float2> segment,
    TorchTensor<bool> outValid,
    TorchTensor<float2> outNear,
    TorchTensor<float2> outFar)
{
    uint3 blockSize = uint3(1, 1, 1);
    uint3 blockCount = uint3(1, 1, 1);

    // Launch the kernel.
    __dispatch_kernel(aabb_intersection_kernel, blockCount, blockSize)(
        aabb,
        segment,
        outValid,
        outNear,
        outFar);
}

[CudaKernel]
void triangle_sample_from_edge_kernel(
    TensorView<float2> vertices,
    TensorView<float2> aabb,
    TensorView<float> sample1D,
    TensorView<bool> outValid,
    TensorView<float2> outPt,
    TensorView<float2> outNormal
    )
{
    // Make triangle.
    Triangle triangle = { vertices[0], vertices[1], vertices[1], float3(1.f), 0.02 };

    Maybe<EdgeSample> es_maybe = triangle.sampleFromEdge(
        vertices[0], vertices[1], sample1D[0], {aabb[0], aabb[1]});

    // Write the results.
    outValid[0] = es_maybe.is_valid;
    outPt[0] = es_maybe.value.pt;
    outNormal[0] = es_maybe.value.n;
}

[TorchEntryPoint]
void triangle_sample_from_edge(
    TorchTensor<float2> aabb,
    TorchTensor<float2> segment,
    TorchTensor<float> sample1D,
    TorchTensor<bool> outValid,
    TorchTensor<float2> outPt,
    TorchTensor<float2> outNormal)
{
    uint3 blockSize = uint3(1, 1, 1);
    uint3 blockCount = uint3(1, 1, 1);

    // Launch the kernel.
    __dispatch_kernel(triangle_sample_from_edge_kernel, blockCount, blockSize)(
        aabb,
        segment,
        sample1D,
        outValid,
        outPt,
        outNormal);
}

[CudaKernel]
void triangle_sample_from_boundary_kernel(
    TensorView<float2> vertices,
    TensorView<float2> aabb,
    TensorView<float> sample1D,
    TensorView<bool> outValid,
    TensorView<float2> outPt,
    TensorView<float2> outNormal
    )
{
    // Make triangle.
    Triangle triangle = { vertices[0], vertices[1], vertices[2], float3(1.f), 0.02 };

    Maybe<EdgeSample> es_maybe = triangle.sampleFromBoundary(sample1D[0], { aabb[0], aabb[1] });

    // Write the results.
    outValid[0] = es_maybe.is_valid;
    outPt[0] = es_maybe.value.pt;
    outNormal[0] = es_maybe.value.n;
}

[TorchEntryPoint]
void triangle_sample_from_boundary(
    TorchTensor<float2> aabb,
    TorchTensor<float2> segment,
    TorchTensor<float> sample1D,
    TorchTensor<bool> outValid,
    TorchTensor<float2> outPt,
    TorchTensor<float2> outNormal)
{
    uint3 blockSize = uint3(1, 1, 1);
    uint3 blockCount = uint3(1, 1, 1);

    // Launch the kernel.
    __dispatch_kernel(triangle_sample_from_boundary_kernel, blockCount, blockSize)(
        aabb,
        segment,
        sample1D,
        outValid,
        outPt,
        outNormal);
}

[CudaKernel]
void render_pixel_fwd_kernel(
    TensorView<float2> vertices,
    TensorView<float2> d_vertices,
    TensorView<float2> pixel,
    TensorView<int> rngState,
    TensorView<float3> outDResult
    )
{
    // Make triangle.
    Triangle triangle = { vertices[0], vertices[1], vertices[2], float3(1.f, 0.f, 0.f), 0.02 };
    Camera camera = { float2(0.0, 0.0), float2(1.0, 1.0), float2(256, 256) };

    Triangle.Differential d_triangle = { d_vertices[0], d_vertices[1], d_vertices[2], 0.0, 0.0 };
    DifferentialPair<Triangle> dp_triangle = diffPair(triangle, d_triangle);
    
    PRNG prng = PRNG(reinterpret<uint, int>(rngState[0]));
    var dpcolor = fwd_diff(render_pixel)(pixel[0], camera, dp_triangle, prng);

    // Write the results.
    outDResult[0] = dpcolor.d;
}

// Dispatch the kernel.
[TorchEntryPoint]
void render_pixel_fwd(
    TorchTensor<float2> vertices,
    TorchTensor<float2> d_vertices,
    TorchTensor<float2> pixel,
    TorchTensor<int> rngState,
    TorchTensor<float3> outDResult)
{
    uint3 blockSize = uint3(1, 1, 1);
    uint3 blockCount = uint3(1, 1, 1);

    // Launch the kernel.
    __dispatch_kernel(render_pixel_fwd_kernel, blockCount, blockSize)(
        vertices,
        d_vertices,
        pixel,
        rngState,
        outDResult);
}