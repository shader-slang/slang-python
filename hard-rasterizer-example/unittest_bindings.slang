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
    TensorView<bool> outValid,
    TensorView<float2> outPt,
    TensorView<float2> outNormal
    )
{
    // Make triangle.
    Triangle triangle = { vertices[0], vertices[1], vertices[2], float3(1.f), 0.02 };

    float sample1D = 0.5;
    Maybe<EdgeSample> es_maybe = triangle.sampleFromBoundary(sample1D, {aabb[0], aabb[1]});

    // Write the results.
    outValid[0] = es_maybe.is_valid;
    outPt[0] = es_maybe.value.pt;
    outNormal[0] = es_maybe.value.n;
}

[TorchEntryPoint]
void triangle_sample_from_edge(
    TorchTensor<float2> aabb,
    TorchTensor<float2> segment,
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
        outValid,
        outPt,
        outNormal);
}
