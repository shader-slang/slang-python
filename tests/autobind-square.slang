[AutoPyBindCUDA]
[CudaKernel]
void square(TensorView<float> A, TensorView<float> result)
{
    uint2 location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).xy;
    if (location.x < A.size(0) && location.y < A.size(1))
        result[location] = A[location] * A[location];
}
 