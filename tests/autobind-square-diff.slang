[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void square(DiffTensorView<float, AtomicAdd> A, DiffTensorView<float, AtomicAdd> result)
{
    uint location = (cudaBlockDim() * cudaBlockIdx() + cudaThreadIdx()).x;
    if (location < A.size(0))
        result.store(location, A.load(location) * A.load(location));
}
