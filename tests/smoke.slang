//DISABLE_TEST:SIMPLE: -target cuda -line-directive-mode none

// Verify that we can output a cuda device function with [CudaDeviceExport].
// Disabled until we have FileCheck.


[CudaKernel]
void myKernel(TensorView<float> inValues, TensorView<float> outValues)
{
    var inputVal = inValues.load(cudaThreadIdx().x, cudaThreadIdx().y);
    outValues.store(cudaThreadIdx().x, cudaThreadIdx().y, 1.0 + inputVal);
}

[TorchEntryPoint]
public __extern_cpp TorchTensor<float> runCompute(TorchTensor<float> inValues)
{
    var outValues = TorchTensor<float>.alloc(inValues.size(0), inValues.size(1));
    __dispatch_kernel(myKernel, uint3(1, 1, 1), uint3(inValues.size(0), inValues.size(1), 1))(inValues, outValues);
    return outValues;
}