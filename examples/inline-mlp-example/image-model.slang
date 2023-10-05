import inline_mlp;

#define PI 3.14159265358979323846f

[Differentiable]
[CUDAKernel]
[AutoPyBindCUDA]
void renderImage(MLP<16, 3> mlp, DiffTensorView imageOutput)
{
    // Build a 16-element vector of pixel coordinates & their positional encodings.
    // The positional encodings are computed using the sine and cosine functions.
    //

    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= imageOutput.size(0) || dispatchIdx.y >= imageOutput.size(1))
        return;

    float feature[16];

    float normX = ((float)dispatchIdx.x) / imageOutput.size(0);
    float normY = ((float)dispatchIdx.y) / imageOutput.size(1);

    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        feature[2 * i] = sin(2 * pow(2, i) * PI * normX);
        feature[2 * i + 1] = sin(2 * pow(2, i) * PI * normY);
    }

    // Run the MLP on the input features.
    //
    mlp.Output output = mlp.eval({ feature }); 

    // First 3 elements of the output are the RGB color values.
    //
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 0), output.vals[0]);
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 1), output.vals[1]);
    imageOutput.storeOnce(uint3(dispatchIdx.x, dispatchIdx.y, 2), output.vals[2]);
}
