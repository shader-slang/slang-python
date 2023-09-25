# Writing a 2D triangle rasterizer in Slang.

This example will demonstrate how to write a simple 2D triangle rasterizer in the Slang shading language, and incorporate it into a PyTorch pipeline using the `slangpy` python package.

All files discussed in this tutorial are available in this directory [here](../fwd-rasterizer-example/)


## Setup

You will need to install the following packages:
1. `slangpy`
2. `torch` with CUDA support

Then, we'll create a new `.slang` file to write our rasterizer. The IDE you use doesn't particularly matter, but if you use Visual Studio Code, the [Slang plugin](https://marketplace.visualstudio.com/items?itemName=shader-slang.slang-language-extension) provides Intellisense and interactive code-completion.


## Converting a 2D triangle to an image.

Our algorithm is fairly simple, and will serve to highlight the ease of writing graphics programs in *Slang*'s `SIMT` model (as opposed to *PyTorch*'s `NDArray` model)

A quick overview: 
1. Test the center point of the pixel against the three *half-planes* described by the three sides of the triangle. 
2. If the point is in all 3 half-planes, then we'll set that pixel to the given color.
3. If not, then use the background color.
4. Repeat this test for each pixel.

In the single-instruction-multiple-threads (SIMT) programming model, we can simply write logic for each pixel, and launch a thread for each pixel on the GPU. Note that this is in contrast to the `NDArray` model, where we describe logic for *all* pixels (as a tensor).

Here's the test function `triangle()` in Slang:

```csharp
// xy: 2D test position
// v1: vertex position 1
// v2: vertex position 2
// v3: vertex position 3
//
bool triangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // 2D triangle test (return 1.0 if the triangle contains the point (x,y), 0.0 otherwise).
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;

    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, xy - v1);
    float d1 = dot(k1, xy - v2);
    float d2 = dot(k2, xy - v3);

    // Check the three half-plane values.
    if(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
        return true;
    else
        return false;
}
```

Note here that, as a shading language, Slang provides built-in vector types such as `float2`, `float3` & `float4`, which can be convenient for 3D graphics coordinate operations. In this case, we leverage them to represent 2D points. All built-in operations & functions defined on scalar types also apply to their vector versions.


Now that we have a function that tests at a single point, let's look at the kernel which will invoke this function on the pixel center value:

```csharp
float3 render_pixel(float2 pixel, float2 frameDim, float2 o, float2 scale, 
                   float2 v1, float2 v2, float2 v3, float3 color)
{

    // Use white as the default background color.
    float2 background_color = float3(1.0, 1.0, 1.0);

    // Center of the pixel will be offset by 50% of the pixel size,
    // which we will assume is 1 unit by 1 unit.
    // 
    float2 sample = pixel + 0.5;

    bool hit = triangle(sample, v1, v2, v3);

    // If we hit the triangle return the provided color, otherwise 0.
    float3 result = hit ? color : float3(1.f);

    return result;
}
```
