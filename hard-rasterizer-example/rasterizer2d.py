import slangpy
import torch
import numpy as np 
import timeit
import os
import matplotlib.pyplot as plt
from torch.autograd import Function

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()

def setup_rasterizer():
    rasterizer2d = slangpy.loadModule("hard_rasterizer2d.slang", verbose=True)

    class Rasterizer2d(Function):
        @staticmethod
        def forward(width, height, vertices, color):
            outputImage = rasterizer2d.rasterize(width, height, vertices, color)
            return outputImage

        @staticmethod
        def setup_context(ctx, inputs, output):
            width, height, vertices, color = inputs
            return ctx.save_for_backward(vertices, color)
        
        @staticmethod
        def backward(ctx, grad_output):
            vertices, color = ctx.saved_tensors 
            grad_vertices = torch.zeros_like(vertices)
            grad_color = torch.zeros_like(color)
            width, height = grad_output.shape[0], grad_output.shape[1]
            # Timing.
            start = timeit.default_timer()
            rasterizer2d.rasterize_bwd(width, height, vertices, grad_vertices, color, grad_color, grad_output)
            end = timeit.default_timer()
            print("Backward pass: %f seconds" % (end - start))
            return None, None, grad_vertices, grad_color
    
    return Rasterizer2d()

rasterizer = setup_rasterizer()

# Render a simple target image.
targetVertices = torch.tensor([[0.7,-0.3], [-0.3,0.2], [-0.6,-0.6]]).type(torch.float).cuda()
targetColor = torch.tensor([0.3, 0.8, 0.3]).type(torch.float).cuda()
targetImage = rasterizer.apply(1024, 1024, targetVertices, targetColor)

# Setup our training loop.
learningRate = 5e-3
numIterations = 400

# Initialize our parameters.
vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
vertices.requires_grad = True
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()
color.requires_grad = True

# Run our training loop.
for i in range(numIterations):
    print("Iteration %d" % i)
    # Forward pass: render the image.
    outputImage = rasterizer.apply(1024, 1024, vertices, color)

    # Compute the loss.
    loss = torch.mean((outputImage - targetImage) ** 2)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    with torch.no_grad():
        # Update vertices.
        vertices -= learningRate * vertices.grad
        vertices.grad.zero_()

        # Update color.
        color -= learningRate * color.grad
        color.grad.zero_()

        print(vertices)
        print(color)

plt.imshow(outputImage.detach().cpu().numpy())
plt.show()
