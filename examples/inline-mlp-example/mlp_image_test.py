import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from image_model import RenderImage

# Make three random 16x16 matrices
torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
w1 = I + torch.randn((16, 16), dtype=torch.float, requires_grad=True).cuda() * 0.05
w2 = I + torch.randn((16, 16), dtype=torch.float, requires_grad=True).cuda() * 0.05
w3 = I + torch.randn((16, 16), dtype=torch.float, requires_grad=True).cuda() * 0.05
#w1 = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
#w2 = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
#w3 = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
b1 = torch.zeros(16, dtype=torch.float, requires_grad=True).cuda().contiguous()
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True).cuda().contiguous()
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True).cuda().contiguous()

image = RenderImage.apply(512, 512, w1, w2, w3, b1, b2, b3)

import numpy as np

# Compare against pytorch version. Make a grid of point coordinates.
# Then construct a feture vector for each pixel: 
# (sin(2pi * x/width), sin(2pi * y/height), sin(2pi * 2 * x/width) * sin(2pi * 2 * y/height), ...)
# Then multiply by the weight matrices and add the biases.
#
# The result should be the same as the slangpy version.
#

# Make a grid of point coordinates
x = torch.linspace(0, 1, 512, dtype=torch.float).cuda()
y = torch.linspace(0, 1, 512, dtype=torch.float).cuda()
xv, yv = torch.meshgrid(x, y)

# Construct the 16-channel feature vector
channels = []
for i in range(8):
    channels.append(torch.sin(2 * np.pi * (2**i) * xv))
    channels.append(torch.sin(2 * np.pi * (2**i) * yv))

# Concatenate the channels
feature_vector = torch.stack(channels, dim=2)

out_feature = torch.einsum('ij,whi->whj', w1, feature_vector) + b1[None, None, :]
out_feature = torch.relu(out_feature)
out_feature = torch.einsum('ij,whi->whj', w2, out_feature) + b2[None, None, :]
out_feature = torch.relu(out_feature)
out_feature = torch.einsum('ij,whi->whj', w3, out_feature) + b3[None, None, :]
torch_image = torch.relu(out_feature)

# Display images side-by-side
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.detach().cpu().numpy())
ax2.imshow(torch_image[:,:,:3].detach().cpu().numpy())

plt.show()

# Exit
import sys
sys.exit(0)

# Setup optimization loop
optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3], lr=0.01)
loss_fn = torch.nn.MSELoss()

for i in range(100):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = RenderImage.apply(16, 16, w1, w2, w3, b1, b2, b3)

    # Compute and print loss.
    loss = loss_fn(y_pred, image)
    print(f'Iteration {i}, loss = {loss.item()}')

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model).
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()