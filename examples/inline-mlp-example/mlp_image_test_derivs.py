import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from image_model import RenderImage

# Load 'nvidia-logo.png' as a torch tensor
target_image = torch.from_numpy(plt.imread('nvidia-logo.png')).cuda()[:, :, :3].contiguous()

# Make three random 16x16 matrices
torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
w1_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w2_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w3_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w1 = I + w1_ * 0.05
w2 = I + w2_ * 0.05
w3 = I + w3_ * 0.05
b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')

image = RenderImage.apply(512, 512, w1, w2, w3, b1, b2, b3)

# Backpropagate the slangpy version and print the gradient of w1
image.backward(target_image)

# Compare derivatives against pytorch version. Make a grid of point coordinates.
# Then construct a feture vector for each pixel: 
# (sin(2pi * x/width), sin(2pi * y/height), sin(2pi * 2 * x/width) * sin(2pi * 2 * y/height), ...)
# Then multiply by the weight matrices and add the biases.
#
# The result should be the same as the slangpy version.
#

# Make three random 16x16 matrices using the same seed.
torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
t_w1_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
t_w2_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
t_w3_ = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w1 = I + t_w1_ * 0.05
w2 = I + t_w2_ * 0.05
w3 = I + t_w3_ * 0.05
b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')

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
torch_image = torch.relu(out_feature)[:, :, :3]

# Backpropagate the torch version and print the gradient of w1
torch_image.backward(target_image)

# Display images side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(w1_.grad.detach().cpu().numpy())
ax2.imshow(t_w1_.grad.detach().cpu().numpy())

plt.show()

