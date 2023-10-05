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

b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')


# Backpropagate the slangpy version and print the gradient of w1
#image.backward(target_image)

# Setup optimization loop
optimizer = torch.optim.Adam([w1_, w2_, w3_, b1, b2, b3], lr=0.03)
loss_fn = torch.nn.MSELoss()

intermediate_images = []

for i in range(2000):
    w1 = I + w1_ * 0.05
    w2 = I + w2_ * 0.05
    w3 = I + w3_ * 0.05

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = RenderImage.apply(512, 512, w1, w2, w3, b1, b2, b3)

    # Compute and print loss.
    loss = loss_fn(y_pred, target_image)
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

    if i % 100 == 0:
        intermediate_images.append(y_pred.detach().cpu().numpy())

# Display images side-by-side
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(y_pred.detach().cpu().numpy())
ax2.imshow(target_image.detach().cpu().numpy())

# Save a video.
import cv2
height, width, layers = y_pred.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, 30, (width * 2, height))
for image in intermediate_images:
    image = np.clip(image, 0, 1)
    # concatenate the target_image to the right of the image
    image = np.concatenate([image, target_image.detach().cpu().numpy()], axis=1)
    video.write((image * 255).astype(np.uint8))

cv2.destroyAllWindows()
video.release()


plt.show()