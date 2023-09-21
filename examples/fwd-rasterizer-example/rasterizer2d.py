import slangpy
import torch
import numpy as np 
import timeit
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

rasterizer2d = slangpy.loadModule("rasterizer2d.slang", verbose=True)

vertices = torch.tensor([[0.5,-0.5], [-0.5,0.5], [-0.5,-0.5]]).type(torch.float).cuda()
color = torch.tensor([0.8, 0.3, 0.3]).type(torch.float).cuda()

outputImage = torch.zeros((1024, 1024, 3), dtype=torch.float).cuda()
rasterizer2d.rasterize(
    vertices=vertices,
    color=color,
    output=outputImage
).launchRaw(
    blockSize=(16, 16, 1),
    gridSize=(64, 64, 1)
)

plt.imshow(outputImage.cpu().numpy())
plt.show()
