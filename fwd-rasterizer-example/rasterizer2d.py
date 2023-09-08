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

outputImage = rasterizer2d.rasterize(1024, 1024, vertices, color)

plt.imshow(outputImage.cpu().numpy())
plt.show()
