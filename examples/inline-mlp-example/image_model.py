import slangpy
import torch

launchBlockSize = (32, 8, 1)

m = slangpy.loadModule('image-model.slang', 
                       defines={
                            'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] * launchBlockSize[2],
                            'WARP_SIZE': 32})

class RenderImage(torch.autograd.Function):
    def forward(ctx, width, height, feature_grid, *args):
        weights = args[0: 3]
        biases = args[3: 6]
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        
        linear_layers = [m.Linear(weights=weights[i], bias=biases[i]) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage(mlp=mlp, featureGrid=feature_grid, imageOutput=output).launchRaw(blockSize=blockSize, gridSize=gridSize)

        ctx.save_for_backward(output, feature_grid, *args)

        return output
    
    def backward(ctx, grad_output):
        output, feature_grid, *args = ctx.saved_tensors
        weights = args[0: 3]
        biases = args[3: 6]

        weights_d = [torch.zeros_like(w) for w in weights]
        biases_d = [torch.zeros_like(b) for b in biases]
        feature_grid_d = torch.zeros_like(feature_grid)

        width, height, _ = output.shape
        
        linear_layers = [m.Linear(weights=(weights[i], weights_d[i]), bias=(biases[i], biases_d[i])) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage.bwd(mlp=mlp, featureGrid=(feature_grid, feature_grid_d), imageOutput=(output, grad_output)).launchRaw(blockSize=blockSize, gridSize=gridSize)

        return None, None, feature_grid_d, *weights_d, *biases_d
