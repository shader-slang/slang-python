import slangpy
import torch

m = slangpy.loadModule('image-model.slang', verbose=True)

class RenderImage(torch.autograd.Function):
    def forward(ctx, width, height, *args):
        weights = args[0: 3]
        biases = args[3: 6]
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        
        linear_layers = [m.Linear(weights=weights[i], bias=biases[i]) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = (32, 8, 1)
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage(mlp=mlp, imageOutput=output).launchRaw(blockSize=blockSize, gridSize=gridSize)

        ctx.save_for_backward(output, *args)

        return output
    
    def backward(ctx, grad_output):
        output, *args = ctx.saved_tensors
        weights = args[0: 3]
        biases = args[3: 6]

        weights_d = [torch.zeros_like(w) for w in weights]
        biases_d = [torch.zeros_like(b) for b in biases]

        width, height, _ = output.shape
        
        linear_layers = [m.Linear(weights=(weights[i], weights_d[i]), bias=(biases[i], biases_d[i])) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = (32, 8, 1)
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage.bwd(mlp=mlp, imageOutput=(output, grad_output)).launchRaw(blockSize=blockSize, gridSize=gridSize)

        return None, None, *weights_d, *biases_d
