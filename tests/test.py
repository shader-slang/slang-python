import unittest
import os
import torch

import slangpy

class TestSlangPySmoke(unittest.TestCase):
    def test_smoke(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'smoke.slang')
        module = slangpy.loadModule(slangModuleFile)
        
        X = torch.zeros(2, 2)
        Y = module.runCompute([X, 1.0])[1].cpu()
        expected = torch.tensor([[1,1],[1,1]])
        assert(torch.all(torch.eq(Y, expected))
)