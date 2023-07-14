import unittest
import os
import torch

import slangpy

class TestSlangPySmoke(unittest.TestCase):
    def test_smoke(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'smoke.slang')
        module = slangpy.loadModule(slangModuleFile)
        
        X = torch.zeros(2, 2).cuda()
        Y = module.runCompute([X, 1.0])[1].cpu()

        expected = torch.tensor([[1, 1],[1, 1]])

        assert(torch.all(torch.eq(Y, expected)))

@unittest.skip("implicit cast fails currently")
class TestImplicitTypeCast(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'addarrays.slang')
        module = slangpy.loadModule(slangModuleFile)
        self.module = module
    
    def test_no_cast(self):
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11., 22.],[33., 44.]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))
    
    def test_implicit_cast(self):
        X = torch.tensor([[1, 2], [3, 4]]).cuda()
        Y = torch.tensor([[10, 20], [30, 40]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11, 22],[33, 44]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))

@unittest.skip("implicit device transfer fails currently")
class TestDeviceCast(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'addarrays.slang')
        module = slangpy.loadModule(slangModuleFile)
        self.module = module
    
    def test_baseline(self):
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11., 22.],[33., 44.]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))
    
    def test_cpu_to_cuda_implicit_transfer(self):
        X = torch.tensor([[1, 2], [3, 4]]).cpu()
        Y = torch.tensor([[10, 20], [30, 40]]).cpu()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11, 22],[33, 44]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))