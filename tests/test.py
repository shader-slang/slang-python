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

class TestOptions(unittest.TestCase):
    def test_load_different_options(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'multiply.slang')

        module1 = slangpy.loadModule(slangModuleFile, defines={'FACTOR': '2.0'})
        module2 = slangpy.loadModule(slangModuleFile, defines={'FACTOR': '1.0'})

        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        Y2 = module2.multiply(X).cpu()

        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()

        assert(torch.all(torch.eq(Y1, expected1)))
        assert(torch.all(torch.eq(Y2, expected2)))

    def test_invalid_load(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'multiply.slang')

        with self.assertRaises(RuntimeError):
            module = slangpy.loadModule(slangModuleFile, defines={})

class TestHotReload(unittest.TestCase):
    def test_hot_reload(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleTemplateFile = os.path.join(test_dir, 'multiply_template.slang')

        # Get a temporary directory.
        import tempfile
        tmpdir = tempfile.mkdtemp()

        slangModuleFile = os.path.join(tmpdir, 'multiply.slang')

        # Read template file, replace %FACTOR% with 2.0, and write to temporary directory.
        with open(slangModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '2.0')
            with open(slangModuleFile, 'w') as f2:
                f2.write(template)

        module1 = slangpy.loadModule(slangModuleFile)
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))

        # Read template file, replace %FACTOR% with 1.0, and write to temporary directory.
        with open(slangModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '1.0')
            with open(slangModuleFile, 'w') as f2:
                f2.write(template)

        module2 = slangpy.loadModule(slangModuleFile)
        Y2 = module2.multiply(X).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()
        assert(torch.all(torch.eq(Y2, expected2)))


class TestMultiFileModule(unittest.TestCase):
    def test_multi_file_reload(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))

        importModuleTemplateFile = os.path.join(test_dir, 'import-module.slang')
        importedModuleTemplateFile = os.path.join(test_dir, 'imported-module.slang')

        # Get a temporary directory.
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()

        importedModuleFile = os.path.join(tmpdir, 'imported-module.slang')
        importModuleFile = os.path.join(tmpdir, 'import-module.slang')

        # Read template file, replace %FACTOR% with 2.0, and write to temporary directory.
        with open(importedModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '2.0')
            with open(importedModuleFile, 'w') as f2:
                f2.write(template)
        # Simply copy the other template file to the temporary directory.
        shutil.copy(importModuleTemplateFile, tmpdir)

        module1 = slangpy.loadModule(importModuleFile)
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))

        # Read template file, replace %FACTOR% with 1.0, and write to temporary directory.
        with open(importedModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '1.0')
            with open(importedModuleFile, 'w') as f2:
                f2.write(template)
        
        # Should trigger a recompile. If not, we'll see invalid results.

        module2 = slangpy.loadModule(importModuleFile)
        Y2 = module2.multiply(X).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()
        assert(torch.all(torch.eq(Y2, expected2)))

class TestCacheState(unittest.TestCase):
    def test_cache_state_on_build_failure(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'multiply.slang')

        # Get a temporary directory.
        import tempfile
        tmpdir = tempfile.mkdtemp()

        slangModuleFile = os.path.join(tmpdir, 'multiply.slang')

        # Copy the source file to the temporary directory.
        import shutil
        shutil.copy(slangModuleSourceFile, slangModuleFile)

        # Expect a RuntimeError (define is not set)
        with self.assertRaises(RuntimeError):
            slangpy.loadModule(slangModuleFile)
        
        # Run again, should succeed assuming the cache is in proper state.
        module = slangpy.loadModule(slangModuleFile, defines={'FACTOR': '2.0'})

        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))


class TestAutoPyBind(unittest.TestCase):
    def test_autopybind(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-square.slang')
        
        module = slangpy.loadModule(slangModuleSourceFile)

        X = torch.tensor([1., 2., 3., 4.]).cuda()
        Y = torch.zeros_like(X).cuda()

        module.square(input=X, output=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([1., 4., 9., 16.]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

class TestAutoPyBindDiff(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-square-diff.slang')
        
        module = slangpy.loadModule(slangModuleSourceFile)
        self.module = module
        
    def test_primal_call(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        expected = torch.tensor([1., 4., 9., 16.]).cpu()
        
        # Test call by direct argument
        Y = torch.zeros_like(X).cuda()
        self.module.square(input=X, output=Y).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))

        # Test call by singleton tuple
        Y = torch.zeros_like(X).cuda()
        self.module.square(input=(X,), output=(Y,)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))

    def test_fwd_diff(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        X_d = torch.tensor([1., 0., 0., 1.]).cuda()
        expected = torch.tensor([1., 4., 9., 16.]).cpu()
        expected_d = torch.tensor([2., 0., 0., 8.]).cpu()
        
        # Test call by direct argument
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.zeros_like(X_d).cuda()
        self.module.square.fwd(input=(X, X_d), output=(Y, Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))
        assert(torch.all(torch.eq(Y_d.cpu(), expected_d)))

        # Test call by named tuple
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.zeros_like(X_d).cuda()
        self.module.square.fwd(
            input=self.module.DiffTensorView(value=X, grad=X_d),
            output=self.module.DiffTensorView(value=Y, grad=Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))
        assert(torch.all(torch.eq(Y_d.cpu(), expected_d)))
    
    def test_bwd_diff(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.tensor([1., 0., 1., 0.]).cuda()
        
        expected_d = torch.tensor([2., 0., 6., 0.]).cpu()

        # Test call by direct argument
        X_d = torch.zeros_like(X).cuda()
        self.module.square.bwd(input=(X, X_d), output=(Y, Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(X_d.cpu(), expected_d)))

        # Test call by named tuple
        X_d = torch.zeros_like(X).cuda()
        Y_d = torch.tensor([1., 0., 1., 0.]).cuda()
        self.module.square.bwd(
            input=self.module.DiffTensorView(value=X, grad=X_d),
            output=self.module.DiffTensorView(value=Y, grad=Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(X_d.cpu(), expected_d)))

class TestAutoPyBindStruct(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-multiply-struct.slang')
        
        module = slangpy.loadModule(slangModuleSourceFile)
        self.module = module

    def test_struct_input(self):
        # Test call struct by tuple
        A = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        B = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Y = torch.zeros_like(A).cuda()

        self.module.multiply(foo=(A, B), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([[10., 40.],[90., 160.]]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

        # Reset Y
        Y = torch.zeros_like(A).cuda()

        # Test call struct by dict
        self.module.multiply(foo={'A': A, 'B': B}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

        # Reset Y
        Y = torch.zeros_like(A).cuda()

        # Test call struct by named tuple
        footype = self.module.Foo
        self.module.multiply(foo=footype(A=A, B=B), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        assert(torch.all(torch.eq(Y.cpu(), expected1)))

    def test_struct_failed_input(self):
        A = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        B = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Y = torch.zeros_like(A).cuda()

        with self.assertRaises(TypeError):
            self.module.multiply(foo=(A,), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo=(A, B, A), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo={'A': A}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo={'A': A, 'Ba': B}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
