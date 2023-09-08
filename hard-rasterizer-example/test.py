# Set up unittests for the hard rasterizer

import unittest
import torch
import slangpy
import timeit
from torch.autograd import Function

class TestAABBIntersection(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rasterizer2d = slangpy.loadModule("unittest_bindings.slang", verbose=True)

    def test_aabb_intersection_1(self):
        aabb = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).type(torch.float).cuda()
        segment = torch.tensor([[-0.5, 0.5], [1.5, 0.5]]).type(torch.float).cuda()
        outValid = torch.zeros(1).type(torch.bool).cuda()
        outNear = torch.zeros((1,2)).type(torch.float).cuda()
        outFar = torch.zeros((1,2)).type(torch.float).cuda()

        self.rasterizer2d.aabb_intersection(aabb, segment, outValid, outNear, outFar)

        self.assertEqual(outValid[0], 1.0)
        self.assertEqual(outNear[0,:], (0.0, 0.5))
        self.assertEqual(outFar[0,:], (1.0, 0.5))
 

class TestTriangle(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rasterizer2d = slangpy.loadModule("unittest_bindings.slang", verbose=True)

    def test_triangle_sample_from_edge(self):
        aabb = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).type(torch.float).cuda()
        segment = torch.tensor([[-0.5, 0.5], [1.5, 0.5]]).type(torch.float).cuda()
        outValid = torch.zeros(1).type(torch.bool).cuda()
        outNear = torch.zeros((1,2)).type(torch.float).cuda()
        outFar = torch.zeros((1,2)).type(torch.float).cuda()

        self.rasterizer2d.aabb_intersection(aabb, segment, outValid, outNear, outFar)

        self.assertEqual(outValid[0], 1.0)
        self.assertEqual(outNear[0,:], (0.0, 0.5))
        self.assertEqual(outFar[0,:], (1.0, 0.5))
