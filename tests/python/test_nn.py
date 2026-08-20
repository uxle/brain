"""
Unit tests for Brain Python Neural Network Layers (brain.nn).
"""

import unittest
import brain

class TestBrainNN(unittest.TestCase):
    def test_linear_layer(self):
        lin = brain.nn.Linear(4, 8, bias=True)
        params = lin.parameters()
        # Expect weight [8, 4] and bias [8]
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [8, 4])
        self.assertEqual(params[1].shape, [8])

        # Forward pass on batch [2, 4] -> [2, 8]
        x = brain.Tensor([1.0]*8, shape=[2, 4])
        out = lin.forward(x)
        self.assertEqual(out.shape, [2, 8])

    def test_conv2d_layer(self):
        # in_c=3, out_c=16, kernel=3
        conv = brain.nn.Conv2d(3, 16, 3, bias=True)
        params = conv.parameters()
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [16, 3, 3, 3])

        # Input: [1, 3, 8, 8]
        x = brain.ones([1, 3, 8, 8])
        out = conv.forward(x)
        # Out: [1, 16, 8, 8] (8 - 3 + 1 = 6)
        self.assertEqual(out.shape, [1, 16, 8, 8])

    def test_layer_norm(self):
        ln = brain.nn.LayerNorm([4], eps=1e-5)
        params = ln.parameters()
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [4])
        self.assertEqual(params[1].shape, [4])

        x = brain.Tensor([1.0, 2.0, 3.0, 4.0], shape=[1, 4])
        out = ln.forward(x)
        self.assertEqual(out.shape, [1, 4])
        # Mean should be ~0.0
        self.assertAlmostEqual(out.mean().item(), 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
