"""
Unit tests for Brain Python Autograd & Value graph backpropagation.
"""

import unittest
import math
import brain

class TestBrainAutograd(unittest.TestCase):
    def test_value_creation(self):
        v = brain.Value([1.0, 2.0, 3.0], shape=[3], requires_grad=True)
        self.assertEqual(v.shape, [3])
        self.assertTrue(v.requires_grad)
        self.assertIsNone(v.grad)

    def test_simple_polynomial_backward(self):
        # f(x) = x^2 + 2x + 1
        # f'(x) = 2x + 2
        # at x = 3.0 -> f(3) = 16, f'(3) = 8.0
        x = brain.Value([3.0], shape=[1], requires_grad=True)
        y = x * x + x + x + brain.Value([1.0], shape=[1], requires_grad=False)
        self.assertAlmostEqual(y.item(), 16.0)
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertAlmostEqual(x.grad.item(), 8.0)

    def test_multivariate_autograd(self):
        # f(a, b) = a * b + a
        # df/da = b + 1, df/db = a
        # a = 2.0, b = 5.0 -> df/da = 6.0, df/db = 2.0
        a = brain.Value([2.0], shape=[1], requires_grad=True)
        b = brain.Value([5.0], shape=[1], requires_grad=True)
        out = a * b + a
        self.assertAlmostEqual(out.item(), 12.0)
        out.backward()
        self.assertAlmostEqual(a.grad.item(), 6.0)
        self.assertAlmostEqual(b.grad.item(), 2.0)

    def test_activations_backward(self):
        # Sigmoid: d/dx sigmoid(x) = sigmoid(x)*(1 - sigmoid(x))
        # for x = 0.0 -> sig(0) = 0.5, grad = 0.25
        x = brain.Value([0.0], shape=[1], requires_grad=True)
        y = x.sigmoid()
        self.assertAlmostEqual(y.item(), 0.5)
        y.backward()
        self.assertAlmostEqual(x.grad.item(), 0.25)

        # Tanh: d/dx tanh(x) = 1 - tanh^2(x)
        # for x = 0.0 -> tanh(0) = 0, grad = 1.0
        x2 = brain.Value([0.0], shape=[1], requires_grad=True)
        y2 = x2.tanh()
        self.assertAlmostEqual(y2.item(), 0.0)
        y2.backward()
        self.assertAlmostEqual(x2.grad.item(), 1.0)

        # ReLU: for x > 0 -> grad = 1.0, for x < 0 -> grad = 0.0
        x_pos = brain.Value([2.5], shape=[1], requires_grad=True)
        y_pos = x_pos.relu()
        y_pos.backward()
        self.assertAlmostEqual(x_pos.grad.item(), 1.0)

    def test_matmul_autograd(self):
        # A: [2, 2], B: [2, 2]
        # C = A @ B
        # Loss = sum(C)
        # dLoss/dA = ones @ B^T
        A = brain.Value([1.0, 2.0, 3.0, 4.0], shape=[2, 2], requires_grad=True)
        B = brain.Value([0.5, -0.5, 1.0, 2.0], shape=[2, 2], requires_grad=True)
        C = A @ B
        loss = C.sum()
        loss.backward()
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)
        self.assertEqual(A.grad.shape, [2, 2])
        self.assertEqual(B.grad.shape, [2, 2])

if __name__ == "__main__":
    unittest.main()
