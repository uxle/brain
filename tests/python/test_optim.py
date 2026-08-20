"""
Unit tests for Brain Python Optimizers (brain.optim).
"""

import unittest
import brain

class TestBrainOptim(unittest.TestCase):
    def test_sgd_optimization_step(self):
        w = brain.Value([10.0], shape=[1], requires_grad=True)
        opt = brain.optim.SGD([w], lr=0.05, momentum=0.0)

        for _ in range(80):
            opt.zero_grad()
            # Target is 2.0: loss = (w - 2)^2
            loss = (w * brain.Value([1.0], requires_grad=False) - brain.Value([2.0], requires_grad=False))
            loss = loss * loss
            loss.backward()
            opt.step()

        self.assertAlmostEqual(w.item(), 2.0, places=2)

    def test_adam_optimization_step(self):
        w = brain.Value([8.0], shape=[1], requires_grad=True)
        opt = brain.optim.Adam([w], lr=0.1)

        for _ in range(200):
            opt.zero_grad()
            loss = (w * brain.Value([1.0], requires_grad=False) - brain.Value([2.0], requires_grad=False))
            loss = loss * loss
            loss.backward()
            opt.step()

        self.assertAlmostEqual(w.item(), 2.0, delta=0.05)

    def test_adamw_optimization_step(self):
        w = brain.Value([5.0], shape=[1], requires_grad=True)
        opt = brain.optim.AdamW([w], lr=0.08, weight_decay=0.001)

        for _ in range(180):
            opt.zero_grad()
            loss = (w * brain.Value([1.0], requires_grad=False) - brain.Value([1.0], requires_grad=False))
            loss = loss * loss
            loss.backward()
            opt.step()

        self.assertAlmostEqual(w.item(), 1.0, delta=0.05)

    def test_zero_grad_resets_gradients(self):
        w = brain.Value([3.0], shape=[1], requires_grad=True)
        opt = brain.optim.SGD([w], lr=0.1)

        loss = w * w
        loss.backward()
        self.assertIsNotNone(w.grad)
        self.assertAlmostEqual(w.grad.item(), 6.0)

        opt.zero_grad()
        self.assertIsNone(w.grad)

if __name__ == "__main__":
    unittest.main()
