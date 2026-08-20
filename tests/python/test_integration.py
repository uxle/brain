"""
Integration tests for Brain Python end-to-end pipeline.
"""

import unittest
import brain

class CustomMLP(brain.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = brain.Value([0.5, -0.5, 0.2, 0.8, 0.1, -0.3], shape=[3, 2], requires_grad=True)
        self.b1 = brain.Value([0.0, 0.0], shape=[1, 2], requires_grad=True)
        self.w2 = brain.Value([0.4, 0.6], shape=[2, 1], requires_grad=True)
        self.b2 = brain.Value([0.0], shape=[1, 1], requires_grad=True)

    def forward(self, x):
        h = (x @ self.w1 + self.b1).relu()
        out = h @ self.w2 + self.b2
        return out

class TestBrainIntegration(unittest.TestCase):
    def test_custom_module_and_training_loop(self):
        model = CustomMLP()
        params = model.parameters()
        self.assertEqual(len(params), 4)

        opt = brain.optim.SGD(params, lr=0.02, momentum=0.0)
        criterion = brain.nn.MSELoss()

        # Training data: x -> target
        x = brain.Value([1.0, 0.5, -1.0], shape=[1, 3], requires_grad=False)
        target = brain.Value([1.5], shape=[1, 1], requires_grad=False)

        initial_loss = None
        final_loss = None

        for step in range(50):
            opt.zero_grad()
            pred = model(x)
            loss = criterion.forward(pred, target)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.05)

if __name__ == "__main__":
    unittest.main()
