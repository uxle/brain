"""
Unit tests for Brain Python Tensor operations.
"""

import unittest
import brain

class TestBrainTensor(unittest.TestCase):
    def test_tensor_creation(self):
        t = brain.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
        self.assertEqual(t.shape, [2, 2])
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.numel, 4)
        self.assertEqual(t.to_list(), [1.0, 2.0, 3.0, 4.0])

    def test_zeros_and_ones(self):
        z = brain.zeros([3, 4])
        self.assertEqual(z.shape, [3, 4])
        self.assertEqual(z.numel, 12)
        self.assertTrue(all(v == 0.0 for v in z.to_list()))

        o = brain.ones([2, 5])
        self.assertEqual(o.shape, [2, 5])
        self.assertEqual(o.numel, 10)
        self.assertTrue(all(v == 1.0 for v in o.to_list()))

    def test_tensor_factory(self):
        t = brain.tensor([5.0, 6.0], shape=[2])
        self.assertEqual(t.shape, [2])
        self.assertEqual(t.to_list(), [5.0, 6.0])

    def test_shape_mismatch_error(self):
        with self.assertRaises(ValueError):
            brain.Tensor([1.0, 2.0], shape=[3, 3])

    def test_item_extraction(self):
        scalar = brain.Tensor([42.0], shape=[1])
        self.assertEqual(scalar.item(), 42.0)

        multi = brain.Tensor([1.0, 2.0], shape=[2])
        with self.assertRaises(ValueError):
            multi.item()

    def test_reshape(self):
        t = brain.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        r = t.reshape([3, 2])
        self.assertEqual(r.shape, [3, 2])
        self.assertEqual(r.to_list(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_transpose(self):
        t = brain.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        tr = t.transpose(0, 1)
        self.assertEqual(tr.shape, [3, 2])
        self.assertEqual(tr.to_list(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_reductions(self):
        t = brain.Tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
        self.assertEqual(t.sum().item(), 10.0)
        self.assertEqual(t.mean().item(), 2.5)

    def test_arithmetic(self):
        a = brain.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
        b = brain.Tensor([10.0, 20.0, 30.0, 40.0], shape=[2, 2])

        c_add = a + b
        self.assertEqual(c_add.to_list(), [11.0, 22.0, 33.0, 44.0])

        c_sub = b - a
        self.assertEqual(c_sub.to_list(), [9.0, 18.0, 27.0, 36.0])

        c_mul = a * b
        self.assertEqual(c_mul.to_list(), [10.0, 40.0, 90.0, 160.0])

    def test_matmul(self):
        a = brain.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = brain.Tensor([7.0, 8.0, 9.0, 1.0, 2.0, 3.0], shape=[3, 2])
        c = a @ b
        self.assertEqual(c.shape, [2, 2])
        self.assertEqual(c.to_list(), [31.0, 19.0, 85.0, 55.0])

if __name__ == "__main__":
    unittest.main()
