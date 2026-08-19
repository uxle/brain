//! Arbitrary strided views, zero-copy transformations, and memory layouts.
//!
//! This module provides [`TensorView`] with support for arbitrary and negative strides,
//! zero-copy slicing, flipping, permuting, and layout contiguous verification.

use crate::tensor::Tensor;

/// A non-owning strided view over a contiguous slice of tensor elements.
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    data: &'a [f64],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<'a> TensorView<'a> {
    /// Creates a new view from a slice, shape, strides, and starting offset.
    pub fn new(data: &'a [f64], shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
        TensorView {
            data,
            shape,
            strides,
            offset,
        }
    }

    /// Creates a contiguous view over an entire tensor.
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        let strides: Vec<isize> = tensor.strides().iter().map(|&s| s as isize).collect();
        TensorView {
            data: tensor.data(),
            shape: tensor.shape().to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the rank of the view.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total virtual number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Gets an element at multi-dimensional coordinates.
    pub fn get(&self, coords: &[usize]) -> f64 {
        assert_eq!(coords.len(), self.shape.len());
        let mut idx = self.offset as isize;
        for (i, &c) in coords.iter().enumerate() {
            idx += (c as isize) * self.strides[i];
        }
        self.data[idx as usize]
    }

    /// Transposes two dimensions zero-copy.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        TensorView {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    /// Flips/reverses a dimension zero-copy by inverting its stride.
    pub fn flip(&self, dim: usize) -> Self {
        assert!(dim < self.ndim());
        let dim_len = self.shape[dim];
        if dim_len <= 1 {
            return self.clone();
        }
        let mut new_strides = self.strides.clone();
        let stride = new_strides[dim];
        new_strides[dim] = -stride;
        let new_offset = (self.offset as isize + (dim_len as isize - 1) * stride) as usize;

        TensorView {
            data: self.data,
            shape: self.shape.clone(),
            strides: new_strides,
            offset: new_offset,
        }
    }

    /// Materializes the view into an owned contiguous [`Tensor`].
    pub fn to_tensor(&self) -> Tensor {
        let numel = self.numel();
        let mut out = Vec::with_capacity(numel);
        let rank = self.ndim();
        let mut coords = vec![0usize; rank];

        for _ in 0..numel {
            out.push(self.get(&coords));
            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < self.shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
        Tensor::new(out, self.shape.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_basic() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.get(&[0, 1]), 2.0);
        let vt = v.transpose(0, 1);
        assert_eq!(vt.get(&[1, 0]), 2.0);
    }

    #[test]
    fn test_tensor_view_flip() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let v = TensorView::from_tensor(&t);
        let vf = v.flip(0);
        let mat = vf.to_tensor();
        assert_eq!(mat.data(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_view_transformations() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = a.transpose(0, 1);
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.get_2d(0, 1), 4.0);

        let flat = a.flatten(0, 1);
        assert_eq!(flat.shape(), &[6]);

        let sq = Tensor::ones(vec![1, 2, 1, 3]).squeeze();
        assert_eq!(sq.shape(), &[2, 3]);
    }
}
