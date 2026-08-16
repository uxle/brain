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
    fn test_view_stress_case_001() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_002() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_003() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_004() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_005() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_006() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_007() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_008() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_009() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_010() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_011() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_012() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_013() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_014() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_015() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_016() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_017() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_018() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_019() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_020() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_021() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_022() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_023() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_024() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_025() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_026() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_027() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_028() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_029() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_030() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_031() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_032() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_033() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_034() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_035() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_036() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_037() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_038() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_039() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_040() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_041() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_042() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_043() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_044() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_045() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_046() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_047() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_048() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_049() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_050() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_051() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_052() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_053() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_054() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_055() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_056() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_057() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_058() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_059() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_060() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_061() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_062() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_063() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_064() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_065() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_066() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_067() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_068() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_069() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_070() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_071() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_072() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_073() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_074() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_075() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_076() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_077() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_078() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_079() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_080() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_081() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_082() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_083() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_084() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_085() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_086() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_087() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_088() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_089() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_090() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_091() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_092() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_093() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_094() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_095() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_096() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_097() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_098() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_099() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_100() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_101() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_102() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_103() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_104() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_105() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_106() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_107() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_108() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_109() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_110() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_111() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_112() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_113() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_114() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_115() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_116() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_117() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_118() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_119() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_120() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_121() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_122() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_123() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_124() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_125() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_126() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_127() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_128() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_129() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_130() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_131() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_132() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_133() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_134() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_135() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_136() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_137() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_138() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_139() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_140() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_141() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_142() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_143() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_144() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_145() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_146() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_147() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_148() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_149() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_150() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_151() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_152() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_153() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_154() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_155() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_156() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_157() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_158() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_159() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_160() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_161() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_162() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_163() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_164() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_165() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_166() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_167() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_168() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_169() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_170() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_171() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_172() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_173() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_174() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_175() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_176() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_177() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_178() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_179() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_180() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_181() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_182() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_183() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_184() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_185() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_186() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_187() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_188() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_189() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_190() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_191() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_192() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_193() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_194() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_195() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_196() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_197() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_198() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_199() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_200() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_201() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_202() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_203() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_204() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_205() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_206() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_207() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_208() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_209() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_210() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_211() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_212() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_213() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_214() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_215() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_216() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_217() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_218() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_219() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_220() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_221() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_222() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_223() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_224() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_225() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_226() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_227() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_228() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_229() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_230() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_231() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_232() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_233() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_234() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_235() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_236() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_237() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_238() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_239() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_240() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_241() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_242() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_243() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_244() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_245() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_246() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_247() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_248() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_249() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_250() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_251() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_252() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_253() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_254() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_255() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_256() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_257() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_258() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_259() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_260() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_261() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_262() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_263() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_264() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_265() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_266() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_267() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_268() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_269() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_270() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_271() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_272() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_273() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_274() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_275() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_276() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_277() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_278() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_279() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_280() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_281() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_282() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_283() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_284() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_285() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_286() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_287() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_288() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_289() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_290() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_291() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_292() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_293() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_294() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_295() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_296() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_297() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_298() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_299() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_300() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_301() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_302() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_303() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_304() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_305() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_306() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_307() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_308() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_309() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_310() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_311() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_312() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_313() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_314() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_315() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_316() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_317() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_318() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_319() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_320() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_321() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_322() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_323() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_324() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_325() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_326() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_327() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_328() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_329() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_330() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_331() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_332() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_333() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_334() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_335() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_336() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_337() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_338() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_339() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_340() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_341() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_342() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_343() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }

    #[test]
    fn test_view_stress_case_344() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let v = TensorView::from_tensor(&t);
        assert_eq!(v.numel(), 10);
        let vf = v.flip(0);
        assert_eq!(vf.get(&[0]), 9.0);
        assert_eq!(vf.get(&[9]), 0.0);
    }
}
