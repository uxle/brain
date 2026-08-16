//! Advanced indexing operations for tensors in the Brain deep learning framework.
//!
//! This module provides various indexing methods including basic indexing,
//! slice indexing, gather/scatter operations, masked operations, and
//! advanced (fancy) indexing.

use crate::tensor::Tensor;

// =============================================================================
// Basic Indexing
// =============================================================================

/// Gets the element at multi-dimensional indices.
pub fn get_index(a: &Tensor, indices: &[usize]) -> f64 {
    a.get_index(indices)
}

/// Sets the element at multi-dimensional indices.
pub fn set_index(a: &mut Tensor, indices: &[usize], value: f64) {
    a.set_index(indices, value);
}

// =============================================================================
// Slice Indexing
// =============================================================================

/// A slice specification for one dimension.
#[derive(Debug, Clone)]
pub struct Slice {
    /// Start index (inclusive), None means 0.
    pub start: Option<isize>,
    /// End index (exclusive), None means end of dimension.
    pub end: Option<isize>,
    /// Step size, defaults to 1.
    pub step: usize,
}

impl Slice {
    /// Creates a full range slice (equivalent to `::`).
    pub fn all() -> Self { Slice { start: None, end: None, step: 1 } }

    /// Creates a slice from start to end (exclusive) with step 1.
    pub fn new(start: isize, end: isize) -> Self { Slice { start: Some(start), end: Some(end), step: 1 } }

    /// Creates a slice with a step.
    pub fn with_step(start: isize, end: isize, step: usize) -> Self { Slice { start: Some(start), end: Some(end), step } }

    /// Resolves the slice for a given dimension size.
    pub fn resolve(&self, dim_size: usize) -> (usize, usize, usize) {
        let start = match self.start {
            Some(s) => if s < 0 { (dim_size as isize + s).max(0) as usize } else { s.min(dim_size as isize) as usize },
            None => 0,
        };
        let end = match self.end {
            Some(e) => if e < 0 { (dim_size as isize + e).max(0) as usize } else { e.min(dim_size as isize) as usize },
            None => dim_size,
        };
        (start, end, self.step)
    }

    /// Returns the length of the resolved slice.
    pub fn len(&self, dim_size: usize) -> usize {
        let (start, end, step) = self.resolve(dim_size);
        if start >= end { return 0; }
        (end - start + step - 1) / step
    }
}

/// Slices a tensor along one or more dimensions.
pub fn slice_tensor(a: &Tensor, slices: &[Slice]) -> Tensor {
    let ndim = a.ndim();
    assert!(slices.len() <= ndim, "Too many slices for {}D tensor", ndim);

    let mut output_shape = Vec::new();
    let mut resolved = Vec::new();

    for (i, sl) in slices.iter().enumerate() {
        let dim = a.shape()[i];
        let (start, end, step) = sl.resolve(dim);
        let len = sl.len(dim);
        if len == 1 && sl.start.is_some() && sl.end.map_or(false, |e| e - sl.start.unwrap() == 1) && sl.step == 1 {
            // Single index: dimension removed (don't add to output shape)
        } else {
            output_shape.push(len);
        }
        resolved.push((start, end, step));
    }
    // Preserve trailing dimensions not covered by slices
    for i in slices.len()..ndim { output_shape.push(a.shape()[i]); }

    let out_numel: usize = output_shape.iter().product();
    let mut data = vec![0.0; out_numel];
    let mut out_idx = 0;

    // Iterate over output elements
    let mut out_multi = vec![0usize; output_shape.len()];
    for _ in 0..out_numel {
        // Build source multi-index
        let mut src_multi = Vec::with_capacity(ndim);
        let mut out_i = 0;
        for (i, sl) in slices.iter().enumerate() {
            let dim = a.shape()[i];
            let (start, _, step) = sl.resolve(dim);
            let len = sl.len(dim);
            if len == 1 && sl.start.is_some() && sl.end.map_or(false, |e| e - sl.start.unwrap() == 1) && sl.step == 1 {
                src_multi.push(start);
            } else {
                src_multi.push(start + out_multi[out_i] * step);
                out_i += 1;
            }
        }
        for i in slices.len()..ndim {
            src_multi.push(out_multi[out_i]);
            out_i += 1;
        }

        let src_flat = a.get_index(&src_multi);
        data[out_idx] = src_flat;
        out_idx += 1;

        // Increment out_multi
        let mut carry = true;
        for i in (0..output_shape.len()).rev() {
            if carry {
                out_multi[i] += 1;
                if out_multi[i] >= output_shape[i] { out_multi[i] = 0; } else { carry = false; }
            }
        }
    }

    Tensor::new(data, output_shape)
}

// =============================================================================
// Gather and Scatter
// =============================================================================

/// Gathers elements along a specified axis.
pub fn gather(a: &Tensor, dim: usize, index: &Tensor) -> Tensor {
    assert!(dim < a.ndim());
    assert!(index.ndim() == a.ndim(), "Index tensor must have same ndim as input");

    let mut out_shape = index.shape().to_vec();
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_numel];

    for flat in 0..out_numel {
        let idx_multi = decompose(flat, &out_shape);
        let idx_val = index.get_index(&idx_multi) as usize;

        let mut src_multi = idx_multi.clone();
        src_multi[dim] = idx_val;

        data[flat] = a.get_index(&src_multi);
    }

    Tensor::new(data, out_shape)
}

/// Scatters values into a tensor along a specified axis.
pub fn scatter(src: &Tensor, dim: usize, index: &Tensor, out: &mut Tensor) {
    assert!(dim < src.ndim());
    assert!(index.ndim() == src.ndim());

    for flat in 0..src.numel() {
        let idx_multi = decompose(flat, src.shape());
        let idx_val = index.get_index(&idx_multi) as usize;
        let val = src.get_index(&idx_multi);

        let mut dest_multi = idx_multi.clone();
        dest_multi[dim] = idx_val;
        out.set_index(&dest_multi, val);
    }
}

// =============================================================================
// Index Select, Add, Copy, Put
// =============================================================================

/// Selects elements from the first dimension using an index tensor.
pub fn index_select(a: &Tensor, dim: usize, index: &Tensor) -> Tensor {
    assert!(dim < a.ndim());
    assert!(index.ndim() == 1, "Index must be 1D");

    let idx_count = index.numel();
    let dim_size = a.shape()[dim];
    let mut out_shape = a.shape().to_vec();
    out_shape[dim] = idx_count;
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_numel];

    for flat in 0..out_numel {
        let out_multi = decompose(flat, &out_shape);
        let idx_val = index.get(out_multi[dim]) as usize;
        assert!(idx_val < dim_size, "Index {} out of bounds for dimension of size {}", idx_val, dim_size);

        let mut src_multi = out_multi.clone();
        src_multi[dim] = idx_val;
        data[flat] = a.get_index(&src_multi);
    }

    Tensor::new(data, out_shape)
}

/// Adds values to specific indices along a dimension.
pub fn index_add(a: &mut Tensor, dim: usize, index: &Tensor, source: &Tensor) {
    assert!(dim < a.ndim());
    assert!(index.ndim() == 1);

    let dim_size = a.shape()[dim];
    for i in 0..index.numel() {
        let idx = index.get(i) as usize;
        assert!(idx < dim_size);

        let sub_size: usize = a.shape()[dim + 1..].iter().product().max(1);
        let src_start = i * sub_size;
        let dst_start = idx * sub_size;

        for j in 0..sub_size {
            let src_val = source.get(src_start + j);
            let dst_val = a.get(dst_start + j);
            a.set(dst_start + j, dst_val + src_val);
        }
    }
}

/// Copies values from source to specific indices along a dimension.
pub fn index_copy(a: &mut Tensor, dim: usize, index: &Tensor, source: &Tensor) {
    assert!(dim < a.ndim());
    assert!(index.ndim() == 1);

    let dim_size = a.shape()[dim];
    for i in 0..index.numel() {
        let idx = index.get(i) as usize;
        assert!(idx < dim_size);

        let sub_size: usize = a.shape()[dim + 1..].iter().product().max(1);
        let src_start = i * sub_size;
        let dst_start = idx * sub_size;

        for j in 0..sub_size {
            a.set(dst_start + j, source.get(src_start + j));
        }
    }
}

/// Puts values at specified indices.
pub fn index_put(a: &mut Tensor, indices: &[usize], values: &[f64]) {
    assert_eq!(indices.len(), values.len());
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        assert!(idx < a.numel());
        a.set(idx, val);
    }
}

// =============================================================================
// Masked Operations
// =============================================================================

/// Selects elements where the mask is true (nonzero).
pub fn masked_select(a: &Tensor, mask: &Tensor) -> Tensor {
    assert_eq!(a.numel(), mask.numel(), "Tensor and mask must have same number of elements");
    let data: Vec<f64> = a.data().iter().zip(mask.data().iter())
        .filter(|(_, &m)| m != 0.0)
        .map(|(&v, _)| v)
        .collect();
    let count = data.len();
    Tensor::new(data, vec![count])
}

/// Fills elements where the mask is true with the given value.
pub fn masked_fill(a: &mut Tensor, mask: &Tensor, value: f64) {
    assert_eq!(a.numel(), mask.numel());
    for i in 0..a.numel() {
        if mask.get(i) != 0.0 {
            a.set(i, value);
        }
    }
}

/// Selects elements from one of two tensors based on a condition.
pub fn where_fn(condition: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
    let out_shape = crate::tensor::binary_broadcast_shape(condition.shape(), x.shape());
    let out_shape = crate::tensor::binary_broadcast_shape(&out_shape, y.shape());
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_numel];

    for i in 0..out_numel {
        let c_idx = crate::tensor::broadcast_flat_index(i, &out_shape, condition.shape());
        let x_idx = crate::tensor::broadcast_flat_index(i, &out_shape, x.shape());
        let y_idx = crate::tensor::broadcast_flat_index(i, &out_shape, y.shape());

        if condition.get(c_idx) != 0.0 { data[i] = x.get(x_idx); }
        else { data[i] = y.get(y_idx); }
    }

    Tensor::new(data, out_shape)
}

// =============================================================================
// Narrow, Select, Take
// =============================================================================

/// Narrows the tensor at the given dimension and range.
pub fn narrow(a: &Tensor, dim: usize, start: usize, length: usize) -> Tensor {
    a.narrow(dim, start, length)
}

/// Selects elements from a dimension at the given index.
pub fn select(a: &Tensor, dim: usize, index: usize) -> Tensor {
    a.narrow(dim, index, 1)
}

/// Takes elements from the tensor at the given indices.
pub fn take_fn(a: &Tensor, indices: &[usize]) -> Tensor {
    a.take(indices)
}

// =============================================================================
// Advanced (Fancy) Indexing
// =============================================================================

/// Gathers elements using advanced (fancy) indexing with index tensors.
pub fn fancy_index(a: &Tensor, indices: &[Tensor]) -> Tensor {
    let idx_count = indices.len();
    assert!(idx_count > 0);

    // All index tensors must be 1D and have the same length
    let idx_len = indices[0].numel();
    for idx in indices { assert!(idx.ndim() == 1 && idx.numel() == idx_len); }
    assert!(idx_count <= a.ndim());

    let mut data = Vec::with_capacity(idx_len);
    for i in 0..idx_len {
        let mut multi = Vec::with_capacity(a.ndim());
        for (d, idx) in indices.iter().enumerate() {
            multi.push(idx.get(i) as usize);
        }
        // Fill remaining dimensions with 0
        while multi.len() < a.ndim() { multi.push(0); }

        if multi.len() == a.ndim() && multi.iter().zip(a.shape().iter()).all(|(&m, &s)| m < s) {
            data.push(a.get_index(&multi));
        }
    }

    Tensor::new(data, vec![data.len()])
}

/// Boolean mask indexing: returns elements where mask is true.
pub fn boolean_index(a: &Tensor, mask: &[bool]) -> Tensor {
    assert_eq!(a.numel(), mask.len());
    let data: Vec<f64> = a.data().iter().zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();
    Tensor::new(data, vec![data.len()])
}

// =============================================================================
// Helper Functions
// =============================================================================

fn decompose(flat: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut multi = vec![0usize; n];
    let mut idx = flat;
    for i in (0..n).rev() {
        if shape[i] > 0 {
            multi[i] = idx % shape[i];
            idx /= shape[i];
        }
    }
    multi
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_set_index() {
        let mut a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(get_index(&a, &[0, 0]), 1.0);
        assert_eq!(get_index(&a, &[1, 1]), 4.0);
        set_index(&mut a, &[0, 1], 10.0);
        assert_eq!(get_index(&a, &[0, 1]), 10.0);
    }

    #[test]
    fn test_slice_all() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let s = slice_tensor(&a, &[Slice::all()]);
        assert_eq!(s.shape(), &[3, 4]);
        for i in 0..12 { assert!((s.get(i) - a.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_slice_row() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let s = slice_tensor(&a, &[Slice::new(1, 3)]);
        assert_eq!(s.shape(), &[2, 4]);
        assert_eq!(s.get(0), 4.0);
        assert_eq!(s.get(4), 8.0);
    }

    #[test]
    fn test_slice_element() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let s = slice_tensor(&a, &[Slice::new(1, 2)]);
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.get(0), 4.0);
    }

    #[test]
    fn test_slice_negative() {
        let a = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], vec![4]);
        let s = slice_tensor(&a, &[Slice::new(-2, -1)]);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(s.get(0), 2.0);
    }

    #[test]
    fn test_slice_with_step() {
        let a = Tensor::arange(0.0, 10.0, 1.0);
        let s = slice_tensor(&a, &[Slice::with_step(0, 10, 2)]);
        assert_eq!(s.shape(), &[5]);
        assert_eq!(s.get(0), 0.0);
        assert_eq!(s.get(1), 2.0);
        assert_eq!(s.get(4), 8.0);
    }

    #[test]
    fn test_gather() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let idx = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        let g = gather(&a, 0, &idx);
        assert_eq!(g.get_index(&[0, 0]), 1.0);
        assert_eq!(g.get_index(&[1, 0]), 3.0);
    }

    #[test]
    fn test_scatter() {
        let src = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        let idx = Tensor::from_slice(&[0.0, 2.0, 1.0], vec![3]);
        let mut out = Tensor::zeros(vec![4]);
        scatter(&src, 0, &idx, &mut out);
        assert_eq!(out.get(0), 10.0);
        assert_eq!(out.get(1), 30.0);
        assert_eq!(out.get(2), 20.0);
    }

    #[test]
    fn test_index_select() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let idx = Tensor::from_slice(&[2.0, 0.0, 1.0], vec![3]);
        let s = index_select(&a, 0, &idx);
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(s.get_index(&[0, 0]), 8.0); // row 2
        assert_eq!(s.get_index(&[1, 0]), 0.0); // row 0
        assert_eq!(s.get_index(&[2, 0]), 4.0); // row 1
    }

    #[test]
    fn test_index_add() {
        let mut a = Tensor::zeros(vec![3, 3]);
        let idx = Tensor::from_slice(&[0.0, 2.0], vec![2]);
        let src = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        index_add(&mut a, 0, &idx, &src);
        assert_eq!(a.get_index(&[0, 0]), 1.0);
        assert_eq!(a.get_index(&[1, 0]), 0.0);
        assert_eq!(a.get_index(&[2, 0]), 4.0);
    }

    #[test]
    fn test_index_copy() {
        let mut a = Tensor::zeros(vec![3, 2]);
        let idx = Tensor::from_slice(&[1.0, 0.0], vec![2]);
        let src = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        index_copy(&mut a, 0, &idx, &src);
        assert_eq!(a.get_index(&[0, 0]), 30.0); // row 0 copied from src[1]
        assert_eq!(a.get_index(&[1, 0]), 10.0); // row 1 copied from src[0]
    }

    #[test]
    fn test_masked_select() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let mask = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0], vec![5]);
        let s = masked_select(&a, &mask);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(1), 3.0);
        assert_eq!(s.get(2), 5.0);
    }

    #[test]
    fn test_masked_fill() {
        let mut a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0], vec![4]);
        masked_fill(&mut a, &mask, -1.0);
        assert_eq!(a.get(0), -1.0);
        assert_eq!(a.get(1), 2.0);
        assert_eq!(a.get(2), -1.0);
        assert_eq!(a.get(3), 4.0);
    }

    #[test]
    fn test_where_fn() {
        let cond = Tensor::from_slice(&[1.0, 0.0, 1.0], vec![3]);
        let x = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        let y = Tensor::from_slice(&[-10.0, -20.0, -30.0], vec![3]);
        let r = where_fn(&cond, &x, &y);
        assert_eq!(r.get(0), 10.0);
        assert_eq!(r.get(1), -20.0);
        assert_eq!(r.get(2), 30.0);
    }

    #[test]
    fn test_narrow() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let n = narrow(&a, 0, 1, 2);
        assert_eq!(n.shape(), &[2, 4]);
        assert_eq!(n.get_index(&[0, 0]), 4.0);
        assert_eq!(n.get_index(&[1, 0]), 8.0);
    }

    #[test]
    fn test_select() {
        let a = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let s = select(&a, 1, 2);
        assert_eq!(s.shape(), &[1, 4]);
        assert_eq!(s.get_index(&[0, 0]), 8.0);
    }

    #[test]
    fn test_take_fn() {
        let a = Tensor::arange(0.0, 10.0, 1.0);
        let t = take_fn(&a, &[0, 3, 7, 9]);
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.get(0), 0.0);
        assert_eq!(t.get(1), 3.0);
        assert_eq!(t.get(2), 7.0);
        assert_eq!(t.get(3), 9.0);
    }

    #[test]
    fn test_fancy_index() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let idx = Tensor::from_slice(&[0.0, 2.0, 3.0], vec![3]);
        let r = fancy_index(&a, &[idx]);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.get(0), 10.0);
        assert_eq!(r.get(1), 30.0);
        assert_eq!(r.get(2), 40.0);
    }

    #[test]
    fn test_boolean_index() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let mask = vec![true, false, true, false, true];
        let r = boolean_index(&a, &mask);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.get(0), 1.0);
        assert_eq!(r.get(1), 3.0);
        assert_eq!(r.get(2), 5.0);
    }

    #[test]
    fn test_index_put() {
        let mut a = Tensor::zeros(vec![5]);
        index_put(&mut a, &[0, 2, 4], &[1.0, 3.0, 5.0]);
        assert_eq!(a.get(0), 1.0);
        assert_eq!(a.get(1), 0.0);
        assert_eq!(a.get(2), 3.0);
        assert_eq!(a.get(4), 5.0);
    }

    #[test]
    fn test_slice_resolve() {
        let s = Slice::new(2, 7);
        assert_eq!(s.resolve(10), (2, 7, 1));
        assert_eq!(s.len(10), 5);
    }

    #[test]
    fn test_slice_negative_resolve() {
        let s = Slice::new(-3, -1);
        assert_eq!(s.resolve(10), (7, 9, 1));
        assert_eq!(s.len(10), 2);
    }

    #[test]
    fn test_gather_2d() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let idx = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let g = gather(&a, 1, &idx);
        assert_eq!(g.get_index(&[0, 0]), 1.0);
        assert_eq!(g.get_index(&[0, 1]), 2.0);
        assert_eq!(g.get_index(&[1, 0]), 4.0);
        assert_eq!(g.get_index(&[1, 1]), 5.0);
    }

    #[test]
    fn test_masked_select_empty() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let mask = Tensor::zeros(vec![3]);
        let s = masked_select(&a, &mask);
        assert_eq!(s.numel(), 0);
    }
}
