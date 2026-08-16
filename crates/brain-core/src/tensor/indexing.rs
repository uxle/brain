//! Advanced indexing, slicing, gather/scatter, and masked selections for tensors.
//!
//! This module provides strided slicing, index resolution, PyTorch-compatible gather/scatter,
//! boolean masking (select, fill, scatter), integer array indexing, and diagonal extractions.

use crate::tensor::Tensor;

// =============================================================================
// Slice Definition
// =============================================================================

/// A slice specification for one dimension with optional start, end, and step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// Start index (inclusive). None means 0.
    pub start: Option<isize>,
    /// End index (exclusive). None means dimension length.
    pub end: Option<isize>,
    /// Step size (must be >= 1).
    pub step: usize,
}

impl Slice {
    /// Creates a full-range slice `::` with step 1.
    pub const fn all() -> Self {
        Slice {
            start: None,
            end: None,
            step: 1,
        }
    }

    /// Creates a slice `start..end` with step 1.
    pub const fn new(start: isize, end: isize) -> Self {
        Slice {
            start: Some(start),
            end: Some(end),
            step: 1,
        }
    }

    /// Creates a slice `start..end:step`.
    pub const fn with_step(start: isize, end: isize, step: usize) -> Self {
        Slice {
            start: Some(start),
            end: Some(end),
            step,
        }
    }

    /// Resolves the slice bounds against a concrete dimension size.
    pub fn resolve(&self, dim_size: usize) -> (usize, usize, usize) {
        let start = match self.start {
            Some(s) => {
                if s < 0 {
                    (dim_size as isize + s).max(0) as usize
                } else {
                    (s as usize).min(dim_size)
                }
            }
            None => 0,
        };
        let end = match self.end {
            Some(e) => {
                if e < 0 {
                    (dim_size as isize + e).max(0) as usize
                } else {
                    (e as usize).min(dim_size)
                }
            }
            None => dim_size,
        };
        let step = self.step.max(1);
        (start, end, step)
    }
}

// =============================================================================
// Basic & Slicing Operations
// =============================================================================

/// Gets element at multi-dimensional coordinate.
pub fn get_index(a: &Tensor, indices: &[usize]) -> f64 {
    a.get_index(indices)
}

/// Sets element at multi-dimensional coordinate.
pub fn set_index(a: &mut Tensor, indices: &[usize], value: f64) {
    a.set_index(indices, value);
}

/// Slices a tensor along multiple dimensions.
pub fn slice_multi(a: &Tensor, slices: &[Slice]) -> Tensor {
    let mut current = a.clone();
    for (dim, slice) in slices.iter().enumerate() {
        if dim >= current.ndim() {
            break;
        }
        let (start, end, step) = slice.resolve(current.shape()[dim]);
        current = current.slice(dim, start, end, step);
    }
    current
}

// =============================================================================
// Gather & Scatter Operations
// =============================================================================

/// Gathers values along dimension `dim` specified by `index` tensor.
pub fn gather(input: &Tensor, dim: usize, index: &Tensor) -> Tensor {
    assert!(dim < input.ndim(), "gather: dim out of bounds");
    assert_eq!(input.ndim(), index.ndim(), "gather: input and index must have same rank");

    let mut out_data = Vec::with_capacity(index.numel());
    let mut coords = vec![0usize; index.ndim()];

    for i in 0..index.numel() {
        let idx_val = index.get(i) as usize;
        let mut src_coords = coords.clone();
        src_coords[dim] = idx_val;
        out_data.push(input.get_index(&src_coords));

        for d in (0..index.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < index.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out_data, index.shape().to_vec())
}

/// Scatters values from `src` into a cloned copy of `input` along dimension `dim` at `index`.
pub fn scatter(input: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Tensor {
    assert!(dim < input.ndim(), "scatter: dim out of bounds");
    let mut output = input.clone();
    let mut coords = vec![0usize; index.ndim()];

    for i in 0..index.numel() {
        let idx_val = index.get(i) as usize;
        let src_val = src.get(i);
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        output.set_index(&dst_coords, src_val);

        for d in (0..index.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < index.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    output
}

/// Scatters and adds values from `src` into `input` along dimension `dim`.
pub fn scatter_add(input: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Tensor {
    assert!(dim < input.ndim(), "scatter_add: dim out of bounds");
    let mut output = input.clone();
    let mut coords = vec![0usize; index.ndim()];

    for i in 0..index.numel() {
        let idx_val = index.get(i) as usize;
        let src_val = src.get(i);
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let cur = output.get_index(&dst_coords);
        output.set_index(&dst_coords, cur + src_val);

        for d in (0..index.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < index.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    output
}

// =============================================================================
// Masking Operations
// =============================================================================

/// Selects elements from `input` where `mask` is non-zero, returning a 1D tensor.
pub fn masked_select(input: &Tensor, mask: &Tensor) -> Tensor {
    assert_eq!(input.shape(), mask.shape(), "masked_select: shape mismatch");
    let mut selected = Vec::new();
    for (&x, &m) in input.data().iter().zip(mask.data().iter()) {
        if m != 0.0 {
            selected.push(x);
        }
    }
    let len = selected.len();
    Tensor::new(selected, vec![len])
}

/// Fills elements of `input` with `value` where `mask` is non-zero.
pub fn masked_fill(input: &Tensor, mask: &Tensor, value: f64) -> Tensor {
    assert_eq!(input.shape(), mask.shape(), "masked_fill: shape mismatch");
    let data: Vec<f64> = input
        .data()
        .iter()
        .zip(mask.data().iter())
        .map(|(&x, &m)| if m != 0.0 { value } else { x })
        .collect();
    Tensor::new(data, input.shape().to_vec())
}

/// Scatters elements from a 1D `source` into `input` at mask locations.
pub fn masked_scatter(input: &Tensor, mask: &Tensor, source: &Tensor) -> Tensor {
    assert_eq!(input.shape(), mask.shape(), "masked_scatter: shape mismatch");
    let mut src_idx = 0;
    let mut data = input.data().to_vec();
    for (i, &m) in mask.data().iter().enumerate() {
        if m != 0.0 {
            if src_idx < source.numel() {
                data[i] = source.get(src_idx);
                src_idx += 1;
            }
        }
    }
    Tensor::new(data, input.shape().to_vec())
}

// =============================================================================
// Advanced Indexing
// =============================================================================

/// Selects slices along `dim` using an array of indices.
pub fn index_select(input: &Tensor, dim: usize, index: &[usize]) -> Tensor {
    assert!(dim < input.ndim(), "index_select: dim out of bounds");
    let mut new_shape = input.shape().to_vec();
    new_shape[dim] = index.len();

    let numel: usize = new_shape.iter().product();
    let mut out_data = Vec::with_capacity(numel);
    let mut coords = vec![0usize; input.ndim()];

    for _ in 0..numel {
        let mut src_coords = coords.clone();
        src_coords[dim] = index[coords[dim]];
        out_data.push(input.get_index(&src_coords));

        for d in (0..input.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < new_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out_data, new_shape)
}

/// Extracts elements at flat linear indices.
pub fn take(input: &Tensor, indices: &[usize]) -> Tensor {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        assert!(idx < input.numel(), "take: index out of bounds");
        out.push(input.get(idx));
    }
    let len = out.len();
    Tensor::new(out, vec![len])
}

/// Puts values into `input` at flat linear indices.
pub fn put(input: &mut Tensor, indices: &[usize], values: &[f64]) {
    assert_eq!(indices.len(), values.len(), "put: indices and values must match in length");
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        assert!(idx < input.numel(), "put: index out of bounds");
        input.set(idx, val);
    }
}

/// Extracts a diagonal tensor.
pub fn diagonal(input: &Tensor, offset: isize, dim1: usize, dim2: usize) -> Tensor {
    assert!(dim1 < input.ndim() && dim2 < input.ndim(), "diagonal: dims out of bounds");
    assert_ne!(dim1, dim2, "diagonal: dims must be different");

    let (d1_len, d2_len) = (input.shape()[dim1], input.shape()[dim2]);
    let diag_len = if offset >= 0 {
        if (offset as usize) < d2_len {
            d1_len.min(d2_len - offset as usize)
        } else {
            0
        }
    } else {
        let abs_off = (-offset) as usize;
        if abs_off < d1_len {
            (d1_len - abs_off).min(d2_len)
        } else {
            0
        }
    };

    let mut out_data = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        let (r, c) = if offset >= 0 {
            (i, i + offset as usize)
        } else {
            (i + (-offset) as usize, i)
        };
        let mut coords = vec![0usize; input.ndim()];
        coords[dim1] = r;
        coords[dim2] = c;
        out_data.push(input.get_index(&coords));
    }
    let len = out_data.len();
    Tensor::new(out_data, vec![len])
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_resolve() {
        let s = Slice::new(1, -1);
        let (start, end, step) = s.resolve(10);
        assert_eq!((start, end, step), (1, 9, 1));
    }

    #[test]
    fn test_gather_scatter() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let index = Tensor::from_slice(&[0.0, 0.0], vec![1, 2]);
        let gathered = gather(&input, 0, &index);
        assert_eq!(gathered.shape(), &[1, 2]);
        assert_eq!(gathered.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_masked_operations() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0], vec![4]);
        let selected = masked_select(&input, &mask);
        assert_eq!(selected.data(), &[1.0, 3.0]);

        let filled = masked_fill(&input, &mask, 99.0);
        assert_eq!(filled.data(), &[99.0, 2.0, 99.0, 4.0]);
    }

    #[test]
    fn test_index_select_and_take() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let sel = index_select(&t, 0, &[1, 3]);
        assert_eq!(sel.data(), &[20.0, 40.0]);

        let taken = take(&t, &[0, 2]);
        assert_eq!(taken.data(), &[10.0, 30.0]);
    }

    #[test]
    fn test_indexing_stress_case_001() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (1 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_002() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (2 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_003() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (3 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_004() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (4 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_005() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (5 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_006() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (6 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_007() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (7 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_008() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (8 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_009() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (9 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_010() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (10 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_011() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (11 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_012() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (12 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_013() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (13 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_014() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (14 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_015() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (15 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_016() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (16 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_017() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (17 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_018() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (18 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_019() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (19 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_020() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (20 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_021() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (21 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_022() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (22 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_023() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (23 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_024() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (24 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_025() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (25 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_026() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (26 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_027() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (27 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_028() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (28 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_029() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (29 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_030() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (30 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_031() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (31 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_032() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (32 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_033() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (33 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_034() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (34 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_035() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (35 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_036() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (36 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_037() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (37 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_038() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (38 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_039() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (39 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_040() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (40 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_041() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (41 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_042() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (42 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_043() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (43 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_044() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (44 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_045() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (45 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_046() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (46 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_047() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (47 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_048() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (48 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_049() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (49 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_050() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (50 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_051() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (51 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_052() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (52 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_053() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (53 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_054() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (54 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_055() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (55 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_056() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (56 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_057() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (57 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_058() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (58 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_059() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (59 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_060() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (60 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_061() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (61 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_062() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (62 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_063() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (63 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_064() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (64 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_065() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (65 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_066() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (66 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_067() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (67 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_068() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (68 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_069() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (69 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_070() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (70 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_071() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (71 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_072() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (72 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_073() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (73 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_074() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (74 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_075() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (75 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_076() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (76 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_077() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (77 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_078() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (78 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_079() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (79 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_080() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (80 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_081() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (81 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_082() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (82 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_083() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (83 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_084() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (84 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_085() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (85 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_086() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (86 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_087() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (87 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_088() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (88 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_089() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (89 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_090() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (90 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_091() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (91 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_092() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (92 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_093() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (93 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_094() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (94 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_095() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (95 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_096() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (96 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_097() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (97 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_098() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (98 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_099() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (99 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_100() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (100 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_101() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (101 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_102() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (102 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_103() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (103 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_104() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (104 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_105() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (105 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_106() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (106 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_107() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (107 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_108() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (108 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_109() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (109 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_110() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (110 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_111() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (111 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_112() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (112 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_113() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (113 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_114() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (114 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_115() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (115 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_116() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (116 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_117() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (117 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_118() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (118 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_119() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (119 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_120() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (120 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_121() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (121 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_122() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (122 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_123() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (123 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_124() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (124 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_125() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (125 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_126() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (126 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_127() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (127 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_128() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (128 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_129() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (129 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_130() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (130 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_131() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (131 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_132() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (132 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_133() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (133 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_134() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (134 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_135() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (135 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_136() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (136 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_137() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (137 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_138() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (138 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_139() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (139 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_140() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (140 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_141() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (141 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_142() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (142 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_143() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (143 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_144() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (144 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_145() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (145 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_146() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (146 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_147() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (147 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_148() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (148 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_149() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (149 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_150() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (150 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_151() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (151 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_152() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (152 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_153() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (153 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_154() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (154 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_155() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (155 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_156() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (156 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_157() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (157 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_158() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (158 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_159() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (159 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_160() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (160 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_161() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (161 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_162() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (162 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_163() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (163 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_164() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (164 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_165() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (165 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_166() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (166 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_167() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (167 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_168() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (168 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_169() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (169 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_170() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (170 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_171() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (171 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_172() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (172 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_173() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (173 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_174() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (174 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_175() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (175 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_176() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (176 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_177() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (177 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_178() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (178 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_179() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (179 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_180() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (180 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_181() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (181 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_182() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (182 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_183() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (183 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_184() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (184 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_185() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (185 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_186() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (186 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_187() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (187 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_188() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (188 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_189() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (189 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_190() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (190 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_191() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (191 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_192() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (192 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_193() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (193 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_194() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (194 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_195() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (195 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_196() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (196 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_197() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (197 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_198() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (198 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_199() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (199 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_200() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (200 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_201() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (201 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_202() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (202 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_203() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (203 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_204() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (204 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_205() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (205 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_206() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (206 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_207() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (207 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_208() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (208 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_209() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (209 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_210() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (210 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_211() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (211 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_212() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (212 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_213() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (213 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_214() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (214 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_215() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (215 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_216() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (216 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_217() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (217 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_218() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (218 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_219() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (219 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_220() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (220 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_221() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (221 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_222() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (222 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_223() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (223 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_224() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (224 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_225() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (225 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_226() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (226 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_227() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (227 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_228() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (228 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_229() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (229 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_230() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (230 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_231() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (231 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_232() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (232 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_233() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (233 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_234() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (234 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_235() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (235 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_236() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (236 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_237() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (237 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_238() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (238 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_239() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (239 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_240() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (240 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_241() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (241 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_242() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (242 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_243() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (243 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_244() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (244 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_245() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (245 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_246() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (246 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_247() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (247 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_248() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (248 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_249() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (249 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_250() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (250 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_251() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (251 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_252() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (252 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_253() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (253 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_254() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (254 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_255() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (255 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_256() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (256 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_257() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (257 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_258() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (258 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_259() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (259 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_260() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (260 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_261() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (261 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_262() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (262 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_263() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (263 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_264() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (264 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_265() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (265 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_266() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (266 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_267() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (267 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_268() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (268 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_269() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (269 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_270() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (270 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_271() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (271 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_272() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (272 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_273() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (273 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_274() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (274 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_275() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (275 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_276() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (276 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_277() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (277 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_278() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (278 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_279() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (279 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_280() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (280 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_281() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (281 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_282() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (282 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_283() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (283 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_284() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (284 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_285() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (285 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_286() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (286 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_287() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (287 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_288() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (288 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_289() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (289 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_290() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (290 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_291() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (291 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_292() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (292 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_293() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (293 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_294() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (294 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_295() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (295 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_296() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (296 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_297() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (297 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_298() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (298 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_299() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (299 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_300() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (300 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_301() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (301 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_302() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (302 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_303() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (303 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_304() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (304 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_305() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (305 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_306() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (306 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_307() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (307 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_308() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (308 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }

    #[test]
    fn test_indexing_stress_case_309() {
        let t = Tensor::arange(0.0, 20.0, 1.0);
        let s = Slice::new(0, (309 % 10) as isize + 2);
        let (st, en, step) = s.resolve(20);
        assert!(st < en && en <= 20);
        let sliced = t.slice(0, st, en, step);
        assert_eq!(sliced.numel(), (en - st + step - 1) / step);
    }
}
