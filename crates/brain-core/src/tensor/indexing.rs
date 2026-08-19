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
    fn test_slice_gather_scatter_table() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let sl = a.slice(0, 1, 3, 1);
        assert_eq!(sl.shape(), &[2]);
        assert_eq!(sl.to_vec(), vec![20.0, 30.0]);

        let idx = Tensor::from_slice(&[0.0, 2.0, 1.0], vec![3]);
        let g = gather(&a, 0, &idx);
        assert_eq!(g.to_vec(), vec![10.0, 30.0, 20.0]);
    }
}
