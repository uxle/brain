//! Advanced tensor factory and creation routines.
//!
//! This module provides diagonal matrices, meshgrid grids, triangular matrices (tril, triu),
//! logarithmic ranges (logspace, geomspace), and *_like tensor factories.

use crate::tensor::Tensor;

/// Creates a tensor of zeros with the same shape, dtype, and device as `input`.
pub fn zeros_like(input: &Tensor) -> Tensor {
    let mut t = Tensor::zeros(input.shape().to_vec());
    t.set_dtype(input.dtype());
    t.to_device(input.device());
    t
}

/// Creates a tensor of ones with the same shape, dtype, and device as `input`.
pub fn ones_like(input: &Tensor) -> Tensor {
    let mut t = Tensor::ones(input.shape().to_vec());
    t.set_dtype(input.dtype());
    t.to_device(input.device());
    t
}

/// Creates a tensor filled with `value` matching `input`'s metadata.
pub fn full_like(input: &Tensor, value: f64) -> Tensor {
    let mut t = Tensor::full(input.shape().to_vec(), value);
    t.set_dtype(input.dtype());
    t.to_device(input.device());
    t
}

/// Creates a 1D tensor of `steps` points logarithmically spaced between base^start and base^end.
pub fn logspace(start: f64, end: f64, steps: usize, base: f64) -> Tensor {
    let lin = Tensor::linspace(start, end, steps);
    lin.map(|x| base.powf(x))
}

/// Creates a 1D tensor of `steps` points geometrically spaced between `start` and `end`.
pub fn geomspace(start: f64, end: f64, steps: usize) -> Tensor {
    assert!(
        start > 0.0 && end > 0.0,
        "geomspace: start and end must be positive"
    );
    let log_start = start.ln();
    let log_end = end.ln();
    let lin = Tensor::linspace(log_start, log_end, steps);
    lin.map(|x| x.exp())
}

/// Constructs a 2D diagonal matrix from a 1D diagonal vector.
pub fn diag(input: &Tensor) -> Tensor {
    assert_eq!(input.ndim(), 1);
    let n = input.numel();
    let mut out = Tensor::zeros(vec![n, n]);
    for i in 0..n {
        out.set_2d(i, i, input.get(i));
    }
    out
}

/// Returns the lower triangular part of a 2D matrix.
pub fn tril(input: &Tensor, diagonal: isize) -> Tensor {
    assert_eq!(input.ndim(), 2);
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut out = Tensor::zeros(vec![rows, cols]);
    for r in 0..rows {
        for c in 0..cols {
            if (c as isize) <= (r as isize) + diagonal {
                out.set_2d(r, c, input.get_2d(r, c));
            }
        }
    }
    out
}

/// Returns the upper triangular part of a 2D matrix.
pub fn triu(input: &Tensor, diagonal: isize) -> Tensor {
    assert_eq!(input.ndim(), 2);
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut out = Tensor::zeros(vec![rows, cols]);
    for r in 0..rows {
        for c in 0..cols {
            if (c as isize) >= (r as isize) + diagonal {
                out.set_2d(r, c, input.get_2d(r, c));
            }
        }
    }
    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_like_factories() {
        let t = Tensor::full(vec![2, 3], 7.0);
        let z = zeros_like(&t);
        assert_eq!(z.shape(), &[2, 3]);
        assert_eq!(z.get(0), 0.0);

        let o = ones_like(&t);
        assert_eq!(o.get(0), 1.0);
    }

    #[test]
    fn test_diag_and_tri() {
        let v = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let d = diag(&v);
        assert_eq!(d.get_2d(0, 0), 1.0);
        assert_eq!(d.get_2d(1, 1), 2.0);
        assert_eq!(d.get_2d(0, 1), 0.0);

        let mat = Tensor::ones(vec![3, 3]);
        let l = tril(&mat, 0);
        assert_eq!(l.get_2d(0, 1), 0.0);
        assert_eq!(l.get_2d(1, 0), 1.0);
    }

    #[test]
    fn test_factories_table() {
        let z = Tensor::zeros(vec![2, 3]);
        assert_eq!(z.data(), &[0.0; 6]);

        let o = Tensor::ones(vec![2, 2]);
        assert_eq!(o.data(), &[1.0; 4]);

        let eye = Tensor::eye(3);
        assert_eq!(eye.get_2d(0, 0), 1.0);
        assert_eq!(eye.get_2d(0, 1), 0.0);

        let ar = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(ar.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let ls = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(ls.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
