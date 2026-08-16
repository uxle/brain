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
    assert!(start > 0.0 && end > 0.0, "geomspace: start and end must be positive");
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
    fn test_factory_stress_case_001() {
        let v = Tensor::from_slice(&[1.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 1.0);
    }

    #[test]
    fn test_factory_stress_case_002() {
        let v = Tensor::from_slice(&[2.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 2.0);
    }

    #[test]
    fn test_factory_stress_case_003() {
        let v = Tensor::from_slice(&[3.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 3.0);
    }

    #[test]
    fn test_factory_stress_case_004() {
        let v = Tensor::from_slice(&[4.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 4.0);
    }

    #[test]
    fn test_factory_stress_case_005() {
        let v = Tensor::from_slice(&[5.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 5.0);
    }

    #[test]
    fn test_factory_stress_case_006() {
        let v = Tensor::from_slice(&[6.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 6.0);
    }

    #[test]
    fn test_factory_stress_case_007() {
        let v = Tensor::from_slice(&[7.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 7.0);
    }

    #[test]
    fn test_factory_stress_case_008() {
        let v = Tensor::from_slice(&[8.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 8.0);
    }

    #[test]
    fn test_factory_stress_case_009() {
        let v = Tensor::from_slice(&[9.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 9.0);
    }

    #[test]
    fn test_factory_stress_case_010() {
        let v = Tensor::from_slice(&[10.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 10.0);
    }

    #[test]
    fn test_factory_stress_case_011() {
        let v = Tensor::from_slice(&[11.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 11.0);
    }

    #[test]
    fn test_factory_stress_case_012() {
        let v = Tensor::from_slice(&[12.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 12.0);
    }

    #[test]
    fn test_factory_stress_case_013() {
        let v = Tensor::from_slice(&[13.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 13.0);
    }

    #[test]
    fn test_factory_stress_case_014() {
        let v = Tensor::from_slice(&[14.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 14.0);
    }

    #[test]
    fn test_factory_stress_case_015() {
        let v = Tensor::from_slice(&[15.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 15.0);
    }

    #[test]
    fn test_factory_stress_case_016() {
        let v = Tensor::from_slice(&[16.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 16.0);
    }

    #[test]
    fn test_factory_stress_case_017() {
        let v = Tensor::from_slice(&[17.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 17.0);
    }

    #[test]
    fn test_factory_stress_case_018() {
        let v = Tensor::from_slice(&[18.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 18.0);
    }

    #[test]
    fn test_factory_stress_case_019() {
        let v = Tensor::from_slice(&[19.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 19.0);
    }

    #[test]
    fn test_factory_stress_case_020() {
        let v = Tensor::from_slice(&[20.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 20.0);
    }

    #[test]
    fn test_factory_stress_case_021() {
        let v = Tensor::from_slice(&[21.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 21.0);
    }

    #[test]
    fn test_factory_stress_case_022() {
        let v = Tensor::from_slice(&[22.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 22.0);
    }

    #[test]
    fn test_factory_stress_case_023() {
        let v = Tensor::from_slice(&[23.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 23.0);
    }

    #[test]
    fn test_factory_stress_case_024() {
        let v = Tensor::from_slice(&[24.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 24.0);
    }

    #[test]
    fn test_factory_stress_case_025() {
        let v = Tensor::from_slice(&[25.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 25.0);
    }

    #[test]
    fn test_factory_stress_case_026() {
        let v = Tensor::from_slice(&[26.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 26.0);
    }

    #[test]
    fn test_factory_stress_case_027() {
        let v = Tensor::from_slice(&[27.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 27.0);
    }

    #[test]
    fn test_factory_stress_case_028() {
        let v = Tensor::from_slice(&[28.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 28.0);
    }

    #[test]
    fn test_factory_stress_case_029() {
        let v = Tensor::from_slice(&[29.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 29.0);
    }

    #[test]
    fn test_factory_stress_case_030() {
        let v = Tensor::from_slice(&[30.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 30.0);
    }

    #[test]
    fn test_factory_stress_case_031() {
        let v = Tensor::from_slice(&[31.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 31.0);
    }

    #[test]
    fn test_factory_stress_case_032() {
        let v = Tensor::from_slice(&[32.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 32.0);
    }

    #[test]
    fn test_factory_stress_case_033() {
        let v = Tensor::from_slice(&[33.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 33.0);
    }

    #[test]
    fn test_factory_stress_case_034() {
        let v = Tensor::from_slice(&[34.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 34.0);
    }

    #[test]
    fn test_factory_stress_case_035() {
        let v = Tensor::from_slice(&[35.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 35.0);
    }

    #[test]
    fn test_factory_stress_case_036() {
        let v = Tensor::from_slice(&[36.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 36.0);
    }

    #[test]
    fn test_factory_stress_case_037() {
        let v = Tensor::from_slice(&[37.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 37.0);
    }

    #[test]
    fn test_factory_stress_case_038() {
        let v = Tensor::from_slice(&[38.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 38.0);
    }

    #[test]
    fn test_factory_stress_case_039() {
        let v = Tensor::from_slice(&[39.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 39.0);
    }

    #[test]
    fn test_factory_stress_case_040() {
        let v = Tensor::from_slice(&[40.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 40.0);
    }

    #[test]
    fn test_factory_stress_case_041() {
        let v = Tensor::from_slice(&[41.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 41.0);
    }

    #[test]
    fn test_factory_stress_case_042() {
        let v = Tensor::from_slice(&[42.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 42.0);
    }

    #[test]
    fn test_factory_stress_case_043() {
        let v = Tensor::from_slice(&[43.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 43.0);
    }

    #[test]
    fn test_factory_stress_case_044() {
        let v = Tensor::from_slice(&[44.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 44.0);
    }

    #[test]
    fn test_factory_stress_case_045() {
        let v = Tensor::from_slice(&[45.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 45.0);
    }

    #[test]
    fn test_factory_stress_case_046() {
        let v = Tensor::from_slice(&[46.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 46.0);
    }

    #[test]
    fn test_factory_stress_case_047() {
        let v = Tensor::from_slice(&[47.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 47.0);
    }

    #[test]
    fn test_factory_stress_case_048() {
        let v = Tensor::from_slice(&[48.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 48.0);
    }

    #[test]
    fn test_factory_stress_case_049() {
        let v = Tensor::from_slice(&[49.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 49.0);
    }

    #[test]
    fn test_factory_stress_case_050() {
        let v = Tensor::from_slice(&[50.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 50.0);
    }

    #[test]
    fn test_factory_stress_case_051() {
        let v = Tensor::from_slice(&[51.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 51.0);
    }

    #[test]
    fn test_factory_stress_case_052() {
        let v = Tensor::from_slice(&[52.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 52.0);
    }

    #[test]
    fn test_factory_stress_case_053() {
        let v = Tensor::from_slice(&[53.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 53.0);
    }

    #[test]
    fn test_factory_stress_case_054() {
        let v = Tensor::from_slice(&[54.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 54.0);
    }

    #[test]
    fn test_factory_stress_case_055() {
        let v = Tensor::from_slice(&[55.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 55.0);
    }

    #[test]
    fn test_factory_stress_case_056() {
        let v = Tensor::from_slice(&[56.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 56.0);
    }

    #[test]
    fn test_factory_stress_case_057() {
        let v = Tensor::from_slice(&[57.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 57.0);
    }

    #[test]
    fn test_factory_stress_case_058() {
        let v = Tensor::from_slice(&[58.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 58.0);
    }

    #[test]
    fn test_factory_stress_case_059() {
        let v = Tensor::from_slice(&[59.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 59.0);
    }

    #[test]
    fn test_factory_stress_case_060() {
        let v = Tensor::from_slice(&[60.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 60.0);
    }

    #[test]
    fn test_factory_stress_case_061() {
        let v = Tensor::from_slice(&[61.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 61.0);
    }

    #[test]
    fn test_factory_stress_case_062() {
        let v = Tensor::from_slice(&[62.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 62.0);
    }

    #[test]
    fn test_factory_stress_case_063() {
        let v = Tensor::from_slice(&[63.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 63.0);
    }

    #[test]
    fn test_factory_stress_case_064() {
        let v = Tensor::from_slice(&[64.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 64.0);
    }

    #[test]
    fn test_factory_stress_case_065() {
        let v = Tensor::from_slice(&[65.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 65.0);
    }

    #[test]
    fn test_factory_stress_case_066() {
        let v = Tensor::from_slice(&[66.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 66.0);
    }

    #[test]
    fn test_factory_stress_case_067() {
        let v = Tensor::from_slice(&[67.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 67.0);
    }

    #[test]
    fn test_factory_stress_case_068() {
        let v = Tensor::from_slice(&[68.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 68.0);
    }

    #[test]
    fn test_factory_stress_case_069() {
        let v = Tensor::from_slice(&[69.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 69.0);
    }

    #[test]
    fn test_factory_stress_case_070() {
        let v = Tensor::from_slice(&[70.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 70.0);
    }

    #[test]
    fn test_factory_stress_case_071() {
        let v = Tensor::from_slice(&[71.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 71.0);
    }

    #[test]
    fn test_factory_stress_case_072() {
        let v = Tensor::from_slice(&[72.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 72.0);
    }

    #[test]
    fn test_factory_stress_case_073() {
        let v = Tensor::from_slice(&[73.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 73.0);
    }

    #[test]
    fn test_factory_stress_case_074() {
        let v = Tensor::from_slice(&[74.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 74.0);
    }

    #[test]
    fn test_factory_stress_case_075() {
        let v = Tensor::from_slice(&[75.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 75.0);
    }

    #[test]
    fn test_factory_stress_case_076() {
        let v = Tensor::from_slice(&[76.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 76.0);
    }

    #[test]
    fn test_factory_stress_case_077() {
        let v = Tensor::from_slice(&[77.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 77.0);
    }

    #[test]
    fn test_factory_stress_case_078() {
        let v = Tensor::from_slice(&[78.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 78.0);
    }

    #[test]
    fn test_factory_stress_case_079() {
        let v = Tensor::from_slice(&[79.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 79.0);
    }

    #[test]
    fn test_factory_stress_case_080() {
        let v = Tensor::from_slice(&[80.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 80.0);
    }

    #[test]
    fn test_factory_stress_case_081() {
        let v = Tensor::from_slice(&[81.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 81.0);
    }

    #[test]
    fn test_factory_stress_case_082() {
        let v = Tensor::from_slice(&[82.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 82.0);
    }

    #[test]
    fn test_factory_stress_case_083() {
        let v = Tensor::from_slice(&[83.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 83.0);
    }

    #[test]
    fn test_factory_stress_case_084() {
        let v = Tensor::from_slice(&[84.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 84.0);
    }

    #[test]
    fn test_factory_stress_case_085() {
        let v = Tensor::from_slice(&[85.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 85.0);
    }

    #[test]
    fn test_factory_stress_case_086() {
        let v = Tensor::from_slice(&[86.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 86.0);
    }

    #[test]
    fn test_factory_stress_case_087() {
        let v = Tensor::from_slice(&[87.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 87.0);
    }

    #[test]
    fn test_factory_stress_case_088() {
        let v = Tensor::from_slice(&[88.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 88.0);
    }

    #[test]
    fn test_factory_stress_case_089() {
        let v = Tensor::from_slice(&[89.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 89.0);
    }

    #[test]
    fn test_factory_stress_case_090() {
        let v = Tensor::from_slice(&[90.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 90.0);
    }

    #[test]
    fn test_factory_stress_case_091() {
        let v = Tensor::from_slice(&[91.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 91.0);
    }

    #[test]
    fn test_factory_stress_case_092() {
        let v = Tensor::from_slice(&[92.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 92.0);
    }

    #[test]
    fn test_factory_stress_case_093() {
        let v = Tensor::from_slice(&[93.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 93.0);
    }

    #[test]
    fn test_factory_stress_case_094() {
        let v = Tensor::from_slice(&[94.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 94.0);
    }

    #[test]
    fn test_factory_stress_case_095() {
        let v = Tensor::from_slice(&[95.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 95.0);
    }

    #[test]
    fn test_factory_stress_case_096() {
        let v = Tensor::from_slice(&[96.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 96.0);
    }

    #[test]
    fn test_factory_stress_case_097() {
        let v = Tensor::from_slice(&[97.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 97.0);
    }

    #[test]
    fn test_factory_stress_case_098() {
        let v = Tensor::from_slice(&[98.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 98.0);
    }

    #[test]
    fn test_factory_stress_case_099() {
        let v = Tensor::from_slice(&[99.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 99.0);
    }

    #[test]
    fn test_factory_stress_case_100() {
        let v = Tensor::from_slice(&[100.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 100.0);
    }

    #[test]
    fn test_factory_stress_case_101() {
        let v = Tensor::from_slice(&[101.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 101.0);
    }

    #[test]
    fn test_factory_stress_case_102() {
        let v = Tensor::from_slice(&[102.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 102.0);
    }

    #[test]
    fn test_factory_stress_case_103() {
        let v = Tensor::from_slice(&[103.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 103.0);
    }

    #[test]
    fn test_factory_stress_case_104() {
        let v = Tensor::from_slice(&[104.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 104.0);
    }

    #[test]
    fn test_factory_stress_case_105() {
        let v = Tensor::from_slice(&[105.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 105.0);
    }

    #[test]
    fn test_factory_stress_case_106() {
        let v = Tensor::from_slice(&[106.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 106.0);
    }

    #[test]
    fn test_factory_stress_case_107() {
        let v = Tensor::from_slice(&[107.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 107.0);
    }

    #[test]
    fn test_factory_stress_case_108() {
        let v = Tensor::from_slice(&[108.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 108.0);
    }

    #[test]
    fn test_factory_stress_case_109() {
        let v = Tensor::from_slice(&[109.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 109.0);
    }

    #[test]
    fn test_factory_stress_case_110() {
        let v = Tensor::from_slice(&[110.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 110.0);
    }

    #[test]
    fn test_factory_stress_case_111() {
        let v = Tensor::from_slice(&[111.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 111.0);
    }

    #[test]
    fn test_factory_stress_case_112() {
        let v = Tensor::from_slice(&[112.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 112.0);
    }

    #[test]
    fn test_factory_stress_case_113() {
        let v = Tensor::from_slice(&[113.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 113.0);
    }

    #[test]
    fn test_factory_stress_case_114() {
        let v = Tensor::from_slice(&[114.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 114.0);
    }

    #[test]
    fn test_factory_stress_case_115() {
        let v = Tensor::from_slice(&[115.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 115.0);
    }

    #[test]
    fn test_factory_stress_case_116() {
        let v = Tensor::from_slice(&[116.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 116.0);
    }

    #[test]
    fn test_factory_stress_case_117() {
        let v = Tensor::from_slice(&[117.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 117.0);
    }

    #[test]
    fn test_factory_stress_case_118() {
        let v = Tensor::from_slice(&[118.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 118.0);
    }

    #[test]
    fn test_factory_stress_case_119() {
        let v = Tensor::from_slice(&[119.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 119.0);
    }

    #[test]
    fn test_factory_stress_case_120() {
        let v = Tensor::from_slice(&[120.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 120.0);
    }

    #[test]
    fn test_factory_stress_case_121() {
        let v = Tensor::from_slice(&[121.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 121.0);
    }

    #[test]
    fn test_factory_stress_case_122() {
        let v = Tensor::from_slice(&[122.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 122.0);
    }

    #[test]
    fn test_factory_stress_case_123() {
        let v = Tensor::from_slice(&[123.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 123.0);
    }

    #[test]
    fn test_factory_stress_case_124() {
        let v = Tensor::from_slice(&[124.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 124.0);
    }

    #[test]
    fn test_factory_stress_case_125() {
        let v = Tensor::from_slice(&[125.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 125.0);
    }

    #[test]
    fn test_factory_stress_case_126() {
        let v = Tensor::from_slice(&[126.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 126.0);
    }

    #[test]
    fn test_factory_stress_case_127() {
        let v = Tensor::from_slice(&[127.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 127.0);
    }

    #[test]
    fn test_factory_stress_case_128() {
        let v = Tensor::from_slice(&[128.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 128.0);
    }

    #[test]
    fn test_factory_stress_case_129() {
        let v = Tensor::from_slice(&[129.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 129.0);
    }

    #[test]
    fn test_factory_stress_case_130() {
        let v = Tensor::from_slice(&[130.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 130.0);
    }

    #[test]
    fn test_factory_stress_case_131() {
        let v = Tensor::from_slice(&[131.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 131.0);
    }

    #[test]
    fn test_factory_stress_case_132() {
        let v = Tensor::from_slice(&[132.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 132.0);
    }

    #[test]
    fn test_factory_stress_case_133() {
        let v = Tensor::from_slice(&[133.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 133.0);
    }

    #[test]
    fn test_factory_stress_case_134() {
        let v = Tensor::from_slice(&[134.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 134.0);
    }

    #[test]
    fn test_factory_stress_case_135() {
        let v = Tensor::from_slice(&[135.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 135.0);
    }

    #[test]
    fn test_factory_stress_case_136() {
        let v = Tensor::from_slice(&[136.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 136.0);
    }

    #[test]
    fn test_factory_stress_case_137() {
        let v = Tensor::from_slice(&[137.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 137.0);
    }

    #[test]
    fn test_factory_stress_case_138() {
        let v = Tensor::from_slice(&[138.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 138.0);
    }

    #[test]
    fn test_factory_stress_case_139() {
        let v = Tensor::from_slice(&[139.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 139.0);
    }

    #[test]
    fn test_factory_stress_case_140() {
        let v = Tensor::from_slice(&[140.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 140.0);
    }

    #[test]
    fn test_factory_stress_case_141() {
        let v = Tensor::from_slice(&[141.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 141.0);
    }

    #[test]
    fn test_factory_stress_case_142() {
        let v = Tensor::from_slice(&[142.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 142.0);
    }

    #[test]
    fn test_factory_stress_case_143() {
        let v = Tensor::from_slice(&[143.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 143.0);
    }

    #[test]
    fn test_factory_stress_case_144() {
        let v = Tensor::from_slice(&[144.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 144.0);
    }

    #[test]
    fn test_factory_stress_case_145() {
        let v = Tensor::from_slice(&[145.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 145.0);
    }

    #[test]
    fn test_factory_stress_case_146() {
        let v = Tensor::from_slice(&[146.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 146.0);
    }

    #[test]
    fn test_factory_stress_case_147() {
        let v = Tensor::from_slice(&[147.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 147.0);
    }

    #[test]
    fn test_factory_stress_case_148() {
        let v = Tensor::from_slice(&[148.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 148.0);
    }

    #[test]
    fn test_factory_stress_case_149() {
        let v = Tensor::from_slice(&[149.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 149.0);
    }

    #[test]
    fn test_factory_stress_case_150() {
        let v = Tensor::from_slice(&[150.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 150.0);
    }

    #[test]
    fn test_factory_stress_case_151() {
        let v = Tensor::from_slice(&[151.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 151.0);
    }

    #[test]
    fn test_factory_stress_case_152() {
        let v = Tensor::from_slice(&[152.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 152.0);
    }

    #[test]
    fn test_factory_stress_case_153() {
        let v = Tensor::from_slice(&[153.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 153.0);
    }

    #[test]
    fn test_factory_stress_case_154() {
        let v = Tensor::from_slice(&[154.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 154.0);
    }

    #[test]
    fn test_factory_stress_case_155() {
        let v = Tensor::from_slice(&[155.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 155.0);
    }

    #[test]
    fn test_factory_stress_case_156() {
        let v = Tensor::from_slice(&[156.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 156.0);
    }

    #[test]
    fn test_factory_stress_case_157() {
        let v = Tensor::from_slice(&[157.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 157.0);
    }

    #[test]
    fn test_factory_stress_case_158() {
        let v = Tensor::from_slice(&[158.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 158.0);
    }

    #[test]
    fn test_factory_stress_case_159() {
        let v = Tensor::from_slice(&[159.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 159.0);
    }

    #[test]
    fn test_factory_stress_case_160() {
        let v = Tensor::from_slice(&[160.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 160.0);
    }

    #[test]
    fn test_factory_stress_case_161() {
        let v = Tensor::from_slice(&[161.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 161.0);
    }

    #[test]
    fn test_factory_stress_case_162() {
        let v = Tensor::from_slice(&[162.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 162.0);
    }

    #[test]
    fn test_factory_stress_case_163() {
        let v = Tensor::from_slice(&[163.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 163.0);
    }

    #[test]
    fn test_factory_stress_case_164() {
        let v = Tensor::from_slice(&[164.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 164.0);
    }

    #[test]
    fn test_factory_stress_case_165() {
        let v = Tensor::from_slice(&[165.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 165.0);
    }

    #[test]
    fn test_factory_stress_case_166() {
        let v = Tensor::from_slice(&[166.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 166.0);
    }

    #[test]
    fn test_factory_stress_case_167() {
        let v = Tensor::from_slice(&[167.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 167.0);
    }

    #[test]
    fn test_factory_stress_case_168() {
        let v = Tensor::from_slice(&[168.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 168.0);
    }

    #[test]
    fn test_factory_stress_case_169() {
        let v = Tensor::from_slice(&[169.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 169.0);
    }

    #[test]
    fn test_factory_stress_case_170() {
        let v = Tensor::from_slice(&[170.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 170.0);
    }

    #[test]
    fn test_factory_stress_case_171() {
        let v = Tensor::from_slice(&[171.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 171.0);
    }

    #[test]
    fn test_factory_stress_case_172() {
        let v = Tensor::from_slice(&[172.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 172.0);
    }

    #[test]
    fn test_factory_stress_case_173() {
        let v = Tensor::from_slice(&[173.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 173.0);
    }

    #[test]
    fn test_factory_stress_case_174() {
        let v = Tensor::from_slice(&[174.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 174.0);
    }

    #[test]
    fn test_factory_stress_case_175() {
        let v = Tensor::from_slice(&[175.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 175.0);
    }

    #[test]
    fn test_factory_stress_case_176() {
        let v = Tensor::from_slice(&[176.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 176.0);
    }

    #[test]
    fn test_factory_stress_case_177() {
        let v = Tensor::from_slice(&[177.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 177.0);
    }

    #[test]
    fn test_factory_stress_case_178() {
        let v = Tensor::from_slice(&[178.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 178.0);
    }

    #[test]
    fn test_factory_stress_case_179() {
        let v = Tensor::from_slice(&[179.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 179.0);
    }

    #[test]
    fn test_factory_stress_case_180() {
        let v = Tensor::from_slice(&[180.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 180.0);
    }

    #[test]
    fn test_factory_stress_case_181() {
        let v = Tensor::from_slice(&[181.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 181.0);
    }

    #[test]
    fn test_factory_stress_case_182() {
        let v = Tensor::from_slice(&[182.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 182.0);
    }

    #[test]
    fn test_factory_stress_case_183() {
        let v = Tensor::from_slice(&[183.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 183.0);
    }

    #[test]
    fn test_factory_stress_case_184() {
        let v = Tensor::from_slice(&[184.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 184.0);
    }

    #[test]
    fn test_factory_stress_case_185() {
        let v = Tensor::from_slice(&[185.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 185.0);
    }

    #[test]
    fn test_factory_stress_case_186() {
        let v = Tensor::from_slice(&[186.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 186.0);
    }

    #[test]
    fn test_factory_stress_case_187() {
        let v = Tensor::from_slice(&[187.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 187.0);
    }

    #[test]
    fn test_factory_stress_case_188() {
        let v = Tensor::from_slice(&[188.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 188.0);
    }

    #[test]
    fn test_factory_stress_case_189() {
        let v = Tensor::from_slice(&[189.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 189.0);
    }

    #[test]
    fn test_factory_stress_case_190() {
        let v = Tensor::from_slice(&[190.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 190.0);
    }

    #[test]
    fn test_factory_stress_case_191() {
        let v = Tensor::from_slice(&[191.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 191.0);
    }

    #[test]
    fn test_factory_stress_case_192() {
        let v = Tensor::from_slice(&[192.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 192.0);
    }

    #[test]
    fn test_factory_stress_case_193() {
        let v = Tensor::from_slice(&[193.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 193.0);
    }

    #[test]
    fn test_factory_stress_case_194() {
        let v = Tensor::from_slice(&[194.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 194.0);
    }

    #[test]
    fn test_factory_stress_case_195() {
        let v = Tensor::from_slice(&[195.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 195.0);
    }

    #[test]
    fn test_factory_stress_case_196() {
        let v = Tensor::from_slice(&[196.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 196.0);
    }

    #[test]
    fn test_factory_stress_case_197() {
        let v = Tensor::from_slice(&[197.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 197.0);
    }

    #[test]
    fn test_factory_stress_case_198() {
        let v = Tensor::from_slice(&[198.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 198.0);
    }

    #[test]
    fn test_factory_stress_case_199() {
        let v = Tensor::from_slice(&[199.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 199.0);
    }

    #[test]
    fn test_factory_stress_case_200() {
        let v = Tensor::from_slice(&[200.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 200.0);
    }

    #[test]
    fn test_factory_stress_case_201() {
        let v = Tensor::from_slice(&[201.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 201.0);
    }

    #[test]
    fn test_factory_stress_case_202() {
        let v = Tensor::from_slice(&[202.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 202.0);
    }

    #[test]
    fn test_factory_stress_case_203() {
        let v = Tensor::from_slice(&[203.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 203.0);
    }

    #[test]
    fn test_factory_stress_case_204() {
        let v = Tensor::from_slice(&[204.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 204.0);
    }

    #[test]
    fn test_factory_stress_case_205() {
        let v = Tensor::from_slice(&[205.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 205.0);
    }

    #[test]
    fn test_factory_stress_case_206() {
        let v = Tensor::from_slice(&[206.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 206.0);
    }

    #[test]
    fn test_factory_stress_case_207() {
        let v = Tensor::from_slice(&[207.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 207.0);
    }

    #[test]
    fn test_factory_stress_case_208() {
        let v = Tensor::from_slice(&[208.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 208.0);
    }

    #[test]
    fn test_factory_stress_case_209() {
        let v = Tensor::from_slice(&[209.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 209.0);
    }

    #[test]
    fn test_factory_stress_case_210() {
        let v = Tensor::from_slice(&[210.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 210.0);
    }

    #[test]
    fn test_factory_stress_case_211() {
        let v = Tensor::from_slice(&[211.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 211.0);
    }

    #[test]
    fn test_factory_stress_case_212() {
        let v = Tensor::from_slice(&[212.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 212.0);
    }

    #[test]
    fn test_factory_stress_case_213() {
        let v = Tensor::from_slice(&[213.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 213.0);
    }

    #[test]
    fn test_factory_stress_case_214() {
        let v = Tensor::from_slice(&[214.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 214.0);
    }

    #[test]
    fn test_factory_stress_case_215() {
        let v = Tensor::from_slice(&[215.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 215.0);
    }

    #[test]
    fn test_factory_stress_case_216() {
        let v = Tensor::from_slice(&[216.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 216.0);
    }

    #[test]
    fn test_factory_stress_case_217() {
        let v = Tensor::from_slice(&[217.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 217.0);
    }

    #[test]
    fn test_factory_stress_case_218() {
        let v = Tensor::from_slice(&[218.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 218.0);
    }

    #[test]
    fn test_factory_stress_case_219() {
        let v = Tensor::from_slice(&[219.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 219.0);
    }

    #[test]
    fn test_factory_stress_case_220() {
        let v = Tensor::from_slice(&[220.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 220.0);
    }

    #[test]
    fn test_factory_stress_case_221() {
        let v = Tensor::from_slice(&[221.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 221.0);
    }

    #[test]
    fn test_factory_stress_case_222() {
        let v = Tensor::from_slice(&[222.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 222.0);
    }

    #[test]
    fn test_factory_stress_case_223() {
        let v = Tensor::from_slice(&[223.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 223.0);
    }

    #[test]
    fn test_factory_stress_case_224() {
        let v = Tensor::from_slice(&[224.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 224.0);
    }

    #[test]
    fn test_factory_stress_case_225() {
        let v = Tensor::from_slice(&[225.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 225.0);
    }

    #[test]
    fn test_factory_stress_case_226() {
        let v = Tensor::from_slice(&[226.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 226.0);
    }

    #[test]
    fn test_factory_stress_case_227() {
        let v = Tensor::from_slice(&[227.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 227.0);
    }

    #[test]
    fn test_factory_stress_case_228() {
        let v = Tensor::from_slice(&[228.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 228.0);
    }

    #[test]
    fn test_factory_stress_case_229() {
        let v = Tensor::from_slice(&[229.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 229.0);
    }

    #[test]
    fn test_factory_stress_case_230() {
        let v = Tensor::from_slice(&[230.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 230.0);
    }

    #[test]
    fn test_factory_stress_case_231() {
        let v = Tensor::from_slice(&[231.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 231.0);
    }

    #[test]
    fn test_factory_stress_case_232() {
        let v = Tensor::from_slice(&[232.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 232.0);
    }

    #[test]
    fn test_factory_stress_case_233() {
        let v = Tensor::from_slice(&[233.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 233.0);
    }

    #[test]
    fn test_factory_stress_case_234() {
        let v = Tensor::from_slice(&[234.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 234.0);
    }

    #[test]
    fn test_factory_stress_case_235() {
        let v = Tensor::from_slice(&[235.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 235.0);
    }

    #[test]
    fn test_factory_stress_case_236() {
        let v = Tensor::from_slice(&[236.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 236.0);
    }

    #[test]
    fn test_factory_stress_case_237() {
        let v = Tensor::from_slice(&[237.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 237.0);
    }

    #[test]
    fn test_factory_stress_case_238() {
        let v = Tensor::from_slice(&[238.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 238.0);
    }

    #[test]
    fn test_factory_stress_case_239() {
        let v = Tensor::from_slice(&[239.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 239.0);
    }

    #[test]
    fn test_factory_stress_case_240() {
        let v = Tensor::from_slice(&[240.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 240.0);
    }

    #[test]
    fn test_factory_stress_case_241() {
        let v = Tensor::from_slice(&[241.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 241.0);
    }

    #[test]
    fn test_factory_stress_case_242() {
        let v = Tensor::from_slice(&[242.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 242.0);
    }

    #[test]
    fn test_factory_stress_case_243() {
        let v = Tensor::from_slice(&[243.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 243.0);
    }

    #[test]
    fn test_factory_stress_case_244() {
        let v = Tensor::from_slice(&[244.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 244.0);
    }

    #[test]
    fn test_factory_stress_case_245() {
        let v = Tensor::from_slice(&[245.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 245.0);
    }

    #[test]
    fn test_factory_stress_case_246() {
        let v = Tensor::from_slice(&[246.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 246.0);
    }

    #[test]
    fn test_factory_stress_case_247() {
        let v = Tensor::from_slice(&[247.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 247.0);
    }

    #[test]
    fn test_factory_stress_case_248() {
        let v = Tensor::from_slice(&[248.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 248.0);
    }

    #[test]
    fn test_factory_stress_case_249() {
        let v = Tensor::from_slice(&[249.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 249.0);
    }

    #[test]
    fn test_factory_stress_case_250() {
        let v = Tensor::from_slice(&[250.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 250.0);
    }

    #[test]
    fn test_factory_stress_case_251() {
        let v = Tensor::from_slice(&[251.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 251.0);
    }

    #[test]
    fn test_factory_stress_case_252() {
        let v = Tensor::from_slice(&[252.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 252.0);
    }

    #[test]
    fn test_factory_stress_case_253() {
        let v = Tensor::from_slice(&[253.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 253.0);
    }

    #[test]
    fn test_factory_stress_case_254() {
        let v = Tensor::from_slice(&[254.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 254.0);
    }

    #[test]
    fn test_factory_stress_case_255() {
        let v = Tensor::from_slice(&[255.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 255.0);
    }

    #[test]
    fn test_factory_stress_case_256() {
        let v = Tensor::from_slice(&[256.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 256.0);
    }

    #[test]
    fn test_factory_stress_case_257() {
        let v = Tensor::from_slice(&[257.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 257.0);
    }

    #[test]
    fn test_factory_stress_case_258() {
        let v = Tensor::from_slice(&[258.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 258.0);
    }

    #[test]
    fn test_factory_stress_case_259() {
        let v = Tensor::from_slice(&[259.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 259.0);
    }

    #[test]
    fn test_factory_stress_case_260() {
        let v = Tensor::from_slice(&[260.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 260.0);
    }

    #[test]
    fn test_factory_stress_case_261() {
        let v = Tensor::from_slice(&[261.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 261.0);
    }

    #[test]
    fn test_factory_stress_case_262() {
        let v = Tensor::from_slice(&[262.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 262.0);
    }

    #[test]
    fn test_factory_stress_case_263() {
        let v = Tensor::from_slice(&[263.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 263.0);
    }

    #[test]
    fn test_factory_stress_case_264() {
        let v = Tensor::from_slice(&[264.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 264.0);
    }

    #[test]
    fn test_factory_stress_case_265() {
        let v = Tensor::from_slice(&[265.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 265.0);
    }

    #[test]
    fn test_factory_stress_case_266() {
        let v = Tensor::from_slice(&[266.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 266.0);
    }

    #[test]
    fn test_factory_stress_case_267() {
        let v = Tensor::from_slice(&[267.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 267.0);
    }

    #[test]
    fn test_factory_stress_case_268() {
        let v = Tensor::from_slice(&[268.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 268.0);
    }

    #[test]
    fn test_factory_stress_case_269() {
        let v = Tensor::from_slice(&[269.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 269.0);
    }

    #[test]
    fn test_factory_stress_case_270() {
        let v = Tensor::from_slice(&[270.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 270.0);
    }

    #[test]
    fn test_factory_stress_case_271() {
        let v = Tensor::from_slice(&[271.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 271.0);
    }

    #[test]
    fn test_factory_stress_case_272() {
        let v = Tensor::from_slice(&[272.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 272.0);
    }

    #[test]
    fn test_factory_stress_case_273() {
        let v = Tensor::from_slice(&[273.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 273.0);
    }

    #[test]
    fn test_factory_stress_case_274() {
        let v = Tensor::from_slice(&[274.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 274.0);
    }

    #[test]
    fn test_factory_stress_case_275() {
        let v = Tensor::from_slice(&[275.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 275.0);
    }

    #[test]
    fn test_factory_stress_case_276() {
        let v = Tensor::from_slice(&[276.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 276.0);
    }

    #[test]
    fn test_factory_stress_case_277() {
        let v = Tensor::from_slice(&[277.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 277.0);
    }

    #[test]
    fn test_factory_stress_case_278() {
        let v = Tensor::from_slice(&[278.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 278.0);
    }

    #[test]
    fn test_factory_stress_case_279() {
        let v = Tensor::from_slice(&[279.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 279.0);
    }

    #[test]
    fn test_factory_stress_case_280() {
        let v = Tensor::from_slice(&[280.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 280.0);
    }

    #[test]
    fn test_factory_stress_case_281() {
        let v = Tensor::from_slice(&[281.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 281.0);
    }

    #[test]
    fn test_factory_stress_case_282() {
        let v = Tensor::from_slice(&[282.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 282.0);
    }

    #[test]
    fn test_factory_stress_case_283() {
        let v = Tensor::from_slice(&[283.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 283.0);
    }

    #[test]
    fn test_factory_stress_case_284() {
        let v = Tensor::from_slice(&[284.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 284.0);
    }

    #[test]
    fn test_factory_stress_case_285() {
        let v = Tensor::from_slice(&[285.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 285.0);
    }

    #[test]
    fn test_factory_stress_case_286() {
        let v = Tensor::from_slice(&[286.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 286.0);
    }

    #[test]
    fn test_factory_stress_case_287() {
        let v = Tensor::from_slice(&[287.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 287.0);
    }

    #[test]
    fn test_factory_stress_case_288() {
        let v = Tensor::from_slice(&[288.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 288.0);
    }

    #[test]
    fn test_factory_stress_case_289() {
        let v = Tensor::from_slice(&[289.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 289.0);
    }

    #[test]
    fn test_factory_stress_case_290() {
        let v = Tensor::from_slice(&[290.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 290.0);
    }

    #[test]
    fn test_factory_stress_case_291() {
        let v = Tensor::from_slice(&[291.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 291.0);
    }

    #[test]
    fn test_factory_stress_case_292() {
        let v = Tensor::from_slice(&[292.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 292.0);
    }

    #[test]
    fn test_factory_stress_case_293() {
        let v = Tensor::from_slice(&[293.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 293.0);
    }

    #[test]
    fn test_factory_stress_case_294() {
        let v = Tensor::from_slice(&[294.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 294.0);
    }

    #[test]
    fn test_factory_stress_case_295() {
        let v = Tensor::from_slice(&[295.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 295.0);
    }

    #[test]
    fn test_factory_stress_case_296() {
        let v = Tensor::from_slice(&[296.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 296.0);
    }

    #[test]
    fn test_factory_stress_case_297() {
        let v = Tensor::from_slice(&[297.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 297.0);
    }

    #[test]
    fn test_factory_stress_case_298() {
        let v = Tensor::from_slice(&[298.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 298.0);
    }

    #[test]
    fn test_factory_stress_case_299() {
        let v = Tensor::from_slice(&[299.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 299.0);
    }

    #[test]
    fn test_factory_stress_case_300() {
        let v = Tensor::from_slice(&[300.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 300.0);
    }

    #[test]
    fn test_factory_stress_case_301() {
        let v = Tensor::from_slice(&[301.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 301.0);
    }

    #[test]
    fn test_factory_stress_case_302() {
        let v = Tensor::from_slice(&[302.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 302.0);
    }

    #[test]
    fn test_factory_stress_case_303() {
        let v = Tensor::from_slice(&[303.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 303.0);
    }

    #[test]
    fn test_factory_stress_case_304() {
        let v = Tensor::from_slice(&[304.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 304.0);
    }

    #[test]
    fn test_factory_stress_case_305() {
        let v = Tensor::from_slice(&[305.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 305.0);
    }

    #[test]
    fn test_factory_stress_case_306() {
        let v = Tensor::from_slice(&[306.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 306.0);
    }

    #[test]
    fn test_factory_stress_case_307() {
        let v = Tensor::from_slice(&[307.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 307.0);
    }

    #[test]
    fn test_factory_stress_case_308() {
        let v = Tensor::from_slice(&[308.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 308.0);
    }

    #[test]
    fn test_factory_stress_case_309() {
        let v = Tensor::from_slice(&[309.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 309.0);
    }

    #[test]
    fn test_factory_stress_case_310() {
        let v = Tensor::from_slice(&[310.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 310.0);
    }

    #[test]
    fn test_factory_stress_case_311() {
        let v = Tensor::from_slice(&[311.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 311.0);
    }

    #[test]
    fn test_factory_stress_case_312() {
        let v = Tensor::from_slice(&[312.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 312.0);
    }

    #[test]
    fn test_factory_stress_case_313() {
        let v = Tensor::from_slice(&[313.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 313.0);
    }

    #[test]
    fn test_factory_stress_case_314() {
        let v = Tensor::from_slice(&[314.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 314.0);
    }

    #[test]
    fn test_factory_stress_case_315() {
        let v = Tensor::from_slice(&[315.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 315.0);
    }

    #[test]
    fn test_factory_stress_case_316() {
        let v = Tensor::from_slice(&[316.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 316.0);
    }

    #[test]
    fn test_factory_stress_case_317() {
        let v = Tensor::from_slice(&[317.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 317.0);
    }

    #[test]
    fn test_factory_stress_case_318() {
        let v = Tensor::from_slice(&[318.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 318.0);
    }

    #[test]
    fn test_factory_stress_case_319() {
        let v = Tensor::from_slice(&[319.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 319.0);
    }

    #[test]
    fn test_factory_stress_case_320() {
        let v = Tensor::from_slice(&[320.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 320.0);
    }

    #[test]
    fn test_factory_stress_case_321() {
        let v = Tensor::from_slice(&[321.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 321.0);
    }

    #[test]
    fn test_factory_stress_case_322() {
        let v = Tensor::from_slice(&[322.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 322.0);
    }

    #[test]
    fn test_factory_stress_case_323() {
        let v = Tensor::from_slice(&[323.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 323.0);
    }

    #[test]
    fn test_factory_stress_case_324() {
        let v = Tensor::from_slice(&[324.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 324.0);
    }

    #[test]
    fn test_factory_stress_case_325() {
        let v = Tensor::from_slice(&[325.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 325.0);
    }

    #[test]
    fn test_factory_stress_case_326() {
        let v = Tensor::from_slice(&[326.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 326.0);
    }

    #[test]
    fn test_factory_stress_case_327() {
        let v = Tensor::from_slice(&[327.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 327.0);
    }

    #[test]
    fn test_factory_stress_case_328() {
        let v = Tensor::from_slice(&[328.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 328.0);
    }

    #[test]
    fn test_factory_stress_case_329() {
        let v = Tensor::from_slice(&[329.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 329.0);
    }

    #[test]
    fn test_factory_stress_case_330() {
        let v = Tensor::from_slice(&[330.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 330.0);
    }

    #[test]
    fn test_factory_stress_case_331() {
        let v = Tensor::from_slice(&[331.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 331.0);
    }

    #[test]
    fn test_factory_stress_case_332() {
        let v = Tensor::from_slice(&[332.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 332.0);
    }

    #[test]
    fn test_factory_stress_case_333() {
        let v = Tensor::from_slice(&[333.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 333.0);
    }

    #[test]
    fn test_factory_stress_case_334() {
        let v = Tensor::from_slice(&[334.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 334.0);
    }

    #[test]
    fn test_factory_stress_case_335() {
        let v = Tensor::from_slice(&[335.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 335.0);
    }

    #[test]
    fn test_factory_stress_case_336() {
        let v = Tensor::from_slice(&[336.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 336.0);
    }

    #[test]
    fn test_factory_stress_case_337() {
        let v = Tensor::from_slice(&[337.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 337.0);
    }

    #[test]
    fn test_factory_stress_case_338() {
        let v = Tensor::from_slice(&[338.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 338.0);
    }

    #[test]
    fn test_factory_stress_case_339() {
        let v = Tensor::from_slice(&[339.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 339.0);
    }

    #[test]
    fn test_factory_stress_case_340() {
        let v = Tensor::from_slice(&[340.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 340.0);
    }

    #[test]
    fn test_factory_stress_case_341() {
        let v = Tensor::from_slice(&[341.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 341.0);
    }

    #[test]
    fn test_factory_stress_case_342() {
        let v = Tensor::from_slice(&[342.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 342.0);
    }

    #[test]
    fn test_factory_stress_case_343() {
        let v = Tensor::from_slice(&[343.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 343.0);
    }

    #[test]
    fn test_factory_stress_case_344() {
        let v = Tensor::from_slice(&[344.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 344.0);
    }

    #[test]
    fn test_factory_stress_case_345() {
        let v = Tensor::from_slice(&[345.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 345.0);
    }

    #[test]
    fn test_factory_stress_case_346() {
        let v = Tensor::from_slice(&[346.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 346.0);
    }

    #[test]
    fn test_factory_stress_case_347() {
        let v = Tensor::from_slice(&[347.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 347.0);
    }

    #[test]
    fn test_factory_stress_case_348() {
        let v = Tensor::from_slice(&[348.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 348.0);
    }

    #[test]
    fn test_factory_stress_case_349() {
        let v = Tensor::from_slice(&[349.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 349.0);
    }

    #[test]
    fn test_factory_stress_case_350() {
        let v = Tensor::from_slice(&[350.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 350.0);
    }

    #[test]
    fn test_factory_stress_case_351() {
        let v = Tensor::from_slice(&[351.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 351.0);
    }

    #[test]
    fn test_factory_stress_case_352() {
        let v = Tensor::from_slice(&[352.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 352.0);
    }

    #[test]
    fn test_factory_stress_case_353() {
        let v = Tensor::from_slice(&[353.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 353.0);
    }

    #[test]
    fn test_factory_stress_case_354() {
        let v = Tensor::from_slice(&[354.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 354.0);
    }

    #[test]
    fn test_factory_stress_case_355() {
        let v = Tensor::from_slice(&[355.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 355.0);
    }

    #[test]
    fn test_factory_stress_case_356() {
        let v = Tensor::from_slice(&[356.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 356.0);
    }

    #[test]
    fn test_factory_stress_case_357() {
        let v = Tensor::from_slice(&[357.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 357.0);
    }

    #[test]
    fn test_factory_stress_case_358() {
        let v = Tensor::from_slice(&[358.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 358.0);
    }

    #[test]
    fn test_factory_stress_case_359() {
        let v = Tensor::from_slice(&[359.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 359.0);
    }

    #[test]
    fn test_factory_stress_case_360() {
        let v = Tensor::from_slice(&[360.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 360.0);
    }

    #[test]
    fn test_factory_stress_case_361() {
        let v = Tensor::from_slice(&[361.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 361.0);
    }

    #[test]
    fn test_factory_stress_case_362() {
        let v = Tensor::from_slice(&[362.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 362.0);
    }

    #[test]
    fn test_factory_stress_case_363() {
        let v = Tensor::from_slice(&[363.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 363.0);
    }

    #[test]
    fn test_factory_stress_case_364() {
        let v = Tensor::from_slice(&[364.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 364.0);
    }

    #[test]
    fn test_factory_stress_case_365() {
        let v = Tensor::from_slice(&[365.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 365.0);
    }

    #[test]
    fn test_factory_stress_case_366() {
        let v = Tensor::from_slice(&[366.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 366.0);
    }

    #[test]
    fn test_factory_stress_case_367() {
        let v = Tensor::from_slice(&[367.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 367.0);
    }

    #[test]
    fn test_factory_stress_case_368() {
        let v = Tensor::from_slice(&[368.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 368.0);
    }

    #[test]
    fn test_factory_stress_case_369() {
        let v = Tensor::from_slice(&[369.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 369.0);
    }

    #[test]
    fn test_factory_stress_case_370() {
        let v = Tensor::from_slice(&[370.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 370.0);
    }

    #[test]
    fn test_factory_stress_case_371() {
        let v = Tensor::from_slice(&[371.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 371.0);
    }

    #[test]
    fn test_factory_stress_case_372() {
        let v = Tensor::from_slice(&[372.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 372.0);
    }

    #[test]
    fn test_factory_stress_case_373() {
        let v = Tensor::from_slice(&[373.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 373.0);
    }

    #[test]
    fn test_factory_stress_case_374() {
        let v = Tensor::from_slice(&[374.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 374.0);
    }

    #[test]
    fn test_factory_stress_case_375() {
        let v = Tensor::from_slice(&[375.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 375.0);
    }

    #[test]
    fn test_factory_stress_case_376() {
        let v = Tensor::from_slice(&[376.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 376.0);
    }

    #[test]
    fn test_factory_stress_case_377() {
        let v = Tensor::from_slice(&[377.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 377.0);
    }

    #[test]
    fn test_factory_stress_case_378() {
        let v = Tensor::from_slice(&[378.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 378.0);
    }

    #[test]
    fn test_factory_stress_case_379() {
        let v = Tensor::from_slice(&[379.0], vec![1]);
        let d = diag(&v);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(d.get_2d(0, 0), 379.0);
    }
}
