//! # Basic Uniform, Normal & Orthogonal Initializers
//!
//! Standard statistical distributions and orthogonal matrix initialization.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Initialization scheme descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitScheme {
    #[default]
    Uniform,
    Normal,
    Kaiming,
    Xavier,
    Orthogonal,
    Zeros,
    Ones,
}

/// Generates a Tensor filled with uniform random values in [min_val, max_val].
pub fn uniform_init(shape: &[usize], min_val: f64, max_val: f64) -> Tensor {
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        let norm = ((i * 1103515245 + 12345) % 65536) as f64 / 65536.0;
        data.push(min_val + norm * (max_val - min_val));
    }
    Tensor::from_vec(data, shape.to_vec())
}

/// Generates a Tensor filled with normal random values N(mean, std^2).
pub fn normal_init(shape: &[usize], mean: f64, std: f64) -> Tensor {
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        let u1 = (((i * 1664525 + 1013904223) % 65536) as f64 / 65536.0).max(1e-12);
        let u2 = ((i * 22695477 + 1) % 65536) as f64 / 65536.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push(mean + z * std);
    }
    Tensor::from_vec(data, shape.to_vec())
}

/// Generates an orthogonal 2D matrix using Gram-Schmidt orthogonalization.
pub fn orthogonal_init(rows: usize, cols: usize, gain: f64) -> Tensor {
    let mut mat = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            let u1 = (((i * 1664525 + 1013904223) % 65536) as f64 / 65536.0).max(1e-12);
            let u2 = ((i * 22695477 + 1) % 65536) as f64 / 65536.0;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mat[r][c] = z;
        }
    }

    if rows <= cols {
        for r in 0..rows {
            for prev_r in 0..r {
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += mat[r][c] * mat[prev_r][c];
                }
                for c in 0..cols {
                    mat[r][c] -= dot * mat[prev_r][c];
                }
            }
            let mut norm_sq = 0.0;
            for c in 0..cols {
                norm_sq += mat[r][c] * mat[r][c];
            }
            let norm = norm_sq.sqrt().max(1e-12);
            for c in 0..cols {
                mat[r][c] = (mat[r][c] / norm) * gain;
            }
        }
    } else {
        for c in 0..cols {
            for prev_c in 0..c {
                let mut dot = 0.0;
                for r in 0..rows {
                    dot += mat[r][c] * mat[r][prev_c];
                }
                for r in 0..rows {
                    mat[r][c] -= dot * mat[r][prev_c];
                }
            }
            let mut norm_sq = 0.0;
            for r in 0..rows {
                norm_sq += mat[r][c] * mat[r][c];
            }
            let norm = norm_sq.sqrt().max(1e-12);
            for r in 0..rows {
                mat[r][c] = (mat[r][c] / norm) * gain;
            }
        }
    }

    let mut flat = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            flat.push(mat[r][c]);
        }
    }
    Tensor::from_vec(flat, vec![rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonal_init_is_orthogonal() {
        let q = orthogonal_init(32, 32, 1.0);
        let q_t = q.transpose(0, 1);
        let prod = brain_core::tensor::arithmetic::matmul(&q, &q_t);
        let eye = Tensor::eye(32);
        for r in 0..32 {
            for c in 0..32 {
                let diff = (prod.get_2d(r, c) - eye.get_2d(r, c)).abs();
                assert!(
                    diff < 1e-5,
                    "Orthogonality failed at ({}, {}): diff={}",
                    r,
                    c,
                    diff
                );
            }
        }
    }
}
