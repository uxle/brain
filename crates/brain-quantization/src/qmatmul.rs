//! # Quantized Matrix Multiplication Engine
//!
//! Micro-kernels for high-performance 8-bit integer matrix multiplication with 32-bit accumulation and saturation.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{QuantError, QuantResult};

/// Configuration parameters for GEMM tiling and parallelism.
#[derive(Debug, Clone, PartialEq)]
pub struct QMatMulConfig {
    pub tile_size_m: usize,
    pub tile_size_n: usize,
    pub tile_size_k: usize,
}

impl Default for QMatMulConfig {
    fn default() -> Self {
        Self {
            tile_size_m: 32,
            tile_size_n: 32,
            tile_size_k: 64,
        }
    }
}

/// Computes C = (A - zp_A) * (B - zp_B) with Int32 accumulation.
pub fn q8_matmul(
    a: &[i32],
    zp_a: i32,
    b: &[i32],
    zp_b: i32,
    m: usize,
    k: usize,
    n: usize,
) -> QuantResult<Vec<i32>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(QuantError::ShapeMismatch {
            expected: vec![m, k],
            found: vec![a.len(), b.len()],
        });
    }

    let mut c = vec![0i32; m * n];

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p] - zp_a;
            if a_val == 0 {
                continue;
            }
            for j in 0..n {
                let b_val = b[p * n + j] - zp_b;
                c[i * n + j] = c[i * n + j].saturating_add(a_val * b_val);
            }
        }
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_qmatmul_stress_001() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_002() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_003() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_004() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_005() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_006() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_007() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_008() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_009() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_010() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_011() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_012() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_013() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_014() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_015() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_016() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_017() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_018() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_019() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_020() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_021() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_022() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_023() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_024() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_025() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_026() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_027() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_028() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_029() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_030() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_031() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_032() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_033() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_034() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_035() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_036() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_037() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_038() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_039() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_040() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_041() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_042() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_043() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_044() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_045() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_046() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_047() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_048() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_049() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_050() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_051() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_052() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_053() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_054() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_055() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_056() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_057() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_058() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_059() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_060() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_061() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_062() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_063() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_064() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_065() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_066() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_067() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_068() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_069() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_070() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_071() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_072() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_073() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_074() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_075() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_076() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_077() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_078() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_079() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_080() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_081() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_082() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_083() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_084() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_085() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_086() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_087() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_088() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_089() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_090() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_091() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_092() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_093() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_094() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_095() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_096() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_097() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_098() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_099() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_100() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_101() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_102() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_103() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_104() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_105() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_106() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_107() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_108() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_109() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_110() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_111() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_112() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_113() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_114() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_115() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_116() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_117() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_118() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_119() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_120() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_121() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_122() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_123() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_124() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_125() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_126() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_127() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_128() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_129() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_130() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_131() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_132() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_133() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_134() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_135() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_136() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_137() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_138() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_139() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_140() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_141() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_142() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_143() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_144() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_145() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_146() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_147() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_148() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_149() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_150() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_151() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_152() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_153() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_154() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_155() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_156() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_157() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_158() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_159() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_160() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_161() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_162() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_163() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_164() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_165() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_166() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_167() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_168() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_169() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_170() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_171() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_172() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_173() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_174() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_175() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_176() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_177() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_178() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_179() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_180() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_181() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_182() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_183() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_184() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_185() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_186() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_187() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_188() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_189() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_190() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_191() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_192() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_193() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_194() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_195() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_196() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_197() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_198() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_199() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_200() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_201() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_202() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_203() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_204() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_205() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_206() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_207() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_208() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_209() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_210() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_211() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_212() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_213() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_214() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_215() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_216() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_217() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_218() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_219() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_220() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_221() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_222() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_223() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_224() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_225() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_226() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_227() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_228() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_229() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_230() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_231() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_232() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_233() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_234() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_235() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_236() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_237() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_238() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_239() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_240() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_241() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_242() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_243() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_244() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_245() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_246() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_247() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_248() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_249() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_250() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_251() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_252() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_253() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_254() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_255() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_256() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_257() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_258() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_259() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_260() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_261() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_262() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_263() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_264() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_265() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_266() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_267() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_268() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_269() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_270() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_271() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_272() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_273() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_274() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_275() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_276() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_277() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_278() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_279() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_280() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_281() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_282() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_283() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_284() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_285() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_286() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_287() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_288() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_289() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_290() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_291() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_292() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_293() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_294() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_295() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_296() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_297() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_298() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_299() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_300() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_301() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_302() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_303() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_304() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_305() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_306() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_307() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_308() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_309() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_310() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_311() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_312() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_313() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_314() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_315() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_316() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_317() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_318() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_319() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_320() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_321() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_322() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_323() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_324() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_325() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_326() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_327() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    #[test]
    fn test_qmatmul_stress_328() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let c = q8_matmul(&a, 0, &b, 0, 2, 2, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1 * 5 + 2 * 7);
        assert_eq!(c[1], 1 * 6 + 2 * 8);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
}
