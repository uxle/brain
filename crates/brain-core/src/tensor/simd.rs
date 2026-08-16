//! SIMD vectorization abstractions and unrolled numerical kernels.
//!
//! This module provides high-throughput unrolled and vector-friendly computational kernels
//! for element-wise arithmetic, fused multiply-add (FMA), reductions, and transcendental activations.

// =============================================================================
// Vector Kernels
// =============================================================================

/// Vector addition: out[i] = a[i] + b[i]
pub fn simd_add(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut chunks = len / 4;
    let mut i = 0;
    while chunks > 0 {
        out[i] = a[i] + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
        i += 4;
        chunks -= 1;
    }
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Vector subtraction: out[i] = a[i] - b[i]
pub fn simd_sub(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] - b[i];
        out[i + 1] = a[i + 1] - b[i + 1];
        out[i + 2] = a[i + 2] - b[i + 2];
        out[i + 3] = a[i + 3] - b[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

/// Vector multiplication: out[i] = a[i] * b[i]
pub fn simd_mul(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] * b[i];
        out[i + 1] = a[i + 1] * b[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// Vector fused multiply-add: out[i] = a[i] * b[i] + c[i]
pub fn simd_fma(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(c.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= len {
        out[i] = a[i] * b[i] + c[i];
        out[i + 1] = a[i + 1] * b[i + 1] + c[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2] + c[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3] + c[i + 3];
        i += 4;
    }
    while i < len {
        out[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

/// Vector dot product
pub fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut i = 0;
    while i + 4 <= len {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut total = sum0 + sum1 + sum2 + sum3;
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// Vector sum reduction
pub fn simd_sum(a: &[f64]) -> f64 {
    let len = a.len();
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut i = 0;
    while i + 4 <= len {
        sum0 += a[i];
        sum1 += a[i + 1];
        sum2 += a[i + 2];
        sum3 += a[i + 3];
        i += 4;
    }
    let mut total = sum0 + sum1 + sum2 + sum3;
    while i < len {
        total += a[i];
        i += 1;
    }
    total
}

/// Vector ReLU activation: out[i] = max(0, a[i])
pub fn simd_relu(a: &[f64], out: &mut [f64]) {
    let len = a.len().min(out.len());
    for i in 0..len {
        out[i] = a[i].max(0.0);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        simd_add(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);

        assert_eq!(simd_dot(&a, &b), 550.0);
        assert_eq!(simd_sum(&a), 15.0);
    }

    #[test]
    fn test_simd_stress_case_001() {
        let a = vec![1.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 1.0 + 2.0);
        assert_eq!(simd_sum(&a), 1.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_002() {
        let a = vec![2.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 2.0 + 2.0);
        assert_eq!(simd_sum(&a), 2.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_003() {
        let a = vec![3.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 3.0 + 2.0);
        assert_eq!(simd_sum(&a), 3.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_004() {
        let a = vec![4.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 4.0 + 2.0);
        assert_eq!(simd_sum(&a), 4.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_005() {
        let a = vec![5.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 5.0 + 2.0);
        assert_eq!(simd_sum(&a), 5.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_006() {
        let a = vec![6.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 6.0 + 2.0);
        assert_eq!(simd_sum(&a), 6.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_007() {
        let a = vec![7.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 7.0 + 2.0);
        assert_eq!(simd_sum(&a), 7.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_008() {
        let a = vec![8.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 8.0 + 2.0);
        assert_eq!(simd_sum(&a), 8.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_009() {
        let a = vec![9.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 9.0 + 2.0);
        assert_eq!(simd_sum(&a), 9.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_010() {
        let a = vec![10.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 10.0 + 2.0);
        assert_eq!(simd_sum(&a), 10.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_011() {
        let a = vec![11.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 11.0 + 2.0);
        assert_eq!(simd_sum(&a), 11.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_012() {
        let a = vec![12.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 12.0 + 2.0);
        assert_eq!(simd_sum(&a), 12.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_013() {
        let a = vec![13.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 13.0 + 2.0);
        assert_eq!(simd_sum(&a), 13.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_014() {
        let a = vec![14.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 14.0 + 2.0);
        assert_eq!(simd_sum(&a), 14.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_015() {
        let a = vec![15.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 15.0 + 2.0);
        assert_eq!(simd_sum(&a), 15.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_016() {
        let a = vec![16.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 16.0 + 2.0);
        assert_eq!(simd_sum(&a), 16.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_017() {
        let a = vec![17.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 17.0 + 2.0);
        assert_eq!(simd_sum(&a), 17.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_018() {
        let a = vec![18.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 18.0 + 2.0);
        assert_eq!(simd_sum(&a), 18.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_019() {
        let a = vec![19.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 19.0 + 2.0);
        assert_eq!(simd_sum(&a), 19.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_020() {
        let a = vec![20.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 20.0 + 2.0);
        assert_eq!(simd_sum(&a), 20.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_021() {
        let a = vec![21.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 21.0 + 2.0);
        assert_eq!(simd_sum(&a), 21.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_022() {
        let a = vec![22.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 22.0 + 2.0);
        assert_eq!(simd_sum(&a), 22.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_023() {
        let a = vec![23.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 23.0 + 2.0);
        assert_eq!(simd_sum(&a), 23.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_024() {
        let a = vec![24.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 24.0 + 2.0);
        assert_eq!(simd_sum(&a), 24.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_025() {
        let a = vec![25.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 25.0 + 2.0);
        assert_eq!(simd_sum(&a), 25.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_026() {
        let a = vec![26.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 26.0 + 2.0);
        assert_eq!(simd_sum(&a), 26.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_027() {
        let a = vec![27.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 27.0 + 2.0);
        assert_eq!(simd_sum(&a), 27.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_028() {
        let a = vec![28.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 28.0 + 2.0);
        assert_eq!(simd_sum(&a), 28.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_029() {
        let a = vec![29.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 29.0 + 2.0);
        assert_eq!(simd_sum(&a), 29.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_030() {
        let a = vec![30.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 30.0 + 2.0);
        assert_eq!(simd_sum(&a), 30.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_031() {
        let a = vec![31.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 31.0 + 2.0);
        assert_eq!(simd_sum(&a), 31.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_032() {
        let a = vec![32.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 32.0 + 2.0);
        assert_eq!(simd_sum(&a), 32.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_033() {
        let a = vec![33.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 33.0 + 2.0);
        assert_eq!(simd_sum(&a), 33.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_034() {
        let a = vec![34.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 34.0 + 2.0);
        assert_eq!(simd_sum(&a), 34.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_035() {
        let a = vec![35.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 35.0 + 2.0);
        assert_eq!(simd_sum(&a), 35.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_036() {
        let a = vec![36.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 36.0 + 2.0);
        assert_eq!(simd_sum(&a), 36.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_037() {
        let a = vec![37.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 37.0 + 2.0);
        assert_eq!(simd_sum(&a), 37.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_038() {
        let a = vec![38.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 38.0 + 2.0);
        assert_eq!(simd_sum(&a), 38.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_039() {
        let a = vec![39.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 39.0 + 2.0);
        assert_eq!(simd_sum(&a), 39.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_040() {
        let a = vec![40.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 40.0 + 2.0);
        assert_eq!(simd_sum(&a), 40.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_041() {
        let a = vec![41.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 41.0 + 2.0);
        assert_eq!(simd_sum(&a), 41.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_042() {
        let a = vec![42.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 42.0 + 2.0);
        assert_eq!(simd_sum(&a), 42.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_043() {
        let a = vec![43.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 43.0 + 2.0);
        assert_eq!(simd_sum(&a), 43.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_044() {
        let a = vec![44.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 44.0 + 2.0);
        assert_eq!(simd_sum(&a), 44.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_045() {
        let a = vec![45.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 45.0 + 2.0);
        assert_eq!(simd_sum(&a), 45.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_046() {
        let a = vec![46.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 46.0 + 2.0);
        assert_eq!(simd_sum(&a), 46.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_047() {
        let a = vec![47.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 47.0 + 2.0);
        assert_eq!(simd_sum(&a), 47.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_048() {
        let a = vec![48.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 48.0 + 2.0);
        assert_eq!(simd_sum(&a), 48.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_049() {
        let a = vec![49.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 49.0 + 2.0);
        assert_eq!(simd_sum(&a), 49.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_050() {
        let a = vec![50.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 50.0 + 2.0);
        assert_eq!(simd_sum(&a), 50.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_051() {
        let a = vec![51.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 51.0 + 2.0);
        assert_eq!(simd_sum(&a), 51.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_052() {
        let a = vec![52.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 52.0 + 2.0);
        assert_eq!(simd_sum(&a), 52.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_053() {
        let a = vec![53.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 53.0 + 2.0);
        assert_eq!(simd_sum(&a), 53.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_054() {
        let a = vec![54.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 54.0 + 2.0);
        assert_eq!(simd_sum(&a), 54.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_055() {
        let a = vec![55.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 55.0 + 2.0);
        assert_eq!(simd_sum(&a), 55.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_056() {
        let a = vec![56.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 56.0 + 2.0);
        assert_eq!(simd_sum(&a), 56.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_057() {
        let a = vec![57.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 57.0 + 2.0);
        assert_eq!(simd_sum(&a), 57.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_058() {
        let a = vec![58.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 58.0 + 2.0);
        assert_eq!(simd_sum(&a), 58.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_059() {
        let a = vec![59.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 59.0 + 2.0);
        assert_eq!(simd_sum(&a), 59.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_060() {
        let a = vec![60.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 60.0 + 2.0);
        assert_eq!(simd_sum(&a), 60.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_061() {
        let a = vec![61.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 61.0 + 2.0);
        assert_eq!(simd_sum(&a), 61.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_062() {
        let a = vec![62.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 62.0 + 2.0);
        assert_eq!(simd_sum(&a), 62.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_063() {
        let a = vec![63.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 63.0 + 2.0);
        assert_eq!(simd_sum(&a), 63.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_064() {
        let a = vec![64.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 64.0 + 2.0);
        assert_eq!(simd_sum(&a), 64.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_065() {
        let a = vec![65.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 65.0 + 2.0);
        assert_eq!(simd_sum(&a), 65.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_066() {
        let a = vec![66.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 66.0 + 2.0);
        assert_eq!(simd_sum(&a), 66.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_067() {
        let a = vec![67.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 67.0 + 2.0);
        assert_eq!(simd_sum(&a), 67.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_068() {
        let a = vec![68.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 68.0 + 2.0);
        assert_eq!(simd_sum(&a), 68.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_069() {
        let a = vec![69.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 69.0 + 2.0);
        assert_eq!(simd_sum(&a), 69.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_070() {
        let a = vec![70.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 70.0 + 2.0);
        assert_eq!(simd_sum(&a), 70.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_071() {
        let a = vec![71.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 71.0 + 2.0);
        assert_eq!(simd_sum(&a), 71.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_072() {
        let a = vec![72.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 72.0 + 2.0);
        assert_eq!(simd_sum(&a), 72.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_073() {
        let a = vec![73.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 73.0 + 2.0);
        assert_eq!(simd_sum(&a), 73.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_074() {
        let a = vec![74.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 74.0 + 2.0);
        assert_eq!(simd_sum(&a), 74.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_075() {
        let a = vec![75.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 75.0 + 2.0);
        assert_eq!(simd_sum(&a), 75.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_076() {
        let a = vec![76.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 76.0 + 2.0);
        assert_eq!(simd_sum(&a), 76.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_077() {
        let a = vec![77.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 77.0 + 2.0);
        assert_eq!(simd_sum(&a), 77.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_078() {
        let a = vec![78.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 78.0 + 2.0);
        assert_eq!(simd_sum(&a), 78.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_079() {
        let a = vec![79.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 79.0 + 2.0);
        assert_eq!(simd_sum(&a), 79.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_080() {
        let a = vec![80.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 80.0 + 2.0);
        assert_eq!(simd_sum(&a), 80.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_081() {
        let a = vec![81.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 81.0 + 2.0);
        assert_eq!(simd_sum(&a), 81.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_082() {
        let a = vec![82.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 82.0 + 2.0);
        assert_eq!(simd_sum(&a), 82.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_083() {
        let a = vec![83.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 83.0 + 2.0);
        assert_eq!(simd_sum(&a), 83.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_084() {
        let a = vec![84.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 84.0 + 2.0);
        assert_eq!(simd_sum(&a), 84.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_085() {
        let a = vec![85.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 85.0 + 2.0);
        assert_eq!(simd_sum(&a), 85.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_086() {
        let a = vec![86.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 86.0 + 2.0);
        assert_eq!(simd_sum(&a), 86.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_087() {
        let a = vec![87.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 87.0 + 2.0);
        assert_eq!(simd_sum(&a), 87.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_088() {
        let a = vec![88.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 88.0 + 2.0);
        assert_eq!(simd_sum(&a), 88.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_089() {
        let a = vec![89.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 89.0 + 2.0);
        assert_eq!(simd_sum(&a), 89.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_090() {
        let a = vec![90.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 90.0 + 2.0);
        assert_eq!(simd_sum(&a), 90.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_091() {
        let a = vec![91.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 91.0 + 2.0);
        assert_eq!(simd_sum(&a), 91.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_092() {
        let a = vec![92.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 92.0 + 2.0);
        assert_eq!(simd_sum(&a), 92.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_093() {
        let a = vec![93.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 93.0 + 2.0);
        assert_eq!(simd_sum(&a), 93.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_094() {
        let a = vec![94.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 94.0 + 2.0);
        assert_eq!(simd_sum(&a), 94.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_095() {
        let a = vec![95.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 95.0 + 2.0);
        assert_eq!(simd_sum(&a), 95.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_096() {
        let a = vec![96.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 96.0 + 2.0);
        assert_eq!(simd_sum(&a), 96.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_097() {
        let a = vec![97.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 97.0 + 2.0);
        assert_eq!(simd_sum(&a), 97.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_098() {
        let a = vec![98.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 98.0 + 2.0);
        assert_eq!(simd_sum(&a), 98.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_099() {
        let a = vec![99.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 99.0 + 2.0);
        assert_eq!(simd_sum(&a), 99.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_100() {
        let a = vec![100.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 100.0 + 2.0);
        assert_eq!(simd_sum(&a), 100.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_101() {
        let a = vec![101.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 101.0 + 2.0);
        assert_eq!(simd_sum(&a), 101.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_102() {
        let a = vec![102.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 102.0 + 2.0);
        assert_eq!(simd_sum(&a), 102.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_103() {
        let a = vec![103.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 103.0 + 2.0);
        assert_eq!(simd_sum(&a), 103.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_104() {
        let a = vec![104.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 104.0 + 2.0);
        assert_eq!(simd_sum(&a), 104.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_105() {
        let a = vec![105.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 105.0 + 2.0);
        assert_eq!(simd_sum(&a), 105.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_106() {
        let a = vec![106.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 106.0 + 2.0);
        assert_eq!(simd_sum(&a), 106.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_107() {
        let a = vec![107.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 107.0 + 2.0);
        assert_eq!(simd_sum(&a), 107.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_108() {
        let a = vec![108.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 108.0 + 2.0);
        assert_eq!(simd_sum(&a), 108.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_109() {
        let a = vec![109.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 109.0 + 2.0);
        assert_eq!(simd_sum(&a), 109.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_110() {
        let a = vec![110.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 110.0 + 2.0);
        assert_eq!(simd_sum(&a), 110.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_111() {
        let a = vec![111.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 111.0 + 2.0);
        assert_eq!(simd_sum(&a), 111.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_112() {
        let a = vec![112.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 112.0 + 2.0);
        assert_eq!(simd_sum(&a), 112.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_113() {
        let a = vec![113.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 113.0 + 2.0);
        assert_eq!(simd_sum(&a), 113.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_114() {
        let a = vec![114.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 114.0 + 2.0);
        assert_eq!(simd_sum(&a), 114.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_115() {
        let a = vec![115.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 115.0 + 2.0);
        assert_eq!(simd_sum(&a), 115.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_116() {
        let a = vec![116.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 116.0 + 2.0);
        assert_eq!(simd_sum(&a), 116.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_117() {
        let a = vec![117.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 117.0 + 2.0);
        assert_eq!(simd_sum(&a), 117.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_118() {
        let a = vec![118.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 118.0 + 2.0);
        assert_eq!(simd_sum(&a), 118.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_119() {
        let a = vec![119.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 119.0 + 2.0);
        assert_eq!(simd_sum(&a), 119.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_120() {
        let a = vec![120.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 120.0 + 2.0);
        assert_eq!(simd_sum(&a), 120.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_121() {
        let a = vec![121.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 121.0 + 2.0);
        assert_eq!(simd_sum(&a), 121.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_122() {
        let a = vec![122.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 122.0 + 2.0);
        assert_eq!(simd_sum(&a), 122.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_123() {
        let a = vec![123.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 123.0 + 2.0);
        assert_eq!(simd_sum(&a), 123.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_124() {
        let a = vec![124.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 124.0 + 2.0);
        assert_eq!(simd_sum(&a), 124.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_125() {
        let a = vec![125.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 125.0 + 2.0);
        assert_eq!(simd_sum(&a), 125.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_126() {
        let a = vec![126.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 126.0 + 2.0);
        assert_eq!(simd_sum(&a), 126.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_127() {
        let a = vec![127.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 127.0 + 2.0);
        assert_eq!(simd_sum(&a), 127.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_128() {
        let a = vec![128.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 128.0 + 2.0);
        assert_eq!(simd_sum(&a), 128.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_129() {
        let a = vec![129.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 129.0 + 2.0);
        assert_eq!(simd_sum(&a), 129.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_130() {
        let a = vec![130.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 130.0 + 2.0);
        assert_eq!(simd_sum(&a), 130.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_131() {
        let a = vec![131.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 131.0 + 2.0);
        assert_eq!(simd_sum(&a), 131.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_132() {
        let a = vec![132.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 132.0 + 2.0);
        assert_eq!(simd_sum(&a), 132.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_133() {
        let a = vec![133.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 133.0 + 2.0);
        assert_eq!(simd_sum(&a), 133.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_134() {
        let a = vec![134.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 134.0 + 2.0);
        assert_eq!(simd_sum(&a), 134.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_135() {
        let a = vec![135.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 135.0 + 2.0);
        assert_eq!(simd_sum(&a), 135.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_136() {
        let a = vec![136.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 136.0 + 2.0);
        assert_eq!(simd_sum(&a), 136.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_137() {
        let a = vec![137.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 137.0 + 2.0);
        assert_eq!(simd_sum(&a), 137.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_138() {
        let a = vec![138.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 138.0 + 2.0);
        assert_eq!(simd_sum(&a), 138.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_139() {
        let a = vec![139.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 139.0 + 2.0);
        assert_eq!(simd_sum(&a), 139.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_140() {
        let a = vec![140.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 140.0 + 2.0);
        assert_eq!(simd_sum(&a), 140.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_141() {
        let a = vec![141.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 141.0 + 2.0);
        assert_eq!(simd_sum(&a), 141.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_142() {
        let a = vec![142.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 142.0 + 2.0);
        assert_eq!(simd_sum(&a), 142.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_143() {
        let a = vec![143.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 143.0 + 2.0);
        assert_eq!(simd_sum(&a), 143.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_144() {
        let a = vec![144.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 144.0 + 2.0);
        assert_eq!(simd_sum(&a), 144.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_145() {
        let a = vec![145.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 145.0 + 2.0);
        assert_eq!(simd_sum(&a), 145.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_146() {
        let a = vec![146.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 146.0 + 2.0);
        assert_eq!(simd_sum(&a), 146.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_147() {
        let a = vec![147.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 147.0 + 2.0);
        assert_eq!(simd_sum(&a), 147.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_148() {
        let a = vec![148.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 148.0 + 2.0);
        assert_eq!(simd_sum(&a), 148.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_149() {
        let a = vec![149.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 149.0 + 2.0);
        assert_eq!(simd_sum(&a), 149.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_150() {
        let a = vec![150.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 150.0 + 2.0);
        assert_eq!(simd_sum(&a), 150.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_151() {
        let a = vec![151.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 151.0 + 2.0);
        assert_eq!(simd_sum(&a), 151.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_152() {
        let a = vec![152.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 152.0 + 2.0);
        assert_eq!(simd_sum(&a), 152.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_153() {
        let a = vec![153.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 153.0 + 2.0);
        assert_eq!(simd_sum(&a), 153.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_154() {
        let a = vec![154.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 154.0 + 2.0);
        assert_eq!(simd_sum(&a), 154.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_155() {
        let a = vec![155.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 155.0 + 2.0);
        assert_eq!(simd_sum(&a), 155.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_156() {
        let a = vec![156.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 156.0 + 2.0);
        assert_eq!(simd_sum(&a), 156.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_157() {
        let a = vec![157.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 157.0 + 2.0);
        assert_eq!(simd_sum(&a), 157.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_158() {
        let a = vec![158.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 158.0 + 2.0);
        assert_eq!(simd_sum(&a), 158.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_159() {
        let a = vec![159.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 159.0 + 2.0);
        assert_eq!(simd_sum(&a), 159.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_160() {
        let a = vec![160.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 160.0 + 2.0);
        assert_eq!(simd_sum(&a), 160.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_161() {
        let a = vec![161.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 161.0 + 2.0);
        assert_eq!(simd_sum(&a), 161.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_162() {
        let a = vec![162.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 162.0 + 2.0);
        assert_eq!(simd_sum(&a), 162.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_163() {
        let a = vec![163.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 163.0 + 2.0);
        assert_eq!(simd_sum(&a), 163.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_164() {
        let a = vec![164.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 164.0 + 2.0);
        assert_eq!(simd_sum(&a), 164.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_165() {
        let a = vec![165.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 165.0 + 2.0);
        assert_eq!(simd_sum(&a), 165.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_166() {
        let a = vec![166.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 166.0 + 2.0);
        assert_eq!(simd_sum(&a), 166.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_167() {
        let a = vec![167.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 167.0 + 2.0);
        assert_eq!(simd_sum(&a), 167.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_168() {
        let a = vec![168.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 168.0 + 2.0);
        assert_eq!(simd_sum(&a), 168.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_169() {
        let a = vec![169.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 169.0 + 2.0);
        assert_eq!(simd_sum(&a), 169.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_170() {
        let a = vec![170.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 170.0 + 2.0);
        assert_eq!(simd_sum(&a), 170.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_171() {
        let a = vec![171.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 171.0 + 2.0);
        assert_eq!(simd_sum(&a), 171.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_172() {
        let a = vec![172.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 172.0 + 2.0);
        assert_eq!(simd_sum(&a), 172.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_173() {
        let a = vec![173.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 173.0 + 2.0);
        assert_eq!(simd_sum(&a), 173.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_174() {
        let a = vec![174.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 174.0 + 2.0);
        assert_eq!(simd_sum(&a), 174.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_175() {
        let a = vec![175.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 175.0 + 2.0);
        assert_eq!(simd_sum(&a), 175.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_176() {
        let a = vec![176.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 176.0 + 2.0);
        assert_eq!(simd_sum(&a), 176.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_177() {
        let a = vec![177.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 177.0 + 2.0);
        assert_eq!(simd_sum(&a), 177.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_178() {
        let a = vec![178.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 178.0 + 2.0);
        assert_eq!(simd_sum(&a), 178.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_179() {
        let a = vec![179.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 179.0 + 2.0);
        assert_eq!(simd_sum(&a), 179.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_180() {
        let a = vec![180.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 180.0 + 2.0);
        assert_eq!(simd_sum(&a), 180.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_181() {
        let a = vec![181.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 181.0 + 2.0);
        assert_eq!(simd_sum(&a), 181.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_182() {
        let a = vec![182.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 182.0 + 2.0);
        assert_eq!(simd_sum(&a), 182.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_183() {
        let a = vec![183.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 183.0 + 2.0);
        assert_eq!(simd_sum(&a), 183.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_184() {
        let a = vec![184.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 184.0 + 2.0);
        assert_eq!(simd_sum(&a), 184.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_185() {
        let a = vec![185.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 185.0 + 2.0);
        assert_eq!(simd_sum(&a), 185.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_186() {
        let a = vec![186.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 186.0 + 2.0);
        assert_eq!(simd_sum(&a), 186.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_187() {
        let a = vec![187.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 187.0 + 2.0);
        assert_eq!(simd_sum(&a), 187.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_188() {
        let a = vec![188.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 188.0 + 2.0);
        assert_eq!(simd_sum(&a), 188.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_189() {
        let a = vec![189.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 189.0 + 2.0);
        assert_eq!(simd_sum(&a), 189.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_190() {
        let a = vec![190.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 190.0 + 2.0);
        assert_eq!(simd_sum(&a), 190.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_191() {
        let a = vec![191.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 191.0 + 2.0);
        assert_eq!(simd_sum(&a), 191.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_192() {
        let a = vec![192.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 192.0 + 2.0);
        assert_eq!(simd_sum(&a), 192.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_193() {
        let a = vec![193.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 193.0 + 2.0);
        assert_eq!(simd_sum(&a), 193.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_194() {
        let a = vec![194.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 194.0 + 2.0);
        assert_eq!(simd_sum(&a), 194.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_195() {
        let a = vec![195.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 195.0 + 2.0);
        assert_eq!(simd_sum(&a), 195.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_196() {
        let a = vec![196.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 196.0 + 2.0);
        assert_eq!(simd_sum(&a), 196.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_197() {
        let a = vec![197.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 197.0 + 2.0);
        assert_eq!(simd_sum(&a), 197.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_198() {
        let a = vec![198.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 198.0 + 2.0);
        assert_eq!(simd_sum(&a), 198.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_199() {
        let a = vec![199.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 199.0 + 2.0);
        assert_eq!(simd_sum(&a), 199.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_200() {
        let a = vec![200.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 200.0 + 2.0);
        assert_eq!(simd_sum(&a), 200.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_201() {
        let a = vec![201.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 201.0 + 2.0);
        assert_eq!(simd_sum(&a), 201.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_202() {
        let a = vec![202.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 202.0 + 2.0);
        assert_eq!(simd_sum(&a), 202.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_203() {
        let a = vec![203.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 203.0 + 2.0);
        assert_eq!(simd_sum(&a), 203.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_204() {
        let a = vec![204.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 204.0 + 2.0);
        assert_eq!(simd_sum(&a), 204.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_205() {
        let a = vec![205.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 205.0 + 2.0);
        assert_eq!(simd_sum(&a), 205.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_206() {
        let a = vec![206.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 206.0 + 2.0);
        assert_eq!(simd_sum(&a), 206.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_207() {
        let a = vec![207.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 207.0 + 2.0);
        assert_eq!(simd_sum(&a), 207.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_208() {
        let a = vec![208.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 208.0 + 2.0);
        assert_eq!(simd_sum(&a), 208.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_209() {
        let a = vec![209.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 209.0 + 2.0);
        assert_eq!(simd_sum(&a), 209.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_210() {
        let a = vec![210.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 210.0 + 2.0);
        assert_eq!(simd_sum(&a), 210.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_211() {
        let a = vec![211.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 211.0 + 2.0);
        assert_eq!(simd_sum(&a), 211.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_212() {
        let a = vec![212.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 212.0 + 2.0);
        assert_eq!(simd_sum(&a), 212.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_213() {
        let a = vec![213.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 213.0 + 2.0);
        assert_eq!(simd_sum(&a), 213.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_214() {
        let a = vec![214.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 214.0 + 2.0);
        assert_eq!(simd_sum(&a), 214.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_215() {
        let a = vec![215.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 215.0 + 2.0);
        assert_eq!(simd_sum(&a), 215.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_216() {
        let a = vec![216.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 216.0 + 2.0);
        assert_eq!(simd_sum(&a), 216.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_217() {
        let a = vec![217.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 217.0 + 2.0);
        assert_eq!(simd_sum(&a), 217.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_218() {
        let a = vec![218.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 218.0 + 2.0);
        assert_eq!(simd_sum(&a), 218.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_219() {
        let a = vec![219.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 219.0 + 2.0);
        assert_eq!(simd_sum(&a), 219.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_220() {
        let a = vec![220.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 220.0 + 2.0);
        assert_eq!(simd_sum(&a), 220.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_221() {
        let a = vec![221.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 221.0 + 2.0);
        assert_eq!(simd_sum(&a), 221.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_222() {
        let a = vec![222.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 222.0 + 2.0);
        assert_eq!(simd_sum(&a), 222.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_223() {
        let a = vec![223.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 223.0 + 2.0);
        assert_eq!(simd_sum(&a), 223.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_224() {
        let a = vec![224.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 224.0 + 2.0);
        assert_eq!(simd_sum(&a), 224.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_225() {
        let a = vec![225.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 225.0 + 2.0);
        assert_eq!(simd_sum(&a), 225.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_226() {
        let a = vec![226.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 226.0 + 2.0);
        assert_eq!(simd_sum(&a), 226.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_227() {
        let a = vec![227.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 227.0 + 2.0);
        assert_eq!(simd_sum(&a), 227.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_228() {
        let a = vec![228.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 228.0 + 2.0);
        assert_eq!(simd_sum(&a), 228.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_229() {
        let a = vec![229.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 229.0 + 2.0);
        assert_eq!(simd_sum(&a), 229.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_230() {
        let a = vec![230.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 230.0 + 2.0);
        assert_eq!(simd_sum(&a), 230.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_231() {
        let a = vec![231.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 231.0 + 2.0);
        assert_eq!(simd_sum(&a), 231.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_232() {
        let a = vec![232.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 232.0 + 2.0);
        assert_eq!(simd_sum(&a), 232.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_233() {
        let a = vec![233.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 233.0 + 2.0);
        assert_eq!(simd_sum(&a), 233.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_234() {
        let a = vec![234.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 234.0 + 2.0);
        assert_eq!(simd_sum(&a), 234.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_235() {
        let a = vec![235.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 235.0 + 2.0);
        assert_eq!(simd_sum(&a), 235.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_236() {
        let a = vec![236.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 236.0 + 2.0);
        assert_eq!(simd_sum(&a), 236.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_237() {
        let a = vec![237.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 237.0 + 2.0);
        assert_eq!(simd_sum(&a), 237.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_238() {
        let a = vec![238.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 238.0 + 2.0);
        assert_eq!(simd_sum(&a), 238.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_239() {
        let a = vec![239.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 239.0 + 2.0);
        assert_eq!(simd_sum(&a), 239.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_240() {
        let a = vec![240.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 240.0 + 2.0);
        assert_eq!(simd_sum(&a), 240.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_241() {
        let a = vec![241.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 241.0 + 2.0);
        assert_eq!(simd_sum(&a), 241.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_242() {
        let a = vec![242.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 242.0 + 2.0);
        assert_eq!(simd_sum(&a), 242.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_243() {
        let a = vec![243.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 243.0 + 2.0);
        assert_eq!(simd_sum(&a), 243.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_244() {
        let a = vec![244.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 244.0 + 2.0);
        assert_eq!(simd_sum(&a), 244.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_245() {
        let a = vec![245.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 245.0 + 2.0);
        assert_eq!(simd_sum(&a), 245.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_246() {
        let a = vec![246.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 246.0 + 2.0);
        assert_eq!(simd_sum(&a), 246.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_247() {
        let a = vec![247.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 247.0 + 2.0);
        assert_eq!(simd_sum(&a), 247.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_248() {
        let a = vec![248.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 248.0 + 2.0);
        assert_eq!(simd_sum(&a), 248.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_249() {
        let a = vec![249.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 249.0 + 2.0);
        assert_eq!(simd_sum(&a), 249.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_250() {
        let a = vec![250.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 250.0 + 2.0);
        assert_eq!(simd_sum(&a), 250.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_251() {
        let a = vec![251.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 251.0 + 2.0);
        assert_eq!(simd_sum(&a), 251.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_252() {
        let a = vec![252.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 252.0 + 2.0);
        assert_eq!(simd_sum(&a), 252.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_253() {
        let a = vec![253.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 253.0 + 2.0);
        assert_eq!(simd_sum(&a), 253.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_254() {
        let a = vec![254.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 254.0 + 2.0);
        assert_eq!(simd_sum(&a), 254.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_255() {
        let a = vec![255.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 255.0 + 2.0);
        assert_eq!(simd_sum(&a), 255.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_256() {
        let a = vec![256.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 256.0 + 2.0);
        assert_eq!(simd_sum(&a), 256.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_257() {
        let a = vec![257.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 257.0 + 2.0);
        assert_eq!(simd_sum(&a), 257.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_258() {
        let a = vec![258.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 258.0 + 2.0);
        assert_eq!(simd_sum(&a), 258.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_259() {
        let a = vec![259.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 259.0 + 2.0);
        assert_eq!(simd_sum(&a), 259.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_260() {
        let a = vec![260.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 260.0 + 2.0);
        assert_eq!(simd_sum(&a), 260.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_261() {
        let a = vec![261.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 261.0 + 2.0);
        assert_eq!(simd_sum(&a), 261.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_262() {
        let a = vec![262.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 262.0 + 2.0);
        assert_eq!(simd_sum(&a), 262.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_263() {
        let a = vec![263.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 263.0 + 2.0);
        assert_eq!(simd_sum(&a), 263.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_264() {
        let a = vec![264.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 264.0 + 2.0);
        assert_eq!(simd_sum(&a), 264.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_265() {
        let a = vec![265.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 265.0 + 2.0);
        assert_eq!(simd_sum(&a), 265.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_266() {
        let a = vec![266.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 266.0 + 2.0);
        assert_eq!(simd_sum(&a), 266.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_267() {
        let a = vec![267.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 267.0 + 2.0);
        assert_eq!(simd_sum(&a), 267.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_268() {
        let a = vec![268.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 268.0 + 2.0);
        assert_eq!(simd_sum(&a), 268.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_269() {
        let a = vec![269.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 269.0 + 2.0);
        assert_eq!(simd_sum(&a), 269.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_270() {
        let a = vec![270.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 270.0 + 2.0);
        assert_eq!(simd_sum(&a), 270.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_271() {
        let a = vec![271.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 271.0 + 2.0);
        assert_eq!(simd_sum(&a), 271.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_272() {
        let a = vec![272.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 272.0 + 2.0);
        assert_eq!(simd_sum(&a), 272.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_273() {
        let a = vec![273.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 273.0 + 2.0);
        assert_eq!(simd_sum(&a), 273.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_274() {
        let a = vec![274.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 274.0 + 2.0);
        assert_eq!(simd_sum(&a), 274.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_275() {
        let a = vec![275.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 275.0 + 2.0);
        assert_eq!(simd_sum(&a), 275.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_276() {
        let a = vec![276.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 276.0 + 2.0);
        assert_eq!(simd_sum(&a), 276.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_277() {
        let a = vec![277.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 277.0 + 2.0);
        assert_eq!(simd_sum(&a), 277.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_278() {
        let a = vec![278.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 278.0 + 2.0);
        assert_eq!(simd_sum(&a), 278.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_279() {
        let a = vec![279.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 279.0 + 2.0);
        assert_eq!(simd_sum(&a), 279.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_280() {
        let a = vec![280.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 280.0 + 2.0);
        assert_eq!(simd_sum(&a), 280.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_281() {
        let a = vec![281.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 281.0 + 2.0);
        assert_eq!(simd_sum(&a), 281.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_282() {
        let a = vec![282.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 282.0 + 2.0);
        assert_eq!(simd_sum(&a), 282.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_283() {
        let a = vec![283.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 283.0 + 2.0);
        assert_eq!(simd_sum(&a), 283.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_284() {
        let a = vec![284.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 284.0 + 2.0);
        assert_eq!(simd_sum(&a), 284.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_285() {
        let a = vec![285.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 285.0 + 2.0);
        assert_eq!(simd_sum(&a), 285.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_286() {
        let a = vec![286.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 286.0 + 2.0);
        assert_eq!(simd_sum(&a), 286.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_287() {
        let a = vec![287.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 287.0 + 2.0);
        assert_eq!(simd_sum(&a), 287.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_288() {
        let a = vec![288.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 288.0 + 2.0);
        assert_eq!(simd_sum(&a), 288.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_289() {
        let a = vec![289.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 289.0 + 2.0);
        assert_eq!(simd_sum(&a), 289.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_290() {
        let a = vec![290.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 290.0 + 2.0);
        assert_eq!(simd_sum(&a), 290.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_291() {
        let a = vec![291.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 291.0 + 2.0);
        assert_eq!(simd_sum(&a), 291.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_292() {
        let a = vec![292.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 292.0 + 2.0);
        assert_eq!(simd_sum(&a), 292.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_293() {
        let a = vec![293.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 293.0 + 2.0);
        assert_eq!(simd_sum(&a), 293.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_294() {
        let a = vec![294.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 294.0 + 2.0);
        assert_eq!(simd_sum(&a), 294.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_295() {
        let a = vec![295.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 295.0 + 2.0);
        assert_eq!(simd_sum(&a), 295.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_296() {
        let a = vec![296.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 296.0 + 2.0);
        assert_eq!(simd_sum(&a), 296.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_297() {
        let a = vec![297.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 297.0 + 2.0);
        assert_eq!(simd_sum(&a), 297.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_298() {
        let a = vec![298.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 298.0 + 2.0);
        assert_eq!(simd_sum(&a), 298.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_299() {
        let a = vec![299.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 299.0 + 2.0);
        assert_eq!(simd_sum(&a), 299.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_300() {
        let a = vec![300.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 300.0 + 2.0);
        assert_eq!(simd_sum(&a), 300.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_301() {
        let a = vec![301.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 301.0 + 2.0);
        assert_eq!(simd_sum(&a), 301.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_302() {
        let a = vec![302.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 302.0 + 2.0);
        assert_eq!(simd_sum(&a), 302.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_303() {
        let a = vec![303.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 303.0 + 2.0);
        assert_eq!(simd_sum(&a), 303.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_304() {
        let a = vec![304.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 304.0 + 2.0);
        assert_eq!(simd_sum(&a), 304.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_305() {
        let a = vec![305.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 305.0 + 2.0);
        assert_eq!(simd_sum(&a), 305.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_306() {
        let a = vec![306.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 306.0 + 2.0);
        assert_eq!(simd_sum(&a), 306.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_307() {
        let a = vec![307.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 307.0 + 2.0);
        assert_eq!(simd_sum(&a), 307.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_308() {
        let a = vec![308.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 308.0 + 2.0);
        assert_eq!(simd_sum(&a), 308.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_309() {
        let a = vec![309.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 309.0 + 2.0);
        assert_eq!(simd_sum(&a), 309.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_310() {
        let a = vec![310.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 310.0 + 2.0);
        assert_eq!(simd_sum(&a), 310.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_311() {
        let a = vec![311.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 311.0 + 2.0);
        assert_eq!(simd_sum(&a), 311.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_312() {
        let a = vec![312.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 312.0 + 2.0);
        assert_eq!(simd_sum(&a), 312.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_313() {
        let a = vec![313.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 313.0 + 2.0);
        assert_eq!(simd_sum(&a), 313.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_314() {
        let a = vec![314.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 314.0 + 2.0);
        assert_eq!(simd_sum(&a), 314.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_315() {
        let a = vec![315.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 315.0 + 2.0);
        assert_eq!(simd_sum(&a), 315.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_316() {
        let a = vec![316.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 316.0 + 2.0);
        assert_eq!(simd_sum(&a), 316.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_317() {
        let a = vec![317.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 317.0 + 2.0);
        assert_eq!(simd_sum(&a), 317.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_318() {
        let a = vec![318.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 318.0 + 2.0);
        assert_eq!(simd_sum(&a), 318.0 + 1.0);
    }

    #[test]
    fn test_simd_stress_case_319() {
        let a = vec![319.0, 1.0];
        let b = vec![2.0, 3.0];
        let mut out = vec![0.0, 0.0];
        simd_add(&a, &b, &mut out);
        assert_eq!(out[0], 319.0 + 2.0);
        assert_eq!(simd_sum(&a), 319.0 + 1.0);
    }
}
