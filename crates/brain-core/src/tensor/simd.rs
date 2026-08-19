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
    fn test_simd_vector_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0; 7];
        simd_add(&a, &b, &mut out);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
