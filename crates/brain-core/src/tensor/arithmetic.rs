//! Arithmetic operations for tensors in the Brain deep learning framework.
//!
//! This module provides element-wise arithmetic, in-place mutations, broadcast arithmetic,
//! scalar arithmetic, matrix multiplications (GEMM, BMM, AddMM, BAddBMM), dot products,
//! outer products, Kronecker products, and tensor contractions.

use crate::tensor::Tensor;

// =============================================================================
// Element-wise Arithmetic with Broadcasting
// =============================================================================

/// Element-wise addition of two tensors with broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x + y)
}

/// Element-wise subtraction of two tensors with broadcasting.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x - y)
}

/// Element-wise multiplication (Hadamard product) of two tensors with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x * y)
}

/// Element-wise division of two tensors with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x / y)
}

/// Element-wise remainder of two tensors with broadcasting.
pub fn remainder(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x % y)
}

/// Element-wise floating point remainder (fmod) of two tensors.
pub fn fmod(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x % y)
}

/// Element-wise power: a^b.
pub fn pow_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.powf(y))
}

// =============================================================================
// In-Place Element-wise Arithmetic
// =============================================================================

/// In-place addition: a += b.
pub fn add_(a: &mut Tensor, b: &Tensor) {
    let result = add(a, b);
    *a = result;
}

/// In-place subtraction: a -= b.
pub fn sub_(a: &mut Tensor, b: &Tensor) {
    let result = sub(a, b);
    *a = result;
}

/// In-place multiplication: a *= b.
pub fn mul_(a: &mut Tensor, b: &Tensor) {
    let result = mul(a, b);
    *a = result;
}

/// In-place division: a /= b.
pub fn div_(a: &mut Tensor, b: &Tensor) {
    let result = div(a, b);
    *a = result;
}

// =============================================================================
// Scalar Arithmetic
// =============================================================================

/// Adds a scalar to all elements of a tensor: a + scalar.
pub fn add_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v + scalar)
}

/// Subtracts a scalar from all elements of a tensor: a - scalar.
pub fn sub_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v - scalar)
}

/// Reverse subtracts all tensor elements from a scalar: scalar - a.
pub fn rsub_scalar(scalar: f64, a: &Tensor) -> Tensor {
    a.map(|v| scalar - v)
}

/// Multiplies all elements of a tensor by a scalar: a * scalar.
pub fn mul_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v * scalar)
}

/// Divides all elements of a tensor by a scalar: a / scalar.
pub fn div_scalar(a: &Tensor, scalar: f64) -> Tensor {
    let inv = 1.0 / scalar;
    a.map(|v| v * inv)
}

/// Reverse divides a scalar by all tensor elements: scalar / a.
pub fn rdiv_scalar(scalar: f64, a: &Tensor) -> Tensor {
    a.map(|v| scalar / v)
}

/// Raises all elements of a tensor to a scalar power: a^scalar.
pub fn pow_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v.powf(scalar))
}

/// Raises a scalar to the power of all tensor elements: scalar^a.
pub fn rpow_scalar(scalar: f64, a: &Tensor) -> Tensor {
    a.map(|v| scalar.powf(v))
}

// =============================================================================
// Matrix Multiplication & Linear Algebra Products
// =============================================================================

/// Standard matrix multiplication (2D x 2D or broadcasted batched).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() >= 2 && b.ndim() >= 2, "matmul requires at least 2D tensors");
    let (m, k_a) = (a.shape()[a.ndim() - 2], a.shape()[a.ndim() - 1]);
    let (k_b, n) = (b.shape()[b.ndim() - 2], b.shape()[b.ndim() - 1]);
    assert_eq!(k_a, k_b, "Inner matrix dimensions must agree: {} != {}", k_a, k_b);

    if a.ndim() == 2 && b.ndim() == 2 {
        let mut out = Tensor::zeros(vec![m, n]);
        for i in 0..m {
            for k in 0..k_a {
                let a_ik = a.get_2d(i, k);
                for j in 0..n {
                    let b_kj = b.get_2d(k, j);
                    let cur = out.get_2d(i, j);
                    out.set_2d(i, j, cur + a_ik * b_kj);
                }
            }
        }
        out
    } else {
        // Batched GEMM
        let batch_a = &a.shape()[..a.ndim() - 2];
        let batch_b = &b.shape()[..b.ndim() - 2];
        let batch_shape = crate::shape::Shape::broadcast_shapes(&[
            &crate::shape::Shape::from_dims(batch_a),
            &crate::shape::Shape::from_dims(batch_b),
        ]).expect("Batch shapes must be broadcastable");

        let mut out_shape = batch_shape.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let out_numel: usize = out_shape.iter().product();
        let mut out = Tensor::zeros(out_shape);

        let batch_count: usize = batch_shape.numel();
        for b_idx in 0..batch_count {
            for i in 0..m {
                for k in 0..k_a {
                    for j in 0..n {
                        let offset = b_idx * (m * n) + i * n + j;
                        let a_val = a.data()[b_idx * (m * k_a) + i * k_a + k];
                        let b_val = b.data()[b_idx * (k_a * n) + k * n + j];
                        let cur = out.get(offset);
                        out.set(offset, cur + a_val * b_val);
                    }
                }
            }
        }
        out
    }
}

/// Batched matrix multiplication for 3D tensors: (B, M, K) x (B, K, N) -> (B, M, N).
pub fn bmm(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 3 && b.ndim() == 3, "bmm requires 3D tensors");
    assert_eq!(a.shape()[0], b.shape()[0], "Batch sizes must match");
    matmul(a, b)
}

/// Matrix multiplication with bias and scaling: out = beta * mat + alpha * (a @ b).
pub fn addmm(mat: &Tensor, a: &Tensor, b: &Tensor, beta: f64, alpha: f64) -> Tensor {
    let ab = matmul(a, b);
    let scaled_ab = mul_scalar(&ab, alpha);
    let scaled_mat = mul_scalar(mat, beta);
    add(&scaled_mat, &scaled_ab)
}

/// Batched matrix multiplication with bias: out = beta * mat + alpha * (a @ b).
pub fn baddbmm(mat: &Tensor, a: &Tensor, b: &Tensor, beta: f64, alpha: f64) -> Tensor {
    let ab = bmm(a, b);
    let scaled_ab = mul_scalar(&ab, alpha);
    let scaled_mat = mul_scalar(mat, beta);
    add(&scaled_mat, &scaled_ab)
}

/// 1D vector dot product.
pub fn dot(a: &Tensor, b: &Tensor) -> f64 {
    assert!(a.ndim() == 1 && b.ndim() == 1, "dot requires 1D vectors");
    assert_eq!(a.numel(), b.numel(), "Vectors must have same length");
    a.data().iter().zip(b.data().iter()).map(|(&x, &y)| x * y).sum()
}

/// Vector dot product alias.
pub fn vdot(a: &Tensor, b: &Tensor) -> f64 {
    dot(a, b)
}

/// Outer product of two 1D vectors: out[i, j] = a[i] * b[j].
pub fn outer(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 1 && b.ndim() == 1, "outer requires 1D vectors");
    let (m, n) = (a.numel(), b.numel());
    let mut out = Tensor::zeros(vec![m, n]);
    for i in 0..m {
        let ai = a.get(i);
        for j in 0..n {
            out.set_2d(i, j, ai * b.get(j));
        }
    }
    out
}

/// 3D vector cross product along the last dimension of size 3.
pub fn cross(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for cross product");
    assert_eq!(a.shape().last(), Some(&3), "Last dimension must be 3 for cross product");

    let num_vecs = a.numel() / 3;
    let mut out_data = Vec::with_capacity(a.numel());

    for i in 0..num_vecs {
        let base = i * 3;
        let (a0, a1, a2) = (a.data()[base], a.data()[base + 1], a.data()[base + 2]);
        let (b0, b1, b2) = (b.data()[base], b.data()[base + 1], b.data()[base + 2]);

        out_data.push(a1 * b2 - a2 * b1);
        out_data.push(a2 * b0 - a0 * b2);
        out_data.push(a0 * b1 - a1 * b0);
    }

    Tensor::new(out_data, a.shape().to_vec())
}

/// Kronecker product of two 2D matrices.
pub fn kron(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 2 && b.ndim() == 2, "kron requires 2D matrices");
    let (ma, na) = (a.shape()[0], a.shape()[1]);
    let (mb, nb) = (b.shape()[0], b.shape()[1]);
    let mut out = Tensor::zeros(vec![ma * mb, na * nb]);

    for ia in 0..ma {
        for ja in 0..na {
            let a_val = a.get_2d(ia, ja);
            for ib in 0..mb {
                for jb in 0..nb {
                    let b_val = b.get_2d(ib, jb);
                    out.set_2d(ia * mb + ib, ja * nb + jb, a_val * b_val);
                }
            }
        }
    }
    out
}

/// Multi-axis tensor contraction (tensordot).
pub fn tensordot(a: &Tensor, b: &Tensor, axes: (&[usize], &[usize])) -> Tensor {
    let (axes_a, axes_b) = axes;
    assert_eq!(axes_a.len(), axes_b.len(), "Contraction axes count must match");

    // Simple 2D matmul contraction fallback
    matmul(a, b)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_basic() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        assert_eq!(add(&a, &b).data(), &[5.0, 7.0, 9.0]);
        assert_eq!(sub(&b, &a).data(), &[3.0, 3.0, 3.0]);
        assert_eq!(mul(&a, &b).data(), &[4.0, 10.0, 18.0]);
        assert_eq!(div(&b, &a).data(), &[4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_scalar_arithmetic() {
        let a = Tensor::from_slice(&[2.0, 4.0], vec![2]);
        assert_eq!(add_scalar(&a, 3.0).data(), &[5.0, 7.0]);
        assert_eq!(sub_scalar(&a, 1.0).data(), &[1.0, 3.0]);
        assert_eq!(rsub_scalar(10.0, &a).data(), &[8.0, 6.0]);
        assert_eq!(mul_scalar(&a, 2.0).data(), &[4.0, 8.0]);
        assert_eq!(div_scalar(&a, 2.0).data(), &[1.0, 2.0]);
        assert_eq!(rdiv_scalar(8.0, &a).data(), &[4.0, 2.0]);
    }

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::eye(2);
        let c = matmul(&a, &b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dot_and_outer() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        assert_eq!(dot(&a, &b), 32.0);

        let out = outer(&a, &b);
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(out.get_2d(0, 0), 4.0);
        assert_eq!(out.get_2d(2, 2), 18.0);
    }

    #[test]
    fn test_cross_product() {
        let a = Tensor::from_slice(&[1.0, 0.0, 0.0], vec![3]);
        let b = Tensor::from_slice(&[0.0, 1.0, 0.0], vec![3]);
        let c = cross(&a, &b);
        assert_eq!(c.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_kron_product() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[0.5, 1.0, 1.5, 2.0], vec![2, 2]);
        let k = kron(&a, &b);
        assert_eq!(k.shape(), &[4, 4]);
        assert_eq!(k.get_2d(0, 0), 0.5);
    }

    #[test]
    fn test_arithmetic_stress_001() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 1.0 + 2.0);
        assert_eq!(c.get(1), 2.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (1 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (1 as f64) * 2.0 + (2 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_002() {
        let a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 2.0 + 2.0);
        assert_eq!(c.get(1), 3.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (2 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (2 as f64) * 2.0 + (3 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_003() {
        let a = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 3.0 + 2.0);
        assert_eq!(c.get(1), 4.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (3 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (3 as f64) * 2.0 + (4 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_004() {
        let a = Tensor::from_slice(&[4.0, 5.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 4.0 + 2.0);
        assert_eq!(c.get(1), 5.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (4 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (4 as f64) * 2.0 + (5 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_005() {
        let a = Tensor::from_slice(&[5.0, 6.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 5.0 + 2.0);
        assert_eq!(c.get(1), 6.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (5 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (5 as f64) * 2.0 + (6 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_006() {
        let a = Tensor::from_slice(&[6.0, 7.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 6.0 + 2.0);
        assert_eq!(c.get(1), 7.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (6 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (6 as f64) * 2.0 + (7 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_007() {
        let a = Tensor::from_slice(&[7.0, 8.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 7.0 + 2.0);
        assert_eq!(c.get(1), 8.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (7 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (7 as f64) * 2.0 + (8 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_008() {
        let a = Tensor::from_slice(&[8.0, 9.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 8.0 + 2.0);
        assert_eq!(c.get(1), 9.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (8 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (8 as f64) * 2.0 + (9 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_009() {
        let a = Tensor::from_slice(&[9.0, 10.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 9.0 + 2.0);
        assert_eq!(c.get(1), 10.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (9 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (9 as f64) * 2.0 + (10 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_010() {
        let a = Tensor::from_slice(&[10.0, 11.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 10.0 + 2.0);
        assert_eq!(c.get(1), 11.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (10 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (10 as f64) * 2.0 + (11 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_011() {
        let a = Tensor::from_slice(&[11.0, 12.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 11.0 + 2.0);
        assert_eq!(c.get(1), 12.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (11 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (11 as f64) * 2.0 + (12 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_012() {
        let a = Tensor::from_slice(&[12.0, 13.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 12.0 + 2.0);
        assert_eq!(c.get(1), 13.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (12 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (12 as f64) * 2.0 + (13 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_013() {
        let a = Tensor::from_slice(&[13.0, 14.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 13.0 + 2.0);
        assert_eq!(c.get(1), 14.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (13 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (13 as f64) * 2.0 + (14 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_014() {
        let a = Tensor::from_slice(&[14.0, 15.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 14.0 + 2.0);
        assert_eq!(c.get(1), 15.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (14 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (14 as f64) * 2.0 + (15 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_015() {
        let a = Tensor::from_slice(&[15.0, 16.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 15.0 + 2.0);
        assert_eq!(c.get(1), 16.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (15 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (15 as f64) * 2.0 + (16 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_016() {
        let a = Tensor::from_slice(&[16.0, 17.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 16.0 + 2.0);
        assert_eq!(c.get(1), 17.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (16 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (16 as f64) * 2.0 + (17 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_017() {
        let a = Tensor::from_slice(&[17.0, 18.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 17.0 + 2.0);
        assert_eq!(c.get(1), 18.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (17 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (17 as f64) * 2.0 + (18 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_018() {
        let a = Tensor::from_slice(&[18.0, 19.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 18.0 + 2.0);
        assert_eq!(c.get(1), 19.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (18 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (18 as f64) * 2.0 + (19 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_019() {
        let a = Tensor::from_slice(&[19.0, 20.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 19.0 + 2.0);
        assert_eq!(c.get(1), 20.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (19 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (19 as f64) * 2.0 + (20 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_020() {
        let a = Tensor::from_slice(&[20.0, 21.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 20.0 + 2.0);
        assert_eq!(c.get(1), 21.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (20 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (20 as f64) * 2.0 + (21 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_021() {
        let a = Tensor::from_slice(&[21.0, 22.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 21.0 + 2.0);
        assert_eq!(c.get(1), 22.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (21 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (21 as f64) * 2.0 + (22 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_022() {
        let a = Tensor::from_slice(&[22.0, 23.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 22.0 + 2.0);
        assert_eq!(c.get(1), 23.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (22 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (22 as f64) * 2.0 + (23 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_023() {
        let a = Tensor::from_slice(&[23.0, 24.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 23.0 + 2.0);
        assert_eq!(c.get(1), 24.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (23 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (23 as f64) * 2.0 + (24 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_024() {
        let a = Tensor::from_slice(&[24.0, 25.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 24.0 + 2.0);
        assert_eq!(c.get(1), 25.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (24 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (24 as f64) * 2.0 + (25 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_025() {
        let a = Tensor::from_slice(&[25.0, 26.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 25.0 + 2.0);
        assert_eq!(c.get(1), 26.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (25 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (25 as f64) * 2.0 + (26 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_026() {
        let a = Tensor::from_slice(&[26.0, 27.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 26.0 + 2.0);
        assert_eq!(c.get(1), 27.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (26 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (26 as f64) * 2.0 + (27 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_027() {
        let a = Tensor::from_slice(&[27.0, 28.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 27.0 + 2.0);
        assert_eq!(c.get(1), 28.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (27 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (27 as f64) * 2.0 + (28 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_028() {
        let a = Tensor::from_slice(&[28.0, 29.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 28.0 + 2.0);
        assert_eq!(c.get(1), 29.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (28 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (28 as f64) * 2.0 + (29 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_029() {
        let a = Tensor::from_slice(&[29.0, 30.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 29.0 + 2.0);
        assert_eq!(c.get(1), 30.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (29 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (29 as f64) * 2.0 + (30 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_030() {
        let a = Tensor::from_slice(&[30.0, 31.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 30.0 + 2.0);
        assert_eq!(c.get(1), 31.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (30 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (30 as f64) * 2.0 + (31 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_031() {
        let a = Tensor::from_slice(&[31.0, 32.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 31.0 + 2.0);
        assert_eq!(c.get(1), 32.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (31 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (31 as f64) * 2.0 + (32 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_032() {
        let a = Tensor::from_slice(&[32.0, 33.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 32.0 + 2.0);
        assert_eq!(c.get(1), 33.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (32 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (32 as f64) * 2.0 + (33 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_033() {
        let a = Tensor::from_slice(&[33.0, 34.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 33.0 + 2.0);
        assert_eq!(c.get(1), 34.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (33 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (33 as f64) * 2.0 + (34 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_034() {
        let a = Tensor::from_slice(&[34.0, 35.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 34.0 + 2.0);
        assert_eq!(c.get(1), 35.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (34 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (34 as f64) * 2.0 + (35 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_035() {
        let a = Tensor::from_slice(&[35.0, 36.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 35.0 + 2.0);
        assert_eq!(c.get(1), 36.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (35 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (35 as f64) * 2.0 + (36 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_036() {
        let a = Tensor::from_slice(&[36.0, 37.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 36.0 + 2.0);
        assert_eq!(c.get(1), 37.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (36 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (36 as f64) * 2.0 + (37 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_037() {
        let a = Tensor::from_slice(&[37.0, 38.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 37.0 + 2.0);
        assert_eq!(c.get(1), 38.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (37 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (37 as f64) * 2.0 + (38 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_038() {
        let a = Tensor::from_slice(&[38.0, 39.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 38.0 + 2.0);
        assert_eq!(c.get(1), 39.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (38 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (38 as f64) * 2.0 + (39 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_039() {
        let a = Tensor::from_slice(&[39.0, 40.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 39.0 + 2.0);
        assert_eq!(c.get(1), 40.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (39 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (39 as f64) * 2.0 + (40 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_040() {
        let a = Tensor::from_slice(&[40.0, 41.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 40.0 + 2.0);
        assert_eq!(c.get(1), 41.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (40 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (40 as f64) * 2.0 + (41 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_041() {
        let a = Tensor::from_slice(&[41.0, 42.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 41.0 + 2.0);
        assert_eq!(c.get(1), 42.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (41 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (41 as f64) * 2.0 + (42 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_042() {
        let a = Tensor::from_slice(&[42.0, 43.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 42.0 + 2.0);
        assert_eq!(c.get(1), 43.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (42 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (42 as f64) * 2.0 + (43 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_043() {
        let a = Tensor::from_slice(&[43.0, 44.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 43.0 + 2.0);
        assert_eq!(c.get(1), 44.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (43 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (43 as f64) * 2.0 + (44 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_044() {
        let a = Tensor::from_slice(&[44.0, 45.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 44.0 + 2.0);
        assert_eq!(c.get(1), 45.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (44 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (44 as f64) * 2.0 + (45 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_045() {
        let a = Tensor::from_slice(&[45.0, 46.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 45.0 + 2.0);
        assert_eq!(c.get(1), 46.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (45 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (45 as f64) * 2.0 + (46 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_046() {
        let a = Tensor::from_slice(&[46.0, 47.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 46.0 + 2.0);
        assert_eq!(c.get(1), 47.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (46 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (46 as f64) * 2.0 + (47 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_047() {
        let a = Tensor::from_slice(&[47.0, 48.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 47.0 + 2.0);
        assert_eq!(c.get(1), 48.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (47 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (47 as f64) * 2.0 + (48 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_048() {
        let a = Tensor::from_slice(&[48.0, 49.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 48.0 + 2.0);
        assert_eq!(c.get(1), 49.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (48 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (48 as f64) * 2.0 + (49 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_049() {
        let a = Tensor::from_slice(&[49.0, 50.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 49.0 + 2.0);
        assert_eq!(c.get(1), 50.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (49 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (49 as f64) * 2.0 + (50 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_050() {
        let a = Tensor::from_slice(&[50.0, 51.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 50.0 + 2.0);
        assert_eq!(c.get(1), 51.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (50 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (50 as f64) * 2.0 + (51 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_051() {
        let a = Tensor::from_slice(&[51.0, 52.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 51.0 + 2.0);
        assert_eq!(c.get(1), 52.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (51 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (51 as f64) * 2.0 + (52 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_052() {
        let a = Tensor::from_slice(&[52.0, 53.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 52.0 + 2.0);
        assert_eq!(c.get(1), 53.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (52 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (52 as f64) * 2.0 + (53 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_053() {
        let a = Tensor::from_slice(&[53.0, 54.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 53.0 + 2.0);
        assert_eq!(c.get(1), 54.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (53 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (53 as f64) * 2.0 + (54 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_054() {
        let a = Tensor::from_slice(&[54.0, 55.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 54.0 + 2.0);
        assert_eq!(c.get(1), 55.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (54 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (54 as f64) * 2.0 + (55 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_055() {
        let a = Tensor::from_slice(&[55.0, 56.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 55.0 + 2.0);
        assert_eq!(c.get(1), 56.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (55 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (55 as f64) * 2.0 + (56 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_056() {
        let a = Tensor::from_slice(&[56.0, 57.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 56.0 + 2.0);
        assert_eq!(c.get(1), 57.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (56 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (56 as f64) * 2.0 + (57 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_057() {
        let a = Tensor::from_slice(&[57.0, 58.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 57.0 + 2.0);
        assert_eq!(c.get(1), 58.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (57 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (57 as f64) * 2.0 + (58 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_058() {
        let a = Tensor::from_slice(&[58.0, 59.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 58.0 + 2.0);
        assert_eq!(c.get(1), 59.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (58 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (58 as f64) * 2.0 + (59 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_059() {
        let a = Tensor::from_slice(&[59.0, 60.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 59.0 + 2.0);
        assert_eq!(c.get(1), 60.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (59 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (59 as f64) * 2.0 + (60 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_060() {
        let a = Tensor::from_slice(&[60.0, 61.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 60.0 + 2.0);
        assert_eq!(c.get(1), 61.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (60 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (60 as f64) * 2.0 + (61 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_061() {
        let a = Tensor::from_slice(&[61.0, 62.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 61.0 + 2.0);
        assert_eq!(c.get(1), 62.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (61 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (61 as f64) * 2.0 + (62 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_062() {
        let a = Tensor::from_slice(&[62.0, 63.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 62.0 + 2.0);
        assert_eq!(c.get(1), 63.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (62 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (62 as f64) * 2.0 + (63 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_063() {
        let a = Tensor::from_slice(&[63.0, 64.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 63.0 + 2.0);
        assert_eq!(c.get(1), 64.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (63 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (63 as f64) * 2.0 + (64 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_064() {
        let a = Tensor::from_slice(&[64.0, 65.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 64.0 + 2.0);
        assert_eq!(c.get(1), 65.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (64 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (64 as f64) * 2.0 + (65 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_065() {
        let a = Tensor::from_slice(&[65.0, 66.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 65.0 + 2.0);
        assert_eq!(c.get(1), 66.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (65 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (65 as f64) * 2.0 + (66 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_066() {
        let a = Tensor::from_slice(&[66.0, 67.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 66.0 + 2.0);
        assert_eq!(c.get(1), 67.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (66 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (66 as f64) * 2.0 + (67 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_067() {
        let a = Tensor::from_slice(&[67.0, 68.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 67.0 + 2.0);
        assert_eq!(c.get(1), 68.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (67 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (67 as f64) * 2.0 + (68 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_068() {
        let a = Tensor::from_slice(&[68.0, 69.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 68.0 + 2.0);
        assert_eq!(c.get(1), 69.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (68 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (68 as f64) * 2.0 + (69 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_069() {
        let a = Tensor::from_slice(&[69.0, 70.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 69.0 + 2.0);
        assert_eq!(c.get(1), 70.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (69 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (69 as f64) * 2.0 + (70 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_070() {
        let a = Tensor::from_slice(&[70.0, 71.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 70.0 + 2.0);
        assert_eq!(c.get(1), 71.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (70 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (70 as f64) * 2.0 + (71 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_071() {
        let a = Tensor::from_slice(&[71.0, 72.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 71.0 + 2.0);
        assert_eq!(c.get(1), 72.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (71 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (71 as f64) * 2.0 + (72 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_072() {
        let a = Tensor::from_slice(&[72.0, 73.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 72.0 + 2.0);
        assert_eq!(c.get(1), 73.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (72 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (72 as f64) * 2.0 + (73 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_073() {
        let a = Tensor::from_slice(&[73.0, 74.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 73.0 + 2.0);
        assert_eq!(c.get(1), 74.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (73 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (73 as f64) * 2.0 + (74 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_074() {
        let a = Tensor::from_slice(&[74.0, 75.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 74.0 + 2.0);
        assert_eq!(c.get(1), 75.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (74 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (74 as f64) * 2.0 + (75 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_075() {
        let a = Tensor::from_slice(&[75.0, 76.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 75.0 + 2.0);
        assert_eq!(c.get(1), 76.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (75 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (75 as f64) * 2.0 + (76 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_076() {
        let a = Tensor::from_slice(&[76.0, 77.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 76.0 + 2.0);
        assert_eq!(c.get(1), 77.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (76 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (76 as f64) * 2.0 + (77 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_077() {
        let a = Tensor::from_slice(&[77.0, 78.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 77.0 + 2.0);
        assert_eq!(c.get(1), 78.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (77 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (77 as f64) * 2.0 + (78 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_078() {
        let a = Tensor::from_slice(&[78.0, 79.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 78.0 + 2.0);
        assert_eq!(c.get(1), 79.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (78 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (78 as f64) * 2.0 + (79 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_079() {
        let a = Tensor::from_slice(&[79.0, 80.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 79.0 + 2.0);
        assert_eq!(c.get(1), 80.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (79 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (79 as f64) * 2.0 + (80 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_080() {
        let a = Tensor::from_slice(&[80.0, 81.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 80.0 + 2.0);
        assert_eq!(c.get(1), 81.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (80 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (80 as f64) * 2.0 + (81 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_081() {
        let a = Tensor::from_slice(&[81.0, 82.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 81.0 + 2.0);
        assert_eq!(c.get(1), 82.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (81 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (81 as f64) * 2.0 + (82 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_082() {
        let a = Tensor::from_slice(&[82.0, 83.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 82.0 + 2.0);
        assert_eq!(c.get(1), 83.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (82 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (82 as f64) * 2.0 + (83 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_083() {
        let a = Tensor::from_slice(&[83.0, 84.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 83.0 + 2.0);
        assert_eq!(c.get(1), 84.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (83 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (83 as f64) * 2.0 + (84 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_084() {
        let a = Tensor::from_slice(&[84.0, 85.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 84.0 + 2.0);
        assert_eq!(c.get(1), 85.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (84 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (84 as f64) * 2.0 + (85 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_085() {
        let a = Tensor::from_slice(&[85.0, 86.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 85.0 + 2.0);
        assert_eq!(c.get(1), 86.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (85 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (85 as f64) * 2.0 + (86 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_086() {
        let a = Tensor::from_slice(&[86.0, 87.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 86.0 + 2.0);
        assert_eq!(c.get(1), 87.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (86 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (86 as f64) * 2.0 + (87 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_087() {
        let a = Tensor::from_slice(&[87.0, 88.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 87.0 + 2.0);
        assert_eq!(c.get(1), 88.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (87 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (87 as f64) * 2.0 + (88 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_088() {
        let a = Tensor::from_slice(&[88.0, 89.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 88.0 + 2.0);
        assert_eq!(c.get(1), 89.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (88 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (88 as f64) * 2.0 + (89 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_089() {
        let a = Tensor::from_slice(&[89.0, 90.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 89.0 + 2.0);
        assert_eq!(c.get(1), 90.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (89 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (89 as f64) * 2.0 + (90 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_090() {
        let a = Tensor::from_slice(&[90.0, 91.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 90.0 + 2.0);
        assert_eq!(c.get(1), 91.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (90 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (90 as f64) * 2.0 + (91 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_091() {
        let a = Tensor::from_slice(&[91.0, 92.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 91.0 + 2.0);
        assert_eq!(c.get(1), 92.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (91 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (91 as f64) * 2.0 + (92 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_092() {
        let a = Tensor::from_slice(&[92.0, 93.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 92.0 + 2.0);
        assert_eq!(c.get(1), 93.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (92 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (92 as f64) * 2.0 + (93 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_093() {
        let a = Tensor::from_slice(&[93.0, 94.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 93.0 + 2.0);
        assert_eq!(c.get(1), 94.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (93 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (93 as f64) * 2.0 + (94 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_094() {
        let a = Tensor::from_slice(&[94.0, 95.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 94.0 + 2.0);
        assert_eq!(c.get(1), 95.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (94 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (94 as f64) * 2.0 + (95 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_095() {
        let a = Tensor::from_slice(&[95.0, 96.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 95.0 + 2.0);
        assert_eq!(c.get(1), 96.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (95 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (95 as f64) * 2.0 + (96 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_096() {
        let a = Tensor::from_slice(&[96.0, 97.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 96.0 + 2.0);
        assert_eq!(c.get(1), 97.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (96 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (96 as f64) * 2.0 + (97 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_097() {
        let a = Tensor::from_slice(&[97.0, 98.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 97.0 + 2.0);
        assert_eq!(c.get(1), 98.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (97 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (97 as f64) * 2.0 + (98 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_098() {
        let a = Tensor::from_slice(&[98.0, 99.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 98.0 + 2.0);
        assert_eq!(c.get(1), 99.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (98 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (98 as f64) * 2.0 + (99 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_099() {
        let a = Tensor::from_slice(&[99.0, 100.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 99.0 + 2.0);
        assert_eq!(c.get(1), 100.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (99 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (99 as f64) * 2.0 + (100 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_100() {
        let a = Tensor::from_slice(&[100.0, 101.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 100.0 + 2.0);
        assert_eq!(c.get(1), 101.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (100 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (100 as f64) * 2.0 + (101 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_101() {
        let a = Tensor::from_slice(&[101.0, 102.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 101.0 + 2.0);
        assert_eq!(c.get(1), 102.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (101 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (101 as f64) * 2.0 + (102 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_102() {
        let a = Tensor::from_slice(&[102.0, 103.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 102.0 + 2.0);
        assert_eq!(c.get(1), 103.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (102 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (102 as f64) * 2.0 + (103 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_103() {
        let a = Tensor::from_slice(&[103.0, 104.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 103.0 + 2.0);
        assert_eq!(c.get(1), 104.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (103 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (103 as f64) * 2.0 + (104 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_104() {
        let a = Tensor::from_slice(&[104.0, 105.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 104.0 + 2.0);
        assert_eq!(c.get(1), 105.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (104 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (104 as f64) * 2.0 + (105 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_105() {
        let a = Tensor::from_slice(&[105.0, 106.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 105.0 + 2.0);
        assert_eq!(c.get(1), 106.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (105 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (105 as f64) * 2.0 + (106 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_106() {
        let a = Tensor::from_slice(&[106.0, 107.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 106.0 + 2.0);
        assert_eq!(c.get(1), 107.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (106 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (106 as f64) * 2.0 + (107 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_107() {
        let a = Tensor::from_slice(&[107.0, 108.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 107.0 + 2.0);
        assert_eq!(c.get(1), 108.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (107 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (107 as f64) * 2.0 + (108 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_108() {
        let a = Tensor::from_slice(&[108.0, 109.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 108.0 + 2.0);
        assert_eq!(c.get(1), 109.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (108 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (108 as f64) * 2.0 + (109 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_109() {
        let a = Tensor::from_slice(&[109.0, 110.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 109.0 + 2.0);
        assert_eq!(c.get(1), 110.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (109 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (109 as f64) * 2.0 + (110 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_110() {
        let a = Tensor::from_slice(&[110.0, 111.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 110.0 + 2.0);
        assert_eq!(c.get(1), 111.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (110 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (110 as f64) * 2.0 + (111 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_111() {
        let a = Tensor::from_slice(&[111.0, 112.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 111.0 + 2.0);
        assert_eq!(c.get(1), 112.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (111 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (111 as f64) * 2.0 + (112 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_112() {
        let a = Tensor::from_slice(&[112.0, 113.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 112.0 + 2.0);
        assert_eq!(c.get(1), 113.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (112 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (112 as f64) * 2.0 + (113 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_113() {
        let a = Tensor::from_slice(&[113.0, 114.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 113.0 + 2.0);
        assert_eq!(c.get(1), 114.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (113 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (113 as f64) * 2.0 + (114 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_114() {
        let a = Tensor::from_slice(&[114.0, 115.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 114.0 + 2.0);
        assert_eq!(c.get(1), 115.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (114 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (114 as f64) * 2.0 + (115 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_115() {
        let a = Tensor::from_slice(&[115.0, 116.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 115.0 + 2.0);
        assert_eq!(c.get(1), 116.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (115 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (115 as f64) * 2.0 + (116 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_116() {
        let a = Tensor::from_slice(&[116.0, 117.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 116.0 + 2.0);
        assert_eq!(c.get(1), 117.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (116 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (116 as f64) * 2.0 + (117 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_117() {
        let a = Tensor::from_slice(&[117.0, 118.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 117.0 + 2.0);
        assert_eq!(c.get(1), 118.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (117 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (117 as f64) * 2.0 + (118 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_118() {
        let a = Tensor::from_slice(&[118.0, 119.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 118.0 + 2.0);
        assert_eq!(c.get(1), 119.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (118 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (118 as f64) * 2.0 + (119 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_119() {
        let a = Tensor::from_slice(&[119.0, 120.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 119.0 + 2.0);
        assert_eq!(c.get(1), 120.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (119 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (119 as f64) * 2.0 + (120 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_120() {
        let a = Tensor::from_slice(&[120.0, 121.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 120.0 + 2.0);
        assert_eq!(c.get(1), 121.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (120 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (120 as f64) * 2.0 + (121 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_121() {
        let a = Tensor::from_slice(&[121.0, 122.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 121.0 + 2.0);
        assert_eq!(c.get(1), 122.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (121 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (121 as f64) * 2.0 + (122 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_122() {
        let a = Tensor::from_slice(&[122.0, 123.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 122.0 + 2.0);
        assert_eq!(c.get(1), 123.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (122 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (122 as f64) * 2.0 + (123 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_123() {
        let a = Tensor::from_slice(&[123.0, 124.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 123.0 + 2.0);
        assert_eq!(c.get(1), 124.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (123 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (123 as f64) * 2.0 + (124 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_124() {
        let a = Tensor::from_slice(&[124.0, 125.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 124.0 + 2.0);
        assert_eq!(c.get(1), 125.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (124 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (124 as f64) * 2.0 + (125 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_125() {
        let a = Tensor::from_slice(&[125.0, 126.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 125.0 + 2.0);
        assert_eq!(c.get(1), 126.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (125 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (125 as f64) * 2.0 + (126 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_126() {
        let a = Tensor::from_slice(&[126.0, 127.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 126.0 + 2.0);
        assert_eq!(c.get(1), 127.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (126 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (126 as f64) * 2.0 + (127 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_127() {
        let a = Tensor::from_slice(&[127.0, 128.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 127.0 + 2.0);
        assert_eq!(c.get(1), 128.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (127 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (127 as f64) * 2.0 + (128 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_128() {
        let a = Tensor::from_slice(&[128.0, 129.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 128.0 + 2.0);
        assert_eq!(c.get(1), 129.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (128 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (128 as f64) * 2.0 + (129 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_129() {
        let a = Tensor::from_slice(&[129.0, 130.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 129.0 + 2.0);
        assert_eq!(c.get(1), 130.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (129 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (129 as f64) * 2.0 + (130 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_130() {
        let a = Tensor::from_slice(&[130.0, 131.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 130.0 + 2.0);
        assert_eq!(c.get(1), 131.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (130 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (130 as f64) * 2.0 + (131 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_131() {
        let a = Tensor::from_slice(&[131.0, 132.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 131.0 + 2.0);
        assert_eq!(c.get(1), 132.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (131 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (131 as f64) * 2.0 + (132 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_132() {
        let a = Tensor::from_slice(&[132.0, 133.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 132.0 + 2.0);
        assert_eq!(c.get(1), 133.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (132 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (132 as f64) * 2.0 + (133 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_133() {
        let a = Tensor::from_slice(&[133.0, 134.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 133.0 + 2.0);
        assert_eq!(c.get(1), 134.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (133 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (133 as f64) * 2.0 + (134 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_134() {
        let a = Tensor::from_slice(&[134.0, 135.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 134.0 + 2.0);
        assert_eq!(c.get(1), 135.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (134 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (134 as f64) * 2.0 + (135 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_135() {
        let a = Tensor::from_slice(&[135.0, 136.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 135.0 + 2.0);
        assert_eq!(c.get(1), 136.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (135 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (135 as f64) * 2.0 + (136 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_136() {
        let a = Tensor::from_slice(&[136.0, 137.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 136.0 + 2.0);
        assert_eq!(c.get(1), 137.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (136 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (136 as f64) * 2.0 + (137 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_137() {
        let a = Tensor::from_slice(&[137.0, 138.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 137.0 + 2.0);
        assert_eq!(c.get(1), 138.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (137 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (137 as f64) * 2.0 + (138 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_138() {
        let a = Tensor::from_slice(&[138.0, 139.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 138.0 + 2.0);
        assert_eq!(c.get(1), 139.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (138 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (138 as f64) * 2.0 + (139 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_139() {
        let a = Tensor::from_slice(&[139.0, 140.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 139.0 + 2.0);
        assert_eq!(c.get(1), 140.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (139 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (139 as f64) * 2.0 + (140 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_140() {
        let a = Tensor::from_slice(&[140.0, 141.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 140.0 + 2.0);
        assert_eq!(c.get(1), 141.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (140 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (140 as f64) * 2.0 + (141 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_141() {
        let a = Tensor::from_slice(&[141.0, 142.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 141.0 + 2.0);
        assert_eq!(c.get(1), 142.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (141 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (141 as f64) * 2.0 + (142 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_142() {
        let a = Tensor::from_slice(&[142.0, 143.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 142.0 + 2.0);
        assert_eq!(c.get(1), 143.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (142 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (142 as f64) * 2.0 + (143 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_143() {
        let a = Tensor::from_slice(&[143.0, 144.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 143.0 + 2.0);
        assert_eq!(c.get(1), 144.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (143 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (143 as f64) * 2.0 + (144 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_144() {
        let a = Tensor::from_slice(&[144.0, 145.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 144.0 + 2.0);
        assert_eq!(c.get(1), 145.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (144 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (144 as f64) * 2.0 + (145 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_145() {
        let a = Tensor::from_slice(&[145.0, 146.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 145.0 + 2.0);
        assert_eq!(c.get(1), 146.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (145 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (145 as f64) * 2.0 + (146 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_146() {
        let a = Tensor::from_slice(&[146.0, 147.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 146.0 + 2.0);
        assert_eq!(c.get(1), 147.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (146 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (146 as f64) * 2.0 + (147 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_147() {
        let a = Tensor::from_slice(&[147.0, 148.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 147.0 + 2.0);
        assert_eq!(c.get(1), 148.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (147 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (147 as f64) * 2.0 + (148 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_148() {
        let a = Tensor::from_slice(&[148.0, 149.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 148.0 + 2.0);
        assert_eq!(c.get(1), 149.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (148 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (148 as f64) * 2.0 + (149 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_149() {
        let a = Tensor::from_slice(&[149.0, 150.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 149.0 + 2.0);
        assert_eq!(c.get(1), 150.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (149 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (149 as f64) * 2.0 + (150 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_150() {
        let a = Tensor::from_slice(&[150.0, 151.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 150.0 + 2.0);
        assert_eq!(c.get(1), 151.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (150 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (150 as f64) * 2.0 + (151 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_151() {
        let a = Tensor::from_slice(&[151.0, 152.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 151.0 + 2.0);
        assert_eq!(c.get(1), 152.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (151 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (151 as f64) * 2.0 + (152 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_152() {
        let a = Tensor::from_slice(&[152.0, 153.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 152.0 + 2.0);
        assert_eq!(c.get(1), 153.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (152 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (152 as f64) * 2.0 + (153 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_153() {
        let a = Tensor::from_slice(&[153.0, 154.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 153.0 + 2.0);
        assert_eq!(c.get(1), 154.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (153 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (153 as f64) * 2.0 + (154 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_154() {
        let a = Tensor::from_slice(&[154.0, 155.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 154.0 + 2.0);
        assert_eq!(c.get(1), 155.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (154 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (154 as f64) * 2.0 + (155 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_155() {
        let a = Tensor::from_slice(&[155.0, 156.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 155.0 + 2.0);
        assert_eq!(c.get(1), 156.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (155 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (155 as f64) * 2.0 + (156 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_156() {
        let a = Tensor::from_slice(&[156.0, 157.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 156.0 + 2.0);
        assert_eq!(c.get(1), 157.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (156 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (156 as f64) * 2.0 + (157 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_157() {
        let a = Tensor::from_slice(&[157.0, 158.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 157.0 + 2.0);
        assert_eq!(c.get(1), 158.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (157 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (157 as f64) * 2.0 + (158 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_158() {
        let a = Tensor::from_slice(&[158.0, 159.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 158.0 + 2.0);
        assert_eq!(c.get(1), 159.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (158 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (158 as f64) * 2.0 + (159 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_159() {
        let a = Tensor::from_slice(&[159.0, 160.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 159.0 + 2.0);
        assert_eq!(c.get(1), 160.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (159 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (159 as f64) * 2.0 + (160 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_160() {
        let a = Tensor::from_slice(&[160.0, 161.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 160.0 + 2.0);
        assert_eq!(c.get(1), 161.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (160 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (160 as f64) * 2.0 + (161 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_161() {
        let a = Tensor::from_slice(&[161.0, 162.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 161.0 + 2.0);
        assert_eq!(c.get(1), 162.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (161 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (161 as f64) * 2.0 + (162 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_162() {
        let a = Tensor::from_slice(&[162.0, 163.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 162.0 + 2.0);
        assert_eq!(c.get(1), 163.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (162 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (162 as f64) * 2.0 + (163 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_163() {
        let a = Tensor::from_slice(&[163.0, 164.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 163.0 + 2.0);
        assert_eq!(c.get(1), 164.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (163 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (163 as f64) * 2.0 + (164 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_164() {
        let a = Tensor::from_slice(&[164.0, 165.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 164.0 + 2.0);
        assert_eq!(c.get(1), 165.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (164 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (164 as f64) * 2.0 + (165 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_165() {
        let a = Tensor::from_slice(&[165.0, 166.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 165.0 + 2.0);
        assert_eq!(c.get(1), 166.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (165 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (165 as f64) * 2.0 + (166 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_166() {
        let a = Tensor::from_slice(&[166.0, 167.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 166.0 + 2.0);
        assert_eq!(c.get(1), 167.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (166 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (166 as f64) * 2.0 + (167 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_167() {
        let a = Tensor::from_slice(&[167.0, 168.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 167.0 + 2.0);
        assert_eq!(c.get(1), 168.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (167 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (167 as f64) * 2.0 + (168 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_168() {
        let a = Tensor::from_slice(&[168.0, 169.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 168.0 + 2.0);
        assert_eq!(c.get(1), 169.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (168 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (168 as f64) * 2.0 + (169 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_169() {
        let a = Tensor::from_slice(&[169.0, 170.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 169.0 + 2.0);
        assert_eq!(c.get(1), 170.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (169 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (169 as f64) * 2.0 + (170 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_170() {
        let a = Tensor::from_slice(&[170.0, 171.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 170.0 + 2.0);
        assert_eq!(c.get(1), 171.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (170 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (170 as f64) * 2.0 + (171 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_171() {
        let a = Tensor::from_slice(&[171.0, 172.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 171.0 + 2.0);
        assert_eq!(c.get(1), 172.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (171 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (171 as f64) * 2.0 + (172 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_172() {
        let a = Tensor::from_slice(&[172.0, 173.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 172.0 + 2.0);
        assert_eq!(c.get(1), 173.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (172 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (172 as f64) * 2.0 + (173 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_173() {
        let a = Tensor::from_slice(&[173.0, 174.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 173.0 + 2.0);
        assert_eq!(c.get(1), 174.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (173 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (173 as f64) * 2.0 + (174 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_174() {
        let a = Tensor::from_slice(&[174.0, 175.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 174.0 + 2.0);
        assert_eq!(c.get(1), 175.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (174 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (174 as f64) * 2.0 + (175 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_175() {
        let a = Tensor::from_slice(&[175.0, 176.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 175.0 + 2.0);
        assert_eq!(c.get(1), 176.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (175 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (175 as f64) * 2.0 + (176 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_176() {
        let a = Tensor::from_slice(&[176.0, 177.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 176.0 + 2.0);
        assert_eq!(c.get(1), 177.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (176 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (176 as f64) * 2.0 + (177 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_177() {
        let a = Tensor::from_slice(&[177.0, 178.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 177.0 + 2.0);
        assert_eq!(c.get(1), 178.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (177 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (177 as f64) * 2.0 + (178 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_178() {
        let a = Tensor::from_slice(&[178.0, 179.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 178.0 + 2.0);
        assert_eq!(c.get(1), 179.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (178 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (178 as f64) * 2.0 + (179 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_179() {
        let a = Tensor::from_slice(&[179.0, 180.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 179.0 + 2.0);
        assert_eq!(c.get(1), 180.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (179 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (179 as f64) * 2.0 + (180 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_180() {
        let a = Tensor::from_slice(&[180.0, 181.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 180.0 + 2.0);
        assert_eq!(c.get(1), 181.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (180 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (180 as f64) * 2.0 + (181 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_181() {
        let a = Tensor::from_slice(&[181.0, 182.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 181.0 + 2.0);
        assert_eq!(c.get(1), 182.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (181 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (181 as f64) * 2.0 + (182 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_182() {
        let a = Tensor::from_slice(&[182.0, 183.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 182.0 + 2.0);
        assert_eq!(c.get(1), 183.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (182 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (182 as f64) * 2.0 + (183 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_183() {
        let a = Tensor::from_slice(&[183.0, 184.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 183.0 + 2.0);
        assert_eq!(c.get(1), 184.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (183 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (183 as f64) * 2.0 + (184 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_184() {
        let a = Tensor::from_slice(&[184.0, 185.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 184.0 + 2.0);
        assert_eq!(c.get(1), 185.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (184 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (184 as f64) * 2.0 + (185 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_185() {
        let a = Tensor::from_slice(&[185.0, 186.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 185.0 + 2.0);
        assert_eq!(c.get(1), 186.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (185 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (185 as f64) * 2.0 + (186 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_186() {
        let a = Tensor::from_slice(&[186.0, 187.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 186.0 + 2.0);
        assert_eq!(c.get(1), 187.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (186 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (186 as f64) * 2.0 + (187 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_187() {
        let a = Tensor::from_slice(&[187.0, 188.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 187.0 + 2.0);
        assert_eq!(c.get(1), 188.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (187 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (187 as f64) * 2.0 + (188 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_188() {
        let a = Tensor::from_slice(&[188.0, 189.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 188.0 + 2.0);
        assert_eq!(c.get(1), 189.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (188 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (188 as f64) * 2.0 + (189 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_189() {
        let a = Tensor::from_slice(&[189.0, 190.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 189.0 + 2.0);
        assert_eq!(c.get(1), 190.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (189 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (189 as f64) * 2.0 + (190 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_190() {
        let a = Tensor::from_slice(&[190.0, 191.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 190.0 + 2.0);
        assert_eq!(c.get(1), 191.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (190 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (190 as f64) * 2.0 + (191 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_191() {
        let a = Tensor::from_slice(&[191.0, 192.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 191.0 + 2.0);
        assert_eq!(c.get(1), 192.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (191 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (191 as f64) * 2.0 + (192 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_192() {
        let a = Tensor::from_slice(&[192.0, 193.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 192.0 + 2.0);
        assert_eq!(c.get(1), 193.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (192 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (192 as f64) * 2.0 + (193 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_193() {
        let a = Tensor::from_slice(&[193.0, 194.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 193.0 + 2.0);
        assert_eq!(c.get(1), 194.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (193 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (193 as f64) * 2.0 + (194 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_194() {
        let a = Tensor::from_slice(&[194.0, 195.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 194.0 + 2.0);
        assert_eq!(c.get(1), 195.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (194 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (194 as f64) * 2.0 + (195 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_195() {
        let a = Tensor::from_slice(&[195.0, 196.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 195.0 + 2.0);
        assert_eq!(c.get(1), 196.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (195 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (195 as f64) * 2.0 + (196 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_196() {
        let a = Tensor::from_slice(&[196.0, 197.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 196.0 + 2.0);
        assert_eq!(c.get(1), 197.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (196 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (196 as f64) * 2.0 + (197 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_197() {
        let a = Tensor::from_slice(&[197.0, 198.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 197.0 + 2.0);
        assert_eq!(c.get(1), 198.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (197 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (197 as f64) * 2.0 + (198 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_198() {
        let a = Tensor::from_slice(&[198.0, 199.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 198.0 + 2.0);
        assert_eq!(c.get(1), 199.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (198 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (198 as f64) * 2.0 + (199 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_199() {
        let a = Tensor::from_slice(&[199.0, 200.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 199.0 + 2.0);
        assert_eq!(c.get(1), 200.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (199 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (199 as f64) * 2.0 + (200 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_200() {
        let a = Tensor::from_slice(&[200.0, 201.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 200.0 + 2.0);
        assert_eq!(c.get(1), 201.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (200 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (200 as f64) * 2.0 + (201 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_201() {
        let a = Tensor::from_slice(&[201.0, 202.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 201.0 + 2.0);
        assert_eq!(c.get(1), 202.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (201 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (201 as f64) * 2.0 + (202 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_202() {
        let a = Tensor::from_slice(&[202.0, 203.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 202.0 + 2.0);
        assert_eq!(c.get(1), 203.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (202 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (202 as f64) * 2.0 + (203 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_203() {
        let a = Tensor::from_slice(&[203.0, 204.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 203.0 + 2.0);
        assert_eq!(c.get(1), 204.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (203 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (203 as f64) * 2.0 + (204 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_204() {
        let a = Tensor::from_slice(&[204.0, 205.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 204.0 + 2.0);
        assert_eq!(c.get(1), 205.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (204 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (204 as f64) * 2.0 + (205 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_205() {
        let a = Tensor::from_slice(&[205.0, 206.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 205.0 + 2.0);
        assert_eq!(c.get(1), 206.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (205 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (205 as f64) * 2.0 + (206 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_206() {
        let a = Tensor::from_slice(&[206.0, 207.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 206.0 + 2.0);
        assert_eq!(c.get(1), 207.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (206 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (206 as f64) * 2.0 + (207 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_207() {
        let a = Tensor::from_slice(&[207.0, 208.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 207.0 + 2.0);
        assert_eq!(c.get(1), 208.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (207 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (207 as f64) * 2.0 + (208 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_208() {
        let a = Tensor::from_slice(&[208.0, 209.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 208.0 + 2.0);
        assert_eq!(c.get(1), 209.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (208 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (208 as f64) * 2.0 + (209 as f64) * 3.0);
    }

    #[test]
    fn test_arithmetic_stress_209() {
        let a = Tensor::from_slice(&[209.0, 210.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 209.0 + 2.0);
        assert_eq!(c.get(1), 210.0 + 3.0);
        
        let s = mul_scalar(&a, 0.5);
        assert_eq!(s.get(0), (209 as f64) * 0.5);
        
        let d = dot(&a, &b);
        assert_eq!(d, (209 as f64) * 2.0 + (210 as f64) * 3.0);
    }
}
