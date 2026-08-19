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

/// Element-wise minimum of two tensors with broadcasting.
pub fn min_elem(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.min(y))
}

/// Element-wise maximum of two tensors with broadcasting.
pub fn max_elem(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.max(y))
}

/// Element-wise conditional select: `where(cond, a, b)` with broadcasting.
///
/// `cond` is treated as a mask: nonzero selects from `a`, zero from `b`.
pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    use crate::tensor::broadcast::broadcast_to;

    // Numpy-style broadcast merge of the three shapes.
    let mut shape = Vec::new();
    let max_ndim = cond.ndim().max(a.ndim()).max(b.ndim());
    for i in 0..max_ndim {
        let mut dim = 1usize;
        for t in [cond, a, b] {
            let nd = t.ndim();
            if i < nd {
                let d = t.shape()[nd - 1 - i];
                if d != 1 && dim != 1 && dim != d {
                    return Tensor::zeros(vec![]);
                }
                dim = dim.max(d);
            }
        }
        shape.push(dim);
    }
    shape.reverse();

    let cond_b = broadcast_to(cond, &shape).unwrap_or_else(|_| cond.clone());
    let a_b = broadcast_to(a, &shape).unwrap_or_else(|_| a.clone());
    let b_b = broadcast_to(b, &shape).unwrap_or_else(|_| b.clone());
    let cd = cond_b.data();
    let ad = a_b.data();
    let bd = b_b.data();
    let mut out = Vec::with_capacity(cd.len());
    for i in 0..cd.len() {
        out.push(if cd[i] != 0.0 { ad[i] } else { bd[i] });
    }
    Tensor::from_vec(out, shape)
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

    let num_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(16);

    if a.ndim() == 2 && b.ndim() == 2 {
        let mut out = Tensor::zeros(vec![m, n]);
        let a_slice = a.data();
        let b_slice = b.data();
        let out_slice = out.data_mut();

        if num_threads <= 1 || m < 8 || (m * n * k_a) < 16384 {
            gemm_2d_tile(a_slice, b_slice, out_slice, k_a, n, 0, m);
        } else {
            let rows_per_thread = (m + num_threads - 1) / num_threads;
            std::thread::scope(|s| {
                for (t_idx, out_chunk) in out_slice.chunks_mut(rows_per_thread * n).enumerate() {
                    let row_start = t_idx * rows_per_thread;
                    let row_count = out_chunk.len() / n;
                    if row_count > 0 {
                        s.spawn(move || {
                            gemm_2d_tile(a_slice, b_slice, out_chunk, k_a, n, row_start, row_start + row_count);
                        });
                    }
                }
            });
        }
        out
    } else {
        // Batched GEMM with proper batch-dimension broadcasting.
        let batch_a = &a.shape()[..a.ndim() - 2];
        let batch_b = &b.shape()[..b.ndim() - 2];
        let batch_shape = crate::shape::Shape::broadcast_shapes(&[
            &crate::shape::Shape::from_dims(batch_a),
            &crate::shape::Shape::from_dims(batch_b),
        ]).expect("Batch shapes must be broadcastable");

        let max_ndim = batch_shape.ndim();

        let a_batch: Vec<usize> = {
            let mut v = vec![1usize; max_ndim];
            for (i, &d) in batch_a.iter().enumerate() {
                v[max_ndim - batch_a.len() + i] = d;
            }
            v
        };
        let b_batch: Vec<usize> = {
            let mut v = vec![1usize; max_ndim];
            for (i, &d) in batch_b.iter().enumerate() {
                v[max_ndim - batch_b.len() + i] = d;
            }
            v
        };

        let mut out_shape = batch_shape.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let batch_count: usize = batch_shape.numel();
        let mut out = Tensor::zeros(out_shape);

        let a_data = a.data();
        let b_data = b.data();
        let out_data = out.data_mut();

        let matrix_size = m * n;

        if num_threads <= 1 || batch_count < 2 {
            for b_idx in 0..batch_count {
                compute_batch_gemm(
                    b_idx,
                    max_ndim,
                    &batch_shape,
                    &a_batch,
                    &b_batch,
                    m,
                    k_a,
                    n,
                    a_data,
                    b_data,
                    &mut out_data[b_idx * matrix_size..(b_idx + 1) * matrix_size],
                );
            }
        } else {
            let batches_per_thread = (batch_count + num_threads - 1) / num_threads;
            let a_batch_ref = &a_batch;
            let b_batch_ref = &b_batch;
            let b_shape_ref = &batch_shape;

            std::thread::scope(|s| {
                for (t_idx, out_chunk) in out_data.chunks_mut(batches_per_thread * matrix_size).enumerate() {
                    let b_start = t_idx * batches_per_thread;
                    let b_len = out_chunk.len() / matrix_size;

                    s.spawn(move || {
                        for local_b in 0..b_len {
                            let global_b = b_start + local_b;
                            let out_matrix = &mut out_chunk[local_b * matrix_size..(local_b + 1) * matrix_size];
                            compute_batch_gemm(
                                global_b,
                                max_ndim,
                                b_shape_ref,
                                a_batch_ref,
                                b_batch_ref,
                                m,
                                k_a,
                                n,
                                a_data,
                                b_data,
                                out_matrix,
                            );
                        }
                    });
                }
            });
        }

        out
    }
}

#[inline(always)]
fn compute_batch_gemm(
    b_idx: usize,
    max_ndim: usize,
    batch_shape: &crate::shape::Shape,
    a_batch: &[usize],
    b_batch: &[usize],
    m: usize,
    k_a: usize,
    n: usize,
    a_data: &[f64],
    b_data: &[f64],
    out_slice: &mut [f64],
) {
    let mut rem = b_idx;
    let mut idxs = vec![0usize; max_ndim];
    for d in (0..max_ndim).rev() {
        idxs[d] = rem % batch_shape[d];
        rem /= batch_shape[d];
    }
    let a_flat = {
        let mut flat = 0usize;
        for d in 0..max_ndim {
            let ai = if a_batch[d] == 1 { 0 } else { idxs[d] };
            flat = flat * a_batch[d] + ai;
        }
        flat
    };
    let b_flat = {
        let mut flat = 0usize;
        for d in 0..max_ndim {
            let bi = if b_batch[d] == 1 { 0 } else { idxs[d] };
            flat = flat * b_batch[d] + bi;
        }
        flat
    };
    let a_base = a_flat * (m * k_a);
    let b_base = b_flat * (k_a * n);
    let a_slice = &a_data[a_base..a_base + m * k_a];
    let b_slice = &b_data[b_base..b_base + k_a * n];

    gemm_2d_tile(a_slice, b_slice, out_slice, k_a, n, 0, m);
}

#[inline(always)]
fn gemm_2d_tile(
    a_slice: &[f64],
    b_slice: &[f64],
    out_chunk: &mut [f64],
    k_dim: usize,
    n_dim: usize,
    row_start: usize,
    row_end: usize,
) {
    const BLOCK: usize = 64;
    let local_m = row_end - row_start;

    for i0 in (0..local_m).step_by(BLOCK) {
        let i_max = (i0 + BLOCK).min(local_m);
        for k0 in (0..k_dim).step_by(BLOCK) {
            let k_max = (k0 + BLOCK).min(k_dim);
            for j0 in (0..n_dim).step_by(BLOCK) {
                let j_max = (j0 + BLOCK).min(n_dim);

                for i in i0..i_max {
                    let global_row = row_start + i;
                    let a_row = &a_slice[global_row * k_dim..];
                    let out_row = &mut out_chunk[i * n_dim..];

                    for k in k0..k_max {
                        let a_val = a_row[k];
                        let b_row = &b_slice[k * n_dim..];

                        let mut j = j0;
                        while j + 4 <= j_max {
                            out_row[j] += a_val * b_row[j];
                            out_row[j + 1] += a_val * b_row[j + 1];
                            out_row[j + 2] += a_val * b_row[j + 2];
                            out_row[j + 3] += a_val * b_row[j + 3];
                            j += 4;
                        }
                        while j < j_max {
                            out_row[j] += a_val * b_row[j];
                            j += 1;
                        }
                    }
                }
            }
        }
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

/// Matrix-vector product: (M, N) x (N,) -> (M,).
pub fn matvec(a: &Tensor, v: &Tensor) -> Tensor {
    assert!(a.ndim() == 2, "matvec requires a 2D matrix, got {}D", a.ndim());
    assert!(v.ndim() == 1, "matvec requires a 1D vector, got {}D", v.ndim());
    let (m, k) = (a.shape()[0], a.shape()[1]);
    assert_eq!(v.shape()[0], k, "Matrix columns must match vector length");
    let mut out_data = vec![0.0; m];
    for i in 0..m {
        let row = &a.data()[i * k..(i + 1) * k];
        out_data[i] = row.iter().zip(v.data().iter()).map(|(&x, &y)| x * y).sum();
    }
    Tensor::from_vec(out_data, vec![m])
}

/// Vector dot product alias.
pub fn vdot(a: &Tensor, b: &Tensor) -> f64 {
    dot(a, b)
}

/// Outer product of two 1D vectors: out[i, j] = a[i] * b[j].
pub fn outer(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 1 && b.ndim() == 1, "outer requires 1D vectors");
    let (m, n) = (a.numel(), b.numel());
    let a_data = a.data();
    let b_data = b.data();
    let mut out_data = Vec::with_capacity(m * n);
    for &ai in a_data {
        for &bj in b_data {
            out_data.push(ai * bj);
        }
    }
    Tensor::from_vec(out_data, vec![m, n])
}

/// Cosine similarity of two equal-shaped tensors along a dimension: dot / (||a|| * ||b||).
/// Zero-norm pairs yield 0.0.
pub fn cosine_similarity(a: &Tensor, b: &Tensor, dim: usize) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "cosine_similarity requires equal shapes");
    assert!(dim < a.ndim(), "cosine_similarity: dim out of bounds");
    let dim_size = a.shape()[dim];
    assert!(dim_size > 0, "cosine_similarity: empty reduction dim");

    let out_shape: Vec<usize> = a.shape().iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect();
    let out_numel: usize = out_shape.iter().product();
    let mut dot_acc = vec![0.0; out_numel];
    let mut aa_acc = vec![0.0; out_numel];
    let mut bb_acc = vec![0.0; out_numel];

    let mut coords = vec![0usize; a.ndim()];
    for flat in 0..a.numel() {
        let mut stripped = coords.clone();
        stripped.remove(dim);
        let mut key = 0usize;
        for (&c, &s) in stripped.iter().zip(out_shape.iter()) {
            key = key * s + c;
        }
        let (x, y) = (a.get(flat), b.get(flat));
        dot_acc[key] += x * y;
        aa_acc[key] += x * x;
        bb_acc[key] += y * y;

        for d in (0..a.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < a.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let mut out_data = Vec::with_capacity(out_numel);
    for i in 0..out_numel {
        let denom = (aa_acc[i] * bb_acc[i]).sqrt();
        out_data.push(if denom > 0.0 { dot_acc[i] / denom } else { 0.0 });
    }
    Tensor::from_vec(out_data, out_shape)
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
    let a_data = a.data();
    let b_data = b.data();
    let mut out_data = vec![0.0f64; ma * mb * na * nb];
    let out_cols = na * nb;

    for ia in 0..ma {
        for ja in 0..na {
            let a_val = a_data[ia * na + ja];
            for ib in 0..mb {
                let row_base = (ia * mb + ib) * out_cols + ja * nb;
                let b_row = &b_data[ib * nb..(ib + 1) * nb];
                for (jb, &b_val) in b_row.iter().enumerate() {
                    out_data[row_base + jb] = a_val * b_val;
                }
            }
        }
    }
    Tensor::from_vec(out_data, vec![ma * mb, na * nb])
}

/// Multi-axis tensor contraction (tensordot).
pub fn tensordot(a: &Tensor, b: &Tensor, axes: (&[usize], &[usize])) -> Tensor {
    let (axes_a, axes_b) = axes;
    assert_eq!(axes_a.len(), axes_b.len(), "Contraction axes count must match");

    let a_shape = a.shape();
    let b_shape = b.shape();

    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        assert!(ax_a < a.ndim(), "Axis {} out of bounds for tensor a", ax_a);
        assert!(ax_b < b.ndim(), "Axis {} out of bounds for tensor b", ax_b);
        assert_eq!(
            a_shape[ax_a], b_shape[ax_b],
            "Dimension mismatch along contraction axes: {} != {}",
            a_shape[ax_a], b_shape[ax_b]
        );
    }

    let free_a: Vec<usize> = (0..a.ndim()).filter(|d| !axes_a.contains(d)).collect();
    let free_b: Vec<usize> = (0..b.ndim()).filter(|d| !axes_b.contains(d)).collect();

    let k_size: usize = axes_a.iter().map(|&d| a_shape[d]).product();
    let m_size: usize = free_a.iter().map(|&d| a_shape[d]).product();
    let n_size: usize = free_b.iter().map(|&d| b_shape[d]).product();

    let mut perm_a = free_a.clone();
    perm_a.extend_from_slice(axes_a);

    let mut perm_b = axes_b.to_vec();
    perm_b.extend_from_slice(&free_b);

    let a_perm = a.permute(&perm_a).reshape(vec![m_size.max(1), k_size.max(1)]);
    let b_perm = b.permute(&perm_b).reshape(vec![k_size.max(1), n_size.max(1)]);

    let c_mat = matmul(&a_perm, &b_perm);

    let mut out_shape: Vec<usize> = free_a.iter().map(|&d| a_shape[d]).collect();
    out_shape.extend(free_b.iter().map(|&d| b_shape[d]));

    if out_shape.is_empty() {
        c_mat.reshape(vec![1])
    } else {
        c_mat.reshape(out_shape)
    }
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
    fn test_matvec() {
        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let v = Tensor::from_slice(&[5.0, 6.0], vec![2]);
        let r = matvec(&m, &v);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.to_vec(), vec![17.0, 39.0]);

        // Rectangular matrix
        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = Tensor::from_slice(&[1.0, 1.0, 1.0], vec![3]);
        let r = matvec(&m, &v);
        assert_eq!(r.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        // Along last dim of a [2,3] tensor: two vectors vs two vectors
        let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![2, 3]);
        let s = cosine_similarity(&a, &b, 1);
        assert_eq!(s.shape(), &[2]);
        assert!((s.get(0) - 1.0).abs() < 1e-9);
        assert!((s.get(1) - 0.0).abs() < 1e-9);

        // Along dim 0 of a [2,3] tensor
        let a = Tensor::from_slice(&[1.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]);
        let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let s = cosine_similarity(&a, &b, 0);
        assert_eq!(s.shape(), &[3]);
        // col 0: (1*1 + 0*0) / (1 * 1) = 1
        assert!((s.get(0) - 1.0).abs() < 1e-9);
        // col 1: (0*0 + 0*1) / 0 -> zero-norm guard => 0
        assert!((s.get(1) - 0.0).abs() < 1e-9);
        // col 2: (2*0 + 0*0) / 0 -> 0
        assert!((s.get(2) - 0.0).abs() < 1e-9);

        // 45-degree angle: cos = 1/sqrt(2)
        let a = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let b = Tensor::from_slice(&[1.0, 0.0], vec![2]);
        let s = cosine_similarity(&a, &b, 0);
        assert!((s.get(0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
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
    fn test_empty_tensor_arithmetic() {
        let a = Tensor::from_slice(&[], vec![0]);
        let b = Tensor::from_slice(&[], vec![0]);
        let c = add(&a, &b);
        assert_eq!(c.shape(), &[0]);
        assert_eq!(c.numel(), 0);

        let d = mul_scalar(&a, 5.0);
        assert_eq!(d.shape(), &[0]);
    }

    #[test]
    fn test_nan_inf_propagation() {
        let a = Tensor::from_slice(&[f64::NAN, 1.0, f64::INFINITY], vec![3]);
        let b = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        
        let c_add = add(&a, &b);
        assert!(c_add.get(0).is_nan());
        assert_eq!(c_add.get(1), 4.0);
        assert!(c_add.get(2).is_infinite());

        let c_div = div(&b, &a);
        assert!(c_div.get(0).is_nan());
        assert_eq!(c_div.get(1), 3.0);
        assert_eq!(c_div.get(2), 0.0); // 4.0 / Inf = 0.0
    }

    #[test]
    fn test_non_contiguous_view_arithmetic() {
        // [2, 3] matrix transposed to [3, 2]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let a_t = a.transpose(0, 1); // shape [3, 2]
        let b = Tensor::ones(vec![3, 2]);
        let c = add(&a_t, &b);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.get_2d(0, 0), 2.0); // 1.0 + 1.0
        assert_eq!(c.get_2d(0, 1), 5.0); // 4.0 + 1.0
        assert_eq!(c.get_2d(1, 0), 3.0); // 2.0 + 1.0
        assert_eq!(c.get_2d(2, 1), 7.0); // 6.0 + 1.0
    }

    #[test]
    fn test_broadcast_arithmetic_edge_cases() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3, 1]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let c = add(&a, &b);
        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(c.get_2d(0, 0), 11.0);
        assert_eq!(c.get_2d(2, 3), 34.0);
    }
}
