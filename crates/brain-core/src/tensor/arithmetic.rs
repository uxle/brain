//! Arithmetic operations for tensors in the Brain deep learning framework.
//!
//! This module provides element-wise and matrix arithmetic operations including
//! addition, subtraction, multiplication, division, matrix multiplication,
//! dot products, outer products, Kronecker products, tensor contractions,
//! and einsum operations.

use crate::tensor::Tensor;

// =============================================================================
// Element-wise Arithmetic (Tensor x Tensor with Broadcasting)
// =============================================================================

/// Element-wise addition of two tensors with broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x + y)
}

/// Element-wise subtraction of two tensors with broadcasting.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x - y)
}

/// Element-wise multiplication of two tensors with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x * y)
}

/// Element-wise division of two tensors with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x / y)
}

/// Element-wise remainder of two tensors.
pub fn remainder(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x % y)
}

/// Element-wise power: a^b.
pub fn pow_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.powf(y))
}

// =============================================================================
// Element-wise Arithmetic (Tensor x Scalar)
// =============================================================================

/// Adds a scalar to all elements of a tensor.
pub fn add_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v + scalar)
}

/// Subtracts a scalar from all elements of a tensor.
pub fn sub_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v - scalar)
}

/// Subtracts all tensor elements from a scalar: scalar - a.
pub fn rsub_scalar(scalar: f64, a: &Tensor) -> Tensor {
    a.map(|v| scalar - v)
}

/// Multiplies all elements of a tensor by a scalar.
pub fn mul_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v * scalar)
}

/// Divides all elements of a tensor by a scalar.
pub fn div_scalar(a: &Tensor, scalar: f64) -> Tensor {
    a.map(|v| v / scalar)
}

/// Divides a scalar by all tensor elements: scalar / a.
pub fn rdiv_scalar(scalar: f64, a: &Tensor) -> Tensor {
    a.map(|v| scalar / v)
}

/// Raises all elements to a scalar power: a^p.
pub fn pow_scalar(a: &Tensor, p: f64) -> Tensor {
    a.map(|v| v.powf(p))
}

// =============================================================================
// Matrix Multiplication
// =============================================================================

/// Performs 2D matrix multiplication: C = A @ B.
///
/// # Panics
///
/// Panics if the inner dimensions of A and B don't match:
/// A must be (M, K) and B must be (K, N).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 2 && b.ndim() == 2,
        "matmul requires 2D tensors, got {}D and {}D", a.ndim(), b.ndim());
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];
    assert_eq!(k, b.shape()[1], "Inner dimensions must match for matmul: {} vs {}", k, b.shape()[1]);
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.get(i * k + p) * b.get(p * n + j);
            }
            data[i * n + j] = sum;
        }
    }
    Tensor::new(data, vec![m, n])
}

/// Performs batched matrix multiplication for 3D tensors.
///
/// A: (batch, M, K), B: (batch, K, N) -> C: (batch, M, N)
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() == 3 && b.ndim() == 3,
        "batch_matmul requires 3D tensors");
    let batch = a.shape()[0];
    assert_eq!(batch, b.shape()[0], "Batch sizes must match");
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[1];
    assert_eq!(k, b.shape()[2], "Inner dimensions must match");
    let mut data = vec![0.0; batch * m * n];
    for b_idx in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a_val = a.get(b_idx * m * k + i * k + p);
                    let b_val = b.get(b_idx * k * n + p * n + j);
                    sum += a_val * b_val;
                }
                data[b_idx * m * n + i * n + j] = sum;
            }
        }
    }
    Tensor::new(data, vec![batch, m, n])
}

/// Dot product of two 1D tensors (vectors).
pub fn dot(a: &Tensor, b: &Tensor) -> f64 {
    assert!(a.is_vector() && b.is_vector(), "dot requires 1D tensors");
    assert_eq!(a.numel(), b.numel(), "Vector lengths must match");
    let mut sum = 0.0;
    for i in 0..a.numel() {
        sum += a.get(i) * b.get(i);
    }
    sum
}

/// Outer product of two vectors.
pub fn outer(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.is_vector() && b.is_vector(), "outer requires 1D tensors");
    let m = a.numel();
    let n = b.numel();
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] = a.get(i) * b.get(j);
        }
    }
    Tensor::new(data, vec![m, n])
}

/// Vector dot product (same as dot).
pub fn vecdot(a: &Tensor, b: &Tensor) -> f64 {
    dot(a, b)
}

/// Complex conjugate dot product of two vectors.
/// For real values, this is the same as the regular dot product.
pub fn vdot(a: &Tensor, b: &Tensor) -> f64 {
    // For real tensors, vdot = conjugate(a) . b = a . b
    dot(a, b)
}

/// Cross product of two 3D vectors.
pub fn cross(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.is_vector() && b.is_vector(), "cross requires 1D tensors");
    assert_eq!(a.numel(), 3, "cross requires 3D vectors");
    assert_eq!(b.numel(), 3, "cross requires 3D vectors");
    let ax = a.get(0); let ay = a.get(1); let az = a.get(2);
    let bx = b.get(0); let by = b.get(1); let bz = b.get(2);
    let data = vec![
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ];
    Tensor::new(data, vec![3])
}

// =============================================================================
// Kronecker Product
// =============================================================================

/// Computes the Kronecker product of two 2D tensors (matrices).
pub fn kron(a: &Tensor, b: &Tensor) -> Tensor {
    let (ma, na) = (a.shape()[0], a.shape()[1]);
    let (mb, nb) = (b.shape()[0], b.shape()[1]);
    let mut data = vec![0.0; ma * mb * na * nb];
    for i in 0..ma {
        for j in 0..na {
            let a_val = a.get(i * na + j);
            for p in 0..mb {
                for q in 0..nb {
                    let out_i = i * mb + p;
                    let out_j = j * nb + q;
                    let out_idx = out_i * (na * nb) + out_j;
                    data[out_idx] = a_val * b.get(p * nb + q);
                }
            }
        }
    }
    Tensor::new(data, vec![ma * mb, na * nb])
}

// =============================================================================
// Tensor Dot (General Contraction)
// =============================================================================

/// General tensor contraction along specified axes.
///
/// `axes` specifies pairs of axes to contract: (axis_a, axis_b).
/// The contracted axes are summed over the product of corresponding elements.
pub fn tensordot(a: &Tensor, b: &Tensor, axes: &[(usize, usize)]) -> Tensor {
    let mut a_shape = a.shape().to_vec();
    let mut b_shape = b.shape().to_vec();

    // Compute contracted sizes
    let mut contracted_a = Vec::new();
    let mut contracted_b = Vec::new();
    let mut contract_sizes = Vec::new();
    for &(ia, ib) in axes {
        contracted_a.push(ia);
        contracted_b.push(ib);
        contract_sizes.push(a_shape[ia]);
        assert_eq!(a_shape[ia], b_shape[ib],
            "Contracted dimensions must have equal size: {} vs {}", a_shape[ia], b_shape[ib]);
    }

    // Compute output shape (remaining axes from a, then remaining from b)
    let mut out_shape = Vec::new();
    for (i, &d) in a_shape.iter().enumerate() {
        if !contracted_a.contains(&i) { out_shape.push(d); }
    }
    for (i, &d) in b_shape.iter().enumerate() {
        if !contracted_b.contains(&i) { out_shape.push(d); }
    }

    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_numel];

    // Simple but correct implementation: iterate over all output elements
    // and sum over contracted axes
    let total_contract: usize = contract_sizes.iter().product();
    let mut out_idx = vec![0usize; out_shape.len()];

    for out_flat in 0..out_numel {
        let mut sum = 0.0;

        // Iterate over contracted indices
        let mut contract_idx = vec![0usize; axes.len()];
        for _ in 0..total_contract {
            // Build full index for a
            let mut a_full = Vec::new();
            let mut ai = 0;
            for d in 0..a_shape.len() {
                if let Some(pos) = contracted_a.iter().position(|&x| x == d) {
                    a_full.push(contract_idx[pos]);
                } else {
                    a_full.push(out_idx[ai]);
                    ai += 1;
                }
            }
            // Build full index for b
            let mut b_full = Vec::new();
            let mut bi = 0;
            for d in 0..b_shape.len() {
                if let Some(pos) = contracted_b.iter().position(|&x| x == d) {
                    b_full.push(contract_idx[pos]);
                } else {
                    b_full.push(out_idx[out_shape.len() - b_shape.len() + b_shape.len() - contracted_b.len() + bi]);
                    bi += 1;
                }
            }

            let a_flat = a.get_index(&a_full);
            let b_flat = b.get_index(&b_full);
            sum += a_flat * b_flat;

            // Increment contract_idx
            let mut carry = true;
            for c in (0..contract_idx.len()).rev() {
                if carry {
                    contract_idx[c] += 1;
                    if contract_idx[c] >= contract_sizes[c] { contract_idx[c] = 0; } else { carry = false; }
                }
            }
        }

        data[out_flat] = sum;

        // Increment out_idx
        let mut carry = true;
        for i in (0..out_shape.len()).rev() {
            if carry {
                out_idx[i] += 1;
                if out_idx[i] >= out_shape[i] { out_idx[i] = 0; } else { carry = false; }
            }
        }
    }

    Tensor::new(data, out_shape)
}

// =============================================================================
// Einsum Operations
// =============================================================================

/// Parses and executes an einsum expression.
///
/// Supported patterns:
/// - "ii->i": diagonal extraction
/// - "ij,jk->ik": matrix multiplication
/// - "ij,jk->ik": matrix multiplication (same)
/// - "bij,bjk->bik": batched matrix multiplication
/// - "ij->ji": matrix transpose
/// - "i,i->": dot product (scalar)
/// - "ij->": sum of all elements
/// - "i->": 1D identity (copy)
/// - "ij,kj->ik": matrix multiply with transposed B
pub fn einsum(expression: &str, tensors: &[&Tensor]) -> Tensor {
    let (inputs, output) = parse_einsum(expression);
    assert_eq!(inputs.len(), tensors.len(),
        "Einsum expression expects {} tensors but got {}", inputs.len(), tensors.len());

    // Match common patterns for efficiency
    match expression {
        "ii->i" => {
            let a = tensors[0];
            let n = a.shape()[0];
            let data: Vec<f64> = (0..n).map(|i| a.get(i * n + i)).collect();
            return Tensor::new(data, vec![n]);
        }
        "ij,jk->ik" | "ij,jk->ik" => {
            return matmul(tensors[0], tensors[1]);
        }
        "ij,kj->ik" => {
            let b_t = tensors[1].transpose();
            return matmul(tensors[0], &b_t);
        }
        "ij->ji" => {
            return tensors[0].transpose();
        }
        "i,i->" => {
            let result = dot(tensors[0], tensors[1]);
            return Tensor::scalar(result);
        }
        "ij->" => {
            let sum = tensors[0].reduce(0.0, |a, b| a + b);
            return Tensor::scalar(sum);
        }
        "bij,bjk->bik" => {
            return batch_matmul(tensors[0], tensors[1]);
        }
        "i->" => {
            let sum = tensors[0].reduce(0.0, |a, b| a + b);
            return Tensor::scalar(sum);
        }
        _ => {
            // General einsum implementation
            return general_einsum(&inputs, &output, tensors);
        }
    }
}

/// Parses an einsum expression into input subscripts and output subscripts.
fn parse_einsum(expression: &str) -> (Vec<Vec<char>>, Vec<char>) {
    let parts: Vec<&str> = expression.split("->").collect();
    let output = if parts.len() > 1 {
        parts[1].chars().collect()
    } else {
        // Implicit output: sorted unique indices
        let input_part = parts[0];
        let mut seen = std::collections::HashSet::new();
        let mut output = Vec::new();
        for c in input_part.chars() {
            if c != ',' && !seen.contains(&c) {
                seen.insert(c);
                output.push(c);
            }
        }
        output
    };
    let input_strs: Vec<&str> = parts[0].split(',').collect();
    let inputs: Vec<Vec<char>> = input_strs.iter().map(|s| s.chars().collect()).collect();
    (inputs, output)
}

/// General-purpose einsum implementation.
fn general_einsum(inputs: &[Vec<char>], output: &[char], tensors: &[&Tensor]) -> Tensor {
    // Determine the size of each index
    let mut index_sizes = std::collections::HashMap::new();
    for (input_sub, tensor) in inputs.iter().zip(tensors.iter()) {
        for (i, &c) in input_sub.iter().enumerate() {
            let size = tensor.shape()[i];
            index_sizes.entry(c).or_insert(size);
            assert_eq!(*index_sizes.get(&c).unwrap(), size,
                "Inconsistent sizes for index '{}': {} vs {}", c, index_sizes.get(&c).unwrap(), size);
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = output.iter().map(|c| *index_sizes.get(c).unwrap()).collect();
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_numel];

    // Find all indices that appear in inputs but not in output (summed indices)
    let summed_indices: Vec<char> = index_sizes.keys()
        .filter(|c| !output.contains(c))
        .cloned()
        .collect();

    // Compute total iterations for summed indices
    let total_sum: usize = if summed_indices.is_empty() { 1 }
        else { summed_indices.iter().map(|c| *index_sizes.get(c).unwrap()).product() };

    // Iterate over output
    let mut out_multi = vec![0usize; out_shape.len()];
    for out_flat in 0..out_numel {
        let mut sum = 0.0;

        // Iterate over summed indices
        let mut sum_multi = vec![0usize; summed_indices.len()];
        for _ in 0..total_sum {
            // Build full index mapping
            let mut index_values = std::collections::HashMap::new();

            // Map output indices
            for (i, &c) in output.iter().enumerate() {
                index_values.insert(c, out_multi[i]);
            }

            // Map summed indices
            for (i, c) in summed_indices.iter().enumerate() {
                index_values.insert(*c, sum_multi[i]);
            }

            // Compute product of all tensor elements at these indices
            let mut product = 1.0;
            for (input_sub, tensor) in inputs.iter().zip(tensors.iter()) {
                let mut multi_idx = Vec::new();
                for &c in input_sub {
                    multi_idx.push(*index_values.get(&c).unwrap());
                }
                product *= tensor.get_index(&multi_idx);
            }
            sum += product;

            // Increment sum_multi
            let mut carry = true;
            for i in (0..sum_multi.len()).rev() {
                if carry {
                    sum_multi[i] += 1;
                    let size = index_sizes[&summed_indices[i]];
                    if sum_multi[i] >= size { sum_multi[i] = 0; } else { carry = false; }
                }
            }
        }

        data[out_flat] = sum;

        // Increment out_multi
        let mut carry = true;
        for i in (0..out_shape.len()).rev() {
            if carry {
                out_multi[i] += 1;
                if out_multi[i] >= out_shape[i] { out_multi[i] = 0; } else { carry = false; }
            }
        }
    }

    if out_shape.is_empty() {
        Tensor::scalar(data[0])
    } else {
        Tensor::new(data, out_shape)
    }
}

// =============================================================================
// Additional Arithmetic Operations
// =============================================================================

/// Computes the absolute difference between two tensors element-wise.
pub fn abs_diff(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| (x - y).abs())
}

/// Element-wise minimum of two tensors.
pub fn minimum(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.min(y))
}

/// Element-wise maximum of two tensors.
pub fn maximum(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.max(y))
}

/// Clamps tensor elements to [min, max] range.
pub fn clamp(a: &Tensor, min: f64, max: f64) -> Tensor {
    a.map(|v| v.clamp(min, max))
}

/// Fused multiply-add: a * b + c.
pub fn fma(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    a.map2(b, |x, y| x * y).map2(c, |x, y| x + y)
}

/// Linear interpolation between two tensors: out = a + t * (b - a).
pub fn lerp(a: &Tensor, b: &Tensor, t: f64) -> Tensor {
    a.map2(b, |x, y| x + t * (y - x))
}

/// Element-wise reciprocal: 1/x.
pub fn reciprocal(a: &Tensor) -> Tensor {
    a.map(|v| 1.0 / v)
}

/// Element-wise square: x^2.
pub fn square(a: &Tensor) -> Tensor {
    a.map(|v| v * v)
}

/// Element-wise cube: x^3.
pub fn cube(a: &Tensor) -> Tensor {
    a.map(|v| v * v * v)
}

/// Element-wise square root of sum of squares: sqrt(a^2 + b^2).
pub fn hypot(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.hypot(y))
}

/// Element-wise nextafter: next representable value towards y.
pub fn nextafter(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| {
        if x == y { return y; }
        if x.is_nan() || y.is_nan() { return f64::NAN; }
        let diff = (y - x).abs();
        if diff == 0.0 { return y; }
        if y > x { x + diff * f64::EPSILON } else { x - diff * f64::EPSILON }
    })
}

/// Copies the sign of b to a: |a| * sign(b).
pub fn copysign(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.copysign(y))
}

/// Element-wise fmod: a % b (same sign as a).
pub fn fmod_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x % y)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        let c = add(&a, &b);
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 7.0);
        assert_eq!(c.get(2), 9.0);
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::from_slice(&[5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = sub(&a, &b);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(2), 6.0);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], vec![3]);
        let c = mul(&a, &b);
        assert_eq!(c.get(0), 10.0);
        assert_eq!(c.get(1), 18.0);
        assert_eq!(c.get(2), 28.0);
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 4.0, 5.0], vec![3]);
        let c = div(&a, &b);
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 5.0);
        assert_eq!(c.get(2), 6.0);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[10.0, 20.0], vec![2]);
        let c = add(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(0), 11.0);
        assert_eq!(c.get(1), 22.0);
        assert_eq!(c.get(2), 13.0);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = add_scalar(&a, 10.0);
        assert_eq!(c.get(0), 11.0);
        assert_eq!(c.get(1), 12.0);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let c = mul_scalar(&a, 3.0);
        assert_eq!(c.get(0), 6.0);
        assert_eq!(c.get(1), 9.0);
        assert_eq!(c.get(2), 12.0);
    }

    #[test]
    fn test_sub_scalar() {
        let a = Tensor::from_slice(&[10.0, 20.0], vec![2]);
        let c = sub_scalar(&a, 3.0);
        assert_eq!(c.get(0), 7.0);
    }

    #[test]
    fn test_rsub_scalar() {
        let a = Tensor::from_slice(&[3.0, 5.0], vec![2]);
        let c = rsub_scalar(10.0, &a);
        assert_eq!(c.get(0), 7.0);
        assert_eq!(c.get(1), 5.0);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::from_slice(&[20.0, 30.0], vec![2]);
        let c = div_scalar(&a, 5.0);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 6.0);
    }

    #[test]
    fn test_rdiv_scalar() {
        let a = Tensor::from_slice(&[2.0, 4.0], vec![2]);
        let c = rdiv_scalar(1.0, &a);
        assert_eq!(c.get(0), 0.5);
        assert_eq!(c.get(1), 0.25);
    }

    #[test]
    fn test_pow_scalar() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let c = pow_scalar(&a, 2.0);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 9.0);
        assert_eq!(c.get(2), 16.0);
    }

    #[test]
    fn test_matmul_basic() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get_index(&[0, 0]), 19.0);
        assert_eq!(c.get_index(&[0, 1]), 22.0);
        assert_eq!(c.get_index(&[1, 0]), 43.0);
        assert_eq!(c.get_index(&[1, 1]), 50.0);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let eye = Tensor::identity(3);
        let c = matmul(&a, &eye);
        for i in 0..9 { assert!((c.get(i) - a.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_matmul_rectangular() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get_index(&[0, 0]), 58.0);
        assert_eq!(c.get_index(&[1, 1]), 142.0);
    }

    #[test]
    fn test_batch_matmul() {
        let a = Tensor::identity(2).unsqueeze(0).expand(vec![3, 2, 2]);
        let b = Tensor::identity(2).unsqueeze(0).expand(vec![3, 2, 2]);
        let c = batch_matmul(&a, &b);
        assert_eq!(c.shape(), &[3, 2, 2]);
        assert_eq!(c.get_index(&[0, 0, 0]), 1.0);
        assert_eq!(c.get_index(&[1, 1, 1]), 1.0);
    }

    #[test]
    fn test_dot() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        assert_eq!(dot(&a, &b), 32.0);
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = Tensor::from_slice(&[1.0, 0.0], vec![2]);
        let b = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn test_outer() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 4.0, 5.0], vec![3]);
        let c = outer(&a, &b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.get_index(&[0, 0]), 3.0);
        assert_eq!(c.get_index(&[1, 2]), 10.0);
    }

    #[test]
    fn test_cross() {
        let x = Tensor::from_slice(&[1.0, 0.0, 0.0], vec![3]);
        let y = Tensor::from_slice(&[0.0, 1.0, 0.0], vec![3]);
        let z = cross(&x, &y);
        assert_eq!(z.get(0), 0.0);
        assert_eq!(z.get(1), 0.0);
        assert_eq!(z.get(2), 1.0);
    }

    #[test]
    fn test_kron() {
        let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = kron(&a, &b);
        assert_eq!(c.shape(), &[4, 4]);
        assert_eq!(c.get_index(&[0, 0]), 1.0);
        assert_eq!(c.get_index(&[2, 2]), 4.0);
    }

    #[test]
    fn test_tensordot_matrix_mul() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = tensordot(&a, &b, &[(1, 0)]);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get_index(&[0, 0]), 19.0);
    }

    #[test]
    fn test_einsum_matmul() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = einsum("ij,jk->ik", &[&a, &b]);
        assert_eq!(c.get_index(&[0, 0]), 19.0);
        assert_eq!(c.get_index(&[1, 1]), 50.0);
    }

    #[test]
    fn test_einsum_transpose() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = einsum("ij->ji", &[&a]);
        assert_eq!(c.get_index(&[0, 1]), 2.0);
        assert_eq!(c.get_index(&[1, 0]), 3.0);
    }

    #[test]
    fn test_einsum_diagonal() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let d = einsum("ii->i", &[&a]);
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.get(0), 1.0);
        assert_eq!(d.get(1), 5.0);
        assert_eq!(d.get(2), 9.0);
    }

    #[test]
    fn test_einsum_dot_product() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        let c = einsum("i,i->", &[&a, &b]);
        assert!(c.is_scalar());
        assert_eq!(c.get(0), 32.0);
    }

    #[test]
    fn test_einsum_sum() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = einsum("ij->", &[&a]);
        assert!(c.is_scalar());
        assert_eq!(c.get(0), 10.0);
    }

    #[test]
    fn test_einsum_batch_matmul() {
        let a_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let a = Tensor::new(a_data, vec![3, 2, 2]);
        let b = Tensor::new(b_data, vec![3, 2, 2]);
        let c = einsum("bij,bjk->bik", &[&a, &b]);
        assert_eq!(c.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_minimum() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 5.0, 4.0], vec![3]);
        let c = minimum(&a, &b);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 1.0);
        assert_eq!(c.get(2), 4.0);
    }

    #[test]
    fn test_maximum() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 5.0, 4.0], vec![3]);
        let c = maximum(&a, &b);
        assert_eq!(c.get(0), 3.0);
        assert_eq!(c.get(1), 5.0);
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::from_slice(&[-1.0, 0.5, 2.0], vec![3]);
        let c = clamp(&a, 0.0, 1.0);
        assert_eq!(c.get(0), 0.0);
        assert_eq!(c.get(1), 0.5);
        assert_eq!(c.get(2), 1.0);
    }

    #[test]
    fn test_fma() {
        let a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[4.0, 5.0], vec![2]);
        let c = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let d = fma(&a, &b, &c);
        assert_eq!(d.get(0), 9.0);
        assert_eq!(d.get(1), 16.0);
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let b = Tensor::from_slice(&[10.0, 20.0], vec![2]);
        let c = lerp(&a, &b, 0.5);
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 10.0);
    }

    #[test]
    fn test_reciprocal() {
        let a = Tensor::from_slice(&[2.0, 4.0, 5.0], vec![3]);
        let c = reciprocal(&a);
        assert_eq!(c.get(0), 0.5);
        assert_eq!(c.get(1), 0.25);
    }

    #[test]
    fn test_square() {
        let a = Tensor::from_slice(&[2.0, 3.0, -4.0], vec![3]);
        let c = square(&a);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 9.0);
        assert_eq!(c.get(2), 16.0);
    }

    #[test]
    fn test_cube() {
        let a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let c = cube(&a);
        assert_eq!(c.get(0), 8.0);
        assert_eq!(c.get(1), 27.0);
    }

    #[test]
    fn test_hypot() {
        let a = Tensor::from_slice(&[3.0, 5.0], vec![2]);
        let b = Tensor::from_slice(&[4.0, 12.0], vec![2]);
        let c = hypot(&a, &b);
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 13.0);
    }

    #[test]
    fn test_copysign() {
        let a = Tensor::from_slice(&[3.0, -3.0], vec![2]);
        let b = Tensor::from_slice(&[-1.0, 1.0], vec![2]);
        let c = copysign(&a, &b);
        assert_eq!(c.get(0), -3.0);
        assert_eq!(c.get(1), 3.0);
    }

    #[test]
    fn test_pow_tensors() {
        let a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 2.0], vec![2]);
        let c = pow_tensors(&a, &b);
        assert_eq!(c.get(0), 8.0);
        assert_eq!(c.get(1), 9.0);
    }

    #[test]
    fn test_abs_diff() {
        let a = Tensor::from_slice(&[5.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 5.0], vec![2]);
        let c = abs_diff(&a, &b);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 2.0);
    }

    #[test]
    fn test_vecdot() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        assert_eq!(vecdot(&a, &b), 32.0);
    }

    #[test]
    fn test_vdot() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        assert_eq!(vdot(&a, &b), 32.0);
    }

    #[test]
    fn test_einsum_general() {
        // Trace: sum of diagonal
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let c = einsum("ii->", &[&a]);
        assert!(c.is_scalar());
        assert_eq!(c.get(0), 15.0);
    }

    #[test]
    fn test_einsum_ij_kj() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = einsum("ij,kj->ik", &[&a, &b]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensordot_trace() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = tensordot(&a, &a, &[(0, 1)]);
        assert_eq!(c.get(0), 7.0); // 1*4 + 2*3 = 10
    }

    #[test]
    fn test_remainder() {
        let a = Tensor::from_slice(&[7.0, 10.0, 13.0], vec![3]);
        let b = Tensor::from_slice(&[3.0, 4.0, 5.0], vec![3]);
        let c = remainder(&a, &b);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 2.0);
        assert_eq!(c.get(2), 3.0);
    }

    #[test]
    fn test_matmul_commutativity_with_identity() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let eye = Tensor::identity(3);
        let c1 = matmul(&a, &eye);
        let c2 = matmul(&eye, &a);
        for i in 0..9 {
            assert!((c1.get(i) - c2.get(i)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_outer_symmetry() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = outer(&a, &a);
        assert_eq!(c.shape(), &[3, 3]);
        assert!(c.get_index(&[0, 1]) == c.get_index(&[1, 0])); // symmetric
    }

    #[test]
    fn test_cross_zero_vectors() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let z = Tensor::zeros(vec![3]);
        let c = cross(&x, &z);
        assert_eq!(c.get(0), 0.0);
        assert_eq!(c.get(1), 0.0);
        assert_eq!(c.get(2), 0.0);
    }
}
