//! Element-wise comparisons, boolean masking logic, sorting, and top-k selection.
//!
//! This module provides comparison predicates (`eq`, `ne`, `lt`, `le`, `gt`, `ge`),
//! boolean bitwise logic (`logical_and`, `logical_or`, `logical_not`), sorting, and top-k extractions.

use crate::tensor::Tensor;

/// Element-wise equality: a == b (returns 1.0 for true, 0.0 for false).
pub fn eq_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if (x - y).abs() < 1e-15 { 1.0 } else { 0.0 })
}

/// Element-wise inequality: a != b.
pub fn ne_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if (x - y).abs() >= 1e-15 { 1.0 } else { 0.0 })
}

/// Element-wise less than: a < b.
pub fn lt_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x < y { 1.0 } else { 0.0 })
}

/// Element-wise less than or equal: a <= b.
pub fn le_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x <= y { 1.0 } else { 0.0 })
}

/// Element-wise greater than: a > b.
pub fn gt_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x > y { 1.0 } else { 0.0 })
}

/// Element-wise greater than or equal: a >= b.
pub fn ge_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x >= y { 1.0 } else { 0.0 })
}

/// Returns the k largest elements and their indices along dimension `dim`.
pub fn topk(input: &Tensor, k: usize, dim: usize, largest: bool) -> (Tensor, Vec<usize>) {
    assert!(
        dim < input.ndim(),
        "topk: dim {} out of bounds for tensor of rank {}",
        dim,
        input.ndim()
    );
    let shape = input.shape();
    let d_len = shape[dim];
    let k_actual = k.min(d_len);

    let mut out_shape = shape.to_vec();
    out_shape[dim] = k_actual;

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let in_data = input.to_vec();
    let mut out_data = vec![0.0; outer_size * k_actual * inner_size];
    let mut out_indices = vec![0usize; outer_size * k_actual * inner_size];

    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut pairs = Vec::with_capacity(d_len);
            for d in 0..d_len {
                let in_idx = (o * d_len + d) * inner_size + i;
                pairs.push((d, in_data[in_idx]));
            }

            if largest {
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            for ki in 0..k_actual {
                let out_idx = (o * k_actual + ki) * inner_size + i;
                out_data[out_idx] = pairs[ki].1;
                out_indices[out_idx] = pairs[ki].0;
            }
        }
    }

    (Tensor::new(out_data, out_shape), out_indices)
}

/// Sorts the tensor along a dimension.
pub fn sort(input: &Tensor, dim: usize, descending: bool) -> (Tensor, Vec<usize>) {
    let d_len = input.shape().get(dim).copied().unwrap_or(0);
    topk(input, d_len, dim, descending)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparisons() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 2.0, 2.0], vec![3]);
        assert_eq!(lt_tensor(&a, &b).data(), &[1.0, 0.0, 0.0]);
        assert_eq!(eq_tensor(&a, &b).data(), &[0.0, 1.0, 0.0]);
        assert_eq!(gt_tensor(&a, &b).data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_topk_and_sort() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
        let (v, idx) = topk(&a, 3, 0, true);
        assert_eq!(v.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(idx, vec![4, 2, 0]);
    }

    #[test]
    fn test_topk_and_sort_edge_cases() {
        // Multi-dim topk along dim 0 vs dim 1
        let a = Tensor::from_slice(&[10.0, 2.0, 30.0, 4.0, 5.0, 60.0], vec![2, 3]);
        let (v0, i0) = topk(&a, 1, 0, true);
        assert_eq!(v0.shape(), &[1, 3]);
        assert_eq!(v0.to_vec(), vec![10.0, 5.0, 60.0]);

        let (v1, i1) = topk(&a, 2, 1, true);
        assert_eq!(v1.shape(), &[2, 2]);
        assert_eq!(v1.to_vec(), vec![30.0, 10.0, 60.0, 5.0]);

        // Empty tensor
        let empty = Tensor::from_slice(&[], vec![0, 3]);
        let (ev, ei) = topk(&empty, 1, 1, true);
        assert_eq!(ev.shape(), &[0, 1]);
    }
}
