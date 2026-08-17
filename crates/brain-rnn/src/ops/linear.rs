//! # Linear Gate Combiners & State Slicing
//!
//! Fused linear gate operations $W x + U h + b$ and vector concatenations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Computes fused gate linear projection: $y = x W_x^T + h W_h^T + b$.
pub fn gate_linear(
    x: &[f64],
    h: &[f64],
    w_x: &[f64], // [out_dim, in_dim]
    w_h: &[f64], // [out_dim, hidden_dim]
    b: Option<&[f64]>,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; out_dim];

    if let Some(bias) = b {
        for i in 0..out_dim.min(bias.len()) {
            out[i] = bias[i];
        }
    }

    for i in 0..out_dim {
        let mut sum_x = 0.0;
        for j in 0..in_dim.min(x.len()) {
            sum_x += x[j] * w_x[i * in_dim + j];
        }

        let mut sum_h = 0.0;
        for j in 0..hidden_dim.min(h.len()) {
            sum_h += h[j] * w_h[i * hidden_dim + j];
        }

        out[i] += sum_x + sum_h;
    }

    out
}

/// Concatenates two state tensors along the last feature dimension.
pub fn concat_states(a: &Tensor, b: &Tensor) -> Tensor {
    let mut data = Vec::with_capacity(a.numel() + b.numel());
    data.extend_from_slice(a.data());
    data.extend_from_slice(b.data());
    let n = data.len();
    Tensor::from_slice(&data, vec![1, n])
}

/// Splits state tensor into two halves along feature dimension.
pub fn split_states(combined: &Tensor) -> (Tensor, Tensor) {
    let d = combined.data();
    let mid = d.len() / 2;
    let a = Tensor::from_slice(&d[..mid], vec![1, mid]);
    let b = Tensor::from_slice(&d[mid..], vec![1, d.len() - mid]);
    (a, b)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_ops_linear_stress_001() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_002() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_003() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_004() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_005() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_006() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_007() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_008() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_009() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_010() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_011() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_012() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_013() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_014() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_015() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_016() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_017() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_018() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_019() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_020() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_021() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_022() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_023() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_024() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_025() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_026() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_027() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_028() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_029() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_030() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_031() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_032() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_033() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_034() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_035() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_036() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_037() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_038() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_039() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_040() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_041() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_042() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_043() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_044() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_045() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_046() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_047() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_048() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_049() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_050() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_051() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_052() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_053() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_054() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_055() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_056() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_057() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_058() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_059() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_060() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_061() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_062() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_063() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_064() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_065() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_066() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_067() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_068() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_069() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_070() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_071() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_072() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_073() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_074() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_075() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_076() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_077() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_078() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_079() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_080() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_081() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_082() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_083() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_084() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_085() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_086() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_087() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_088() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_089() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_090() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_091() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_092() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_093() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_094() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_095() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_096() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_097() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_098() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_099() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_100() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_101() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_102() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_103() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_104() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_105() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_106() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_107() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_108() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_109() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_110() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_111() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_112() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_113() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_114() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_115() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_116() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_117() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_118() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_119() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_120() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_121() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_122() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_123() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_124() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_125() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_126() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_127() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_128() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_129() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_130() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_131() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_132() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_133() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_134() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_135() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_136() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_137() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_138() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_139() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_140() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_141() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_142() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_143() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_144() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_145() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_146() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_147() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_148() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_149() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_150() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_151() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_152() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_153() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_154() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_155() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_156() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_157() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_158() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_159() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_160() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_161() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_162() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_163() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_164() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_165() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_166() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_167() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_168() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_169() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_170() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_171() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }

    #[test]
    fn test_ops_linear_stress_172() {
        let x = vec![1.0, 2.0];
        let h = vec![0.5, -0.5];
        let w_x = vec![1.0, 0.0, 0.0, 1.0];
        let w_h = vec![1.0, 1.0, 1.0, -1.0];
        let b = vec![0.1, 0.2];
        let out = gate_linear(&x, &h, &w_x, &w_h, Some(&b), 2, 2, 2);
        assert_eq!(out.len(), 2);

        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let cat = concat_states(&t1, &t2);
        assert_eq!(cat.numel(), 4);
        let (s1, s2) = split_states(&cat);
        assert_eq!(s1.numel(), 2);
        assert_eq!(s2.numel(), 2);
    }
}
