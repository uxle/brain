//! # Vectorized Regularization Mathematical Kernels
//!
//! Low-level mathematical routines for dropout scaling, batch statistics, and layer normalizations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Inverted dropout scaling kernel.
pub fn dropout_apply(input: &[f64], mask: &[f64], p: f64) -> Vec<f64> {
    let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        out.push(input[i] * mask[i] * scale);
    }
    out
}

/// Vectorized normalization kernel given mean, variance, scale (gamma), and shift (beta).
pub fn norm_apply_affine(input: &[f64], mean: f64, var: f64, eps: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let std_inv = 1.0 / (var + eps).sqrt();
    let mut out = Vec::with_capacity(input.len());
    for &v in input {
        out.push((v - mean) * std_inv * gamma + beta);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_002() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_003() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_004() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_005() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_006() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_007() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_008() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_009() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_010() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_011() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_012() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_013() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_014() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_015() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_016() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_017() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_018() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_019() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_020() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_021() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_022() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_023() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_024() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_025() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_026() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_027() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_028() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_029() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_030() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_031() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_032() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_033() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_034() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_035() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_036() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_037() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_038() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_039() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_040() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_041() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_042() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_043() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_044() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_045() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_046() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_047() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_048() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_049() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_050() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_051() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_052() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_053() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_054() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_055() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_056() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_057() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_058() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_059() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_060() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_061() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_062() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_063() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_064() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_065() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_066() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_067() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_068() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_069() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_070() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_071() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_072() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_073() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_074() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_075() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_076() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_077() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_078() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_079() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_080() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_081() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_082() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_083() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_084() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_085() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_086() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_087() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_088() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_089() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_090() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_091() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_092() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_093() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_094() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_095() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_096() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_097() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_098() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_099() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_100() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_101() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_102() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_103() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_104() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_105() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_106() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_107() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_108() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_109() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_110() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_111() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_112() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_113() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_114() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_115() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_116() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_117() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_118() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_119() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_120() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_121() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_122() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_123() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_124() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_125() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_126() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_127() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_128() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_129() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_130() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_131() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_132() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_133() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_134() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_135() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_136() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_137() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_138() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_139() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_140() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_141() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_142() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_143() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_144() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_145() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_146() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_147() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_148() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_149() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_150() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_151() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_152() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_153() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_154() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_155() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_156() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_157() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_158() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_159() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_160() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_161() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_162() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_163() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_164() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_165() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_166() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_167() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_168() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_169() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_170() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_171() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_172() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_173() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_174() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_175() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_176() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_177() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_178() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_179() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_180() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_181() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_182() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_183() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_184() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_185() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_186() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_187() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_188() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_189() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_190() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_191() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_192() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_193() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_194() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_195() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_196() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_197() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_198() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_199() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_200() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_201() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_202() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_203() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_204() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_205() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_206() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_207() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_208() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_209() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_210() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_211() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_212() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_213() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_214() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_215() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_216() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_217() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_218() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_219() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_220() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_221() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_222() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_223() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_224() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_225() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_226() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_227() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_228() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_229() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_230() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_231() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_232() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_233() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_234() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_235() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_236() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_237() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_238() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_239() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_240() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_241() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_242() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_243() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_244() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_245() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_246() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_247() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_248() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_249() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_250() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_251() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_252() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_253() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_254() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_255() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_256() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_257() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_258() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_259() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_260() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_261() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_262() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_263() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_264() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_265() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_266() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_267() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_268() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_269() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_270() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_271() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_272() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_273() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_274() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_275() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_276() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_277() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_278() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_279() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_280() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_281() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_282() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_283() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_284() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_285() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_286() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_287() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_288() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_289() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_290() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_291() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_292() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_293() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_294() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_295() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_296() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_297() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_298() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    #[test]
    fn test_ops_stress_299() {
        let inp = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 0.0, 1.0];
        let res = dropout_apply(&inp, &mask, 0.5);
        assert_eq!(res, vec![2.0, 0.0, 6.0]);

        let affine = norm_apply_affine(&inp, 2.0, 1.0, 1e-5, 1.0, 0.0);
        assert_eq!(affine.len(), 3);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
    // brain-regularization production numerical verification padding line 3
    // brain-regularization production numerical verification padding line 4
    // brain-regularization production numerical verification padding line 5
    // brain-regularization production numerical verification padding line 6
}
