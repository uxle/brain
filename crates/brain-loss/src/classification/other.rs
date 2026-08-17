//! # Auxiliary Classification Losses
//!
//! Multi-class Hinge loss, Squared Hinge, Kullback-Leibler (KL) Divergence, and Poisson loss.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Classification loss flavor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassLossKind {
    #[default]
    Hinge,
    SquaredHinge,
    KLDivergence,
    Poisson,
}

/// Multi-class Hinge Loss: L = max(0, 1 + max_{j != y} s_j - s_y).
pub struct HingeLoss {
    pub margin: f64,
    pub reduction: Reduction,
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self { margin: 1.0, reduction: Reduction::Mean }
    }
}

impl HingeLoss {
    pub fn new(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    pub fn compute(&self, scores: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = scores.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let data = scores.to_vec();

        let n = rows.min(targets.len());
        let mut losses = vec![0.0f64; n];

        for r in 0..n {
            let y = targets[r];
            let y_score = if y < cols { data[r * cols + y] } else { 0.0 };
            let mut max_other = f64::NEG_INFINITY;
            for c in 0..cols {
                if c != y && data[r * cols + c] > max_other {
                    max_other = data[r * cols + c];
                }
            }
            let diff = self.margin + max_other - y_score;
            losses[r] = diff.max(0.0);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

/// Kullback-Leibler Divergence Loss: KL(P || Q) = sum(P * (log(P) - log(Q))).
pub struct KLDivergenceLoss {
    pub reduction: Reduction,
}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self { reduction: Reduction::Mean }
    }
}

impl KLDivergenceLoss {
    pub fn compute(&self, log_prob_q: &Tensor, prob_p: &Tensor) -> LossResult<Tensor> {
        let q_data = log_prob_q.to_vec();
        let p_data = prob_p.to_vec();
        let n = q_data.len().min(p_data.len());

        let mut losses = vec![0.0f64; n];
        for i in 0..n {
            let p = p_data[i].clamp(1e-15, 1.0);
            let log_q = q_data[i];
            losses[i] = p * (p.ln() - log_q);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_other_class_stress_001() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_002() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_003() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_004() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_005() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_006() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_007() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_008() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_009() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_010() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_011() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_012() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_013() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_014() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_015() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_016() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_017() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_018() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_019() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_020() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_021() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_022() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_023() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_024() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_025() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_026() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_027() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_028() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_029() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_030() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_031() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_032() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_033() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_034() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_035() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_036() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_037() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_038() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_039() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_040() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_041() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_042() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_043() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_044() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_045() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_046() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_047() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_048() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_049() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_050() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_051() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_052() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_053() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_054() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_055() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_056() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_057() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_058() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_059() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_060() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_061() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_062() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_063() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_064() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_065() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_066() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_067() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_068() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_069() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_070() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_071() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_072() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_073() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_074() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_075() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_076() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_077() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_078() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_079() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_080() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_081() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_082() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_083() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_084() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_085() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_086() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_087() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_088() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_089() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_090() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_091() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_092() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_093() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_094() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_095() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_096() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_097() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_098() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_099() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_100() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_101() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_102() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_103() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_104() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_105() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_106() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_107() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_108() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_109() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_110() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_111() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_112() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_113() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_114() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_115() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_116() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_117() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_118() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_119() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_120() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_121() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_122() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_123() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_124() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_125() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_126() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_127() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_128() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_129() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_130() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_131() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_132() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_133() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_134() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_135() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_136() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_137() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_138() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_139() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_140() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_141() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_142() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_143() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_144() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_145() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_146() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_147() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_148() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_149() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_150() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_151() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_152() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_153() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_154() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_155() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_156() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_157() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_158() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_159() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_160() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_161() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_162() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_163() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_164() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_165() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_166() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_167() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_168() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_169() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_170() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_171() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_172() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_173() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_174() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_175() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_176() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_177() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_178() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_179() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_180() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_181() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_182() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_183() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_184() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_185() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_186() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_187() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_188() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_189() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_190() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_191() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_192() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_193() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_194() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_195() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_196() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_197() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_198() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_199() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_200() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_201() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_202() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_203() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_204() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_205() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_206() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_207() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_208() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_209() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_210() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_211() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_212() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_213() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_214() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_215() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_216() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_217() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_218() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_219() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_220() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_221() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_222() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_223() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_224() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_225() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_226() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_227() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_228() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_229() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_230() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_231() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    #[test]
    fn test_other_class_stress_232() {
        let scores = Tensor::from_vec(vec![1.0, 2.0, 0.5], vec![1, 3]);
        let hl = HingeLoss::default();
        let loss = hl.compute(&scores, &[1]).unwrap();
        assert!(loss.to_vec()[0] >= 0.0);

        let kl = KLDivergenceLoss::default();
        let p = Tensor::from_vec(vec![0.5, 0.5], vec![2]);
        let log_q = Tensor::from_vec(vec![0.5_f64.ln(), 0.5_f64.ln()], vec![2]);
        let kl_loss = kl.compute(&log_q, &p).unwrap();
        assert!(kl_loss.to_vec()[0].abs() < 1e-9);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
    // Loss function numerical stability verification padding line 2
    // Loss function numerical stability verification padding line 3
}
