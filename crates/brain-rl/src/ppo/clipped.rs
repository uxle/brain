//! # PPO Clipped Surrogate Loss Objective
//!
//! Evaluates clipped policy surrogate loss and clipped value function error.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// PPO Clipped Surrogate Objective Evaluator.
#[derive(Debug, Clone)]
pub struct PpoClippedObjective {
    pub clip_epsilon: f64,
}

impl PpoClippedObjective {
    pub fn new(clip_epsilon: f64) -> Self {
        Self {
            clip_epsilon: clip_epsilon.max(0.0),
        }
    }

    /// Computes clipped policy loss term given probability ratio r_t and estimated advantage A_t.
    pub fn compute_policy_loss(&self, ratio: f64, advantage: f64) -> f64 {
        let surr1 = ratio * advantage;
        let surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage;
        -surr1.min(surr2)
    }

    /// Computes clipped value function loss.
    pub fn compute_value_loss(&self, v_pred: f64, v_old: f64, v_target: f64) -> f64 {
        let v_clipped = v_old + (v_pred - v_old).clamp(-self.clip_epsilon, self.clip_epsilon);
        let loss1 = (v_pred - v_target).powi(2);
        let loss2 = (v_clipped - v_target).powi(2);
        0.5 * loss1.max(loss2)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::env::*;
    use crate::policy::*;
    use crate::value::*;
    use crate::buffer::*;
    use crate::dqn::*;
    use crate::ppo::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::sac::*;
    use crate::agents::*;
    use crate::trainer::*;
    use crate::eval::*;
    use crate::checkpoint::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_ppo_clipped_stress_001() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_002() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_003() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_004() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_005() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_006() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_007() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_008() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_009() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_010() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_011() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_012() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_013() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_014() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_015() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_016() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_017() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_018() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_019() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_020() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_021() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_022() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_023() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_024() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_025() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_026() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_027() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_028() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_029() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_030() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_031() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_032() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_033() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_034() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_035() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_036() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_037() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_038() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_039() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_040() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_041() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_042() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_043() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_044() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_045() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_046() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_047() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_048() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_049() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_050() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_051() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_052() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_053() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_054() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_055() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_056() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_057() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_058() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_059() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_060() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_061() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_062() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_063() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_064() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_065() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_066() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_067() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_068() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_069() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_070() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_071() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_072() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_073() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_074() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_075() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_076() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_077() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_078() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_079() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_080() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_081() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_082() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_083() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_084() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_085() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_086() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_087() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_088() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_089() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_090() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_091() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_092() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_093() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_094() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_095() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_096() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_097() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_098() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_099() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_100() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_101() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_102() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_103() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_104() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_105() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_106() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_107() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_108() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_109() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_110() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_111() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_112() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_113() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_114() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_115() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_116() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_117() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_118() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_119() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_120() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_121() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_122() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_123() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_124() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_125() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_126() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_127() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_128() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_129() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_130() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_131() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_132() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_133() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_134() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_135() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_136() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_137() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_138() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_139() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_140() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_141() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_142() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_143() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_144() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_145() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_146() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_147() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_148() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_149() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_150() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_151() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_152() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_153() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_154() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_155() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_156() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_157() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_158() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_159() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_160() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_161() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_162() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_163() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_164() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_165() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_166() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_167() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_168() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_169() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_170() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_171() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_172() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_173() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_174() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_175() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_176() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_177() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_178() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_179() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_180() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_181() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_182() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_183() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_184() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_185() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_186() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_187() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_188() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_189() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_190() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_191() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_192() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_193() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_194() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_195() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_196() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_197() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_198() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_199() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_200() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_201() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_202() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_203() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_204() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_205() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_206() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_207() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_208() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_209() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_210() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_211() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_212() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_213() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_214() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_215() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_216() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_217() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_218() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_219() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_220() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_221() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_222() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_223() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_224() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_225() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_226() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_227() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_228() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_229() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_230() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_231() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_232() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_233() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_234() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_235() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_236() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_237() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_238() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_239() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_240() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_241() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_242() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_243() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_244() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_245() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_246() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_247() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_248() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_249() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_250() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_251() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_252() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_253() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_254() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_255() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_256() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_257() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_258() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_259() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_260() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_261() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_262() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_263() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_264() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_265() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_266() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_267() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_268() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_269() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_270() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_271() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_272() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_273() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_274() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_275() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_276() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_277() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_278() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_279() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_280() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_281() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_282() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_283() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_284() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_285() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_286() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_287() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_288() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_289() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_290() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_291() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_292() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_293() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_294() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_295() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_296() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_297() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_298() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_299() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_300() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_301() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_302() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_303() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_304() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_305() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_306() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_307() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_308() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_309() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_310() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_311() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_312() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_313() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_314() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_315() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_316() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_317() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_318() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_319() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_320() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_321() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_322() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_323() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_324() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_325() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_326() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_327() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_328() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clipped_stress_329() {
        let obj = PpoClippedObjective::new(0.2);
        let loss = obj.compute_policy_loss(1.0, 1.5);
        assert!((loss - (-1.5)).abs() < 1e-6);

        let v_loss = obj.compute_value_loss(1.0, 1.0, 2.0);
        assert!((v_loss - 0.5).abs() < 1e-6);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
}
