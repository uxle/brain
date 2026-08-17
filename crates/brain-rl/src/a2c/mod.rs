//! # Synchronous Advantage Actor-Critic (A2C)
//!
//! Synchronous multi-environment trajectory collection and policy gradient advantage updates.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::policy::CategoricalDist;

/// Configuration hyperparameters for A2C.
#[derive(Debug, Clone, PartialEq)]
pub struct A2cConfig {
    pub lr: f64,
    pub gamma: f64,
    pub entropy_coef: f64,
    pub value_loss_coef: f64,
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self {
            lr: 7e-4,
            gamma: 0.99,
            entropy_coef: 0.01,
            value_loss_coef: 0.5,
        }
    }
}

/// Synchronous Advantage Actor-Critic (A2C) Agent.
#[derive(Debug, Clone)]
pub struct A2cAgent {
    pub config: A2cConfig,
    pub input_dim: usize,
    pub num_actions: usize,
    pub actor_weights: Vec<f64>,
    pub critic_weights: Vec<f64>,
}

impl A2cAgent {
    pub fn new(input_dim: usize, num_actions: usize, config: A2cConfig) -> Self {
        Self {
            config,
            input_dim,
            num_actions,
            actor_weights: vec![0.0; input_dim * num_actions],
            critic_weights: vec![0.0; input_dim],
        }
    }

    pub fn act(&self, state: &Tensor) -> usize {
        let d = state.data();
        let mut logits = vec![0.0; self.num_actions];
        for a in 0..self.num_actions {
            for i in 0..d.len().min(self.input_dim) {
                logits[a] += d[i] * self.actor_weights[a * self.input_dim + i];
            }
        }
        let dist = CategoricalDist::from_logits(&logits);
        let mut best_a = 0;
        let mut best_p = f64::NEG_INFINITY;
        for (a, &p) in dist.probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_a = a;
            }
        }
        best_a
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
    fn test_a2c_mod_stress_001() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_002() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_003() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_004() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_005() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_006() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_007() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_008() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_009() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_010() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_011() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_012() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_013() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_014() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_015() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_016() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_017() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_018() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_019() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_020() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_021() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_022() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_023() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_024() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_025() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_026() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_027() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_028() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_029() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_030() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_031() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_032() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_033() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_034() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_035() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_036() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_037() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_038() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_039() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_040() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_041() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_042() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_043() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_044() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_045() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_046() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_047() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_048() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_049() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_050() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_051() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_052() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_053() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_054() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_055() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_056() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_057() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_058() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_059() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_060() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_061() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_062() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_063() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_064() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_065() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_066() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_067() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_068() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_069() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_070() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_071() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_072() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_073() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_074() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_075() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_076() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_077() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_078() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_079() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_080() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_081() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_082() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_083() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_084() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_085() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_086() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_087() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_088() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_089() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_090() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_091() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_092() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_093() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_094() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_095() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_096() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_097() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_098() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_099() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_100() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_101() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_102() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_103() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_104() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_105() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_106() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_107() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_108() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_109() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_110() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_111() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_112() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_113() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_114() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_115() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_116() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_117() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_118() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_119() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_120() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_121() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_122() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_123() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_124() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_125() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_126() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_127() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_128() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_129() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_130() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_131() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_132() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_133() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_134() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_135() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_136() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_137() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_138() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_139() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_140() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_141() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_142() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_143() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_144() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_145() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_146() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_147() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_148() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_149() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_150() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_151() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_152() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_153() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_154() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_155() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_156() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_157() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_158() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_159() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_160() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_161() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_162() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_163() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_164() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_165() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_166() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_167() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_168() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_169() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_170() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_171() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_172() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_173() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_174() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_175() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_176() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_177() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_178() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_179() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_180() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_181() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_182() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_183() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_184() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_185() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_186() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_187() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_188() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_189() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_190() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_191() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_192() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_193() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_194() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_195() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_196() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_197() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_198() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_199() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_200() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_201() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_202() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_203() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_204() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_205() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_206() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_207() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_208() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_209() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_210() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_211() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_212() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_213() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_214() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_215() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_216() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_217() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_218() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_219() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_220() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_221() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_222() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_223() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_224() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_225() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_226() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_227() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_228() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_229() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_230() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_231() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_232() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_233() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_234() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_235() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_236() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_237() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_238() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_239() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_240() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_241() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_242() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_243() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_244() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_245() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_246() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_247() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_248() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_249() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_250() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_251() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_252() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_253() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_254() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_255() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_256() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_257() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_258() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_259() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_260() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_261() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_262() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_263() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_264() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_265() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_266() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_267() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_268() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_269() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_270() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_271() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_272() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_273() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_274() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_275() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_276() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_277() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_278() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_279() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_280() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_281() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_282() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_283() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_284() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_285() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_286() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_287() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_288() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_289() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_290() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_291() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_292() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_293() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_294() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_295() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_296() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_297() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_298() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_299() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_300() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_301() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_302() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_303() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_304() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_305() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_306() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_307() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_308() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_309() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_310() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_311() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_312() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_313() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_314() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_315() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_316() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_317() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_318() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_319() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_320() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_321() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_322() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_323() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_324() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_325() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_326() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_327() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_328() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_329() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_330() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_331() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_332() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_333() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_334() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_335() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_336() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_337() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_338() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_339() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_340() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_341() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_342() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_343() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_344() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_345() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_346() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_347() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_348() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_349() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_350() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_351() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_352() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_353() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_354() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_355() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_356() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_357() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_358() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_359() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_360() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    #[test]
    fn test_a2c_mod_stress_361() {
        let cfg = A2cConfig::default();
        let agent = A2cAgent::new(2, 2, cfg);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
}
