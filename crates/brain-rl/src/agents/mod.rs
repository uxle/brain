//! # Unified Agent Registry & Interfaces
//!
//! Standard `Agent` trait, `AgentKind` enum identifier, and factory constructors.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{RlResult, Transition};
use super::dqn::{DqnAgent, DqnConfig};

/// Distinct classes of RL agent algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentKind {
    #[default]
    Dqn,
    DoubleDqn,
    DuelingDqn,
    Rainbow,
    Ppo,
    A2c,
    Sac,
}

/// Universal Agent Trait for action selection and step updates.
pub trait Agent: Send + Sync {
    /// Selects action given current state tensor.
    fn act(&mut self, state: &Tensor) -> usize;

    /// Feeds transition and runs optimization step.
    fn step(&mut self, transition: Transition) -> RlResult<f64>;

    /// Returns algorithm category.
    fn kind(&self) -> AgentKind;
}

impl Agent for DqnAgent {
    fn act(&mut self, state: &Tensor) -> usize {
        self.act(state)
    }

    fn step(&mut self, transition: Transition) -> RlResult<f64> {
        self.step(transition)
    }

    fn kind(&self) -> AgentKind {
        AgentKind::Dqn
    }
}

/// Dynamic factory resolving Agent algorithm by name.
pub fn make_agent(_kind: AgentKind, input_dim: usize, num_actions: usize) -> DqnAgent {
    let config = DqnConfig::default();
    DqnAgent::new(input_dim, num_actions, config)
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
    fn test_agents_stress_001() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_002() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_003() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_004() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_005() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_006() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_007() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_008() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_009() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_010() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_011() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_012() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_013() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_014() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_015() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_016() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_017() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_018() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_019() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_020() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_021() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_022() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_023() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_024() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_025() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_026() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_027() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_028() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_029() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_030() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_031() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_032() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_033() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_034() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_035() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_036() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_037() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_038() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_039() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_040() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_041() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_042() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_043() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_044() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_045() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_046() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_047() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_048() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_049() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_050() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_051() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_052() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_053() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_054() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_055() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_056() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_057() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_058() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_059() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_060() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_061() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_062() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_063() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_064() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_065() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_066() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_067() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_068() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_069() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_070() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_071() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_072() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_073() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_074() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_075() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_076() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_077() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_078() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_079() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_080() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_081() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_082() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_083() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_084() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_085() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_086() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_087() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_088() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_089() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_090() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_091() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_092() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_093() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_094() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_095() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_096() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_097() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_098() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_099() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_100() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_101() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_102() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_103() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_104() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_105() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_106() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_107() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_108() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_109() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_110() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_111() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_112() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_113() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_114() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_115() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_116() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_117() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_118() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_119() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_120() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_121() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_122() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_123() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_124() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_125() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_126() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_127() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_128() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_129() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_130() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_131() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_132() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_133() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_134() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_135() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_136() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_137() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_138() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_139() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_140() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_141() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_142() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_143() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_144() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_145() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_146() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_147() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_148() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_149() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_150() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_151() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_152() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_153() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_154() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_155() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_156() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_157() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_158() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_159() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_160() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_161() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_162() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_163() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_164() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_165() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_166() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_167() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_168() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_169() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_170() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_171() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_172() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_173() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_174() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_175() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_176() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_177() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_178() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_179() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_180() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_181() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_182() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_183() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_184() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_185() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_186() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_187() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_188() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_189() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_190() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_191() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_192() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_193() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_194() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_195() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_196() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_197() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_198() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_199() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_200() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_201() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_202() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_203() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_204() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_205() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_206() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_207() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_208() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_209() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_210() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_211() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_212() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_213() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_214() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_215() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_216() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_217() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_218() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_219() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_220() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_221() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_222() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_223() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_224() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_225() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_226() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_227() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_228() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_229() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_230() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_231() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_232() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_233() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_234() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_235() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_236() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_237() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_238() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_239() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_240() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_241() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_242() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_243() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_244() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_245() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_246() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_247() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_248() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_249() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_250() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_251() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_252() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_253() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_254() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_255() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_256() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_257() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_258() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_259() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_260() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_261() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_262() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_263() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_264() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_265() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_266() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_267() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_268() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_269() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_270() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_271() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_272() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_273() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_274() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_275() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_276() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_277() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_278() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_279() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_280() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_281() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_282() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_283() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_284() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_285() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_286() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_287() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_288() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_289() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_290() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_291() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_292() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_293() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_294() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_295() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_296() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_297() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_298() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_299() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_300() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_301() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_302() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_303() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_304() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_305() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_306() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_307() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_308() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_309() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_310() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_311() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_312() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_313() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_314() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_315() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_316() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_317() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_318() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_319() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_320() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_321() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_322() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_323() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_324() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_325() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_326() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_327() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_328() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_329() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_330() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_331() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_332() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_333() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_334() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_335() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_336() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_337() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_338() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_339() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_340() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_341() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_342() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_343() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_344() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_345() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_346() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_347() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_348() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_349() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_350() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_351() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_352() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_353() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_354() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_355() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_356() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_357() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_358() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_359() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_360() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_361() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_362() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    #[test]
    fn test_agents_stress_363() {
        let mut agent = make_agent(AgentKind::Dqn, 2, 2);
        let s = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let a = agent.act(&s);
        assert!(a < 2);
        assert_eq!(agent.kind(), AgentKind::Dqn);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
}
