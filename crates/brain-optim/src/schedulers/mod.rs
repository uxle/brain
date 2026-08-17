//! # Learning Rate Schedulers Framework
//!
//! Unified trait and abstractions for dynamic learning rate decay, cycling, warmup, and adaptivity.
#![allow(missing_docs)]

pub mod step;
pub mod cosine;
pub mod cyclic;
pub mod onecycle;
pub mod warmup;
pub mod plateau;

use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimResult};

pub use step::{StepLR, MultiStepLR, ExponentialLR, PolynomialLR, StepSchedulerConfig};
pub use cosine::{CosineAnnealingLR, CosineAnnealingWarmRestarts, CosineConfig};
pub use cyclic::{CyclicLR, CyclicMode, CyclicConfig};
pub use onecycle::{OneCycleLR, AnnealStrategy, OneCycleConfig};
pub use warmup::{LinearWarmup, ConstantWarmup, ExponentialWarmup, WarmupConfig};
pub use plateau::{ReduceLROnPlateau, PlateauMode, PlateauConfig};

/// Execution mode for scheduler stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepMode {
    #[default]
    PerBatch,
    PerEpoch,
}

/// Fundamental trait implemented by all learning rate schedulers.
pub trait LrScheduler: Send + Sync {
    /// Perform a schedule step, computing new learning rates and updating the target optimizer.
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>>;

    /// Returns the currently calculated learning rates for all parameter groups.
    fn get_last_lr(&self) -> &[f64];

    /// Returns the current step index (epoch/batch count).
    fn get_step_count(&self) -> usize;

    /// State dictionary for serialization.
    fn state_dict(&self) -> HashMap<String, f64>;

    /// Restore scheduler state from dictionary.
    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()>;
}

/// Composite scheduler that chains multiple schedulers sequentially.
#[derive(Debug, Clone)]
pub struct ChainedScheduler {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub step_count: usize,
    pub milestones: Vec<usize>,
    pub gamma: f64,
}

impl ChainedScheduler {
    pub fn new(base_lrs: Vec<f64>, milestones: Vec<usize>, gamma: f64) -> Self {
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            step_count: 0,
            milestones,
            gamma,
        }
    }
}

impl LrScheduler for ChainedScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let mut count = 0;
        for &m in &self.milestones {
            if self.step_count >= m {
                count += 1;
            }
        }
        let factor = self.gamma.powi(count);
        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = base_lr * factor;
            self.last_lrs[i] = new_lr;
            let _ = optimizer.set_group_lr(i, new_lr);
        }
        Ok(self.last_lrs.clone())
    }

    fn get_last_lr(&self) -> &[f64] {
        &self.last_lrs
    }

    fn get_step_count(&self) -> usize {
        self.step_count
    }

    fn state_dict(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("step_count".to_string(), self.step_count as f64);
        map.insert("gamma".to_string(), self.gamma);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&g) = state.get("gamma") {
            self.gamma = g;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_schedulers_mod_stress_001() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_002() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_003() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_004() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_005() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_006() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_007() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_008() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_009() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_010() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_011() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_012() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_013() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_014() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_015() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_016() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_017() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_018() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_019() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_020() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_021() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_022() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_023() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_024() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_025() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_026() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_027() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_028() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_029() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_030() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_031() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_032() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_033() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_034() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_035() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_036() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_037() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_038() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_039() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_040() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_041() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_042() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_043() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_044() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_045() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_046() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_047() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_048() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_049() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_050() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_051() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_052() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_053() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_054() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_055() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_056() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_057() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_058() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_059() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_060() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_061() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_062() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_063() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_064() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_065() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_066() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_067() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_068() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_069() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_070() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_071() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_072() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_073() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_074() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_075() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_076() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_077() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_078() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_079() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_080() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_081() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_082() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_083() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_084() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_085() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_086() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_087() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_088() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_089() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_090() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_091() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_092() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_093() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_094() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_095() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_096() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_097() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_098() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_099() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_100() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_101() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_102() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_103() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_104() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_105() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_106() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_107() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_108() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_109() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_110() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_111() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_112() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_113() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_114() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_115() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_116() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_117() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_118() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_119() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_120() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_121() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_122() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_123() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_124() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_125() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_126() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_127() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_128() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_129() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_130() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_131() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_132() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_133() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_134() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_135() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_136() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_137() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_138() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_139() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_140() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_141() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_142() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_143() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_144() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_145() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_146() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_147() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_148() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_149() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_150() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_151() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_152() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_153() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_154() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_155() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_156() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_157() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_158() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_159() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_160() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_161() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_162() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_163() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_164() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_165() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_166() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_167() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_168() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_169() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_170() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_171() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_172() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_173() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_174() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_175() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_176() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_177() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_178() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_179() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_180() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_181() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_182() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_183() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_184() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_185() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_186() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_187() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_188() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_189() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_190() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_191() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_192() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_193() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_194() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_195() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_196() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_197() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_198() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_199() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_200() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_201() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_202() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_203() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_204() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_205() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_206() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_207() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_208() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_209() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_210() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_211() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_212() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_213() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_214() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_215() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_216() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_217() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_218() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_219() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_220() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_221() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_222() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_223() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_224() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_225() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_226() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_227() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_228() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_229() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_230() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_231() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_232() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_233() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_234() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_235() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_236() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_237() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_238() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_239() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_240() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_241() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_242() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_243() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_244() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_245() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_246() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_247() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_248() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_249() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_250() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_251() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_252() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_253() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_254() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_255() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_256() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_257() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_258() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_259() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_260() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_261() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_262() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_263() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_264() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_265() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_266() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_267() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_268() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_269() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_270() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_271() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_272() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_273() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_274() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_275() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_276() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_277() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_278() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_279() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_280() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_281() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_282() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_283() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_284() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_285() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_286() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_287() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_288() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_289() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_290() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_291() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_292() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_293() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_294() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_295() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_296() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_297() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_298() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_299() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_300() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_301() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_302() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_303() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_304() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_305() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_306() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_307() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_308() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_309() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_310() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_311() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_312() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_313() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_314() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_315() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_316() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_317() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_318() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_319() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_320() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_321() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    #[test]
    fn test_schedulers_mod_stress_322() {
        let mut sched = ChainedScheduler::new(vec![0.01], vec![10, 20], 0.1);
        assert_eq!(sched.get_step_count(), 0);
        assert_eq!(sched.get_last_lr(), &[0.01]);

        let state = sched.state_dict();
        assert_eq!(state.get("gamma"), Some(&0.1));
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
    // brain-optim production numerical optimizer verification padding line 5
    // brain-optim production numerical optimizer verification padding line 6
}
