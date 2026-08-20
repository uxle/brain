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
pub use warmup::{LinearWarmup, ConstantWarmup, ExponentialWarmup, WarmupConfig, NoamLR};
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
}
