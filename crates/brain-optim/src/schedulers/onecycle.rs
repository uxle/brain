//! # One Cycle Learning Rate Policy (1cycle)
//!
//! Anneals learning rate and momentum through rapid warm-up and cosine/linear cool-down phases (Leslie Smith).
#![allow(missing_docs)]

use super::LrScheduler;
use crate::optimizer::{OptimResult, Optimizer};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Annealing curvature strategy for 1cycle policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnnealStrategy {
    #[default]
    Cosine,
    Linear,
}

/// Configuration settings for OneCycleLR.
#[derive(Debug, Clone, PartialEq)]
pub struct OneCycleConfig {
    pub max_lr: f64,
    pub total_steps: usize,
    pub pct_start: f64,
    pub anneal_strategy: AnnealStrategy,
    pub div_factor: f64,
    pub final_div_factor: f64,
    pub three_phase: bool,
}

impl Default for OneCycleConfig {
    fn default() -> Self {
        Self {
            max_lr: 1e-2,
            total_steps: 1000,
            pct_start: 0.3,
            anneal_strategy: AnnealStrategy::Cosine,
            div_factor: 25.0,
            final_div_factor: 1e4,
            three_phase: false,
        }
    }
}

/// OneCycleLR Scheduler.
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    pub max_lrs: Vec<f64>,
    pub initial_lrs: Vec<f64>,
    pub min_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub total_steps: usize,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub anneal_strategy: AnnealStrategy,
    pub step_count: usize,
}

impl OneCycleLR {
    pub fn new(max_lrs: Vec<f64>, config: OneCycleConfig) -> Self {
        let total_steps = config.total_steps.max(1);
        let step_size_up = (total_steps as f64 * config.pct_start).round() as usize;
        let step_size_down = total_steps - step_size_up;

        let mut initial_lrs = Vec::with_capacity(max_lrs.len());
        let mut min_lrs = Vec::with_capacity(max_lrs.len());
        let mut last_lrs = Vec::with_capacity(max_lrs.len());

        for &max_lr in &max_lrs {
            let init_lr = max_lr / config.div_factor;
            let min_lr = init_lr / config.final_div_factor;
            initial_lrs.push(init_lr);
            min_lrs.push(min_lr);
            last_lrs.push(init_lr);
        }

        Self {
            max_lrs,
            initial_lrs,
            min_lrs,
            last_lrs,
            total_steps,
            step_size_up: step_size_up.max(1),
            step_size_down: step_size_down.max(1),
            anneal_strategy: config.anneal_strategy,
            step_count: 0,
        }
    }
}

impl LrScheduler for OneCycleLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;

        for i in 0..self.max_lrs.len() {
            let max_lr = self.max_lrs[i];
            let init_lr = self.initial_lrs[i];
            let min_lr = self.min_lrs[i];

            let new_lr = if self.step_count <= self.step_size_up {
                let progress = self.step_count as f64 / self.step_size_up as f64;
                match self.anneal_strategy {
                    AnnealStrategy::Cosine => {
                        let cos_val = (1.0 + (progress * PI + PI).cos()) * 0.5;
                        init_lr + (max_lr - init_lr) * cos_val
                    }
                    AnnealStrategy::Linear => init_lr + (max_lr - init_lr) * progress,
                }
            } else {
                let down_step = (self.step_count - self.step_size_up) as f64;
                let progress = (down_step / self.step_size_down as f64).min(1.0);
                match self.anneal_strategy {
                    AnnealStrategy::Cosine => {
                        let cos_val = (1.0 + (progress * PI).cos()) * 0.5;
                        min_lr + (max_lr - min_lr) * cos_val
                    }
                    AnnealStrategy::Linear => max_lr - (max_lr - min_lr) * progress,
                }
            };

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
        map.insert("total_steps".to_string(), self.total_steps as f64);
        map.insert("step_size_up".to_string(), self.step_size_up as f64);
        map.insert("step_size_down".to_string(), self.step_size_down as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&tot) = state.get("total_steps") {
            self.total_steps = tot as usize;
        }
        if let Some(&up) = state.get("step_size_up") {
            self.step_size_up = up as usize;
        }
        if let Some(&down) = state.get("step_size_down") {
            self.step_size_down = down as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
