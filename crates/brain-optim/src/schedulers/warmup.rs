//! # Warmup Schedulers
//!
//! Linear, constant, and exponential learning rate warmup wrappers preventing divergence during early training.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimResult};
use super::LrScheduler;

/// Configuration parameters for warmup schedules.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmupConfig {
    pub warmup_steps: usize,
    pub warmup_start_lr: f64,
    pub target_lr: f64,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 1000,
            warmup_start_lr: 1e-6,
            target_lr: 1e-3,
        }
    }
}

/// Linear Warmup Scheduler.
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    pub target_lrs: Vec<f64>,
    pub start_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub warmup_steps: usize,
    pub step_count: usize,
}

impl LinearWarmup {
    pub fn new(target_lrs: Vec<f64>, start_lrs: Vec<f64>, warmup_steps: usize) -> Self {
        let last_lrs = start_lrs.clone();
        Self {
            target_lrs,
            start_lrs,
            last_lrs,
            warmup_steps: warmup_steps.max(1),
            step_count: 0,
        }
    }
}

impl LrScheduler for LinearWarmup {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let alpha = (self.step_count as f64 / self.warmup_steps as f64).min(1.0);

        for i in 0..self.target_lrs.len() {
            let start = self.start_lrs.get(i).copied().unwrap_or(0.0);
            let target = self.target_lrs[i];
            let new_lr = start + (target - start) * alpha;
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
        map.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&w) = state.get("warmup_steps") {
            self.warmup_steps = w as usize;
        }
        Ok(())
    }
}

/// Constant Warmup Scheduler (holds learning rate at a lower fixed value before jump).
#[derive(Debug, Clone)]
pub struct ConstantWarmup {
    pub target_lrs: Vec<f64>,
    pub warmup_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub warmup_steps: usize,
    pub step_count: usize,
}

impl ConstantWarmup {
    pub fn new(target_lrs: Vec<f64>, warmup_lrs: Vec<f64>, warmup_steps: usize) -> Self {
        let last_lrs = warmup_lrs.clone();
        Self {
            target_lrs,
            warmup_lrs,
            last_lrs,
            warmup_steps: warmup_steps.max(1),
            step_count: 0,
        }
    }
}

impl LrScheduler for ConstantWarmup {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let is_warming = self.step_count <= self.warmup_steps;

        for i in 0..self.target_lrs.len() {
            let new_lr = if is_warming {
                self.warmup_lrs.get(i).copied().unwrap_or(1e-5)
            } else {
                self.target_lrs[i]
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
        map.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&w) = state.get("warmup_steps") {
            self.warmup_steps = w as usize;
        }
        Ok(())
    }
}

/// Exponential Warmup Scheduler.
#[derive(Debug, Clone)]
pub struct ExponentialWarmup {
    pub target_lrs: Vec<f64>,
    pub start_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub warmup_steps: usize,
    pub step_count: usize,
}

impl ExponentialWarmup {
    pub fn new(target_lrs: Vec<f64>, start_lrs: Vec<f64>, warmup_steps: usize) -> Self {
        let last_lrs = start_lrs.clone();
        Self {
            target_lrs,
            start_lrs,
            last_lrs,
            warmup_steps: warmup_steps.max(1),
            step_count: 0,
        }
    }
}

impl LrScheduler for ExponentialWarmup {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let progress = (self.step_count as f64 / self.warmup_steps as f64).min(1.0);

        for i in 0..self.target_lrs.len() {
            let start = self.start_lrs.get(i).copied().unwrap_or(1e-6).max(1e-12);
            let target = self.target_lrs[i].max(start);
            let log_start = start.ln();
            let log_target = target.ln();
            let new_lr = (log_start + (log_target - log_start) * progress).exp();
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
        map.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&w) = state.get("warmup_steps") {
            self.warmup_steps = w as usize;
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

/// Noam Learning Rate Scheduler (Vaswani et al., 2017):
/// lr = factor * (d_model ^ -0.5) * min(step ^ -0.5, step * (warmup_steps ^ -1.5))
#[derive(Debug, Clone)]
pub struct NoamLR {
    pub d_model: usize,
    pub warmup_steps: usize,
    pub factor: f64,
    pub last_lrs: Vec<f64>,
    pub step_count: usize,
}

impl NoamLR {
    pub fn new(d_model: usize, warmup_steps: usize, factor: f64) -> Self {
        Self {
            d_model: d_model.max(1),
            warmup_steps: warmup_steps.max(1),
            factor,
            last_lrs: vec![0.0],
            step_count: 0,
        }
    }

    pub fn compute_lr(&self, step: usize) -> f64 {
        let s = (step.max(1)) as f64;
        let d = self.d_model as f64;
        let w = self.warmup_steps as f64;
        self.factor * d.powf(-0.5) * (s.powf(-0.5)).min(s * w.powf(-1.5))
    }
}

impl LrScheduler for NoamLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let new_lr = self.compute_lr(self.step_count);
        self.last_lrs = vec![new_lr; optimizer.param_groups().len()];
        for (i, group) in optimizer.param_groups_mut().iter_mut().enumerate() {
            group.lr = self.last_lrs.get(i).copied().unwrap_or(new_lr);
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
        map.insert("d_model".to_string(), self.d_model as f64);
        map.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        map.insert("factor".to_string(), self.factor);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        Ok(())
    }
}
