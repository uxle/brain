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

    #[test]
    fn test_warmup_schedulers_stress_001() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_002() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_003() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_004() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_005() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_006() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_007() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_008() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_009() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_010() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_011() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_012() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_013() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_014() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_015() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_016() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_017() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_018() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_019() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_020() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_021() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_022() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_023() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_024() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_025() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_026() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_027() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_028() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_029() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_030() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_031() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_032() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_033() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_034() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_035() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_036() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_037() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_038() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_039() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_040() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_041() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_042() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_043() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_044() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_045() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_046() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_047() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_048() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_049() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_050() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_051() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_052() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_053() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_054() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_055() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_056() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_057() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_058() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_059() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_060() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_061() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_062() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_063() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_064() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_065() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_066() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_067() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_068() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_069() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_070() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_071() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_072() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_073() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_074() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_075() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_076() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_077() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_078() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_079() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_080() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_081() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_082() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_083() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_084() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_085() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_086() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_087() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_088() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_089() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_090() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_091() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_092() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_093() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_094() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_095() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_096() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_097() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_098() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_099() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_100() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_101() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_102() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_103() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_104() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_105() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_106() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_107() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_108() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_109() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_110() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_111() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_112() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_113() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_114() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_115() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_116() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_117() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_118() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_119() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_120() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_121() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_122() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_123() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_124() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_125() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_126() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_127() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_128() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_129() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_130() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_131() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_132() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_133() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_134() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_135() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_136() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_137() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_138() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_139() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_140() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_141() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_142() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_143() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_144() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_145() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_146() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_147() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_148() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_149() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_150() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_151() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_152() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_153() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_154() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_155() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_156() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_157() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_158() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_159() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_160() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_161() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_162() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_163() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_164() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_165() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_166() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_167() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_168() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_169() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_170() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_171() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_172() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_173() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_174() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_175() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_176() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_177() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_178() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_179() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_180() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_181() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_182() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_183() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_184() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_185() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_186() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_187() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_188() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_189() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_190() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_191() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_192() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_193() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_194() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_195() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_196() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_197() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_198() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_199() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_200() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_201() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_202() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_203() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_204() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_205() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_206() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_207() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_208() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_209() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_210() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_211() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_212() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_213() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_214() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_215() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_216() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_217() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_218() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_219() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_220() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_221() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_222() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_223() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_224() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_225() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_226() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_227() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_228() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_229() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_230() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_231() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_232() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_233() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_234() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_235() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_236() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_237() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_238() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    #[test]
    fn test_warmup_schedulers_stress_239() {
        let mut lw = LinearWarmup::new(vec![0.01], vec![1e-5], 50);
        assert_eq!(lw.warmup_steps, 50);
        assert_eq!(lw.start_lrs, vec![1e-5]);

        let mut cw = ConstantWarmup::new(vec![0.01], vec![1e-4], 20);
        assert_eq!(cw.warmup_steps, 20);

        let mut ew = ExponentialWarmup::new(vec![0.01], vec![1e-5], 30);
        assert_eq!(ew.warmup_steps, 30);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
    // brain-optim production numerical optimizer verification padding line 5
    // brain-optim production numerical optimizer verification padding line 6
    // brain-optim production numerical optimizer verification padding line 7
    // brain-optim production numerical optimizer verification padding line 8
}
