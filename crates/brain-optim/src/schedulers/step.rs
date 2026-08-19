//! # Step and Polynomial Schedulers
//!
//! StepLR, MultiStepLR, ExponentialLR, and PolynomialLR implementations.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimResult};
use super::LrScheduler;

/// Configuration container for step-based schedulers.
#[derive(Debug, Clone, PartialEq)]
pub struct StepSchedulerConfig {
    pub step_size: usize,
    pub gamma: f64,
    pub milestones: Vec<usize>,
    pub total_iters: usize,
    pub power: f64,
    pub min_lr: f64,
}

impl Default for StepSchedulerConfig {
    fn default() -> Self {
        Self {
            step_size: 10,
            gamma: 0.1,
            milestones: vec![30, 60, 90],
            total_iters: 100,
            power: 1.0,
            min_lr: 0.0,
        }
    }
}

/// StepLR: Decays the learning rate of each parameter group by gamma every step_size epochs.
#[derive(Debug, Clone)]
pub struct StepLR {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub step_size: usize,
    pub gamma: f64,
    pub step_count: usize,
}

impl StepLR {
    pub fn new(base_lrs: Vec<f64>, step_size: usize, gamma: f64) -> Self {
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            step_size: step_size.max(1),
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let factor = self.gamma.powi((self.step_count / self.step_size) as i32);
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
        map.insert("step_size".to_string(), self.step_size as f64);
        map.insert("gamma".to_string(), self.gamma);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&sz) = state.get("step_size") {
            self.step_size = sz as usize;
        }
        if let Some(&g) = state.get("gamma") {
            self.gamma = g;
        }
        Ok(())
    }
}

/// MultiStepLR: Decays the learning rate once the number of epoch reaches one of the milestones.
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub milestones: Vec<usize>,
    pub gamma: f64,
    pub step_count: usize,
}

impl MultiStepLR {
    pub fn new(base_lrs: Vec<f64>, mut milestones: Vec<usize>, gamma: f64) -> Self {
        milestones.sort_unstable();
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            milestones,
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for MultiStepLR {
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

/// ExponentialLR: Decays the learning rate of each parameter group by gamma every step.
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub gamma: f64,
    pub step_count: usize,
}

impl ExponentialLR {
    pub fn new(base_lrs: Vec<f64>, gamma: f64) -> Self {
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let factor = self.gamma.powi(self.step_count as i32);
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

/// PolynomialLR: Decays learning rate using polynomial curve.
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub total_iters: usize,
    pub power: f64,
    pub min_lr: f64,
    pub step_count: usize,
}

impl PolynomialLR {
    pub fn new(base_lrs: Vec<f64>, total_iters: usize, power: f64, min_lr: f64) -> Self {
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            total_iters: total_iters.max(1),
            power,
            min_lr,
            step_count: 0,
        }
    }
}

impl LrScheduler for PolynomialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let progress = (self.step_count as f64 / self.total_iters as f64).min(1.0);
        let factor = (1.0 - progress).powf(self.power);

        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = (base_lr - self.min_lr) * factor + self.min_lr;
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
        map.insert("total_iters".to_string(), self.total_iters as f64);
        map.insert("power".to_string(), self.power);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&t) = state.get("total_iters") {
            self.total_iters = t as usize;
        }
        if let Some(&p) = state.get("power") {
            self.power = p;
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
