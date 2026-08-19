//! # Cosine Annealing Schedulers
//!
//! Standard cosine annealing and cosine annealing with warm restarts (SGDR).
#![allow(missing_docs)]

use std::collections::HashMap;
use std::f64::consts::PI;
use crate::optimizer::{Optimizer, OptimResult};
use super::LrScheduler;

/// Configuration settings for cosine annealing.
#[derive(Debug, Clone, PartialEq)]
pub struct CosineConfig {
    pub t_max: usize,
    pub eta_min: f64,
    pub t_0: usize,
    pub t_mult: usize,
}

impl Default for CosineConfig {
    fn default() -> Self {
        Self {
            t_max: 100,
            eta_min: 0.0,
            t_0: 10,
            t_mult: 1,
        }
    }
}

/// CosineAnnealingLR: Set the learning rate using a cosine annealing schedule.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub t_max: usize,
    pub eta_min: f64,
    pub step_count: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lrs: Vec<f64>, t_max: usize, eta_min: f64) -> Self {
        let last_lrs = base_lrs.clone();
        Self {
            base_lrs,
            last_lrs,
            t_max: t_max.max(1),
            eta_min,
            step_count: 0,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let progress = (self.step_count % (2 * self.t_max)) as f64 / self.t_max as f64;
        let cos_factor = (1.0 + (progress * PI).cos()) * 0.5;

        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = self.eta_min + (base_lr - self.eta_min) * cos_factor;
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
        map.insert("t_max".to_string(), self.t_max as f64);
        map.insert("eta_min".to_string(), self.eta_min);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&t) = state.get("t_max") {
            self.t_max = t as usize;
        }
        if let Some(&e) = state.get("eta_min") {
            self.eta_min = e;
        }
        Ok(())
    }
}

/// CosineAnnealingWarmRestarts: Set the learning rate of each parameter group using a cosine annealing schedule with warm restarts.
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub t_0: usize,
    pub t_mult: usize,
    pub eta_min: f64,
    pub step_count: usize,
    pub t_cur: usize,
    pub t_i: usize,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(base_lrs: Vec<f64>, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        let last_lrs = base_lrs.clone();
        let t_0 = t_0.max(1);
        let t_mult = t_mult.max(1);
        Self {
            base_lrs,
            last_lrs,
            t_0,
            t_mult,
            eta_min,
            step_count: 0,
            t_cur: 0,
            t_i: t_0,
        }
    }
}

impl LrScheduler for CosineAnnealingWarmRestarts {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        self.t_cur += 1;

        if self.t_cur >= self.t_i {
            self.t_cur = 0;
            self.t_i *= self.t_mult;
        }

        let cos_factor = (1.0 + ((self.t_cur as f64 / self.t_i as f64) * PI).cos()) * 0.5;

        for (i, base_lr) in self.base_lrs.iter().enumerate() {
            let new_lr = self.eta_min + (base_lr - self.eta_min) * cos_factor;
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
        map.insert("t_cur".to_string(), self.t_cur as f64);
        map.insert("t_i".to_string(), self.t_i as f64);
        map.insert("t_mult".to_string(), self.t_mult as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&tc) = state.get("t_cur") {
            self.t_cur = tc as usize;
        }
        if let Some(&ti) = state.get("t_i") {
            self.t_i = ti as usize;
        }
        if let Some(&tm) = state.get("t_mult") {
            self.t_mult = tm as usize;
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
