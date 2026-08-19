//! # Cyclic Learning Rate Schedulers
//!
//! Cycles the learning rate between boundary thresholds according to triangular or exponential amplitude policies.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::optimizer::{Optimizer, OptimResult};
use super::LrScheduler;

/// Operational amplitude policy for cyclic learning rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CyclicMode {
    #[default]
    Triangular,
    Triangular2,
    ExpRange,
}

/// Configuration settings for CyclicLR.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicConfig {
    pub base_lr: f64,
    pub max_lr: f64,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub mode: CyclicMode,
    pub gamma: f64,
    pub cycle_momentum: bool,
}

impl Default for CyclicConfig {
    fn default() -> Self {
        Self {
            base_lr: 1e-3,
            max_lr: 1e-2,
            step_size_up: 2000,
            step_size_down: 2000,
            mode: CyclicMode::Triangular,
            gamma: 1.0,
            cycle_momentum: false,
        }
    }
}

/// CyclicLR Scheduler.
#[derive(Debug, Clone)]
pub struct CyclicLR {
    pub base_lrs: Vec<f64>,
    pub max_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub total_size: usize,
    pub mode: CyclicMode,
    pub gamma: f64,
    pub step_count: usize,
}

impl CyclicLR {
    pub fn new(base_lrs: Vec<f64>, max_lrs: Vec<f64>, step_size_up: usize, step_size_down: usize, mode: CyclicMode, gamma: f64) -> Self {
        let last_lrs = base_lrs.clone();
        let step_size_up = step_size_up.max(1);
        let step_size_down = step_size_down.max(1);
        let total_size = step_size_up + step_size_down;
        Self {
            base_lrs,
            max_lrs,
            last_lrs,
            step_size_up,
            step_size_down,
            total_size,
            mode,
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for CyclicLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        self.step_count += 1;
        let cycle = (1 + self.step_count / self.total_size) as f64;
        let x = (self.step_count % self.total_size) as f64;

        let scale_factor = match self.mode {
            CyclicMode::Triangular => 1.0,
            CyclicMode::Triangular2 => 1.0 / (2.0f64.powf(cycle - 1.0)),
            CyclicMode::ExpRange => self.gamma.powf(self.step_count as f64),
        };

        let unit_progress = if x <= self.step_size_up as f64 {
            x / self.step_size_up as f64
        } else {
            1.0 - (x - self.step_size_up as f64) / self.step_size_down as f64
        };

        for i in 0..self.base_lrs.len() {
            let base = self.base_lrs[i];
            let max = self.max_lrs.get(i).copied().unwrap_or(base * 10.0);
            let new_lr = base + (max - base) * unit_progress.max(0.0) * scale_factor;
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
        map.insert("step_size_up".to_string(), self.step_size_up as f64);
        map.insert("step_size_down".to_string(), self.step_size_down as f64);
        map.insert("gamma".to_string(), self.gamma);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&up) = state.get("step_size_up") {
            self.step_size_up = up as usize;
        }
        if let Some(&down) = state.get("step_size_down") {
            self.step_size_down = down as usize;
        }
        if let Some(&g) = state.get("gamma") {
            self.gamma = g;
        }
        self.total_size = self.step_size_up + self.step_size_down;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
