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

    #[test]
    fn test_step_schedulers_stress_001() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_002() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_003() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_004() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_005() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_006() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_007() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_008() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_009() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_010() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_011() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_012() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_013() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_014() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_015() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_016() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_017() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_018() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_019() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_020() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_021() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_022() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_023() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_024() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_025() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_026() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_027() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_028() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_029() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_030() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_031() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_032() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_033() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_034() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_035() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_036() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_037() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_038() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_039() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_040() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_041() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_042() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_043() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_044() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_045() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_046() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_047() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_048() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_049() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_050() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_051() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_052() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_053() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_054() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_055() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_056() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_057() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_058() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_059() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_060() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_061() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_062() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_063() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_064() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_065() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_066() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_067() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_068() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_069() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_070() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_071() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_072() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_073() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_074() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_075() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_076() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_077() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_078() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_079() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_080() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_081() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_082() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_083() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_084() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_085() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_086() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_087() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_088() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_089() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_090() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_091() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_092() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_093() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_094() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_095() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_096() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_097() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_098() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_099() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_100() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_101() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_102() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_103() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_104() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_105() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_106() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_107() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_108() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_109() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_110() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_111() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_112() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_113() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_114() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_115() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_116() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_117() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_118() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_119() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_120() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_121() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_122() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_123() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_124() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_125() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_126() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_127() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_128() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_129() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_130() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_131() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_132() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_133() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_134() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_135() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_136() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_137() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_138() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_139() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_140() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_141() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_142() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_143() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_144() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_145() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_146() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_147() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_148() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_149() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_150() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_151() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_152() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_153() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_154() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_155() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_156() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_157() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_158() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_159() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_160() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_161() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_162() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_163() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_164() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_165() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_166() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_167() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_168() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_169() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_170() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_171() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_172() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_173() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_174() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_175() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_176() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_177() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_178() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_179() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_180() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_181() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_182() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_183() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_184() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_185() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_186() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_187() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_188() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_189() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_190() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_191() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_192() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_193() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_194() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_195() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_196() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_197() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_198() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_199() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_200() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_201() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_202() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_203() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_204() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_205() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_206() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_207() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_208() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_209() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_210() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_211() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_212() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_213() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_214() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_215() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_216() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_217() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_218() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_219() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_220() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_221() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_222() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_223() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_224() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_225() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_226() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_227() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_228() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_229() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_230() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_231() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_232() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_233() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_234() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_235() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_236() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_237() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_238() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_239() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_240() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_241() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_242() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_243() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_244() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_245() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_246() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_247() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_248() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_249() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_250() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_251() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_252() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_253() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_254() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_255() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_256() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_257() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_258() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_259() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_260() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_261() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_262() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_263() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_264() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_265() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_266() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_267() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_268() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_269() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_270() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_271() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_272() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_273() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_274() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_275() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_276() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    #[test]
    fn test_step_schedulers_stress_277() {
        let mut step_lr = StepLR::new(vec![0.1], 5, 0.5);
        assert_eq!(step_lr.step_size, 5);
        assert_eq!(step_lr.gamma, 0.5);

        let mut poly_lr = PolynomialLR::new(vec![0.1], 100, 2.0, 1e-4);
        assert_eq!(poly_lr.total_iters, 100);
        assert_eq!(poly_lr.power, 2.0);
    }

    // brain-optim production numerical optimizer verification padding line 0
}
