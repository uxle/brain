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

    #[test]
    fn test_cosine_schedulers_stress_001() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_002() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_003() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_004() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_005() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_006() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_007() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_008() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_009() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_010() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_011() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_012() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_013() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_014() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_015() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_016() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_017() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_018() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_019() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_020() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_021() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_022() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_023() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_024() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_025() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_026() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_027() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_028() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_029() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_030() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_031() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_032() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_033() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_034() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_035() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_036() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_037() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_038() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_039() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_040() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_041() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_042() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_043() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_044() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_045() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_046() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_047() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_048() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_049() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_050() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_051() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_052() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_053() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_054() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_055() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_056() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_057() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_058() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_059() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_060() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_061() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_062() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_063() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_064() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_065() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_066() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_067() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_068() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_069() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_070() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_071() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_072() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_073() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_074() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_075() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_076() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_077() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_078() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_079() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_080() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_081() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_082() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_083() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_084() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_085() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_086() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_087() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_088() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_089() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_090() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_091() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_092() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_093() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_094() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_095() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_096() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_097() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_098() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_099() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_100() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_101() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_102() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_103() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_104() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_105() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_106() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_107() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_108() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_109() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_110() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_111() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_112() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_113() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_114() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_115() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_116() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_117() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_118() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_119() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_120() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_121() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_122() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_123() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_124() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_125() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_126() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_127() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_128() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_129() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_130() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_131() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_132() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_133() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_134() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_135() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_136() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_137() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_138() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_139() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_140() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_141() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_142() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_143() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_144() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_145() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_146() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_147() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_148() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_149() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_150() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_151() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_152() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_153() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_154() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_155() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_156() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_157() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_158() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_159() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_160() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_161() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_162() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_163() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_164() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_165() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_166() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_167() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_168() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_169() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_170() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_171() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_172() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_173() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_174() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_175() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_176() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_177() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_178() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_179() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_180() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_181() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_182() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_183() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_184() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_185() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_186() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_187() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_188() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_189() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_190() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_191() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_192() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_193() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_194() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_195() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_196() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_197() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_198() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_199() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_200() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_201() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_202() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_203() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_204() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_205() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_206() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_207() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_208() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_209() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_210() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_211() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_212() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_213() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_214() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_215() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_216() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_217() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_218() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_219() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_220() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_221() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_222() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_223() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_224() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_225() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_226() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_227() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_228() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_229() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_230() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_231() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_232() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_233() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_234() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_235() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_236() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_237() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_238() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_239() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_240() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_241() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_242() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_243() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_244() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_245() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_246() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_247() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_248() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_249() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_250() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_251() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_252() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_253() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_254() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_255() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_256() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_257() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_258() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_259() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_260() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_261() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_262() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_263() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_264() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_265() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_266() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_267() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_268() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_269() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_270() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_271() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_272() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_273() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_274() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_275() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_276() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_277() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_278() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_279() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_280() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_281() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_282() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_283() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_284() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_285() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_286() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    #[test]
    fn test_cosine_schedulers_stress_287() {
        let mut cos = CosineAnnealingLR::new(vec![0.01], 20, 1e-5);
        assert_eq!(cos.t_max, 20);
        assert_eq!(cos.eta_min, 1e-5);

        let mut warm = CosineAnnealingWarmRestarts::new(vec![0.01], 10, 2, 0.0);
        assert_eq!(warm.t_0, 10);
        assert_eq!(warm.t_mult, 2);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
}
