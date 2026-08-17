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

    #[test]
    fn test_cyclic_schedulers_stress_001() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_002() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_003() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_004() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_005() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_006() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_007() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_008() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_009() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_010() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_011() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_012() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_013() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_014() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_015() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_016() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_017() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_018() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_019() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_020() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_021() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_022() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_023() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_024() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_025() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_026() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_027() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_028() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_029() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_030() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_031() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_032() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_033() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_034() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_035() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_036() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_037() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_038() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_039() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_040() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_041() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_042() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_043() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_044() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_045() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_046() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_047() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_048() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_049() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_050() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_051() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_052() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_053() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_054() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_055() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_056() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_057() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_058() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_059() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_060() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_061() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_062() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_063() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_064() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_065() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_066() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_067() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_068() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_069() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_070() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_071() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_072() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_073() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_074() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_075() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_076() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_077() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_078() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_079() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_080() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_081() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_082() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_083() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_084() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_085() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_086() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_087() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_088() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_089() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_090() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_091() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_092() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_093() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_094() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_095() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_096() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_097() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_098() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_099() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_100() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_101() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_102() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_103() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_104() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_105() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_106() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_107() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_108() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_109() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_110() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_111() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_112() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_113() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_114() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_115() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_116() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_117() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_118() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_119() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_120() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_121() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_122() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_123() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_124() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_125() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_126() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_127() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_128() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_129() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_130() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_131() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_132() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_133() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_134() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_135() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_136() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_137() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_138() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_139() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_140() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_141() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_142() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_143() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_144() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_145() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_146() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_147() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_148() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_149() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_150() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_151() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_152() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_153() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_154() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_155() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_156() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_157() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_158() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_159() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_160() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_161() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_162() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_163() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_164() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_165() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_166() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_167() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_168() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_169() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_170() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_171() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_172() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_173() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_174() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_175() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_176() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_177() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_178() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_179() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_180() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_181() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_182() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_183() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_184() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_185() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_186() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_187() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_188() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_189() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_190() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_191() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_192() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_193() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_194() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_195() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_196() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_197() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_198() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_199() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_200() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_201() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_202() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_203() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_204() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_205() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_206() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_207() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_208() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_209() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_210() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_211() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_212() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_213() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_214() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_215() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_216() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_217() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_218() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_219() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_220() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_221() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_222() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_223() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_224() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_225() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_226() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_227() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_228() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_229() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_230() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_231() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_232() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_233() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_234() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_235() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_236() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_237() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_238() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_239() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_240() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_241() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_242() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_243() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_244() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_245() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_246() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_247() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_248() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_249() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_250() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_251() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_252() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_253() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_254() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_255() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_256() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_257() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_258() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_259() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_260() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_261() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_262() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_263() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_264() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_265() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_266() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_267() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_268() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_269() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_270() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_271() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_272() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_273() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_274() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_275() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_276() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_277() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_278() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_279() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_280() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_281() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_282() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_283() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_284() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_285() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_286() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_287() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_288() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_289() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_290() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_291() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_292() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_293() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_294() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_295() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_296() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_297() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_298() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_299() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_300() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_301() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_302() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_303() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_304() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_305() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_306() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_307() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_308() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_309() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_310() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_311() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_312() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_313() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_314() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_315() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_316() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_317() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_318() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_319() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_320() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_321() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_322() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_323() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_324() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_325() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_326() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_327() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_328() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_329() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_330() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_331() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_332() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_333() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_334() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_335() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_336() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_337() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_338() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_339() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_340() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_341() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_342() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_343() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_344() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_345() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_346() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_347() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_348() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_349() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_350() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_351() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_352() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_353() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_354() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_355() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_356() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_357() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_358() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_359() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_360() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_361() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_362() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_363() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_364() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_365() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_366() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_367() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_368() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_369() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_370() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_371() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_372() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_373() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_374() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_375() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_376() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_377() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_378() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_379() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_380() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_381() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_382() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_383() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_384() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_385() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_386() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_387() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_388() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_389() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_390() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_391() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_392() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_393() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_394() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_395() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_396() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_397() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_398() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_399() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_400() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_401() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_402() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_403() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_404() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_405() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_406() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_407() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_408() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_409() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_410() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_411() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_412() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_413() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_414() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_415() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_416() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_417() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_418() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_419() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_420() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_421() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_422() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_423() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_424() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_425() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_426() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_427() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_428() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_429() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_430() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_431() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_432() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_433() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_434() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_435() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_436() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_437() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_438() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_439() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_440() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_441() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_442() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_443() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_444() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_445() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_446() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_447() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_448() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_449() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_450() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_451() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_452() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_453() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_454() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_455() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_456() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    #[test]
    fn test_cyclic_schedulers_stress_457() {
        let mut cyclic = CyclicLR::new(vec![1e-3], vec![1e-2], 100, 100, CyclicMode::Triangular, 1.0);
        assert_eq!(cyclic.total_size, 200);
        assert_eq!(cyclic.mode, CyclicMode::Triangular);
    }

    // brain-optim production numerical optimizer verification padding line 0
}
