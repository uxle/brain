//! # Reduce LR On Plateau
//!
//! Reduce learning rate when a metric has stopped improving.
#![allow(missing_docs)]

use super::LrScheduler;
use crate::optimizer::{OptimResult, Optimizer};
use std::collections::HashMap;

/// Optimization mode for ReduceLROnPlateau.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlateauMode {
    #[default]
    Min,
    Max,
}

/// Threshold evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThresholdMode {
    #[default]
    Rel,
    Abs,
}

/// Configuration settings for plateau scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateauConfig {
    pub mode: PlateauMode,
    pub factor: f64,
    pub patience: usize,
    pub threshold: f64,
    pub threshold_mode: ThresholdMode,
    pub cooldown: usize,
    pub min_lr: f64,
    pub eps: f64,
}

impl Default for PlateauConfig {
    fn default() -> Self {
        Self {
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
        }
    }
}

/// ReduceLROnPlateau Scheduler.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    pub base_lrs: Vec<f64>,
    pub last_lrs: Vec<f64>,
    pub config: PlateauConfig,
    pub best: f64,
    pub num_bad_epochs: usize,
    pub cooldown_counter: usize,
    pub step_count: usize,
}

impl ReduceLROnPlateau {
    pub fn new(base_lrs: Vec<f64>, config: PlateauConfig) -> Self {
        let last_lrs = base_lrs.clone();
        let best = match config.mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        Self {
            base_lrs,
            last_lrs,
            config,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            step_count: 0,
        }
    }

    /// Primary stepping method providing the scalar evaluation metric.
    pub fn step_metric(
        &mut self,
        optimizer: &mut dyn Optimizer,
        metric: f64,
    ) -> OptimResult<Vec<f64>> {
        self.step_count += 1;

        if self.is_better(metric, self.best) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.config.patience {
            self.cooldown_counter = self.config.cooldown;
            self.num_bad_epochs = 0;

            for i in 0..self.last_lrs.len() {
                let new_lr = (self.last_lrs[i] * self.config.factor).max(self.config.min_lr);
                if self.last_lrs[i] - new_lr > self.config.eps {
                    self.last_lrs[i] = new_lr;
                    let _ = optimizer.set_group_lr(i, new_lr);
                }
            }
        }

        Ok(self.last_lrs.clone())
    }

    fn is_better(&self, current: f64, best: f64) -> bool {
        match self.config.mode {
            PlateauMode::Min => match self.config.threshold_mode {
                ThresholdMode::Rel => current < best * (1.0 - self.config.threshold),
                ThresholdMode::Abs => current < best - self.config.threshold,
            },
            PlateauMode::Max => match self.config.threshold_mode {
                ThresholdMode::Rel => current > best * (1.0 + self.config.threshold),
                ThresholdMode::Abs => current > best + self.config.threshold,
            },
        }
    }
}

impl LrScheduler for ReduceLROnPlateau {
    fn step(&mut self, optimizer: &mut dyn Optimizer) -> OptimResult<Vec<f64>> {
        // Fallback without explicit metric assumes neutral step
        self.step_metric(optimizer, self.best)
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
        map.insert("best".to_string(), self.best);
        map.insert("num_bad_epochs".to_string(), self.num_bad_epochs as f64);
        map.insert("cooldown_counter".to_string(), self.cooldown_counter as f64);
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, f64>) -> OptimResult<()> {
        if let Some(&s) = state.get("step_count") {
            self.step_count = s as usize;
        }
        if let Some(&b) = state.get("best") {
            self.best = b;
        }
        if let Some(&nb) = state.get("num_bad_epochs") {
            self.num_bad_epochs = nb as usize;
        }
        if let Some(&cd) = state.get("cooldown_counter") {
            self.cooldown_counter = cd as usize;
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
