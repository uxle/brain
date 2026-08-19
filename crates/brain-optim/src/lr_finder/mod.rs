//! # Learning Rate Range Finder
//!
//! Automated learning rate exploration and steepest descent recommendation engine (Leslie Smith).
#![allow(missing_docs)]

/// Configuration settings for learning rate range test.
#[derive(Debug, Clone, PartialEq)]
pub struct LrFindConfig {
    pub start_lr: f64,
    pub end_lr: f64,
    pub num_iter: usize,
    pub step_mode: LrFindStepMode,
    pub beta: f64,
    pub divergence_threshold: f64,
}

impl Default for LrFindConfig {
    fn default() -> Self {
        Self {
            start_lr: 1e-7,
            end_lr: 10.0,
            num_iter: 100,
            step_mode: LrFindStepMode::Exp,
            beta: 0.98,
            divergence_threshold: 4.0,
        }
    }
}

/// Progression policy across LR range test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LrFindStepMode {
    #[default]
    Exp,
    Linear,
}

/// Result of learning rate range test.
#[derive(Debug, Clone)]
pub struct LrFindResult {
    pub lrs: Vec<f64>,
    pub losses: Vec<f64>,
    pub smoothed_losses: Vec<f64>,
    pub suggested_lr: f64,
    pub min_loss_lr: f64,
}

/// Learning rate finder coordinator.
#[derive(Debug, Clone)]
pub struct LrFinder {
    pub config: LrFindConfig,
    pub lrs: Vec<f64>,
    pub losses: Vec<f64>,
    pub smoothed_losses: Vec<f64>,
    pub best_loss: f64,
    pub current_step: usize,
}

impl LrFinder {
    pub fn new(config: LrFindConfig) -> Self {
        Self {
            config,
            lrs: Vec::new(),
            losses: Vec::new(),
            smoothed_losses: Vec::new(),
            best_loss: f64::INFINITY,
            current_step: 0,
        }
    }

    /// Computes the next scheduled learning rate for step `i`.
    pub fn calculate_lr(&self, step: usize) -> f64 {
        let num_iter = self.config.num_iter.max(1);
        let ratio = (step as f64 / num_iter as f64).min(1.0);
        match self.config.step_mode {
            LrFindStepMode::Exp => {
                self.config.start_lr * (self.config.end_lr / self.config.start_lr).powf(ratio)
            }
            LrFindStepMode::Linear => {
                self.config.start_lr + (self.config.end_lr - self.config.start_lr) * ratio
            }
        }
    }

    /// Records an observed loss value and updates smooth trajectory.
    ///
    /// Returns `true` if training has diverged and test should stop early.
    pub fn record_loss(&mut self, loss: f64) -> bool {
        self.current_step += 1;
        let lr = self.calculate_lr(self.current_step);
        self.lrs.push(lr);
        self.losses.push(loss);

        let smoothed = if self.smoothed_losses.is_empty() {
            loss
        } else {
            let prev = *self.smoothed_losses.last().unwrap();
            self.config.beta * prev + (1.0 - self.config.beta) * loss
        };
        self.smoothed_losses.push(smoothed);

        if smoothed < self.best_loss {
            self.best_loss = smoothed;
        }

        // Divergence check
        if smoothed > self.config.divergence_threshold * self.best_loss {
            return true;
        }

        self.current_step >= self.config.num_iter
    }

    /// Recommends the optimal learning rate based on minimum gradient.
    pub fn get_summary(&self) -> LrFindResult {
        let mut min_grad = f64::INFINITY;
        let mut suggested_lr = self.config.start_lr;
        let mut min_loss = f64::INFINITY;
        let mut min_loss_lr = self.config.start_lr;

        for i in 1..self.smoothed_losses.len() {
            let l_curr = self.smoothed_losses[i];
            let l_prev = self.smoothed_losses[i - 1];
            let grad = (l_curr - l_prev) / (self.lrs[i] - self.lrs[i - 1]).max(1e-12);

            if grad < min_grad {
                min_grad = grad;
                suggested_lr = self.lrs[i];
            }

            if l_curr < min_loss {
                min_loss = l_curr;
                min_loss_lr = self.lrs[i];
            }
        }

        LrFindResult {
            lrs: self.lrs.clone(),
            losses: self.losses.clone(),
            smoothed_losses: self.smoothed_losses.clone(),
            suggested_lr,
            min_loss_lr,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
