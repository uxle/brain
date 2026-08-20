//! # GAN Core Types
//!
//! Fundamental data structures: [`GanState`], [`GanMetrics`], [`GanResult`].
#![allow(missing_docs)]

use brain_core::Tensor;

/// Error type for GAN operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GanError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidConfig(String),
    TrainingFailed(String),
}

impl std::fmt::Display for GanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GanError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            GanError::InvalidConfig(s) => write!(f, "Invalid config: {}", s),
            GanError::TrainingFailed(s) => write!(f, "Training failed: {}", s),
        }
    }
}

/// Result type for GAN operations.
pub type GanResult<T> = Result<T, GanError>;

/// Metrics collected after each training step.
#[derive(Debug, Clone, Default)]
pub struct GanMetrics {
    pub step: usize,
    pub d_loss: f64,
    pub g_loss: f64,
    pub d_real: f64,
    pub d_fake: f64,
    pub grad_norm_g: f64,
    pub grad_norm_d: f64,
    pub gp: f64,
}

/// Full GAN state: weights, optimizer states, step counter.
#[derive(Debug, Clone)]
pub struct GanState {
    pub generator_weights: Vec<Tensor>,
    pub discriminator_weights: Vec<Tensor>,
    pub step: usize,
    pub epoch: usize,
}

impl GanState {
    pub fn new(generator_weights: Vec<Tensor>, discriminator_weights: Vec<Tensor>) -> Self {
        Self {
            generator_weights,
            discriminator_weights,
            step: 0,
            epoch: 0,
        }
    }

    pub fn advance_step(&mut self) {
        self.step += 1;
    }
    pub fn advance_epoch(&mut self) {
        self.epoch += 1;
    }
}

/// Summary statistics for a training epoch.
#[derive(Debug, Clone, Default)]
pub struct EpochSummary {
    pub epoch: usize,
    pub avg_d_loss: f64,
    pub avg_g_loss: f64,
    pub num_steps: usize,
}

impl EpochSummary {
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            avg_d_loss: 0.0,
            avg_g_loss: 0.0,
            num_steps: 0,
        }
    }

    pub fn update(&mut self, metrics: &GanMetrics) {
        self.avg_d_loss += metrics.d_loss;
        self.avg_g_loss += metrics.g_loss;
        self.num_steps += 1;
    }

    pub fn finalize(&mut self) {
        if self.num_steps > 0 {
            self.avg_d_loss /= self.num_steps as f64;
            self.avg_g_loss /= self.num_steps as f64;
        }
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
