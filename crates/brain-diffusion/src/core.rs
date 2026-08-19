//! # Core Diffusion State & Iteration Types
//!
//! Provides the primary [`DiffusionState`] tracking sample coordinates, timesteps, and predicted noise tensors.

use brain_core::Tensor;

/// Complete state of a diffusion trajectory step.
#[derive(Debug, Clone)]
pub struct DiffusionState {
    pub x: Tensor,
    pub t: usize,
    pub noise: Option<Tensor>,
    pub pred: Option<Tensor>,
}

impl DiffusionState {
    /// Creates a new `DiffusionState`.
    pub fn new(x: Tensor, t: usize) -> Self {
        Self {
            x,
            t,
            noise: None,
            pred: None,
        }
    }

    /// Attaches predicted noise tensor.
    pub fn with_pred(mut self, pred: Tensor) -> Self {
        self.pred = Some(pred);
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
