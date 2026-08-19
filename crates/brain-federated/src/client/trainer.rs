//! # Local Training Algorithms
//!
//! SGD and Adam local trainers used within federated client training loops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Trait representing a local training algorithm.
pub trait LocalTrainer: Send + Sync {
    fn train_step(&self, params: &mut Vec<Tensor>, grads: &[Tensor], lr: f64);
}

/// Stochastic Gradient Descent local trainer.
#[derive(Debug, Clone, Default)]
pub struct SgdTrainer;

impl SgdTrainer {
    pub fn new() -> Self { Self }
}

impl LocalTrainer for SgdTrainer {
    fn train_step(&self, params: &mut Vec<Tensor>, grads: &[Tensor], lr: f64) {
        let lr_t = Tensor::scalar(lr);
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p = &*p - &(g * &lr_t);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
