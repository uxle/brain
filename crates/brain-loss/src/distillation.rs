//! # Knowledge Distillation (KD)
//!
//! Temperature-scaled soft-target cross entropy and feature-map distillation.
#![allow(missing_docs)]

use crate::core::{LossResult, Reduction};
use crate::ops::{log_softmax, softmax};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Configuration for Knowledge Distillation.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    pub temperature: f64,
    pub alpha: f64, // Weight between hard target CE and soft target KD
    pub reduction: Reduction,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
            reduction: Reduction::Mean,
        }
    }
}

/// Knowledge Distillation loss module.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeDistillationLoss {
    pub config: DistillConfig,
}

impl KnowledgeDistillationLoss {
    pub fn compute(&self, student_logits: &Tensor, teacher_logits: &Tensor) -> LossResult<Tensor> {
        let t = self.config.temperature;
        let scale = Tensor::scalar(1.0 / t);

        let student_scaled = student_logits * &scale;
        let teacher_scaled = teacher_logits * &scale;

        let log_s = log_softmax(&student_scaled);
        let soft_t = softmax(&teacher_scaled);

        let s_data = log_s.to_vec();
        let t_data = soft_t.to_vec();

        let shape = student_logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };

        let mut losses = vec![0.0f64; rows];
        for r in 0..rows {
            let mut kl = 0.0f64;
            for c in 0..cols {
                let p_t = t_data[r * cols + c];
                let log_p_s = s_data[r * cols + c];
                kl += -p_t * log_p_s;
            }
            losses[r] = kl * (t * t);
        }

        Ok(reduction_apply(&losses, self.config.reduction))
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
