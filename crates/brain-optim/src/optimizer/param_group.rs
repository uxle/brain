//! # Parameter Group Definitions & Overrides
//!
//! Flexible grouping of model parameters with distinct learning rates, weight decay,
//! momentum, and regularizers.
#![allow(missing_docs)]

use brain_core::Tensor;
use std::collections::HashMap;

pub type ParamId = usize;

/// Representation of a gradient entry associated with a parameter.
#[derive(Debug, Clone)]
pub struct GradEntry {
    pub param_id: ParamId,
    pub grad: Tensor,
    pub has_nan: bool,
    pub norm: f64,
}

impl GradEntry {
    pub fn new(param_id: ParamId, grad: Tensor) -> Self {
        let mut has_nan = false;
        let mut sum_sq = 0.0;
        for &val in grad.data() {
            if val.is_nan() || val.is_infinite() {
                has_nan = true;
            }
            sum_sq += val * val;
        }
        Self {
            param_id,
            grad,
            has_nan,
            norm: sum_sq.sqrt(),
        }
    }
}

/// Configuration settings for a parameter group.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamGroupConfig {
    pub name: String,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub lr_multiplier: f64,
    pub is_frozen: bool,
    pub dampening: f64,
}

impl Default for ParamGroupConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr_multiplier: 1.0,
            is_frozen: false,
            dampening: 0.0,
        }
    }
}

/// A parameter group holding a list of parameter IDs and group-level hyperparameter overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamGroup {
    pub params: Vec<ParamId>,
    pub lr: f64,
    pub initial_lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub lr_multiplier: f64,
    pub is_frozen: bool,
    pub name: String,
    pub options: HashMap<String, f64>,
}

impl ParamGroup {
    /// Creates a new parameter group with the given parameter IDs and learning rate.
    pub fn new(params: Vec<ParamId>, lr: f64) -> Self {
        Self {
            params,
            lr,
            initial_lr: lr,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr_multiplier: 1.0,
            is_frozen: false,
            name: "default".to_string(),
            options: HashMap::new(),
        }
    }

    /// Creates a parameter group from configuration.
    pub fn from_config(params: Vec<ParamId>, config: &ParamGroupConfig) -> Self {
        Self {
            params,
            lr: config.lr,
            initial_lr: config.lr,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            lr_multiplier: config.lr_multiplier,
            is_frozen: config.is_frozen,
            name: config.name.clone(),
            options: HashMap::new(),
        }
    }

    /// Builder pattern: set weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder pattern: set betas.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Builder pattern: set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder pattern: set custom option.
    pub fn with_option(mut self, key: impl Into<String>, val: f64) -> Self {
        self.options.insert(key.into(), val);
        self
    }

    /// Returns the effective learning rate including multiplier.
    pub fn effective_lr(&self) -> f64 {
        if self.is_frozen {
            0.0
        } else {
            self.lr * self.lr_multiplier
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
