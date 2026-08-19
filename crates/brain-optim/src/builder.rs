//! # Fluent Optimizer Builder & Registry
//!
//! Declarative builder API for instantiating and configuring any optimizer in the Brain suite.
#![allow(missing_docs)]

use std::collections::HashMap;
use super::optimizer::param_group::ParamGroup;

/// Supported optimizer algorithm variants in builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizerKind {
    #[default]
    Adam,
    AdamW,
    Sgd,
    SgdNesterov,
    Rmsprop,
    Adagrad,
    Adadelta,
    Lamb,
    Lion,
    RAdam,
    Adafactor,
    NovoGrad,
}

/// Fluent builder for assembling optimizer instances with full parameter group customization.
#[derive(Debug, Clone)]
pub struct OptimizerBuilder {
    pub kind: OptimizerKind,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub amsgrad: bool,
    pub centered: bool,
    pub param_groups: Vec<ParamGroup>,
    pub extra_options: HashMap<String, f64>,
}

impl Default for OptimizerBuilder {
    fn default() -> Self {
        Self {
            kind: OptimizerKind::Adam,
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            amsgrad: false,
            centered: false,
            param_groups: Vec::new(),
            extra_options: HashMap::new(),
        }
    }
}

impl OptimizerBuilder {
    /// Creates a new builder initialized with default Adam parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select Adam optimizer.
    pub fn adam(mut self) -> Self {
        self.kind = OptimizerKind::Adam;
        self
    }

    /// Select AdamW optimizer.
    pub fn adamw(mut self) -> Self {
        self.kind = OptimizerKind::AdamW;
        self
    }

    /// Select SGD optimizer.
    pub fn sgd(mut self) -> Self {
        self.kind = OptimizerKind::Sgd;
        self
    }

    /// Select Lion optimizer.
    pub fn lion(mut self) -> Self {
        self.kind = OptimizerKind::Lion;
        self.beta1 = 0.9;
        self.beta2 = 0.99;
        self
    }

    /// Select LAMB optimizer.
    pub fn lamb(mut self) -> Self {
        self.kind = OptimizerKind::Lamb;
        self
    }

    /// Select RMSProp optimizer.
    pub fn rmsprop(mut self) -> Self {
        self.kind = OptimizerKind::Rmsprop;
        self
    }

    /// Select Adagrad optimizer.
    pub fn adagrad(mut self) -> Self {
        self.kind = OptimizerKind::Adagrad;
        self
    }

    /// Select Adadelta optimizer.
    pub fn adadelta(mut self) -> Self {
        self.kind = OptimizerKind::Adadelta;
        self
    }

    /// Select RAdam optimizer.
    pub fn radam(mut self) -> Self {
        self.kind = OptimizerKind::RAdam;
        self
    }

    /// Sets base learning rate.
    pub fn lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Sets weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Sets betas (beta1, beta2).
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Sets eps.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets momentum.
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Sets nesterov flag.
    pub fn nesterov(mut self, n: bool) -> Self {
        self.nesterov = n;
        self
    }

    /// Sets amsgrad flag.
    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }

    /// Adds a parameter group.
    pub fn add_param_group(mut self, group: ParamGroup) -> Self {
        self.param_groups.push(group);
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
