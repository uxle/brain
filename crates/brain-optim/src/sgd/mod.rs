//! # Stochastic Gradient Descent (SGD & Momentum)
//!
//! ## Mathematical Formulation
//!
//! Standard SGD update:
//! $$\theta_t = \theta_{t-1} - \eta g_t$$
//!
//! Classical Polyak Momentum ($v_0 = 0, \mu \in [0, 1)$):
//! $$v_t = \mu v_{t-1} + g_t$$
//! $$\theta_t = \theta_{t-1} - \eta v_t$$
//!
//! Nesterov Accelerated Gradient (NAG, Nesterov, 1983):
//! $$v_t = \mu v_{t-1} + g_t$$
//! $$\theta_t = \theta_{t-1} - \eta (g_t + \mu v_t)$$
//! (SGD)
//!
//! Production implementation of SGD with classical momentum, dampening, L2 regularization,
//! and decoupled weight decay (SGDW).
#![allow(missing_docs)]

pub mod nesterov;

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};

/// Configuration parameters for SGD optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub decoupled_weight_decay: bool,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            decoupled_weight_decay: false,
        }
    }
}

/// Stochastic Gradient Descent Optimizer.
#[derive(Debug, Clone)]
pub struct Sgd {
    pub config: SgdConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub momentum_buffers: HashMap<usize, Vec<f64>>,
}

impl Sgd {
    /// Creates a new SGD optimizer instance with specified configuration and parameter groups.
    pub fn new(param_groups: Vec<ParamGroup>, config: SgdConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Convenience constructor with single default learning rate.
    pub fn with_lr(param_groups: Vec<ParamGroup>, lr: f64) -> Self {
        let config = SgdConfig {
            lr,
            ..Default::default()
        };
        Self::new(param_groups, config)
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter("Params and grads length mismatch".into()));
        }

        self.step_count += 1;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.config.weight_decay);
            let momentum = self.config.momentum;
            let dampening = self.config.dampening;

            for &p_idx in &group.params {
                if p_idx >= params.len() {
                    return Err(OptimizerError::MissingGradient(p_idx));
                }
                let param = &mut params[p_idx];
                let grad = &grads[p_idx];

                if param.shape() != grad.shape() {
                    return Err(OptimizerError::GradientDimensionMismatch {
                        expected: param.shape().to_vec(),
                        found: grad.shape().to_vec(),
                    });
                }

                let p_data = param.data_mut();
                let g_data = grad.data();
                let n = p_data.len();

                let buf = self.momentum_buffers.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if buf.len() != n {
                    *buf = vec![0.0; n];
                }

                for i in 0..n {
                    let g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: g_val });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    let mut d_p = g_val;
                    if wd != 0.0 && !self.config.decoupled_weight_decay {
                        d_p += wd * p_data[i];
                    }

                    if momentum != 0.0 {
                        if self.step_count == 1 {
                            buf[i] = d_p;
                        } else {
                            buf[i] = momentum * buf[i] + (1.0 - dampening) * d_p;
                        }

                        if self.config.nesterov {
                            d_p += momentum * buf[i];
                        } else {
                            d_p = buf[i];
                        }
                    }

                    if self.config.decoupled_weight_decay && wd != 0.0 {
                        p_data[i] -= lr * wd * p_data[i];
                    }

                    p_data[i] -= lr * d_p;
                    total_param_norm_sq += p_data[i] * p_data[i];
                }
                updated += 1;
            }
        }

        Ok(StepInfo {
            step_count: self.step_count,
            grad_norm: total_grad_norm_sq.sqrt(),
            param_norm: total_param_norm_sq.sqrt(),
            num_params_updated: updated,
            lr_current: self.get_lr(),
            loss_value: None,
        })
    }

    fn get_lr(&self) -> f64 {
        self.param_groups.first().map(|g| g.effective_lr()).unwrap_or(self.config.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
        for g in &mut self.param_groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group_idx: usize, lr: f64) -> OptimResult<()> {
        if let Some(g) = self.param_groups.get_mut(group_idx) {
            g.lr = lr;
            Ok(())
        } else {
            Err(OptimizerError::GroupNotFound(group_idx))
        }
    }

    fn get_step_count(&self) -> usize {
        self.step_count
    }

    fn param_groups(&self) -> &[ParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.param_groups
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (idx, buf) in &self.momentum_buffers {
            map.insert(format!("momentum_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.momentum_buffers.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("momentum_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.momentum_buffers.insert(idx, t.data().to_vec());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
