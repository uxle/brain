//! # Adaptive Gradient Algorithm (Adagrad)
//!
//! Subgradient optimization with parameter-specific learning rates based on historical gradient norms.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};

/// Configuration settings for Adagrad optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct AdagradConfig {
    pub lr: f64,
    pub lr_decay: f64,
    pub weight_decay: f64,
    pub initial_accumulator_value: f64,
    pub eps: f64,
}

impl Default for AdagradConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        }
    }
}

/// Adagrad Optimizer.
#[derive(Debug, Clone)]
pub struct Adagrad {
    pub config: AdagradConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub sum_squares: HashMap<usize, Vec<f64>>,
}

impl Adagrad {
    /// Creates a new Adagrad optimizer instance.
    pub fn new(param_groups: Vec<ParamGroup>, config: AdagradConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            sum_squares: HashMap::new(),
        }
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter("Length mismatch".into()));
        }

        self.step_count += 1;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let lr_decay = self.config.lr_decay;
        let eps = self.config.eps;
        let init_acc = self.config.initial_accumulator_value;

        for group in &self.param_groups {
            let mut lr = group.effective_lr();
            if lr_decay > 0.0 {
                lr /= 1.0 + (self.step_count - 1) as f64 * lr_decay;
            }
            let wd = group.weight_decay.max(self.config.weight_decay);

            for &p_idx in &group.params {
                if p_idx >= params.len() {
                    return Err(OptimizerError::MissingGradient(p_idx));
                }
                let param = &mut params[p_idx];
                let grad = &grads[p_idx];

                let p_data = param.data_mut();
                let g_data = grad.data();
                let n = p_data.len();

                let sum_sq = self.sum_squares.entry(p_idx).or_insert_with(|| vec![init_acc; n]);
                if sum_sq.len() != n {
                    *sum_sq = vec![init_acc; n];
                }

                for i in 0..n {
                    let mut g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: g_val });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    if wd != 0.0 {
                        g_val += wd * p_data[i];
                    }

                    sum_sq[i] += g_val * g_val;
                    let std = sum_sq[i].sqrt() + eps;

                    p_data[i] -= lr * (g_val / std);
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
        for (idx, buf) in &self.sum_squares {
            map.insert(format!("adagrad_acc_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.sum_squares.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adagrad_acc_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.sum_squares.insert(idx, t.data().to_vec());
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
