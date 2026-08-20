//! # Adaptive Delta Optimizer (Adadelta)
//!
//! Learning rate-free adaptive optimizer dynamically scaling updates by running averages of squared updates.
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration settings for Adadelta optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct AdadeltaConfig {
    pub lr: f64,
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdadeltaConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
        }
    }
}

/// Adadelta Optimizer.
#[derive(Debug, Clone)]
pub struct Adadelta {
    pub config: AdadeltaConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub square_avg: HashMap<usize, Vec<f64>>,
    pub acc_delta: HashMap<usize, Vec<f64>>,
}

impl Adadelta {
    /// Creates a new Adadelta optimizer instance.
    pub fn new(param_groups: Vec<ParamGroup>, config: AdadeltaConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            square_avg: HashMap::new(),
            acc_delta: HashMap::new(),
        }
    }
}

impl Optimizer for Adadelta {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter(
                "Length mismatch".into(),
            ));
        }

        self.step_count += 1;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let rho = self.config.rho;
        let eps = self.config.eps;

        for group in &self.param_groups {
            let lr = group.effective_lr();
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

                let sq_avg = self.square_avg.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if sq_avg.len() != n {
                    *sq_avg = vec![0.0; n];
                }

                let acc_d = self.acc_delta.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if acc_d.len() != n {
                    *acc_d = vec![0.0; n];
                }

                for i in 0..n {
                    let mut g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient {
                            param_id: p_idx,
                            value: g_val,
                        });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    if wd != 0.0 {
                        g_val += wd * p_data[i];
                    }

                    sq_avg[i] = rho * sq_avg[i] + (1.0 - rho) * g_val * g_val;
                    let std = (sq_avg[i] + eps).sqrt();
                    let delta = (acc_d[i] + eps).sqrt() / std * g_val;

                    p_data[i] -= lr * delta;
                    acc_d[i] = rho * acc_d[i] + (1.0 - rho) * delta * delta;

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
        self.param_groups
            .first()
            .map(|g| g.effective_lr())
            .unwrap_or(self.config.lr)
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
        for (idx, buf) in &self.square_avg {
            map.insert(
                format!("adadelta_sq_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        for (idx, buf) in &self.acc_delta {
            map.insert(
                format!("adadelta_acc_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.square_avg.clear();
        self.acc_delta.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adadelta_sq_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.square_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("adadelta_acc_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.acc_delta.insert(idx, t.data().to_vec());
                }
            }
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
