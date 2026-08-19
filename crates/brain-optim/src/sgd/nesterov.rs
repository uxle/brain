//! # Nesterov Accelerated Gradient (NAG)
//!
//! Specialized implementation and analysis for Nesterov momentum dynamics.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};
use super::SgdConfig;

/// Nesterov Accelerated Gradient optimizer engine.
#[derive(Debug, Clone)]
pub struct SgdNesterov {
    pub config: SgdConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub velocity_buffers: HashMap<usize, Vec<f64>>,
}

impl SgdNesterov {
    /// Creates a dedicated Nesterov momentum optimizer.
    pub fn new(param_groups: Vec<ParamGroup>, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        let config = SgdConfig {
            lr,
            momentum,
            dampening: 0.0,
            weight_decay,
            nesterov: true,
            decoupled_weight_decay: false,
        };
        Self {
            config,
            param_groups,
            step_count: 0,
            velocity_buffers: HashMap::new(),
        }
    }
}

impl Optimizer for SgdNesterov {
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

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.config.weight_decay);
            let momentum = self.config.momentum;

            for &p_idx in &group.params {
                if p_idx >= params.len() {
                    return Err(OptimizerError::MissingGradient(p_idx));
                }
                let param = &mut params[p_idx];
                let grad = &grads[p_idx];

                let p_data = param.data_mut();
                let g_data = grad.data();
                let n = p_data.len();

                let v_buf = self.velocity_buffers.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if v_buf.len() != n {
                    *v_buf = vec![0.0; n];
                }

                for i in 0..n {
                    let g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: g_val });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    let mut d_p = g_val;
                    if wd != 0.0 {
                        d_p += wd * p_data[i];
                    }

                    v_buf[i] = momentum * v_buf[i] + d_p;
                    let update = d_p + momentum * v_buf[i];

                    p_data[i] -= lr * update;
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
        for (idx, buf) in &self.velocity_buffers {
            map.insert(format!("nesterov_v_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.velocity_buffers.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("nesterov_v_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.velocity_buffers.insert(idx, t.data().to_vec());
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
