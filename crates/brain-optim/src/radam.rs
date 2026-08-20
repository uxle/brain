//! # Rectified Adam (RAdam)
//!
//! Variance-rectified Adam optimizer stabilizing early training without explicit warm-up.
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration settings for RAdam optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct RAdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub decoupled_weight_decay: bool,
}

impl Default for RAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: true,
        }
    }
}

/// Rectified Adam Optimizer.
#[derive(Debug, Clone)]
pub struct RAdam {
    pub config: RAdamConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub exp_avg_sq: HashMap<usize, Vec<f64>>,
}

impl RAdam {
    pub fn new(param_groups: Vec<ParamGroup>, config: RAdamConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for RAdam {
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
        let step = self.step_count as f64;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        let beta2_t = beta2.powf(step);
        let rho_t = rho_inf - 2.0 * step * beta2_t / (1.0 - beta2_t);
        let bias_correction1 = 1.0 - beta1.powf(step);

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

                let m_buf = self.exp_avg.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if m_buf.len() != n {
                    *m_buf = vec![0.0; n];
                }

                let v_buf = self.exp_avg_sq.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if v_buf.len() != n {
                    *v_buf = vec![0.0; n];
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

                    if wd != 0.0 && !self.config.decoupled_weight_decay {
                        g_val += wd * p_data[i];
                    }

                    if self.config.decoupled_weight_decay && wd != 0.0 {
                        p_data[i] -= lr * wd * p_data[i];
                    }

                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g_val;
                    v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g_val * g_val;

                    let m_hat = m_buf[i] / bias_correction1;

                    if rho_t > 5.0 {
                        let rect = (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                            .sqrt();
                        let v_hat = (v_buf[i] / (1.0 - beta2_t)).sqrt() + eps;
                        p_data[i] -= lr * rect * (m_hat / v_hat);
                    } else {
                        p_data[i] -= lr * m_hat;
                    }

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
        for (idx, buf) in &self.exp_avg {
            map.insert(
                format!("radam_m_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        for (idx, buf) in &self.exp_avg_sq {
            map.insert(
                format!("radam_v_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("radam_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("radam_v_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg_sq.insert(idx, t.data().to_vec());
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
