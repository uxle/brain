//! # Sophia-G Optimizer
//!
//! Second-order Clipped Stochastic Optimization with Gauss-Newton Hessian estimator (Liu et al. 2023).
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration parameters for Sophia-G optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct SophiaGConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub rho: f64,
    pub gamma: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for SophiaGConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.96,
            beta2: 0.99,
            rho: 0.04,
            gamma: 0.01,
            eps: 1e-15,
            weight_decay: 0.1,
        }
    }
}

/// Sophia-G Optimizer.
#[derive(Debug, Clone)]
pub struct SophiaG {
    pub config: SophiaGConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub hessian_avg: HashMap<usize, Vec<f64>>,
}

impl SophiaG {
    pub fn new(param_groups: Vec<ParamGroup>, config: SophiaGConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            hessian_avg: HashMap::new(),
        }
    }
}

impl Optimizer for SophiaG {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter(
                "Length mismatch between params and grads".into(),
            ));
        }

        self.step_count += 1;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let rho = self.config.rho;
        let gamma = self.config.gamma;
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

                let m_buf = self.exp_avg.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if m_buf.len() != n {
                    *m_buf = vec![0.0; n];
                }
                let h_buf = self
                    .hessian_avg
                    .entry(p_idx)
                    .or_insert_with(|| vec![0.0; n]);
                if h_buf.len() != n {
                    *h_buf = vec![0.0; n];
                }

                for i in 0..n {
                    let g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient {
                            param_id: p_idx,
                            value: g_val,
                        });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    // Decoupled weight decay
                    if wd != 0.0 {
                        p_data[i] -= lr * wd * p_data[i];
                    }

                    // Update EMA of gradient: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g_val;

                    // Update EMA of Hessian diagonal estimate: h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2
                    let diag_h = g_val * g_val;
                    h_buf[i] = beta2 * h_buf[i] + (1.0 - beta2) * diag_h;

                    // u_t = clip(m_t / max(gamma * h_t, eps), -rho, rho)
                    let denom = (gamma * h_buf[i]).max(eps);
                    let ratio = m_buf[i] / denom;
                    let u = ratio.max(-rho).min(rho);

                    p_data[i] -= lr * u;
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
                format!("sophia_m_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        for (idx, buf) in &self.hessian_avg {
            map.insert(
                format!("sophia_h_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.hessian_avg.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("sophia_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("sophia_h_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.hessian_avg.insert(idx, t.data().to_vec());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sophia_g_step() {
        let mut params = vec![Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2, 0.3], vec![3])];
        let group = ParamGroup::new(vec![0], 1e-3);
        let mut opt = SophiaG::new(vec![group], SophiaGConfig::default());

        let info = opt.step(&mut params, &grads).unwrap();
        assert_eq!(info.step_count, 1);
        assert!(params[0].get(0) < 1.0);
    }
}
