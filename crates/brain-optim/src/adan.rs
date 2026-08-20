//! # Adaptive Nesterov Momentum Optimizer (Adan)
//!
//! Xie et al., 2022: "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models".
//! Tracks gradient, gradient differences, and second moments for accelerated convergence.

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};

/// Configuration parameters for Adan optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct AdanConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub beta3: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdanConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.98,
            beta2: 0.92,
            beta3: 0.99,
            eps: 1e-8,
            weight_decay: 0.02,
        }
    }
}

/// Adan Optimizer.
#[derive(Debug, Clone)]
pub struct Adan {
    pub config: AdanConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg_m: HashMap<usize, Vec<f64>>,
    pub exp_avg_v: HashMap<usize, Vec<f64>>,
    pub exp_avg_n: HashMap<usize, Vec<f64>>,
    pub prev_grad: HashMap<usize, Vec<f64>>,
}

impl Adan {
    pub fn new(param_groups: Vec<ParamGroup>, config: AdanConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            exp_avg_m: HashMap::new(),
            exp_avg_v: HashMap::new(),
            exp_avg_n: HashMap::new(),
            prev_grad: HashMap::new(),
        }
    }
}

impl Optimizer for Adan {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter("Length mismatch".into()));
        }

        self.step_count += 1;
        let mut total_grad_norm_sq: f64 = 0.0;
        let mut total_param_norm_sq: f64 = 0.0;
        let mut updated = 0;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let beta3 = self.config.beta3;
        let eps = self.config.eps;

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.config.weight_decay);

            for &p_idx in &group.params {
                if p_idx >= params.len() {
                    return Err(OptimizerError::MissingGradient(p_idx));
                }

                let p = &mut params[p_idx];
                let g = &grads[p_idx];

                let p_data = p.data_mut();
                let g_data = g.data();

                let n = p_data.len();
                let m = self.exp_avg_m.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                let v = self.exp_avg_v.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                let sq = self.exp_avg_n.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                let prev_g = self.prev_grad.entry(p_idx).or_insert_with(|| g_data.to_vec());

                for i in 0..n {
                    let grad_val = g_data[i];
                    if grad_val.is_nan() || grad_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: grad_val });
                    }
                    total_grad_norm_sq += grad_val * grad_val;
                    let grad_diff = grad_val - prev_g[i];

                    m[i] = (1.0 - beta1) * m[i] + beta1 * grad_val;
                    v[i] = (1.0 - beta2) * v[i] + beta2 * grad_diff;
                    let n_term = grad_val + (1.0 - beta2) * grad_diff;
                    sq[i] = (1.0 - beta3) * sq[i] + beta3 * (n_term * n_term);

                    let numer = m[i] + (1.0 - beta2) * v[i];
                    let denom = sq[i].sqrt() + eps;
                    let update = numer / denom;

                    p_data[i] = (p_data[i] - lr * update) / (1.0 + lr * wd);
                    total_param_norm_sq += p_data[i] * p_data[i];
                    prev_g[i] = grad_val;
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
        for (idx, buf) in &self.exp_avg_m {
            map.insert(format!("adan_m_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        for (idx, buf) in &self.exp_avg_v {
            map.insert(format!("adan_v_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        for (idx, buf) in &self.exp_avg_n {
            map.insert(format!("adan_n_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg_m.clear();
        self.exp_avg_v.clear();
        self.exp_avg_n.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adan_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg_m.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("adan_v_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg_v.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("adan_n_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg_n.insert(idx, t.data().to_vec());
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
    fn test_adan_step() {
        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], vec![2])];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2], vec![2])];
        let group = ParamGroup::new(vec![0], 0.01);
        let mut adan = Adan::new(vec![group], AdanConfig::default());
        let info = adan.step(&mut params, &grads).unwrap();
        assert_eq!(info.step_count, 1);
        assert!(params[0].data()[0] < 1.0);
    }
}
