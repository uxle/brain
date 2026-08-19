//! # Advanced Adam Variants
//!
//! Implementations of Adamax (infinity-norm), Nadam (Nesterov-accelerated Adam),
//! and Adafactor (factored second moments for memory efficiency).
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};

/// Enumeration of Adam variant architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdamVariant {
    #[default]
    Adamax,
    Nadam,
    Adafactor,
}

/// Adamax Optimizer (L-infinity norm based Adam).
#[derive(Debug, Clone)]
pub struct Adamax {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub exp_inf: HashMap<usize, Vec<f64>>,
}

impl Adamax {
    pub fn new(param_groups: Vec<ParamGroup>, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_inf: HashMap::new(),
        }
    }
}

impl Optimizer for Adamax {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter("Length mismatch".into()));
        }

        self.step_count += 1;
        let step = self.step_count as f64;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let bias_correction = 1.0 - beta1.powf(step);

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.weight_decay);

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

                let u_buf = self.exp_inf.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                if u_buf.len() != n {
                    *u_buf = vec![0.0; n];
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

                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g_val;
                    u_buf[i] = (beta2 * u_buf[i]).max(g_val.abs());

                    let clr = lr / bias_correction;
                    p_data[i] -= clr * (m_buf[i] / (u_buf[i] + eps));
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
        self.param_groups.first().map(|g| g.effective_lr()).unwrap_or(self.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
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
            map.insert(format!("adamax_m_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        for (idx, buf) in &self.exp_inf {
            map.insert(format!("adamax_u_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.exp_inf.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adamax_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("adamax_u_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_inf.insert(idx, t.data().to_vec());
                }
            }
        }
        Ok(())
    }
}

/// Nadam Optimizer (Nesterov-accelerated Adaptive Moment Estimation).
#[derive(Debug, Clone)]
pub struct Nadam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub exp_avg_sq: HashMap<usize, Vec<f64>>,
}

impl Nadam {
    pub fn new(param_groups: Vec<ParamGroup>, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for Nadam {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo> {
        if self.param_groups.is_empty() {
            return Err(OptimizerError::EmptyParamGroup);
        }
        if params.len() != grads.len() {
            return Err(OptimizerError::InvalidHyperparameter("Length mismatch".into()));
        }

        self.step_count += 1;
        let step = self.step_count as f64;
        let mut total_grad_norm_sq = 0.0;
        let mut total_param_norm_sq = 0.0;
        let mut updated = 0;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.weight_decay);

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
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: g_val });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    if wd != 0.0 {
                        g_val += wd * p_data[i];
                    }

                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g_val;
                    v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g_val * g_val;

                    let g_hat = g_val / bias_correction1;
                    let m_hat = m_buf[i] / bias_correction1;
                    let v_hat = v_buf[i] / bias_correction2;

                    let nes_m = beta1 * m_hat + (1.0 - beta1) * g_hat;
                    p_data[i] -= lr * (nes_m / (v_hat.sqrt() + eps));
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
        self.param_groups.first().map(|g| g.effective_lr()).unwrap_or(self.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
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
            map.insert(format!("nadam_m_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        for (idx, buf) in &self.exp_avg_sq {
            map.insert(format!("nadam_v_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("nadam_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("nadam_v_") {
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
