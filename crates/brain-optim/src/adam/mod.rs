//! # Adaptive Moment Estimation (Adam & AdamW)
//!
//! ## Mathematical Formulation
//!
//! Given objective $f(\theta)$, learning rate $\eta$, decay rates $\beta_1, \beta_2 \in [0, 1)$, and $\epsilon > 0$:
//!
//! First moment (momentum):
//! $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
//!
//! Second moment (uncentered variance):
//! $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
//!
//! Bias corrections:
//! $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
//!
//! Parameter update (Standard Adam with L2 regularization):
//! $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
//!
//! Decoupled Weight Decay (AdamW, Loshchilov & Hutter, 2019):
//! $$\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
#![allow(missing_docs)]

pub mod variants;

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration settings for Adam optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub decoupled_weight_decay: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            decoupled_weight_decay: false,
        }
    }
}

/// Convenience alias for AdamW configuration.
pub type AdamWConfig = AdamConfig;

/// Adam / AdamW Optimizer.
#[derive(Debug, Clone)]
pub struct Adam {
    pub config: AdamConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub exp_avg_sq: HashMap<usize, Vec<f64>>,
    pub max_exp_avg_sq: HashMap<usize, Vec<f64>>,
}

impl Adam {
    /// Creates a new Adam optimizer.
    pub fn new(param_groups: Vec<ParamGroup>, config: AdamConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
        }
    }

    /// Creates an AdamW optimizer with decoupled weight decay.
    pub fn adamw(param_groups: Vec<ParamGroup>, lr: f64, weight_decay: f64) -> Self {
        let config = AdamConfig {
            lr,
            weight_decay,
            decoupled_weight_decay: true,
            ..Default::default()
        };
        Self::new(param_groups, config)
    }
}

impl Optimizer for Adam {
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

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let beta1 = if group.beta1 > 0.0 {
                group.beta1
            } else {
                self.config.beta1
            };
            let beta2 = if group.beta2 > 0.0 {
                group.beta2
            } else {
                self.config.beta2
            };
            let eps = if group.eps > 0.0 {
                group.eps
            } else {
                self.config.eps
            };
            let wd = group.weight_decay.max(self.config.weight_decay);
            let bias_correction1 = 1.0 - beta1.powf(step);
            let bias_correction2 = 1.0 - beta2.powf(step);

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

                let mut v_max_buf = if self.config.amsgrad {
                    let entry = self
                        .max_exp_avg_sq
                        .entry(p_idx)
                        .or_insert_with(|| vec![0.0; n]);
                    if entry.len() != n {
                        *entry = vec![0.0; n];
                    }
                    Some(entry)
                } else {
                    None
                };

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

                    let denom = if let Some(ref mut max_v) = v_max_buf {
                        if v_buf[i] > max_v[i] {
                            max_v[i] = v_buf[i];
                        }
                        let v_hat = max_v[i] / bias_correction2;
                        v_hat.sqrt() + eps
                    } else {
                        let v_hat = v_buf[i] / bias_correction2;
                        v_hat.sqrt() + eps
                    };

                    let m_hat = m_buf[i] / bias_correction1;
                    p_data[i] -= lr * (m_hat / denom);
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
                format!("adam_m_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        for (idx, buf) in &self.exp_avg_sq {
            map.insert(
                format!("adam_v_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adam_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("adam_v_") {
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
