//! # Adafactor Optimizer
//!
//! Adaptive learning rate optimizer with sublinear memory footprint via matrix factorization (Shazeer & Stern 2018).
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration for Adafactor optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct AdafactorConfig {
    pub lr: f64,
    pub beta1: Option<f64>,
    pub beta2: f64,
    pub eps1: f64,
    pub eps2: f64,
    pub clip_threshold: f64,
    pub weight_decay: f64,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: None,
            beta2: 0.999,
            eps1: 1e-30,
            eps2: 1e-3,
            clip_threshold: 1.0,
            weight_decay: 0.0,
        }
    }
}

/// Adafactor state per parameter.
#[derive(Debug, Clone)]
enum AdafactorState {
    Factorized {
        row_var: Vec<f64>,
        col_var: Vec<f64>,
        exp_avg: Option<Vec<f64>>,
    },
    Standard {
        var: Vec<f64>,
        exp_avg: Option<Vec<f64>>,
    },
}

/// Adafactor Optimizer.
#[derive(Debug, Clone)]
pub struct Adafactor {
    pub config: AdafactorConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    state: HashMap<usize, AdafactorState>,
}

impl Adafactor {
    pub fn new(param_groups: Vec<ParamGroup>, config: AdafactorConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Adafactor {
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

        let beta2 = self.config.beta2;
        let eps1 = self.config.eps1;
        let eps2 = self.config.eps2;
        let clip_thresh = self.config.clip_threshold;

        for group in &self.param_groups {
            let lr = group.effective_lr();
            let wd = group.weight_decay.max(self.config.weight_decay);

            for &p_idx in &group.params {
                if p_idx >= params.len() {
                    return Err(OptimizerError::MissingGradient(p_idx));
                }
                let param = &mut params[p_idx];
                let grad = &grads[p_idx];

                let shape = param.shape().to_vec();
                let p_data = param.data_mut();
                let g_data = grad.data();
                let n = p_data.len();

                let is_matrix = shape.len() == 2 && shape[0] > 1 && shape[1] > 1;

                if is_matrix {
                    let rows = shape[0];
                    let cols = shape[1];

                    let state_entry =
                        self.state
                            .entry(p_idx)
                            .or_insert_with(|| AdafactorState::Factorized {
                                row_var: vec![0.0; rows],
                                col_var: vec![0.0; cols],
                                exp_avg: self.config.beta1.map(|_| vec![0.0; n]),
                            });

                    if let AdafactorState::Factorized {
                        ref mut row_var,
                        ref mut col_var,
                        ref mut exp_avg,
                        ..
                    } = state_entry
                    {
                        // Compute squared grad averages across rows and cols
                        for r in 0..rows {
                            let mut sum_sq = 0.0;
                            for c in 0..cols {
                                let g = g_data[r * cols + c];
                                sum_sq += g * g;
                            }
                            let mean_sq = sum_sq / (cols as f64) + eps1;
                            row_var[r] = beta2 * row_var[r] + (1.0 - beta2) * mean_sq;
                        }

                        for c in 0..cols {
                            let mut sum_sq = 0.0;
                            for r in 0..rows {
                                let g = g_data[r * cols + c];
                                sum_sq += g * g;
                            }
                            let mean_sq = sum_sq / (rows as f64) + eps1;
                            col_var[c] = beta2 * col_var[c] + (1.0 - beta2) * mean_sq;
                        }

                        let row_sum: f64 = row_var.iter().sum();
                        let row_mean = row_sum / (rows as f64);

                        for r in 0..rows {
                            for c in 0..cols {
                                let idx = r * cols + c;
                                let g = g_data[idx];
                                total_grad_norm_sq += g * g;

                                if wd != 0.0 {
                                    p_data[idx] -= lr * wd * p_data[idx];
                                }

                                let denom_approx = (row_var[r] * col_var[c] / row_mean.max(eps1))
                                    .sqrt()
                                    .max(eps2);
                                let mut update = g / denom_approx;

                                // RMS clipping
                                let rms = update.abs();
                                if rms > clip_thresh {
                                    update = update / rms * clip_thresh;
                                }

                                if let (Some(b1), Some(ref mut m)) =
                                    (self.config.beta1, exp_avg.as_mut())
                                {
                                    m[idx] = b1 * m[idx] + (1.0 - b1) * update;
                                    update = m[idx];
                                }

                                p_data[idx] -= lr * update;
                                total_param_norm_sq += p_data[idx] * p_data[idx];
                            }
                        }
                    }
                } else {
                    let state_entry =
                        self.state
                            .entry(p_idx)
                            .or_insert_with(|| AdafactorState::Standard {
                                var: vec![0.0; n],
                                exp_avg: self.config.beta1.map(|_| vec![0.0; n]),
                            });

                    if let AdafactorState::Standard {
                        ref mut var,
                        ref mut exp_avg,
                    } = state_entry
                    {
                        for i in 0..n {
                            let g = g_data[i];
                            total_grad_norm_sq += g * g;

                            if wd != 0.0 {
                                p_data[i] -= lr * wd * p_data[i];
                            }

                            var[i] = beta2 * var[i] + (1.0 - beta2) * (g * g + eps1);
                            let denom = var[i].sqrt().max(eps2);
                            let mut update = g / denom;

                            let rms = update.abs();
                            if rms > clip_thresh {
                                update = update / rms * clip_thresh;
                            }

                            if let (Some(b1), Some(ref mut m)) =
                                (self.config.beta1, exp_avg.as_mut())
                            {
                                m[i] = b1 * m[i] + (1.0 - b1) * update;
                                update = m[i];
                            }

                            p_data[i] -= lr * update;
                            total_param_norm_sq += p_data[i] * p_data[i];
                        }
                    }
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
        for (idx, state) in &self.state {
            match state {
                AdafactorState::Factorized {
                    row_var,
                    col_var,
                    exp_avg,
                    ..
                } => {
                    map.insert(
                        format!("adafactor_r_{}", idx),
                        Tensor::from_slice(row_var, vec![row_var.len()]),
                    );
                    map.insert(
                        format!("adafactor_c_{}", idx),
                        Tensor::from_slice(col_var, vec![col_var.len()]),
                    );
                    if let Some(m) = exp_avg {
                        map.insert(
                            format!("adafactor_m_{}", idx),
                            Tensor::from_slice(m, vec![m.len()]),
                        );
                    }
                }
                AdafactorState::Standard { var, exp_avg } => {
                    map.insert(
                        format!("adafactor_v_{}", idx),
                        Tensor::from_slice(var, vec![var.len()]),
                    );
                    if let Some(m) = exp_avg {
                        map.insert(
                            format!("adafactor_m_{}", idx),
                            Tensor::from_slice(m, vec![m.len()]),
                        );
                    }
                }
            }
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.state.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("adafactor_v_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.state.insert(
                        idx,
                        AdafactorState::Standard {
                            var: t.data().to_vec(),
                            exp_avg: None,
                        },
                    );
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
    fn test_adafactor_matrix_and_vector_step() {
        let mut params = vec![
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::from_slice(&[1.0, 2.0], vec![2]),
        ];
        let grads = vec![
            Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![2, 2]),
            Tensor::from_slice(&[0.1, 0.2], vec![2]),
        ];
        let group = ParamGroup::new(vec![0, 1], 1e-3);
        let mut opt = Adafactor::new(vec![group], AdafactorConfig::default());

        let info = opt.step(&mut params, &grads).unwrap();
        assert_eq!(info.step_count, 1);
        assert_eq!(info.num_params_updated, 2);
    }
}
