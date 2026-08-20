//! # Muon Optimizer (Momentum with Newton-Schulz Orthogonalization)
//!
//! High-performance matrix optimizer for 2D transformer weights using 5-step Newton-Schulz iteration (Keller Jordan et al. 2024).
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, OptimizerError, ParamGroup, StepInfo};
use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration for Muon optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct MuonConfig {
    /// Learning rate for 2D matrix weights.
    pub lr: f64,
    /// Learning rate for non-matrix (1D) parameters (AdamW fallback).
    pub lr_adamw: f64,
    /// Momentum factor beta for matrix parameters.
    pub momentum: f64,
    /// AdamW beta1 for 1D parameters.
    pub beta1: f64,
    /// AdamW beta2 for 1D parameters.
    pub beta2: f64,
    /// Weight decay.
    pub weight_decay: f64,
    /// Epsilon for numerical stability.
    pub eps: f64,
    /// Number of Newton-Schulz iterations (typically 5).
    pub ns_steps: usize,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            lr_adamw: 1e-3,
            momentum: 0.95,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            ns_steps: 5,
        }
    }
}

/// Muon Optimizer.
#[derive(Debug, Clone)]
pub struct Muon {
    pub config: MuonConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub momentum_buf: HashMap<usize, Vec<f64>>,
    pub exp_avg_1d: HashMap<usize, Vec<f64>>,
    pub exp_avg_sq_1d: HashMap<usize, Vec<f64>>,
}

impl Muon {
    pub fn new(param_groups: Vec<ParamGroup>, config: MuonConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            momentum_buf: HashMap::new(),
            exp_avg_1d: HashMap::new(),
            exp_avg_sq_1d: HashMap::new(),
        }
    }
}

/// Computes 5th-order Newton-Schulz orthogonalization on a matrix of shape [rows, cols].
/// Returns orthogonalized update matrix of shape [rows, cols].
pub fn newton_schulz_5(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    steps: usize,
    eps: f64,
) -> Vec<f64> {
    let transpose = rows > cols;
    let (r, c) = if transpose {
        (cols, rows)
    } else {
        (rows, cols)
    };

    // Initial matrix X of shape [r, c]
    let mut x = vec![0.0f64; r * c];
    if transpose {
        for i in 0..rows {
            for j in 0..cols {
                x[j * rows + i] = matrix[i * cols + j];
            }
        }
    } else {
        x.copy_from_slice(matrix);
    }

    // Compute Frobenius norm ||X||_F
    let frob_sq: f64 = x.iter().map(|&v| v * v).sum();
    let norm = frob_sq.sqrt().max(eps);
    for v in &mut x {
        *v /= norm;
    }

    // Chebyshev polynomial coefficients for quintic Newton-Schulz
    let alpha = 3.4445f64;
    let beta = -4.7750f64;
    let gamma = 2.0315f64;

    // Iteration: X_{k+1} = a * X + (b * A + c * A^2) * X, where A = X * X^T [r, r]
    for _ in 0..steps {
        // Compute A = X * X^T [r, r]
        let mut a = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut dot = 0.0;
                for k in 0..c {
                    dot += x[i * c + k] * x[j * c + k];
                }
                a[i * r + j] = dot;
            }
        }

        // Compute A^2 = A * A [r, r]
        let mut a2 = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut dot = 0.0;
                for k in 0..r {
                    dot += a[i * r + k] * a[k * r + j];
                }
                a2[i * r + j] = dot;
            }
        }

        // Compute B = beta * A + gamma * A^2 [r, r]
        let mut b = vec![0.0f64; r * r];
        for idx in 0..r * r {
            b[idx] = beta * a[idx] + gamma * a2[idx];
        }

        // Compute X_next = alpha * X + B * X [r, c]
        let mut x_next = vec![0.0f64; r * c];
        for i in 0..r {
            for j in 0..c {
                let mut b_dot = 0.0;
                for k in 0..r {
                    b_dot += b[i * r + k] * x[k * c + j];
                }
                x_next[i * c + j] = alpha * x[i * c + j] + b_dot;
            }
        }

        x = x_next;
    }

    // Scale by RMS adjustment factor: max(1, sqrt(rows / cols))
    let scale = (rows as f64 / cols as f64).sqrt().max(1.0);
    for v in &mut x {
        *v *= scale;
    }

    // If transposed initially, transpose back to [rows, cols]
    if transpose {
        let mut res = vec![0.0f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                res[i * cols + j] = x[j * rows + i];
            }
        }
        res
    } else {
        x
    }
}

impl Optimizer for Muon {
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

        let momentum = self.config.momentum;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.eps;
        let ns_steps = self.config.ns_steps;

        for group in &self.param_groups {
            let lr_mat = group.effective_lr();
            let lr_1d = self.config.lr_adamw;
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

                let is_2d_matrix = shape.len() == 2 && shape[0] > 1 && shape[1] > 1;

                if is_2d_matrix {
                    let rows = shape[0];
                    let cols = shape[1];

                    let m_buf = self
                        .momentum_buf
                        .entry(p_idx)
                        .or_insert_with(|| vec![0.0; n]);
                    if m_buf.len() != n {
                        *m_buf = vec![0.0; n];
                    }

                    for i in 0..n {
                        let g = g_data[i];
                        if g.is_nan() || g.is_infinite() {
                            return Err(OptimizerError::NonFiniteGradient {
                                param_id: p_idx,
                                value: g,
                            });
                        }
                        total_grad_norm_sq += g * g;

                        // Momentum accumulator
                        m_buf[i] = momentum * m_buf[i] + (1.0 - momentum) * g;
                    }

                    // Perform Newton-Schulz orthogonalization on momentum buffer
                    let ortho_update = newton_schulz_5(m_buf, rows, cols, ns_steps, eps);

                    for i in 0..n {
                        if wd != 0.0 {
                            p_data[i] -= lr_mat * wd * p_data[i];
                        }
                        p_data[i] -= lr_mat * ortho_update[i];
                        total_param_norm_sq += p_data[i] * p_data[i];
                    }
                } else {
                    // 1D parameter fallback (AdamW update)
                    let m_buf = self.exp_avg_1d.entry(p_idx).or_insert_with(|| vec![0.0; n]);
                    if m_buf.len() != n {
                        *m_buf = vec![0.0; n];
                    }
                    let v_buf = self
                        .exp_avg_sq_1d
                        .entry(p_idx)
                        .or_insert_with(|| vec![0.0; n]);
                    if v_buf.len() != n {
                        *v_buf = vec![0.0; n];
                    }

                    let bias_c1 = 1.0 - b1.powi(self.step_count as i32);
                    let bias_c2 = 1.0 - b2.powi(self.step_count as i32);

                    for i in 0..n {
                        let g = g_data[i];
                        total_grad_norm_sq += g * g;

                        if wd != 0.0 {
                            p_data[i] -= lr_1d * wd * p_data[i];
                        }

                        m_buf[i] = b1 * m_buf[i] + (1.0 - b1) * g;
                        v_buf[i] = b2 * v_buf[i] + (1.0 - b2) * (g * g);

                        let m_hat = m_buf[i] / bias_c1;
                        let v_hat = v_buf[i] / bias_c2;

                        p_data[i] -= lr_1d * (m_hat / (v_hat.sqrt() + eps));
                        total_param_norm_sq += p_data[i] * p_data[i];
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
        for (idx, buf) in &self.momentum_buf {
            map.insert(
                format!("muon_m_{}", idx),
                Tensor::from_slice(buf, vec![buf.len()]),
            );
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.momentum_buf.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("muon_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.momentum_buf.insert(idx, t.data().to_vec());
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
    fn test_newton_schulz_orthogonalization() {
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let ortho = newton_schulz_5(&mat, 2, 2, 5, 1e-8);
        assert_eq!(ortho.len(), 4);
    }

    #[test]
    fn test_muon_step() {
        let mut params = vec![
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::from_slice(&[1.0, 2.0], vec![2]),
        ];
        let grads = vec![
            Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![2, 2]),
            Tensor::from_slice(&[0.1, 0.2], vec![2]),
        ];
        let group = ParamGroup::new(vec![0, 1], 1e-2);
        let mut opt = Muon::new(vec![group], MuonConfig::default());

        let info = opt.step(&mut params, &grads).unwrap();
        assert_eq!(info.step_count, 1);
        assert_eq!(info.num_params_updated, 2);
    }
}
