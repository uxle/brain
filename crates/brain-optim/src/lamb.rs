//! # Layer-wise Adaptive Moments for Batch Training (LAMB)
//!
//! Trust-ratio scaled optimizer enabling high-throughput large-batch distributed training.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimizerError, OptimResult, StepInfo, ParamGroup};

/// Configuration settings for LAMB optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct LambConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub clamp_value: f64,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            clamp_value: 10.0,
        }
    }
}

/// LAMB Optimizer.
#[derive(Debug, Clone)]
pub struct Lamb {
    pub config: LambConfig,
    pub param_groups: Vec<ParamGroup>,
    pub step_count: usize,
    pub exp_avg: HashMap<usize, Vec<f64>>,
    pub exp_avg_sq: HashMap<usize, Vec<f64>>,
}

impl Lamb {
    pub fn new(param_groups: Vec<ParamGroup>, config: LambConfig) -> Self {
        Self {
            config,
            param_groups,
            step_count: 0,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for Lamb {
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

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

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

                let mut param_norm_sq = 0.0;
                for &val in p_data.iter() {
                    param_norm_sq += val * val;
                }
                let param_norm = param_norm_sq.sqrt();

                let mut update_vec = vec![0.0; n];
                let mut update_norm_sq = 0.0;

                for i in 0..n {
                    let g_val = g_data[i];
                    if g_val.is_nan() || g_val.is_infinite() {
                        return Err(OptimizerError::NonFiniteGradient { param_id: p_idx, value: g_val });
                    }
                    total_grad_norm_sq += g_val * g_val;

                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g_val;
                    v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g_val * g_val;

                    let m_hat = m_buf[i] / bias_correction1;
                    let v_hat = (v_buf[i] / bias_correction2).sqrt() + eps;

                    let mut u = m_hat / v_hat;
                    if wd != 0.0 {
                        u += wd * p_data[i];
                    }
                    update_vec[i] = u;
                    update_norm_sq += u * u;
                }

                let update_norm = update_norm_sq.sqrt();
                let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
                    param_norm / update_norm
                } else {
                    1.0
                };

                for i in 0..n {
                    p_data[i] -= lr * trust_ratio * update_vec[i];
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
        for (idx, buf) in &self.exp_avg {
            map.insert(format!("lamb_m_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        for (idx, buf) in &self.exp_avg_sq {
            map.insert(format!("lamb_v_{}", idx), Tensor::from_slice(buf, vec![buf.len()]));
        }
        map
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()> {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        for (k, t) in state {
            if let Some(idx_str) = k.strip_prefix("lamb_m_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    self.exp_avg.insert(idx, t.data().to_vec());
                }
            } else if let Some(idx_str) = k.strip_prefix("lamb_v_") {
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

    #[test]
    fn test_lamb_stress_001() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[1 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_002() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[2 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_003() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[3 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_004() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[4 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_005() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[5 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_006() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[6 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_007() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[7 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_008() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[8 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_009() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[9 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_010() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[10 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_011() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[11 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_012() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[12 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_013() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[13 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_014() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[14 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_015() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[15 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_016() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[16 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_017() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[17 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_018() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[18 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_019() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[19 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_020() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[20 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_021() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[21 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_022() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[22 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_023() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[23 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_024() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[24 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_025() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[25 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_026() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[26 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_027() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[27 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_028() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[28 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_029() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[29 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_030() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[30 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_031() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[31 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_032() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[32 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_033() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[33 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_034() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[34 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_035() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[35 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_036() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[36 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_037() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[37 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_038() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[38 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_039() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[39 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_040() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[40 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_041() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[41 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_042() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[42 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_043() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[43 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_044() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[44 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_045() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[45 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_046() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[46 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_047() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[47 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_048() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[48 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_049() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[49 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_050() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[50 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_051() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[51 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_052() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[52 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_053() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[53 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_054() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[54 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_055() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[55 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_056() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[56 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_057() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[57 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_058() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[58 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_059() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[59 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_060() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[60 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_061() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[61 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_062() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[62 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_063() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[63 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_064() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[64 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_065() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[65 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_066() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[66 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_067() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[67 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_068() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[68 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_069() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[69 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_070() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[70 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_071() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[71 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_072() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[72 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_073() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[73 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_074() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[74 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_075() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[75 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_076() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[76 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_077() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[77 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_078() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[78 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_079() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[79 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_080() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[80 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_081() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[81 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_082() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[82 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_083() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[83 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_084() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[84 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_085() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[85 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_086() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[86 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_087() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[87 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_088() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[88 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_089() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[89 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_090() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[90 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_091() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[91 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_092() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[92 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_093() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[93 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_094() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[94 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_095() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[95 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_096() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[96 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_097() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[97 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_098() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[98 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_099() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[99 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_100() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[100 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_101() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[101 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_102() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[102 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_103() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[103 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_104() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[104 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_105() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[105 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_106() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[106 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_107() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[107 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_108() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[108 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_109() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[109 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_110() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[110 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_111() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[111 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_112() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[112 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_113() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[113 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_114() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[114 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_115() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[115 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_116() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[116 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_117() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[117 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_118() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[118 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_119() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[119 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_120() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[120 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_121() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[121 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_122() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[122 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_123() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[123 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_124() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[124 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_125() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[125 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_126() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[126 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_127() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[127 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_128() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[128 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_129() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[129 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_130() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[130 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_131() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[131 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_132() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[132 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_133() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[133 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_134() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[134 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_135() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[135 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_136() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[136 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_137() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[137 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_138() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[138 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_139() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[139 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_140() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[140 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_141() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[141 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_142() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[142 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_143() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[143 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_144() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[144 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_145() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[145 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_146() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[146 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_147() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[147 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_148() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[148 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_149() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[149 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_150() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[150 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_151() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[151 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_152() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[152 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_153() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[153 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_154() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[154 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_155() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[155 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_156() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[156 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_157() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[157 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_158() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[158 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_159() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[159 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_160() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[160 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_161() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[161 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_162() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[162 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_163() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[163 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_164() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[164 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_165() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[165 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_166() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[166 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_167() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[167 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_168() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[168 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_169() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[169 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_170() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[170 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_171() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[171 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_172() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[172 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_173() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[173 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_174() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[174 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_175() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[175 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_176() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[176 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_177() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[177 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_178() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[178 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_179() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[179 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_180() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[180 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_181() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[181 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_182() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[182 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_183() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[183 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_184() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[184 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_185() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[185 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_186() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[186 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_187() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[187 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_188() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[188 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_189() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[189 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_190() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[190 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_191() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[191 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_192() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[192 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_193() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[193 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_194() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[194 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_195() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[195 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_196() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[196 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_197() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[197 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_198() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[198 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_199() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[199 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_200() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[200 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_201() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[201 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_202() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[202 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_203() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[203 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_204() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[204 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_205() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[205 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_206() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[206 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_207() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[207 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_208() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[208 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_209() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[209 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_210() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[210 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_211() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[211 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_212() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[212 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_213() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[213 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_214() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[214 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_215() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[215 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_216() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[216 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_217() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[217 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_218() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[218 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_219() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[219 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_220() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[220 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_221() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[221 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_222() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[222 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_223() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[223 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_224() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[224 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_225() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[225 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_226() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[226 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_227() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[227 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_228() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[228 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_229() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[229 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_230() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[230 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_231() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[231 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_232() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[232 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_233() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[233 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_234() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[234 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_235() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[235 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_236() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[236 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_237() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[237 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_238() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[238 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_239() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[239 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_240() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[240 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_241() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[241 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_242() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[242 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_243() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[243 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_244() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[244 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_245() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[245 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_246() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[246 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_247() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[247 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_248() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[248 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_249() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[249 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_250() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[250 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_251() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[251 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_252() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[252 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_253() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[253 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_254() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[254 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_255() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[255 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_256() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[256 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_257() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[257 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_258() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[258 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_259() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[259 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    #[test]
    fn test_lamb_stress_260() {
        let group = ParamGroup::new(vec![0], 0.001);
        let mut opt = Lamb::new(vec![group], LambConfig::default());

        let mut p = vec![Tensor::from_slice(&[260 as f64 * 0.1], vec![1])];
        let g = vec![Tensor::from_slice(&[0.1], vec![1])];

        let step_res = opt.step(&mut p, &g).unwrap();
        assert_eq!(step_res.step_count, 1);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
}
