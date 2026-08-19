//! # Deep Metric Learning Losses
//!
//! Large-margin angular loss functions: ArcFace, CosFace, SphereFace.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Configuration for Angular Margin losses.
#[derive(Debug, Clone)]
pub struct MetricConfig {
    pub scale: f64,
    pub margin: f64,
    pub reduction: Reduction,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            scale: 64.0,
            margin: 0.5,
            reduction: Reduction::Mean,
        }
    }
}

/// ArcFace (Additive Angular Margin Loss): L = -log( e^{s * cos(theta_y + m)} / (e^{s * cos(theta_y + m)} + sum_{j != y} e^{s * cos(theta_j)}) ).
#[derive(Debug, Clone, Default)]
pub struct ArcFaceLoss {
    pub config: MetricConfig,
}

impl ArcFaceLoss {
    pub fn compute(&self, cos_thetas: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = cos_thetas.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let data = cos_thetas.to_vec();

        let s = self.config.scale;
        let m = self.config.margin;

        let n = rows.min(targets.len());
        let mut losses = vec![0.0f64; n];

        for r in 0..n {
            let y = targets[r];
            let mut sum_exp = 0.0f64;
            let mut target_exp = 0.0f64;

            for c in 0..cols {
                let cos_t = data[r * cols + c].clamp(-1.0, 1.0);
                let score = if c == y {
                    let theta = cos_t.acos();
                    let margined_cos = (theta + m).cos();
                    s * margined_cos
                } else {
                    s * cos_t
                };

                let exp_val = score.exp();
                sum_exp += exp_val;
                if c == y {
                    target_exp = exp_val;
                }
            }

            losses[r] = -(target_exp / sum_exp.max(1e-12)).clamp(1e-12, 1.0).ln();
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
