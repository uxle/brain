//! # Segmentation Loss Combinations
//!
//! Combined Cross-Entropy + Soft Dice loss and Lovász Hinge for image segmentation.
#![allow(missing_docs)]

use crate::core::{LossResult, Reduction};
use brain_core::Tensor;

/// Configuration for segmentation losses.
#[derive(Debug, Clone)]
pub struct SegLossConfig {
    pub ce_weight: f64,
    pub dice_weight: f64,
    pub smooth: f64,
    pub reduction: Reduction,
}

impl Default for SegLossConfig {
    fn default() -> Self {
        Self {
            ce_weight: 1.0,
            dice_weight: 1.0,
            smooth: 1e-5,
            reduction: Reduction::Mean,
        }
    }
}

/// Combined CE + Soft Dice Loss module.
#[derive(Debug, Clone, Default)]
pub struct CEDiceLoss {
    pub config: SegLossConfig,
}

impl CEDiceLoss {
    pub fn compute(&self, pred_probs: &Tensor, target_masks: &Tensor) -> LossResult<Tensor> {
        let p = pred_probs.to_vec();
        let t = target_masks.to_vec();
        let n = p.len().min(t.len());

        let mut intersection = 0.0f64;
        let mut sum_p = 0.0f64;
        let mut sum_t = 0.0f64;
        let mut ce_sum = 0.0f64;

        let smooth = self.config.smooth;

        for i in 0..n {
            let pred_val = p[i].clamp(1e-7, 1.0 - 1e-7);
            let target_val = t[i];

            intersection += pred_val * target_val;
            sum_p += pred_val * pred_val;
            sum_t += target_val * target_val;

            ce_sum += -(target_val * pred_val.ln() + (1.0 - target_val) * (1.0 - pred_val).ln());
        }

        let dice = (2.0 * intersection + smooth) / (sum_p + sum_t + smooth);
        let dice_loss = 1.0 - dice;
        let ce_loss = ce_sum / n.max(1) as f64;

        let total = self.config.ce_weight * ce_loss + self.config.dice_weight * dice_loss;
        Ok(Tensor::from_vec(vec![total], vec![1]))
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
