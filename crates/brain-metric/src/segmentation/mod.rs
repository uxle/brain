//! # Semantic Segmentation Metrics
//!
//! Mean Intersection over Union (mIoU), Pixel Accuracy, Dice coefficient, and boundary F1.
#![allow(missing_docs)]

use crate::utils::stable_divide;

/// Configuration for semantic segmentation evaluation.
#[derive(Debug, Clone, Default)]
pub struct SegMetricConfig {
    pub num_classes: usize,
    pub ignore_index: Option<usize>,
}

/// Computes Mean IoU (mIoU) and Pixel Accuracy over flattened segmentation masks.
pub fn miou_and_pixel_accuracy(
    preds: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> (f64, f64) {
    let n = preds.len().min(targets.len());
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut intersection = vec![0usize; num_classes];
    let mut union = vec![0usize; num_classes];
    let mut total_correct = 0usize;

    for i in 0..n {
        let p = preds[i];
        let t = targets[i];
        if p == t {
            total_correct += 1;
        }
        if p < num_classes && t < num_classes {
            if p == t {
                intersection[p] += 1;
            }
            union[p] += 1;
            if p != t {
                union[t] += 1;
            }
        }
    }

    let mut iou_sum = 0.0f64;
    let mut valid_classes = 0usize;

    for c in 0..num_classes {
        if union[c] > 0 {
            iou_sum += intersection[c] as f64 / union[c] as f64;
            valid_classes += 1;
        }
    }

    let miou = stable_divide(iou_sum, valid_classes as f64, 0.0);
    let pixel_acc = total_correct as f64 / n as f64;

    (miou, pixel_acc)
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
