//! # Segmentation Evaluation Metrics
//!
//! Mean Intersection-over-Union (mIoU), Pixel Accuracy, Dice Score, and confusion matrices.

/// Container for segmentation evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct SegMetrics {
    pub mean_iou: f64,
    pub pixel_accuracy: f64,
    pub dice_score: f64,
}

impl SegMetrics {
    /// Creates a new `SegMetrics` container.
    pub fn new(mean_iou: f64, pixel_accuracy: f64, dice_score: f64) -> Self {
        Self {
            mean_iou,
            pixel_accuracy,
            dice_score,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
