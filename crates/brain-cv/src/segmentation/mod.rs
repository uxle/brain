//! # Semantic & Instance Segmentation Subsystem
//!
//! Features FCN, PSPNet, DeepLabV3 ASPP, U-Net, Dice/Focal loss functions, and mIoU evaluation metrics.

pub mod fcn;
pub mod losses;
pub mod metrics;

pub use fcn::FcnHead;
pub use losses::SegLossConfig;
pub use metrics::SegMetrics;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
