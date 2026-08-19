//! # Brain CV — Production-Grade Computer Vision Suite
//!
//! Provides convolutional modules, object detection pipelines, semantic/instance segmentation,
//! deep feature extraction backbones, multi-dimensional pooling, and comprehensive image augmentations.
//!
//! ## Subsystems
//!
//! * [`conv`] - Standard, Deformable, Depthwise, Residual, Transposed, Grouped, WS, and Ghost Convolutions
//! * [`detection`] - Anchors, Detection Heads, Postprocessing, Losses, NMS, RoI, and RoIAlign
//! * [`segmentation`] - FCN, PSPNet, DeepLabV3 ASPP, U-Net, Loss functions, and mIoU Metrics
//! * [`feature`] - Feature Pyramid Networks (FPN) and Backbone Architectures (ResNet, MobileNet, EfficientNet)
//! * [`augmentation`] - Color Jitter, Geometric, MixUp, CutMix, Mosaic, Bounding Box Transforms, and Photometric filters
//! * [`pooling`] - 2D/3D Pooling layers (`AvgPool`, `MaxPool`, `AdaptiveAvgPool`, `LPool`)
//! * [`ops`] - Vectorized Box Mathematics, Affine Grids, Grid Sampling, and Histogram Equalization
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_cv::prelude::*;
//!
//! let conv = Conv2d::new(3, 16, 3);
//! let input = Tensor::zeros(vec![1, 3, 32, 32]);
//! let output = conv.forward(&input);
//! assert_eq!(output.shape()[1], 16);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod augmentation;
pub mod conv;
pub mod detection;
pub mod feature;
pub mod ops;
pub mod pooling;
pub mod segmentation;

// Re-exports
pub use conv::{Conv2d, Conv2dConfig, ConvTranspose2d, DeformableConv2d, DepthwiseSeparableConv2d, GhostModule, GroupedConv2d};
pub use detection::{AnchorGenerator, BBoxFormat, DetectionHead, NmsConfig, RoIAlign};
pub use feature::{BackboneZoo, Fpn};
pub use pooling::{AvgPool2d, MaxPool2d};
pub use segmentation::{FcnHead, SegLossConfig, SegMetrics};

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_cv::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_cv::version_string;
/// assert_eq!(version_string(), "brain-cv v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-cv v{}", VERSION)
}

/// Standard prelude imports for computer vision workflows.
///
/// ```rust
/// use brain_cv::prelude::*;
/// let cfg = Conv2dConfig::default();
/// assert_eq!(cfg.in_channels, 1);
/// ```
pub mod prelude {
    pub use crate::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, DeformableConv2d, DepthwiseSeparableConv2d, GhostModule, GroupedConv2d};
    pub use crate::detection::{AnchorGenerator, BBoxFormat, DetectionHead, NmsConfig, RoIAlign};
    pub use crate::feature::{BackboneZoo, Fpn};
    pub use crate::pooling::{AvgPool2d, MaxPool2d};
    pub use crate::segmentation::{FcnHead, SegLossConfig, SegMetrics};
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
