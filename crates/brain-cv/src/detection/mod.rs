//! # Object Detection Subsystem
//!
//! Features multi-scale anchor generation, detection heads, RoIAlign, IoU variants, and non-maximum suppression.

pub mod anchor;
pub mod head;
pub mod losses;
pub mod nms;
pub mod postprocess;
pub mod roi;
pub mod roi_align;

pub use anchor::AnchorGenerator;
pub use head::DetectionHead;
pub use losses::DetectionLossConfig;
pub use nms::{non_max_suppression, NmsConfig};
pub use postprocess::convert_bbox_format;
pub use roi::RoIPool;
pub use roi_align::RoIAlign;

/// Bounding box coordinate representation formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BBoxFormat {
    #[default]
    XYXY,
    XYWH,
    CXCYWH,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
