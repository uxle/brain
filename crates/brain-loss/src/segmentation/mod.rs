//! # Segmentation Losses
//!
//! Re-exports of combined segmentation loss functions (CE + Dice, IoU).
#![allow(missing_docs)]

pub mod ce_dice;
pub use ce_dice::{CEDiceLoss, SegLossConfig};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
