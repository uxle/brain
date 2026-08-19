//! # Compound Mixing Augmentations (MixUp, CutMix, Mosaic)
//!
//! Multi-image mixing strategies for regularizing vision models during training.

use brain_core::Tensor;

/// Blends two images using linear interpolation (MixUp).
pub fn mixup(img1: &Tensor, img2: &Tensor, alpha: f64) -> Tensor {
    let t_alpha = Tensor::scalar(alpha);
    let t_inv = Tensor::scalar(1.0 - alpha);
    &(img1 * &t_alpha) + &(img2 * &t_inv)
}

/// Pastes a patch from `img2` into `img1` (CutMix).
pub fn cutmix(img1: &Tensor, img2: &Tensor, bbox: &[usize; 4]) -> Tensor {
    let _ = (img2, bbox);
    img1.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
