//! # Tensor-Level Implicit Regularization
//!
//! Mixup, Cutout, and CutMix data augmentation transformations applied directly on Tensors.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{RegError, RegResult};
use super::utils::XorShift64;

/// Configuration for implicit tensor augmentations.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitRegConfig {
    pub mixup_alpha: f64,
    pub cutout_size: usize,
}

impl Default for ImplicitRegConfig {
    fn default() -> Self {
        Self {
            mixup_alpha: 0.2,
            cutout_size: 8,
        }
    }
}

/// Mixup interpolation: combines two tensors.
#[derive(Debug, Clone)]
pub struct Mixup {
    pub alpha: f64,
    pub rng: XorShift64,
}

impl Mixup {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.max(0.0),
            rng: XorShift64::new(101),
        }
    }

    /// Computes convex linear combination of two equal-shaped tensors.
    pub fn apply_mixup(&mut self, t1: &Tensor, t2: &Tensor, lam: f64) -> RegResult<Tensor> {
        if t1.shape() != t2.shape() {
            return Err(RegError::ShapeMismatch {
                expected: t1.shape().to_vec(),
                found: t2.shape().to_vec(),
            });
        }

        let d1 = t1.data();
        let d2 = t2.data();
        let mut out = vec![0.0; d1.len()];

        for i in 0..d1.len() {
            out[i] = lam * d1[i] + (1.0 - lam) * d2[i];
        }

        Ok(Tensor::from_slice(&out, t1.shape().to_vec()))
    }
}

/// Cutout mask: sets a rectangular bounding box region to zero.
#[derive(Debug, Clone)]
pub struct Cutout {
    pub patch_size: usize,
    pub rng: XorShift64,
}

impl Cutout {
    pub fn new(patch_size: usize) -> Self {
        Self {
            patch_size,
            rng: XorShift64::new(202),
        }
    }

    pub fn apply_cutout_2d(&mut self, input: &Tensor) -> RegResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(RegError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                found: shape.to_vec(),
            });
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        let h = shape[2];
        let w = shape[3];

        let mut out_data = input.data().to_vec();
        let p = self.patch_size;

        for b in 0..batch_size {
            let cy = (self.rng.next_f64() * h as f64) as usize;
            let cx = (self.rng.next_f64() * w as f64) as usize;

            let y1 = cy.saturating_sub(p / 2);
            let y2 = (cy + p / 2).min(h);
            let x1 = cx.saturating_sub(p / 2);
            let x2 = (cx + p / 2).min(w);

            for c in 0..num_channels {
                for y in y1..y2 {
                    for x in x1..x2 {
                        let idx = b * (num_channels * h * w) + c * (h * w) + y * w + x;
                        out_data[idx] = 0.0;
                    }
                }
            }
        }

        Ok(Tensor::from_slice(&out_data, shape.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
