//! # Dropout Family Modules
//!
//! Standard inverted dropout, 2D spatial feature map dropout, 3D volumetric dropout, and fused operations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod alpha;
pub mod adaptive;

use brain_core::Tensor;
use super::core::{RegError, RegKind, RegResult, Regularization};
use super::utils::XorShift64;

/// Inverted Dropout layer: zeroing elements with probability p and scaling remaining by 1 / (1 - p).
#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f64,
    pub is_training: bool,
    pub rng: XorShift64,
    pub last_mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(1337),
            last_mask: None,
        }
    }

    pub fn with_seed(p: f64, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(seed),
            last_mask: None,
        }
    }

    /// Computes fused dropout and residual addition.
    pub fn forward_add(&mut self, input: &Tensor, residual: &Tensor) -> RegResult<Tensor> {
        if input.shape() != residual.shape() {
            return Err(RegError::ShapeMismatch {
                expected: input.shape().to_vec(),
                found: residual.shape().to_vec(),
            });
        }
        let dropped = self.apply(input)?;
        let mut out = dropped.clone();
        let out_data = out.data_mut();
        let res_data = residual.data();
        for i in 0..out_data.len() {
            out_data[i] += res_data[i];
        }
        Ok(out)
    }
}

impl Regularization for Dropout {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }
        if self.p == 1.0 {
            return Ok(Tensor::zeros(input.shape().to_vec()));
        }

        let scale = 1.0 / (1.0 - self.p);
        let data = input.data();
        let n = data.len();
        let mut out_data = vec![0.0; n];
        let mut mask_data = vec![0.0; n];

        for i in 0..n {
            let r = self.rng.next_f64();
            if r >= self.p {
                mask_data[i] = 1.0;
                out_data[i] = data[i] * scale;
            }
        }

        self.last_mask = Some(Tensor::from_slice(&mask_data, input.shape().to_vec()));
        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }

    fn train_mode(&mut self) {
        self.is_training = true;
    }

    fn eval_mode(&mut self) {
        self.is_training = false;
    }

    fn kind(&self) -> RegKind {
        RegKind::Dropout
    }
}

/// 2D Spatial Feature Map Dropout (zeroing entire channels independently).
#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    pub is_training: bool,
    pub rng: XorShift64,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(42),
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> RegResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(RegError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                found: shape.to_vec(),
            });
        }
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        let spatial_size = shape[2] * shape[3];
        let scale = 1.0 / (1.0 - self.p);

        let data = input.data();
        let mut out_data = vec![0.0; data.len()];

        for b in 0..batch_size {
            for c in 0..num_channels {
                let keep = self.rng.next_f64() >= self.p;
                if keep {
                    let start = (b * num_channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for i in start..end {
                        out_data[i] = data[i] * scale;
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
