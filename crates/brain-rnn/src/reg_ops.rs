//! # Recurrent Regularizers & Variational Dropout
//!
//! Step-locked recurrent dropout masks (Gal & Ghahramani) and Zoneout state preservation.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown,
    clippy::module_inception,
    clippy::manual_memcpy
)]

use super::utils::RnnRng;
use brain_core::Tensor;

/// Variational Dropout Module applying consistent locked mask across time steps.
#[derive(Debug, Clone)]
pub struct VariationalDropout {
    pub p: f64,
    pub mask: Option<Tensor>,
    pub rng: RnnRng,
}

impl VariationalDropout {
    pub fn new(p: f64, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            mask: None,
            rng: RnnRng::new(seed),
        }
    }

    pub fn reset_mask(&mut self, shape: &[usize]) {
        if self.p == 0.0 {
            self.mask = None;
            return;
        }
        let numel: usize = shape.iter().product();
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_data = vec![0.0; numel];
        for val in mask_data.iter_mut() {
            if self.rng.next_f64() >= self.p {
                *val = scale;
            }
        }
        self.mask = Some(Tensor::from_slice(&mask_data, shape.to_vec()));
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        if let Some(mask) = &self.mask {
            let d_x = x.data();
            let d_m = mask.data();
            let mut out = vec![0.0; d_x.len()];
            for i in 0..d_x.len() {
                out[i] = d_x[i] * d_m[i];
            }
            Tensor::from_slice(&out, x.shape().to_vec())
        } else {
            x.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::cells::*;
    use crate::config::*;
    use crate::core::*;
    use crate::helper::*;
    use crate::init_rnn::*;
    use crate::ops::*;
    use crate::process::*;
    use crate::reg_ops::*;
    use crate::seq::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
