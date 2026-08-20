//! # Adaptive Gradient Clipping (AGC)
//!
//! Layer-wise and per-parameter adaptive gradient clipping based on weight-to-gradient norm ratios (Brock & Geiping).
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration container for Adaptive Gradient Clipping.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveClipConfig {
    pub clipping_rate: f64,
    pub eps: f64,
}

impl Default for AdaptiveClipConfig {
    fn default() -> Self {
        Self {
            clipping_rate: 0.01,
            eps: 1e-3,
        }
    }
}

/// AGC (Adaptive Gradient Clipper) engine.
#[derive(Debug, Clone)]
pub struct AGC {
    pub config: AdaptiveClipConfig,
}

impl AGC {
    pub fn new(clipping_rate: f64, eps: f64) -> Self {
        Self {
            config: AdaptiveClipConfig { clipping_rate, eps },
        }
    }

    pub fn clip(&self, params: &mut [Tensor], grads: &mut [Tensor]) {
        clip_grad_adaptive_(params, grads, self.config.clipping_rate, self.config.eps);
    }
}

/// Applies adaptive gradient clipping per parameter tensor.
///
/// Formula: max_norm = clip_factor * max(||w||, eps)
/// if ||g|| > max_norm: g = g * (max_norm / ||g||)
pub fn clip_grad_adaptive_(
    params: &mut [Tensor],
    grads: &mut [Tensor],
    clipping_rate: f64,
    eps: f64,
) {
    if params.len() != grads.len() || clipping_rate <= 0.0 {
        return;
    }

    for (p, g) in params.iter_mut().zip(grads.iter_mut()) {
        let p_data = p.data();
        let g_data = g.data_mut();
        let n = p_data.len();
        if n != g_data.len() {
            continue;
        }

        let mut p_sq = 0.0;
        let mut g_sq = 0.0;

        for i in 0..n {
            p_sq += p_data[i] * p_data[i];
            g_sq += g_data[i] * g_data[i];
        }

        let p_norm = p_sq.sqrt().max(eps);
        let g_norm = g_sq.sqrt();
        let max_g_norm = p_norm * clipping_rate;

        if g_norm > max_g_norm && g_norm > 0.0 {
            let trigger_factor = max_g_norm / g_norm;
            for val in g_data.iter_mut() {
                *val *= trigger_factor;
            }
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
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
