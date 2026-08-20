//! # GAN Evaluation
//!
//! `GanEvalReport`, FID-lite, IS-lite, sample grids, `eval_gan`.
#![allow(missing_docs)]

pub mod samples;
pub use samples::{assemble_grid, fixed_latent_sample, interpolate_latents_batch};

use brain_core::Tensor;

/// GAN evaluation report with lite metrics.
#[derive(Debug, Clone, Default)]
pub struct GanEvalReport {
    pub fid_lite: f64,
    pub is_lite: f64,
    pub avg_d_real: f64,
    pub avg_d_fake: f64,
    pub num_samples: usize,
}

impl GanEvalReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn summary(&self) -> String {
        format!(
            "EvalReport[FID-lite={:.4} IS-lite={:.4} D(real)={:.4} D(fake)={:.4} n={}]",
            self.fid_lite, self.is_lite, self.avg_d_real, self.avg_d_fake, self.num_samples
        )
    }
}

/// FID-lite: squared L2 distance between real and fake feature statistics.
pub fn fid_lite(real_feats: &[Tensor], fake_feats: &[Tensor]) -> f64 {
    if real_feats.is_empty() || fake_feats.is_empty() {
        return 0.0;
    }
    let real_mean = feature_mean(real_feats);
    let fake_mean = feature_mean(fake_feats);
    real_mean
        .iter()
        .zip(fake_mean.iter())
        .map(|(r, f)| (r - f).powi(2))
        .sum()
}

fn feature_mean(feats: &[Tensor]) -> Vec<f64> {
    if feats.is_empty() {
        return vec![];
    }
    let n = feats.len() as f64;
    let dim = feats[0].to_vec().len();
    let mut mean = vec![0.0f64; dim];
    for t in feats {
        for (m, v) in mean.iter_mut().zip(t.to_vec().iter()) {
            *m += v / n;
        }
    }
    mean
}

/// IS-lite: average KL divergence proxy from logits (inception-free).
pub fn is_lite(logits: &[Tensor]) -> f64 {
    if logits.is_empty() {
        return 1.0;
    }
    let dim = logits[0].to_vec().len().max(1) as f64;
    let avg_entropy: f64 = logits
        .iter()
        .map(|t| {
            let v = t.to_vec();
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            -exp.iter()
                .map(|e| {
                    let p = e / sum;
                    p * p.max(1e-15).ln()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / logits.len() as f64;
    (avg_entropy / dim.ln().max(1e-8)).exp()
}

/// Runs GAN evaluation on a set of real and generated tensors.
pub fn eval_gan(real: &[Tensor], fake: &[Tensor]) -> GanEvalReport {
    GanEvalReport {
        fid_lite: fid_lite(real, fake),
        is_lite: is_lite(fake),
        avg_d_real: real
            .iter()
            .map(|t| t.to_vec().iter().sum::<f64>().tanh())
            .sum::<f64>()
            / real.len().max(1) as f64,
        avg_d_fake: fake
            .iter()
            .map(|t| t.to_vec().iter().sum::<f64>().tanh())
            .sum::<f64>()
            / fake.len().max(1) as f64,
        num_samples: fake.len(),
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
