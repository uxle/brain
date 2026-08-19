//! # Mathematical and Generation Utilities for Transformers
//!
//! Deterministic pseudo-random number generator, weight initializers, tensor slice operations, and floating point comparison helpers.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use brain_core::Tensor;

/// Deterministic 64-bit XorShift pseudo-random number generator for sampling and initialization.
#[derive(Debug, Clone)]
pub struct TransformerRng {
    state: u64,
}

impl TransformerRng {
    /// Creates a new `TransformerRng` with non-zero seed.
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state: s }
    }

    /// Returns the next pseudo-random 64-bit unsigned integer.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform pseudo-random float in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a standard normal distributed float $\mathcal{N}(0, 1)$ using Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Samples an integer index in `[0, upper)` according to unnormalized logits using the Gumbel-Max trick.
    pub fn sample_logits(&mut self, logits: &[f64], temperature: f64) -> usize {
        if logits.is_empty() {
            return 0;
        }
        if temperature <= 1e-6 {
            // Greedy argmax
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }

        let mut max_val = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, &l) in logits.iter().enumerate() {
            let u = self.next_f64().max(1e-15);
            let gumbel = -(-u.ln()).ln();
            let score = (l / temperature) + gumbel;
            if score > max_val {
                max_val = score;
                best_idx = i;
            }
        }

        best_idx
    }
}

impl Default for TransformerRng {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Compares two floating point slices with relative and absolute tolerance.
pub fn all_close(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let tol = atol + rtol * b[i].abs();
        if diff > tol {
            return false;
        }
    }
    true
}

/// Applies Xavier / Glorot uniform weight initialization to a flat buffer.
pub fn init_xavier_uniform(data: &mut [f64], in_dim: usize, out_dim: usize, rng: &mut TransformerRng) {
    let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
    for x in data.iter_mut() {
        *x = (rng.next_f64() * 2.0 - 1.0) * limit;
    }
}

/// Applies Kaiming / He normal weight initialization for ReLU/GELU layers.
pub fn init_kaiming_normal(data: &mut [f64], in_dim: usize, rng: &mut TransformerRng) {
    let std_dev = (2.0 / in_dim as f64).sqrt();
    for x in data.iter_mut() {
        *x = rng.next_gaussian() * std_dev;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero, clippy::all)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::attention::*;
    use crate::attention::scaled::*;
    use crate::attention::multi_head::*;
    use crate::attention::relative::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_query::*;
    use crate::attention::xformers_lite::*;
    use crate::position::*;
    use crate::position::rope::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::embedding_layers::*;
    use crate::ffn::*;
    use crate::encoder::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::decoder::*;
    use crate::decoder::layer::*;
    use crate::decoder::cross::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::generate::*;
    use crate::models::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::llama_lite::*;
    use crate::builder::*;
    use brain_core::Tensor;

    #[test]
    fn test_utils_pipeline_1() {
        let mut rng = TransformerRng::new(1 as u64);
        let val = rng.next_f64();
        assert!((0.0..1.0).contains(&val));

        let g = rng.next_gaussian();
        assert!(g.is_finite());

        let logits = vec![1.0, 2.0, 5.0, 0.5];
        let sampled = rng.sample_logits(&logits, 0.0);
        assert_eq!(sampled, 2);

        let sampled_t = rng.sample_logits(&logits, 1.0);
        assert!(sampled_t < 4);

        let mut data = vec![0.0f64; 64];
        init_xavier_uniform(&mut data, 8, 8, &mut rng);
        assert!(data.iter().any(|&x| x != 0.0));

        let a = [1.0, 2.0, 3.0];
        let b = [1.0000001, 2.0000001, 2.9999999];
        assert!(all_close(&a, &b, 1e-5, 1e-5));
    }
}
