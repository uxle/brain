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

    #[test]
    fn test_utils_pipeline_2() {
        let mut rng = TransformerRng::new(2 as u64);
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

    #[test]
    fn test_utils_pipeline_3() {
        let mut rng = TransformerRng::new(3 as u64);
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

    #[test]
    fn test_utils_pipeline_4() {
        let mut rng = TransformerRng::new(4 as u64);
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

    #[test]
    fn test_utils_pipeline_5() {
        let mut rng = TransformerRng::new(5 as u64);
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

    #[test]
    fn test_utils_pipeline_6() {
        let mut rng = TransformerRng::new(6 as u64);
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

    #[test]
    fn test_utils_pipeline_7() {
        let mut rng = TransformerRng::new(7 as u64);
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

    #[test]
    fn test_utils_pipeline_8() {
        let mut rng = TransformerRng::new(8 as u64);
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

    #[test]
    fn test_utils_pipeline_9() {
        let mut rng = TransformerRng::new(9 as u64);
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

    #[test]
    fn test_utils_pipeline_10() {
        let mut rng = TransformerRng::new(10 as u64);
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

    #[test]
    fn test_utils_pipeline_11() {
        let mut rng = TransformerRng::new(11 as u64);
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

    #[test]
    fn test_utils_pipeline_12() {
        let mut rng = TransformerRng::new(12 as u64);
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

    #[test]
    fn test_utils_pipeline_13() {
        let mut rng = TransformerRng::new(13 as u64);
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

    #[test]
    fn test_utils_pipeline_14() {
        let mut rng = TransformerRng::new(14 as u64);
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

    #[test]
    fn test_utils_pipeline_15() {
        let mut rng = TransformerRng::new(15 as u64);
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

    #[test]
    fn test_utils_pipeline_16() {
        let mut rng = TransformerRng::new(16 as u64);
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

    #[test]
    fn test_utils_pipeline_17() {
        let mut rng = TransformerRng::new(17 as u64);
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

    #[test]
    fn test_utils_pipeline_18() {
        let mut rng = TransformerRng::new(18 as u64);
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

    #[test]
    fn test_utils_pipeline_19() {
        let mut rng = TransformerRng::new(19 as u64);
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

    #[test]
    fn test_utils_pipeline_20() {
        let mut rng = TransformerRng::new(20 as u64);
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

    #[test]
    fn test_utils_pipeline_21() {
        let mut rng = TransformerRng::new(21 as u64);
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

    #[test]
    fn test_utils_pipeline_22() {
        let mut rng = TransformerRng::new(22 as u64);
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

    #[test]
    fn test_utils_pipeline_23() {
        let mut rng = TransformerRng::new(23 as u64);
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

    #[test]
    fn test_utils_pipeline_24() {
        let mut rng = TransformerRng::new(24 as u64);
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

    #[test]
    fn test_utils_pipeline_25() {
        let mut rng = TransformerRng::new(25 as u64);
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

    #[test]
    fn test_utils_pipeline_26() {
        let mut rng = TransformerRng::new(26 as u64);
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

    #[test]
    fn test_utils_pipeline_27() {
        let mut rng = TransformerRng::new(27 as u64);
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

    #[test]
    fn test_utils_pipeline_28() {
        let mut rng = TransformerRng::new(28 as u64);
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

    #[test]
    fn test_utils_pipeline_29() {
        let mut rng = TransformerRng::new(29 as u64);
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

    #[test]
    fn test_utils_pipeline_30() {
        let mut rng = TransformerRng::new(30 as u64);
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

    #[test]
    fn test_utils_pipeline_31() {
        let mut rng = TransformerRng::new(31 as u64);
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

    #[test]
    fn test_utils_pipeline_32() {
        let mut rng = TransformerRng::new(32 as u64);
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

    #[test]
    fn test_utils_pipeline_33() {
        let mut rng = TransformerRng::new(33 as u64);
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

    #[test]
    fn test_utils_pipeline_34() {
        let mut rng = TransformerRng::new(34 as u64);
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

    #[test]
    fn test_utils_pipeline_35() {
        let mut rng = TransformerRng::new(35 as u64);
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

    #[test]
    fn test_utils_pipeline_36() {
        let mut rng = TransformerRng::new(36 as u64);
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

    #[test]
    fn test_utils_pipeline_37() {
        let mut rng = TransformerRng::new(37 as u64);
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

    #[test]
    fn test_utils_pipeline_38() {
        let mut rng = TransformerRng::new(38 as u64);
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

    #[test]
    fn test_utils_pipeline_39() {
        let mut rng = TransformerRng::new(39 as u64);
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

    #[test]
    fn test_utils_pipeline_40() {
        let mut rng = TransformerRng::new(40 as u64);
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

    #[test]
    fn test_utils_pipeline_41() {
        let mut rng = TransformerRng::new(41 as u64);
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

    #[test]
    fn test_utils_pipeline_42() {
        let mut rng = TransformerRng::new(42 as u64);
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

    #[test]
    fn test_utils_pipeline_43() {
        let mut rng = TransformerRng::new(43 as u64);
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

    #[test]
    fn test_utils_pipeline_44() {
        let mut rng = TransformerRng::new(44 as u64);
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

    #[test]
    fn test_utils_pipeline_45() {
        let mut rng = TransformerRng::new(45 as u64);
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

    #[test]
    fn test_utils_pipeline_46() {
        let mut rng = TransformerRng::new(46 as u64);
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

    #[test]
    fn test_utils_pipeline_47() {
        let mut rng = TransformerRng::new(47 as u64);
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

    #[test]
    fn test_utils_pipeline_48() {
        let mut rng = TransformerRng::new(48 as u64);
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

    #[test]
    fn test_utils_pipeline_49() {
        let mut rng = TransformerRng::new(49 as u64);
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

    #[test]
    fn test_utils_pipeline_50() {
        let mut rng = TransformerRng::new(50 as u64);
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

    #[test]
    fn test_utils_pipeline_51() {
        let mut rng = TransformerRng::new(51 as u64);
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

    #[test]
    fn test_utils_pipeline_52() {
        let mut rng = TransformerRng::new(52 as u64);
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

    #[test]
    fn test_utils_pipeline_53() {
        let mut rng = TransformerRng::new(53 as u64);
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

    #[test]
    fn test_utils_pipeline_54() {
        let mut rng = TransformerRng::new(54 as u64);
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

    #[test]
    fn test_utils_pipeline_55() {
        let mut rng = TransformerRng::new(55 as u64);
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

    #[test]
    fn test_utils_pipeline_56() {
        let mut rng = TransformerRng::new(56 as u64);
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

    #[test]
    fn test_utils_pipeline_57() {
        let mut rng = TransformerRng::new(57 as u64);
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

    #[test]
    fn test_utils_pipeline_58() {
        let mut rng = TransformerRng::new(58 as u64);
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

    #[test]
    fn test_utils_pipeline_59() {
        let mut rng = TransformerRng::new(59 as u64);
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

    #[test]
    fn test_utils_pipeline_60() {
        let mut rng = TransformerRng::new(60 as u64);
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

    #[test]
    fn test_utils_pipeline_61() {
        let mut rng = TransformerRng::new(61 as u64);
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

    #[test]
    fn test_utils_pipeline_62() {
        let mut rng = TransformerRng::new(62 as u64);
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

    #[test]
    fn test_utils_pipeline_63() {
        let mut rng = TransformerRng::new(63 as u64);
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

    #[test]
    fn test_utils_pipeline_64() {
        let mut rng = TransformerRng::new(64 as u64);
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

    #[test]
    fn test_utils_pipeline_65() {
        let mut rng = TransformerRng::new(65 as u64);
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

    #[test]
    fn test_utils_pipeline_66() {
        let mut rng = TransformerRng::new(66 as u64);
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

    #[test]
    fn test_utils_pipeline_67() {
        let mut rng = TransformerRng::new(67 as u64);
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

    #[test]
    fn test_utils_pipeline_68() {
        let mut rng = TransformerRng::new(68 as u64);
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

    #[test]
    fn test_utils_pipeline_69() {
        let mut rng = TransformerRng::new(69 as u64);
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

    #[test]
    fn test_utils_pipeline_70() {
        let mut rng = TransformerRng::new(70 as u64);
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

    #[test]
    fn test_utils_pipeline_71() {
        let mut rng = TransformerRng::new(71 as u64);
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

    #[test]
    fn test_utils_pipeline_72() {
        let mut rng = TransformerRng::new(72 as u64);
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

    #[test]
    fn test_utils_pipeline_73() {
        let mut rng = TransformerRng::new(73 as u64);
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

    #[test]
    fn test_utils_pipeline_74() {
        let mut rng = TransformerRng::new(74 as u64);
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

    #[test]
    fn test_utils_pipeline_75() {
        let mut rng = TransformerRng::new(75 as u64);
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

    #[test]
    fn test_utils_pipeline_76() {
        let mut rng = TransformerRng::new(76 as u64);
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

    #[test]
    fn test_utils_pipeline_77() {
        let mut rng = TransformerRng::new(77 as u64);
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

    #[test]
    fn test_utils_pipeline_78() {
        let mut rng = TransformerRng::new(78 as u64);
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

    #[test]
    fn test_utils_pipeline_79() {
        let mut rng = TransformerRng::new(79 as u64);
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

    #[test]
    fn test_utils_pipeline_80() {
        let mut rng = TransformerRng::new(80 as u64);
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

    #[test]
    fn test_utils_pipeline_81() {
        let mut rng = TransformerRng::new(81 as u64);
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

    #[test]
    fn test_utils_pipeline_82() {
        let mut rng = TransformerRng::new(82 as u64);
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

    #[test]
    fn test_utils_pipeline_83() {
        let mut rng = TransformerRng::new(83 as u64);
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

    #[test]
    fn test_utils_pipeline_84() {
        let mut rng = TransformerRng::new(84 as u64);
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

    #[test]
    fn test_utils_pipeline_85() {
        let mut rng = TransformerRng::new(85 as u64);
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

    #[test]
    fn test_utils_pipeline_86() {
        let mut rng = TransformerRng::new(86 as u64);
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

    #[test]
    fn test_utils_pipeline_87() {
        let mut rng = TransformerRng::new(87 as u64);
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

    #[test]
    fn test_utils_pipeline_88() {
        let mut rng = TransformerRng::new(88 as u64);
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

    #[test]
    fn test_utils_pipeline_89() {
        let mut rng = TransformerRng::new(89 as u64);
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

    #[test]
    fn test_utils_pipeline_90() {
        let mut rng = TransformerRng::new(90 as u64);
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

    #[test]
    fn test_utils_pipeline_91() {
        let mut rng = TransformerRng::new(91 as u64);
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

    #[test]
    fn test_utils_pipeline_92() {
        let mut rng = TransformerRng::new(92 as u64);
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

    #[test]
    fn test_utils_pipeline_93() {
        let mut rng = TransformerRng::new(93 as u64);
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

    #[test]
    fn test_utils_pipeline_94() {
        let mut rng = TransformerRng::new(94 as u64);
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

    #[test]
    fn test_utils_pipeline_95() {
        let mut rng = TransformerRng::new(95 as u64);
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

    #[test]
    fn test_utils_pipeline_96() {
        let mut rng = TransformerRng::new(96 as u64);
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

    #[test]
    fn test_utils_pipeline_97() {
        let mut rng = TransformerRng::new(97 as u64);
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

    #[test]
    fn test_utils_pipeline_98() {
        let mut rng = TransformerRng::new(98 as u64);
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

    #[test]
    fn test_utils_pipeline_99() {
        let mut rng = TransformerRng::new(99 as u64);
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

    #[test]
    fn test_utils_pipeline_100() {
        let mut rng = TransformerRng::new(100 as u64);
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

    #[test]
    fn test_utils_pipeline_101() {
        let mut rng = TransformerRng::new(101 as u64);
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

    #[test]
    fn test_utils_pipeline_102() {
        let mut rng = TransformerRng::new(102 as u64);
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

    #[test]
    fn test_utils_pipeline_103() {
        let mut rng = TransformerRng::new(103 as u64);
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

    #[test]
    fn test_utils_pipeline_104() {
        let mut rng = TransformerRng::new(104 as u64);
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

    #[test]
    fn test_utils_pipeline_105() {
        let mut rng = TransformerRng::new(105 as u64);
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

    #[test]
    fn test_utils_pipeline_106() {
        let mut rng = TransformerRng::new(106 as u64);
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

    #[test]
    fn test_utils_pipeline_107() {
        let mut rng = TransformerRng::new(107 as u64);
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

    #[test]
    fn test_utils_pipeline_108() {
        let mut rng = TransformerRng::new(108 as u64);
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

    #[test]
    fn test_utils_pipeline_109() {
        let mut rng = TransformerRng::new(109 as u64);
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

    #[test]
    fn test_utils_pipeline_110() {
        let mut rng = TransformerRng::new(110 as u64);
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

    #[test]
    fn test_utils_pipeline_111() {
        let mut rng = TransformerRng::new(111 as u64);
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

    #[test]
    fn test_utils_pipeline_112() {
        let mut rng = TransformerRng::new(112 as u64);
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

    #[test]
    fn test_utils_pipeline_113() {
        let mut rng = TransformerRng::new(113 as u64);
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

    #[test]
    fn test_utils_pipeline_114() {
        let mut rng = TransformerRng::new(114 as u64);
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

    #[test]
    fn test_utils_pipeline_115() {
        let mut rng = TransformerRng::new(115 as u64);
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

    #[test]
    fn test_utils_pipeline_116() {
        let mut rng = TransformerRng::new(116 as u64);
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

    #[test]
    fn test_utils_pipeline_117() {
        let mut rng = TransformerRng::new(117 as u64);
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

    #[test]
    fn test_utils_pipeline_118() {
        let mut rng = TransformerRng::new(118 as u64);
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

    #[test]
    fn test_utils_pipeline_119() {
        let mut rng = TransformerRng::new(119 as u64);
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

    #[test]
    fn test_utils_pipeline_120() {
        let mut rng = TransformerRng::new(120 as u64);
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

    #[test]
    fn test_utils_pipeline_121() {
        let mut rng = TransformerRng::new(121 as u64);
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

    #[test]
    fn test_utils_pipeline_122() {
        let mut rng = TransformerRng::new(122 as u64);
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

    #[test]
    fn test_utils_pipeline_123() {
        let mut rng = TransformerRng::new(123 as u64);
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

    #[test]
    fn test_utils_pipeline_124() {
        let mut rng = TransformerRng::new(124 as u64);
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

    #[test]
    fn test_utils_pipeline_125() {
        let mut rng = TransformerRng::new(125 as u64);
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

    #[test]
    fn test_utils_pipeline_126() {
        let mut rng = TransformerRng::new(126 as u64);
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

    #[test]
    fn test_utils_pipeline_127() {
        let mut rng = TransformerRng::new(127 as u64);
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

    #[test]
    fn test_utils_pipeline_128() {
        let mut rng = TransformerRng::new(128 as u64);
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
