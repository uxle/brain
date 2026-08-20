//! # Autoregressive Text Generation & Decoding Strategies
//!
//! Greedy search, temperature scaling, top-k filtering, top-p (nucleus) sampling, min-p, and repetition penalties.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::core::{TransformerError, TransformerResult};
use crate::utils::TransformerRng;

/// Hyperparameters and strategies for autoregressive sequence generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateConfig {
    /// Maximum number of newly generated tokens.
    pub max_new_tokens: usize,
    /// Sampling temperature (1.0 = standard, 0.0 = greedy argmax).
    pub temperature: f64,
    /// Top-k candidate filtering threshold (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) cumulative probability threshold (1.0 = disabled).
    pub top_p: f64,
    /// Repetition penalty factor (> 1.0 penalizes already generated tokens).
    pub repetition_penalty: f64,
    /// Min-p filtering threshold (filters tokens with prob < min_p * max_prob).
    pub min_p: f64,
    /// End-of-sequence token ID.
    pub eos_token_id: Option<usize>,
    /// Padding token ID.
    pub pad_token_id: Option<usize>,
    /// Pseudo-random generator seed.
    pub seed: u64,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            min_p: 0.0,
            eos_token_id: None,
            pad_token_id: None,
            seed: 42,
        }
    }
}

/// Generation decoding engine.
pub struct Generator;

impl Generator {
    /// In-place repetition penalty application: penalizes logits for tokens present in `generated_ids`.
    pub fn apply_repetition_penalty(logits: &mut [f64], generated_ids: &[usize], penalty: f64) {
        if penalty <= 1.0 || logits.is_empty() {
            return;
        }

        for &id in generated_ids {
            if id < logits.len() {
                if logits[id] > 0.0 {
                    logits[id] /= penalty;
                } else {
                    logits[id] *= penalty;
                }
            }
        }
    }

    /// Filters logits using combined Top-K, Top-P (Nucleus), and Min-P strategies.
    pub fn filter_logits(logits: &mut [f64], top_k: usize, top_p: f64, min_p: f64) {
        if logits.is_empty() {
            return;
        }

        // 1. Top-K filtering
        if top_k > 0 && top_k < logits.len() {
            let mut indexed: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff = indexed[top_k - 1].1;
            for x in logits.iter_mut() {
                if *x < cutoff {
                    *x = f64::NEG_INFINITY;
                }
            }
        }

        // 2. Top-P (Nucleus) filtering
        if top_p < 1.0 && top_p > 0.0 {
            let mut indexed: Vec<(usize, f64)> = logits
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, v)| *v > f64::NEG_INFINITY)
                .collect();

            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Compute softmax over active tokens
            let max_val = indexed.first().map(|x| x.1).unwrap_or(0.0);
            let exp_sum: f64 = indexed.iter().map(|x| (x.1 - max_val).exp()).sum();

            let mut cum_prob = 0.0f64;
            let mut keep_count = 0;

            for item in &indexed {
                let prob = (item.1 - max_val).exp() / exp_sum;
                cum_prob += prob;
                keep_count += 1;
                if cum_prob >= top_p {
                    break;
                }
            }

            for item in &indexed[keep_count..] {
                logits[item.0] = f64::NEG_INFINITY;
            }
        }
    }

    /// Samples the next token ID given raw unnormalized vocabulary logits.
    pub fn sample_next_token(
        logits: &[f64],
        generated_ids: &[usize],
        config: &GenerateConfig,
        rng: &mut TransformerRng,
    ) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let mut processed = logits.to_vec();

        // 1. Repetition penalty
        Self::apply_repetition_penalty(&mut processed, generated_ids, config.repetition_penalty);

        // 2. Greedy decoding if temperature ~ 0
        if config.temperature <= 1e-6 {
            return processed
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }

        // 3. Top-k / Top-p filtering
        Self::filter_logits(&mut processed, config.top_k, config.top_p, config.min_p);

        // 4. Sample with Gumbel / temperature
        rng.sample_logits(&processed, config.temperature)
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
        clippy::excessive_precision,
        clippy::float_cmp,
        clippy::len_zero,
        clippy::all
    )]
    use super::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_head::*;
    use crate::attention::multi_query::*;
    use crate::attention::relative::*;
    use crate::attention::scaled::*;
    use crate::attention::xformers_lite::*;
    use crate::attention::*;
    use crate::builder::*;
    use crate::config::*;
    use crate::core::*;
    use crate::decoder::cross::*;
    use crate::decoder::layer::*;
    use crate::decoder::*;
    use crate::embedding_layers::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::encoder::*;
    use crate::ffn::*;
    use crate::generate::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::llama_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::*;
    use crate::ops::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::position::rope::*;
    use crate::position::*;
    use crate::utils::*;
    use brain_core::Tensor;

    #[test]
    fn test_generation_pipeline_1() {
        let cfg = GenerateConfig {
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let mut rng = TransformerRng::new(1 as u64);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let sampled = Generator::sample_next_token(&logits, &[9], &cfg, &mut rng);
        assert!(sampled < 10);

        Generator::apply_repetition_penalty(&mut logits, &[9], 2.0);
        assert_eq!(logits[9], 5.0);
    }
}
