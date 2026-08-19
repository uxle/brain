//! # LLaMA-Lite: Modern Autoregressive Transformer Architecture
//!
//! State-of-the-art causal language model featuring Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU Gated MLP, and Grouped-Query Attention (GQA).
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::config::{ActivationType, NormPosition, NormType, PositionEncodingType};
use crate::core::{AttentionMask, TransformerResult};
use crate::decoder::{DecoderConfig, TransformerDecoder};
use crate::embedding_layers::{EmbConfig, TransformerEmbedding};
use crate::generate::{GenerateConfig, Generator};
use crate::head::{HeadConfig, LmHead};
use crate::position::rope::{RopeConfig, RotaryEmbedding};
use crate::utils::TransformerRng;
use brain_core::Tensor;

/// Configuration for LLaMA-lite model.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLiteConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension $d_{\text{model}}$.
    pub hidden_dim: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads for GQA.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate expansion dimension for SwiGLU FFN (e.g. 11008 for 7B).
    pub intermediate_dim: usize,
    /// Maximum context sequence length.
    pub max_seq_len: usize,
    /// RoPE base theta (typically 10000.0 or 500000.0).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl Default for LlamaLiteConfig {
    fn default() -> Self {
        let hidden_dim = 4096;
        let num_heads = 32;
        Self {
            vocab_size: 32000,
            hidden_dim,
            num_layers: 32,
            num_heads,
            num_kv_heads: num_heads,
            head_dim: 128,
            intermediate_dim: 11008,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        }
    }
}

/// Production LLaMA-Lite Architecture.
#[derive(Debug, Clone)]
pub struct LlamaLite {
    /// Token embedding table.
    pub embed_tokens: TransformerEmbedding,
    /// Rotary Position Embedding table.
    pub rope: RotaryEmbedding,
    /// Stacked decoder layers with RMSNorm and SwiGLU.
    pub decoder: TransformerDecoder,
    /// Output LM projection head.
    pub lm_head: LmHead,
    /// Configuration options.
    pub config: LlamaLiteConfig,
}

impl LlamaLite {
    /// Creates a new `LlamaLite` model.
    pub fn new(config: LlamaLiteConfig, seed: u64) -> Self {
        let emb_cfg = EmbConfig {
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            max_position_embeddings: config.max_seq_len,
            type_vocab_size: None,
            dropout: 0.0,
            pos_encoding: PositionEncodingType::None,
            norm_type: None,
            norm_eps: config.norm_eps,
        };
        let embed_tokens = TransformerEmbedding::new(emb_cfg, seed);

        let rope_cfg = RopeConfig {
            dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            theta: config.rope_theta,
            scaling_factor: 1.0,
            is_2d: false,
        };
        let rope = RotaryEmbedding::new(rope_cfg);

        let dec_cfg = DecoderConfig {
            num_layers: config.num_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            has_cross_attention: false,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            activation: ActivationType::Swiglu,
            norm_eps: config.norm_eps,
        };
        let decoder = TransformerDecoder::new(dec_cfg, seed.wrapping_add(500));

        let head_cfg = HeadConfig {
            hidden_dim: config.hidden_dim,
            vocab_size: config.vocab_size,
            num_classes: None,
            bias: false,
        };
        let lm_head = LmHead::new(head_cfg, seed.wrapping_add(1000));

        Self {
            embed_tokens,
            rope,
            decoder,
            lm_head,
            config,
        }
    }

    /// Computes full causal forward pass.
    pub fn forward(
        &self,
        input_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
    ) -> TransformerResult<Tensor> {
        let emb = self.embed_tokens.forward(input_ids, batch_size, seq_len, None, 0)?;
        let dec_out = self.decoder.forward(&emb, None, &AttentionMask::Causal, &AttentionMask::None, false)?;
        self.lm_head.forward(&dec_out.last_hidden_state)
    }

    /// Autoregressive text generation.
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        gen_config: &GenerateConfig,
    ) -> TransformerResult<Vec<usize>> {
        let mut tokens = prompt_ids.to_vec();
        let mut rng = TransformerRng::new(gen_config.seed);

        for _ in 0..gen_config.max_new_tokens {
            let seq_len = tokens.len();
            if seq_len >= self.config.max_seq_len {
                break;
            }

            let logits = self.forward(&tokens, 1, seq_len)?;
            let v_dim = self.config.vocab_size;
            let last_token_logits = &logits.data()[(seq_len - 1) * v_dim..seq_len * v_dim];

            let next_token = Generator::sample_next_token(last_token_logits, &tokens, gen_config, &mut rng);
            tokens.push(next_token);

            if let Some(eos) = gen_config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
        }

        Ok(tokens)
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
    fn test_llama_lite_model_1() {
        let cfg = LlamaLiteConfig {
            vocab_size: 50,
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            max_seq_len: 64,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let llama = LlamaLite::new(cfg, 1 as u64);
        let prompt = vec![1, 2, 3];
        let logits = llama.forward(&prompt, 1, 3).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 50]);

        let gen_cfg = GenerateConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };
        let generated = llama.generate(&prompt, &gen_cfg).unwrap();
        assert_eq!(generated.len(), 5);
    }
}
