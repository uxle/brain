//! # Pre-Packaged Transformer Model Architectures
//!
//! Production implementations of BERT-lite (encoder-only), GPT-lite (decoder-only), T5-lite (encoder-decoder), and LLaMA-lite (modern GQA/RMSNorm/RoPE/SwiGLU).
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod bert_lite;
pub mod gpt_lite;
pub mod llama_lite;
pub mod t5_lite;

pub use bert_lite::{BertLite, BertLiteConfig, BertOutput};
pub use gpt_lite::{GptLite, GptLiteConfig};
pub use llama_lite::{LlamaLite, LlamaLiteConfig};
pub use t5_lite::{T5Lite, T5LiteConfig};

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
    fn test_models_registry_1() {
        let bert_cfg = BertLiteConfig::default();
        assert_eq!(bert_cfg.hidden_dim, 768);

        let gpt_cfg = GptLiteConfig::default();
        assert_eq!(gpt_cfg.vocab_size, 50257);

        let llama_cfg = LlamaLiteConfig::default();
        assert_eq!(llama_cfg.num_kv_heads, 32);

        let t5_cfg = T5LiteConfig::default();
        assert_eq!(t5_cfg.num_buckets, 32);
    }
}
