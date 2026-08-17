//! # Tokenizer Post-Processing: Special Tokens, Truncation, and Padding
//!
//! Framing templates (`[CLS] A [SEP] B [SEP]`), sentence-pair token type mapping, and batch alignment.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TokenId, TokenizedOutput};

/// Strategy for sequence pair truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Iteratively truncate from the longest sequence.
    #[default]
    LongestFirst,
    /// Truncate only from the first sequence.
    OnlyFirst,
    /// Truncate only from the second sequence.
    OnlySecond,
}

/// Configuration for post-processing tokenized outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostConfig {
    /// Maximum allowed sequence length.
    pub max_length: usize,
    /// Whether to pad sequences to max length.
    pub padding: bool,
    /// Numeric pad token ID.
    pub pad_id: TokenId,
    /// Pad token string.
    pub pad_token: String,
    /// Whether to truncate sequences exceeding max length.
    pub truncation: bool,
    /// Sequence pair truncation strategy.
    pub truncation_strategy: TruncationStrategy,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenId>,
    /// End-of-sequence token ID.
    pub eos_token_id: Option<TokenId>,
    /// Classification / document token ID.
    pub cls_token_id: Option<TokenId>,
    /// Separator token ID.
    pub sep_token_id: Option<TokenId>,
    /// Beginning-of-sequence token string.
    pub bos_token: Option<String>,
    /// End-of-sequence token string.
    pub eos_token: Option<String>,
    /// Classification token string.
    pub cls_token: Option<String>,
    /// Separator token string.
    pub sep_token: Option<String>,
}

impl Default for PostConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            padding: false,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            truncation: true,
            truncation_strategy: TruncationStrategy::LongestFirst,
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            bos_token_id: None,
            eos_token_id: None,
            cls_token: Some("[CLS]".to_string()),
            sep_token: Some("[SEP]".to_string()),
            bos_token: None,
            eos_token: None,
        }
    }
}

/// Post-processing engine for formatting tokenized representations.
#[derive(Debug, Clone, Default)]
pub struct PostProcessor {
    /// Configuration options.
    pub config: PostConfig,
}

impl PostProcessor {
    /// Creates a new `PostProcessor`.
    pub fn new(config: PostConfig) -> Self {
        Self { config }
    }

    /// Formats a single sequence with special tokens, truncation, and padding.
    pub fn process_single(&self, mut output: TokenizedOutput) -> TokenizedOutput {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut special_mask = Vec::new();

        if let (Some(cls_id), Some(ref cls_tok)) = (self.config.cls_token_id, &self.config.cls_token) {
            ids.push(cls_id);
            tokens.push(cls_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
        } else if let (Some(bos_id), Some(ref bos_tok)) = (self.config.bos_token_id, &self.config.bos_token) {
            ids.push(bos_id);
            tokens.push(bos_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
        }

        let num_specials = ids.len() + if self.config.sep_token_id.is_some() || self.config.eos_token_id.is_some() { 1 } else { 0 };
        let available_slots = if self.config.truncation {
            self.config.max_length.saturating_sub(num_specials)
        } else {
            output.ids.len()
        };

        let take_len = output.ids.len().min(available_slots);
        ids.extend_from_slice(&output.ids[..take_len]);
        tokens.extend_from_slice(&output.tokens[..take_len]);
        offsets.extend_from_slice(&output.offsets[..take_len]);
        special_mask.extend(std::iter::repeat(0u8).take(take_len));

        if let (Some(sep_id), Some(ref sep_tok)) = (self.config.sep_token_id, &self.config.sep_token) {
            ids.push(sep_id);
            tokens.push(sep_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
        } else if let (Some(eos_id), Some(ref eos_tok)) = (self.config.eos_token_id, &self.config.eos_token) {
            ids.push(eos_id);
            tokens.push(eos_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
        }

        let mut attention_mask = vec![1u8; ids.len()];

        if self.config.padding && ids.len() < self.config.max_length {
            let pad_count = self.config.max_length - ids.len();
            for _ in 0..pad_count {
                ids.push(self.config.pad_id);
                tokens.push(self.config.pad_token.clone());
                offsets.push((0, 0));
                special_mask.push(1u8);
                attention_mask.push(0u8);
            }
        }

        let type_ids = vec![0usize; ids.len()];
        output.ids = ids;
        output.tokens = tokens;
        output.offsets = offsets;
        output.attention_mask = attention_mask;
        output.special_tokens_mask = special_mask;
        output.type_ids = Some(type_ids);
        output
    }

    /// Formats a sentence pair into `[CLS] A [SEP] B [SEP]` with `type_ids` and masks.
    pub fn process_pair(
        &self,
        first: TokenizedOutput,
        second: TokenizedOutput,
    ) -> TokenizedOutput {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut special_mask = Vec::new();
        let mut type_ids = Vec::new();

        if let (Some(cls_id), Some(ref cls_tok)) = (self.config.cls_token_id, &self.config.cls_token) {
            ids.push(cls_id);
            tokens.push(cls_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
            type_ids.push(0usize);
        }

        let special_tokens_count = 3;
        let available_slots = if self.config.truncation {
            self.config.max_length.saturating_sub(special_tokens_count)
        } else {
            first.ids.len() + second.ids.len()
        };

        let mut first_take = first.ids.len();
        let mut second_take = second.ids.len();

        while first_take + second_take > available_slots {
            match self.config.truncation_strategy {
                TruncationStrategy::LongestFirst => {
                    if first_take >= second_take && first_take > 0 {
                        first_take -= 1;
                    } else if second_take > 0 {
                        second_take -= 1;
                    } else {
                        break;
                    }
                }
                TruncationStrategy::OnlyFirst => {
                    if first_take > 0 {
                        first_take -= 1;
                    } else {
                        break;
                    }
                }
                TruncationStrategy::OnlySecond => {
                    if second_take > 0 {
                        second_take -= 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // Add sequence A
        ids.extend_from_slice(&first.ids[..first_take]);
        tokens.extend_from_slice(&first.tokens[..first_take]);
        offsets.extend_from_slice(&first.offsets[..first_take]);
        special_mask.extend(std::iter::repeat(0u8).take(first_take));
        type_ids.extend(std::iter::repeat(0usize).take(first_take));

        // Add first SEP
        if let (Some(sep_id), Some(ref sep_tok)) = (self.config.sep_token_id, &self.config.sep_token) {
            ids.push(sep_id);
            tokens.push(sep_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
            type_ids.push(0usize);
        }

        // Add sequence B
        ids.extend_from_slice(&second.ids[..second_take]);
        tokens.extend_from_slice(&second.tokens[..second_take]);
        offsets.extend_from_slice(&second.offsets[..second_take]);
        special_mask.extend(std::iter::repeat(0u8).take(second_take));
        type_ids.extend(std::iter::repeat(1usize).take(second_take));

        // Add second SEP
        if let (Some(sep_id), Some(ref sep_tok)) = (self.config.sep_token_id, &self.config.sep_token) {
            ids.push(sep_id);
            tokens.push(sep_tok.clone());
            offsets.push((0, 0));
            special_mask.push(1u8);
            type_ids.push(1usize);
        }

        let mut attention_mask = vec![1u8; ids.len()];

        if self.config.padding && ids.len() < self.config.max_length {
            let pad_count = self.config.max_length - ids.len();
            for _ in 0..pad_count {
                ids.push(self.config.pad_id);
                tokens.push(self.config.pad_token.clone());
                offsets.push((0, 0));
                special_mask.push(1u8);
                attention_mask.push(0u8);
                type_ids.push(0usize);
            }
        }

        TokenizedOutput {
            ids,
            tokens,
            offsets,
            attention_mask,
            type_ids: Some(type_ids),
            special_tokens_mask: special_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::vocab::*;
    use crate::text_ops::*;
    use crate::features::*;
    use crate::similarity::*;
    use crate::lm::*;
    use crate::process::*;
    use crate::optimize::*;
    use crate::analyze::*;
    use crate::compute::*;
    use crate::helper::*;
    use crate::transform::*;
    use crate::builder::*;
    use crate::tokenizer::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::post::*;
    use crate::embedding::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::fasttext::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_post_processing_1() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 1], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_2() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 2], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_3() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 3], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_4() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 4], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_5() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 5], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_6() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 6], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_7() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 7], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_8() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 8], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_9() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 9], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_10() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 10], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_11() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 11], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_12() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 12], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_13() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 13], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_14() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 14], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_15() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 15], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_16() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 16], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_17() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 17], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_18() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 18], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_19() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 19], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_20() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 20], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_21() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 21], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_22() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 22], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_23() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 23], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_24() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 24], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_25() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 25], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_26() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 26], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_27() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 27], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_28() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 28], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_29() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 29], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_30() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 30], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_31() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 31], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_32() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 32], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_33() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 33], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_34() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 34], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_35() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 35], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_36() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 36], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_37() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 37], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_38() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 38], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_39() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 39], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_40() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 40], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_41() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 41], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_42() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 42], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_43() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 43], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_44() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 44], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_45() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 45], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_46() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 46], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_47() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 47], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_48() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 48], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_49() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 49], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_50() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 50], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_51() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 51], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_52() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 52], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_53() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 53], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_54() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 54], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_55() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 55], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_56() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 56], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_57() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 57], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_58() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 58], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_59() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 59], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_60() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 60], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_61() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 61], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_62() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 62], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_63() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 63], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_64() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 64], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_65() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 65], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_66() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 66], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_67() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 67], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_68() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 68], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_69() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 69], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_70() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 70], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_71() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 71], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_72() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 72], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_73() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 73], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_74() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 74], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_75() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 75], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_76() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 76], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_77() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 77], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_78() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 78], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_79() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 79], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_80() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 80], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_81() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 81], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_82() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 82], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_83() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 83], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_84() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 84], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_85() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 85], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_86() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 86], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_87() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 87], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_88() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 88], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_89() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 89], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_90() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 90], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_91() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 91], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_92() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 92], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_93() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 93], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_94() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 94], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_95() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 95], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_96() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 96], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_97() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 97], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_98() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 98], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_99() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 99], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_100() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 100], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_101() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 101], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_102() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 102], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_103() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 103], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_104() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 104], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_105() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 105], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_106() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 106], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_107() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 107], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_108() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 108], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_109() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 109], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_110() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 110], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_111() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 111], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_112() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 112], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_113() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 113], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_114() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 114], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_115() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 115], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_116() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 116], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn test_post_processing_117() {
        let cfg = PostConfig {
            max_length: 8,
            padding: true,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            ..Default::default()
        };
        let pp = PostProcessor::new(cfg);
        let out = TokenizedOutput::new(vec![1, 2, 117], vec!["a".to_string(), "b".to_string(), "c".to_string()], vec![(0,1),(1,2),(2,3)]);
        let processed = pp.process_single(out);
        assert_eq!(processed.ids.len(), 8);
        assert_eq!(processed.ids[0], 101);
        assert_eq!(processed.ids[4], 102);
        assert_eq!(processed.ids[5], 0);

        let out_a = TokenizedOutput::new(vec![1, 2], vec!["a".to_string(), "b".to_string()], vec![(0,1),(1,2)]);
        let out_b = TokenizedOutput::new(vec![3, 4], vec!["c".to_string(), "d".to_string()], vec![(0,1),(1,2)]);
        let pair = pp.process_pair(out_a, out_b);
        assert_eq!(pair.ids.len(), 8);
        assert_eq!(pair.type_ids.as_ref().unwrap(), &vec![0, 0, 0, 0, 1, 1, 1, 0]);
    }

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
}
