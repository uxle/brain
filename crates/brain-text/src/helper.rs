//! # NLP Pipeline Helpers, Data Collators, and Text Augmenters
//!
//! T5-style span corruption, batch collators for seq2seq and language modeling, and text perturbation tools.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::core::{TextBatch, TokenId, TokenizedOutput};
use crate::ops::{mask_tokens, pad_sequences};
use crate::utils::TextRng;

/// Data collator for Sequence-to-Sequence (Seq2Seq) tasks.
#[derive(Debug, Clone)]
pub struct DataCollatorForSeq2Seq {
    /// Padding token ID for inputs.
    pub pad_token_id: TokenId,
    /// Padding label ID for loss masking (typically -100).
    pub label_pad_token_id: i64,
    /// Maximum sequence length.
    pub max_length: Option<usize>,
}

impl DataCollatorForSeq2Seq {
    /// Creates a new Seq2Seq collator.
    pub fn new(pad_token_id: TokenId, label_pad_token_id: i64) -> Self {
        Self {
            pad_token_id,
            label_pad_token_id,
            max_length: None,
        }
    }

    /// Collates feature inputs and labels into padded batches.
    pub fn collate(
        &self,
        inputs: &[Vec<TokenId>],
        labels: &[Vec<i64>],
    ) -> (Vec<Vec<TokenId>>, Vec<Vec<i64>>, Vec<Vec<u8>>) {
        let (padded_inputs, attention_mask) = pad_sequences(
            inputs,
            self.pad_token_id,
            self.max_length,
            crate::config::PaddingSide::Right,
        );

        let target_label_len = match self.max_length {
            Some(l) => l,
            None => labels.iter().map(|l| l.len()).max().unwrap_or(0),
        };

        let mut padded_labels = Vec::with_capacity(labels.len());
        for lbl in labels {
            let mut new_lbl = lbl.clone();
            if new_lbl.len() > target_label_len {
                new_lbl.truncate(target_label_len);
            } else {
                while new_lbl.len() < target_label_len {
                    new_lbl.push(self.label_pad_token_id);
                }
            }
            padded_labels.push(new_lbl);
        }

        (padded_inputs, padded_labels, attention_mask)
    }
}

/// Data collator for Language Modeling with dynamic masking.
#[derive(Debug, Clone)]
pub struct DataCollatorForLanguageModeling {
    /// Whether to perform Masked Language Modeling (MLM).
    pub mlm: bool,
    /// Masking probability (e.g. 0.15).
    pub mlm_probability: f32,
    /// Padding token ID.
    pub pad_token_id: TokenId,
    /// Mask token ID.
    pub mask_token_id: TokenId,
    /// Vocabulary capacity.
    pub vocab_size: usize,
}

impl DataCollatorForLanguageModeling {
    /// Collates batch and applies dynamic masking if MLM enabled.
    pub fn collate(
        &self,
        sequences: &[Vec<TokenId>],
        rng: &mut TextRng,
    ) -> (Vec<Vec<TokenId>>, Vec<Vec<i64>>, Vec<Vec<u8>>) {
        let (padded_seqs, attention_masks) = pad_sequences(
            sequences,
            self.pad_token_id,
            None,
            crate::config::PaddingSide::Right,
        );

        if !self.mlm {
            let labels: Vec<Vec<i64>> = padded_seqs
                .iter()
                .map(|seq| seq.iter().map(|&id| id as i64).collect())
                .collect();
            return (padded_seqs, labels, attention_masks);
        }

        let mut input_ids = Vec::with_capacity(padded_seqs.len());
        let mut labels = Vec::with_capacity(padded_seqs.len());

        for seq in padded_seqs {
            let (masked, flags) = mask_tokens(
                &seq,
                self.mlm_probability,
                self.mask_token_id,
                self.vocab_size,
                rng,
            );
            let lbl: Vec<i64> = seq
                .iter()
                .zip(flags.iter())
                .map(|(&orig_id, &is_masked)| {
                    if is_masked && orig_id != self.pad_token_id {
                        orig_id as i64
                    } else {
                        -100i64
                    }
                })
                .collect();
            input_ids.push(masked);
            labels.push(lbl);
        }

        (input_ids, labels, attention_masks)
    }
}

/// Helper for T5-style span corruption with sentinel replacement tokens `<extra_id_0>`, `<extra_id_1>`, &c.
pub struct SpanCorruptionHelper;

impl SpanCorruptionHelper {
    /// Applies span corruption to a token sequence replacing masked spans with sentinel IDs.
    pub fn corrupt_spans(
        tokens: &[TokenId],
        noise_density: f32,
        mean_noise_span_length: f32,
        sentinel_start_id: TokenId,
        rng: &mut TextRng,
    ) -> (Vec<TokenId>, Vec<TokenId>) {
        if tokens.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_noise_tokens = ((tokens.len() as f32) * noise_density).round() as usize;
        let num_spans = (num_noise_tokens as f32 / mean_noise_span_length.max(1.0))
            .round()
            .max(1.0) as usize;

        let mut input_tokens = Vec::new();
        let mut target_tokens = Vec::new();
        let mut sentinel_idx = 0;
        let mut i = 0;

        while i < tokens.len() {
            if rng.next_f32() < (num_spans as f32 / tokens.len() as f32) && sentinel_idx < 100 {
                let span_len = rng.gen_range(mean_noise_span_length as usize + 2).max(1);
                let end = (i + span_len).min(tokens.len());

                let sentinel_id = sentinel_start_id + sentinel_idx;
                sentinel_idx += 1;

                input_tokens.push(sentinel_id);
                target_tokens.push(sentinel_id);
                target_tokens.extend_from_slice(&tokens[i..end]);

                i = end;
            } else {
                input_tokens.push(tokens[i]);
                i += 1;
            }
        }

        (input_tokens, target_tokens)
    }
}

/// Text data augmentation utilities.
pub struct TextAugmenter;

impl TextAugmenter {
    /// Randomly deletes words with probability `prob`.
    pub fn random_deletion(words: &[String], prob: f32, rng: &mut TextRng) -> Vec<String> {
        if words.len() <= 1 {
            return words.to_vec();
        }
        let mut augmented = Vec::new();
        for w in words {
            if rng.next_f32() >= prob {
                augmented.push(w.clone());
            }
        }
        if augmented.is_empty() {
            vec![words[0].clone()]
        } else {
            augmented
        }
    }

    /// Randomly swaps adjacent words with probability `prob`.
    pub fn random_swap(words: &[String], prob: f32, rng: &mut TextRng) -> Vec<String> {
        let mut augmented = words.to_vec();
        if augmented.len() < 2 {
            return augmented;
        }
        for i in 0..(augmented.len() - 1) {
            if rng.next_f32() < prob {
                augmented.swap(i, i + 1);
            }
        }
        augmented
    }

    /// Randomly inserts duplicate words with probability `prob`.
    pub fn random_insertion(words: &[String], prob: f32, rng: &mut TextRng) -> Vec<String> {
        let mut augmented = Vec::with_capacity(words.len() * 2);
        for w in words {
            augmented.push(w.clone());
            if rng.next_f32() < prob {
                augmented.push(w.clone());
            }
        }
        augmented
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
        clippy::len_zero
    )]
    use super::*;
    use crate::analyze::*;
    use crate::builder::*;
    use crate::compute::*;
    use crate::config::*;
    use crate::core::*;
    use crate::embedding::fasttext::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::*;
    use crate::features::*;
    use crate::helper::*;
    use crate::lm::*;
    use crate::ops::*;
    use crate::optimize::*;
    use crate::process::*;
    use crate::similarity::*;
    use crate::text_ops::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::post::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::*;
    use crate::transform::*;
    use crate::utils::*;
    use crate::vocab::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_helper_pipeline_1() {
        let seq2seq = DataCollatorForSeq2Seq::new(0, -100);
        let inps = vec![vec![1, 2, 1], vec![3, 4]];
        let lbls = vec![vec![10, 20], vec![30]];
        let (pi, pl, pm) = seq2seq.collate(&inps, &lbls);
        assert_eq!(pi[0].len(), 3);
        assert_eq!(pl[0].len(), 2);
        assert_eq!(pm[0].len(), 3);

        let mut rng = TextRng::new(1 as u64);
        let lm_collator = DataCollatorForLanguageModeling {
            mlm: true,
            mlm_probability: 0.5,
            pad_token_id: 0,
            mask_token_id: 103,
            vocab_size: 1000,
        };
        let (lmi, lml, lmm) = lm_collator.collate(&inps, &mut rng);
        assert_eq!(lmi.len(), 2);

        let (cin, ctgt) =
            SpanCorruptionHelper::corrupt_spans(&[1, 2, 3, 4, 5], 0.3, 2.0, 1000, &mut rng);
        assert!(!cin.is_empty());

        let words = vec!["the".to_string(), "quick".to_string(), "fox_1".to_string()];
        let del = TextAugmenter::random_deletion(&words, 0.2, &mut rng);
        assert!(!del.is_empty());
        let swap = TextAugmenter::random_swap(&words, 0.5, &mut rng);
        assert_eq!(swap.len(), 3);
        let ins = TextAugmenter::random_insertion(&words, 0.5, &mut rng);
        assert!(ins.len() >= 3);
    }
}
