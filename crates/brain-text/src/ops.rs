//! # Sequence & Token Tensor Operations
//!
//! Padding, truncation, ID transformations, masking, and one-hot encoding routines.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::config::{PaddingSide, TruncationSide};
use crate::core::{TokenId, TokenIds};
use crate::utils::TextRng;
use crate::vocab::Vocab;

/// Pads a list of token sequences to a uniform length.
pub fn pad_sequences(
    sequences: &[Vec<TokenId>],
    pad_id: TokenId,
    max_len: Option<usize>,
    side: PaddingSide,
) -> (Vec<Vec<TokenId>>, Vec<Vec<u8>>) {
    let target_len = match max_len {
        Some(l) => l,
        None => sequences.iter().map(|s| s.len()).max().unwrap_or(0),
    };

    let mut padded_seqs = Vec::with_capacity(sequences.len());
    let mut attention_masks = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let mut new_seq = Vec::with_capacity(target_len);
        let mut mask = Vec::with_capacity(target_len);

        let current_len = seq.len().min(target_len);
        let pad_count = target_len.saturating_sub(current_len);

        match side {
            PaddingSide::Right => {
                for &id in &seq[..current_len] {
                    new_seq.push(id);
                    mask.push(1u8);
                }
                for _ in 0..pad_count {
                    new_seq.push(pad_id);
                    mask.push(0u8);
                }
            }
            PaddingSide::Left => {
                for _ in 0..pad_count {
                    new_seq.push(pad_id);
                    mask.push(0u8);
                }
                for &id in &seq[..current_len] {
                    new_seq.push(id);
                    mask.push(1u8);
                }
            }
        }

        padded_seqs.push(new_seq);
        attention_masks.push(mask);
    }

    (padded_seqs, attention_masks)
}

/// Truncates sequences to a maximum length.
pub fn truncate_sequences(
    sequences: &[Vec<TokenId>],
    max_len: usize,
    side: TruncationSide,
) -> Vec<Vec<TokenId>> {
    sequences
        .iter()
        .map(|seq| {
            if seq.len() <= max_len {
                seq.clone()
            } else {
                match side {
                    TruncationSide::Right => seq[..max_len].to_vec(),
                    TruncationSide::Left => seq[seq.len() - max_len..].to_vec(),
                }
            }
        })
        .collect()
}

/// Maps numeric token IDs to token strings using a vocabulary.
pub fn ids_to_tokens(ids: &[TokenId], vocab: &Vocab) -> Vec<String> {
    ids.iter()
        .map(|&id| vocab.get_token(id).unwrap_or("[UNK]").to_string())
        .collect()
}

/// Maps token strings to numeric token IDs using a vocabulary.
pub fn tokens_to_ids(tokens: &[String], vocab: &Vocab) -> Vec<TokenId> {
    tokens
        .iter()
        .map(|t| vocab.get_id(t).or_else(|| vocab.unk_id()).unwrap_or(0))
        .collect()
}

/// Creates binary attention masks for padded sequences.
pub fn create_attention_mask(sequences: &[Vec<TokenId>], pad_id: TokenId) -> Vec<Vec<u8>> {
    sequences
        .iter()
        .map(|seq| {
            seq.iter()
                .map(|&id| if id == pad_id { 0u8 } else { 1u8 })
                .collect()
        })
        .collect()
}

/// Generates positional indices `[0, 1, ..., seq_len - 1]` for a batch.
pub fn create_position_ids(batch_size: usize, seq_len: usize) -> Vec<Vec<usize>> {
    let pos_row: Vec<usize> = (0..seq_len).collect();
    vec![pos_row; batch_size]
}

/// Generates token type IDs for sentence pair classification (e.g. `0` for A, `1` for B).
pub fn create_token_type_ids(first_len: usize, second_len: usize) -> Vec<usize> {
    let mut type_ids = Vec::with_capacity(first_len + second_len);
    type_ids.extend(std::iter::repeat(0).take(first_len));
    type_ids.extend(std::iter::repeat(1).take(second_len));
    type_ids
}

/// Applies random masked language modeling (MLM) replacement to token IDs.
pub fn mask_tokens(
    ids: &[TokenId],
    mask_prob: f32,
    mask_id: TokenId,
    vocab_size: usize,
    rng: &mut TextRng,
) -> (Vec<TokenId>, Vec<bool>) {
    let mut masked_ids = ids.to_vec();
    let mut mask_flags = vec![false; ids.len()];

    for i in 0..ids.len() {
        if rng.next_f32() < mask_prob {
            mask_flags[i] = true;
            let p = rng.next_f32();
            if p < 0.80 {
                masked_ids[i] = mask_id;
            } else if p < 0.90 {
                masked_ids[i] = rng.gen_range(vocab_size);
            }
        }
    }

    (masked_ids, mask_flags)
}

/// Packs multiple variable-length sequences into contiguous chunks of fixed length.
pub fn pack_sequences(
    sequences: &[Vec<TokenId>],
    max_length: usize,
    eos_id: TokenId,
) -> Vec<Vec<TokenId>> {
    let mut packed = Vec::new();
    let mut current_chunk = Vec::with_capacity(max_length);

    for seq in sequences {
        let mut seq_with_eos = seq.clone();
        seq_with_eos.push(eos_id);

        for &id in &seq_with_eos {
            current_chunk.push(id);
            if current_chunk.len() == max_length {
                packed.push(current_chunk);
                current_chunk = Vec::with_capacity(max_length);
            }
        }
    }

    if !current_chunk.is_empty() {
        packed.push(current_chunk);
    }

    packed
}

/// Converts a sequence of token IDs to one-hot vectors.
pub fn one_hot_encode(ids: &[TokenId], vocab_size: usize) -> Vec<Vec<f32>> {
    ids.iter()
        .map(|&id| {
            let mut vec = vec![0.0f32; vocab_size];
            if id < vocab_size {
                vec[id] = 1.0;
            }
            vec
        })
        .collect()
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
    fn test_ops_processing_1() {
        let seqs = vec![vec![1, 2, 1], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 1, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(1 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }
}
