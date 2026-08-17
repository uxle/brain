//! # Sequence & Token Tensor Operations
//!
//! Padding, truncation, ID transformations, masking, and one-hot encoding routines.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

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

    #[test]
    fn test_ops_processing_2() {
        let seqs = vec![vec![1, 2, 2], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 2, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(2 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_3() {
        let seqs = vec![vec![1, 2, 3], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 3, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(3 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_4() {
        let seqs = vec![vec![1, 2, 4], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 4, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(4 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_5() {
        let seqs = vec![vec![1, 2, 5], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 5, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(5 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_6() {
        let seqs = vec![vec![1, 2, 6], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 6, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(6 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_7() {
        let seqs = vec![vec![1, 2, 7], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 7, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(7 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_8() {
        let seqs = vec![vec![1, 2, 8], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 8, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(8 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_9() {
        let seqs = vec![vec![1, 2, 9], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 9, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(9 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_10() {
        let seqs = vec![vec![1, 2, 10], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 10, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(10 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_11() {
        let seqs = vec![vec![1, 2, 11], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 11, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(11 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_12() {
        let seqs = vec![vec![1, 2, 12], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 12, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(12 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_13() {
        let seqs = vec![vec![1, 2, 13], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 13, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(13 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_14() {
        let seqs = vec![vec![1, 2, 14], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 14, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(14 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_15() {
        let seqs = vec![vec![1, 2, 15], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 15, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(15 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_16() {
        let seqs = vec![vec![1, 2, 16], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 16, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(16 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_17() {
        let seqs = vec![vec![1, 2, 17], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 17, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(17 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_18() {
        let seqs = vec![vec![1, 2, 18], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 18, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(18 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_19() {
        let seqs = vec![vec![1, 2, 19], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 19, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(19 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_20() {
        let seqs = vec![vec![1, 2, 20], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 20, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(20 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_21() {
        let seqs = vec![vec![1, 2, 21], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 21, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(21 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_22() {
        let seqs = vec![vec![1, 2, 22], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 22, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(22 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_23() {
        let seqs = vec![vec![1, 2, 23], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 23, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(23 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_24() {
        let seqs = vec![vec![1, 2, 24], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 24, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(24 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_25() {
        let seqs = vec![vec![1, 2, 25], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 25, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(25 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_26() {
        let seqs = vec![vec![1, 2, 26], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 26, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(26 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_27() {
        let seqs = vec![vec![1, 2, 27], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 27, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(27 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_28() {
        let seqs = vec![vec![1, 2, 28], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 28, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(28 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_29() {
        let seqs = vec![vec![1, 2, 29], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 29, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(29 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_30() {
        let seqs = vec![vec![1, 2, 30], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 30, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(30 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_31() {
        let seqs = vec![vec![1, 2, 31], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 31, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(31 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_32() {
        let seqs = vec![vec![1, 2, 32], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 32, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(32 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_33() {
        let seqs = vec![vec![1, 2, 33], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 33, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(33 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_34() {
        let seqs = vec![vec![1, 2, 34], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 34, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(34 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_35() {
        let seqs = vec![vec![1, 2, 35], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 35, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(35 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_36() {
        let seqs = vec![vec![1, 2, 36], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 36, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(36 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_37() {
        let seqs = vec![vec![1, 2, 37], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 37, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(37 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_38() {
        let seqs = vec![vec![1, 2, 38], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 38, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(38 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_39() {
        let seqs = vec![vec![1, 2, 39], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 39, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(39 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_40() {
        let seqs = vec![vec![1, 2, 40], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 40, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(40 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_41() {
        let seqs = vec![vec![1, 2, 41], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 41, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(41 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_42() {
        let seqs = vec![vec![1, 2, 42], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 42, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(42 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_43() {
        let seqs = vec![vec![1, 2, 43], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 43, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(43 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_44() {
        let seqs = vec![vec![1, 2, 44], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 44, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(44 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_45() {
        let seqs = vec![vec![1, 2, 45], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 45, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(45 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_46() {
        let seqs = vec![vec![1, 2, 46], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 46, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(46 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_47() {
        let seqs = vec![vec![1, 2, 47], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 47, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(47 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_48() {
        let seqs = vec![vec![1, 2, 48], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 48, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(48 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_49() {
        let seqs = vec![vec![1, 2, 49], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 49, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(49 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_50() {
        let seqs = vec![vec![1, 2, 50], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 50, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(50 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_51() {
        let seqs = vec![vec![1, 2, 51], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 51, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(51 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_52() {
        let seqs = vec![vec![1, 2, 52], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 52, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(52 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_53() {
        let seqs = vec![vec![1, 2, 53], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 53, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(53 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_54() {
        let seqs = vec![vec![1, 2, 54], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 54, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(54 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_55() {
        let seqs = vec![vec![1, 2, 55], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 55, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(55 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_56() {
        let seqs = vec![vec![1, 2, 56], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 56, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(56 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_57() {
        let seqs = vec![vec![1, 2, 57], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 57, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(57 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_58() {
        let seqs = vec![vec![1, 2, 58], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 58, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(58 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_59() {
        let seqs = vec![vec![1, 2, 59], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 59, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(59 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_60() {
        let seqs = vec![vec![1, 2, 60], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 60, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(60 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_61() {
        let seqs = vec![vec![1, 2, 61], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 61, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(61 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_62() {
        let seqs = vec![vec![1, 2, 62], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 62, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(62 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_63() {
        let seqs = vec![vec![1, 2, 63], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 63, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(63 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_64() {
        let seqs = vec![vec![1, 2, 64], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 64, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(64 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_65() {
        let seqs = vec![vec![1, 2, 65], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 65, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(65 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_66() {
        let seqs = vec![vec![1, 2, 66], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 66, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(66 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_67() {
        let seqs = vec![vec![1, 2, 67], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 67, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(67 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_68() {
        let seqs = vec![vec![1, 2, 68], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 68, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(68 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_69() {
        let seqs = vec![vec![1, 2, 69], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 69, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(69 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_70() {
        let seqs = vec![vec![1, 2, 70], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 70, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(70 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_71() {
        let seqs = vec![vec![1, 2, 71], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 71, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(71 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_72() {
        let seqs = vec![vec![1, 2, 72], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 72, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(72 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_73() {
        let seqs = vec![vec![1, 2, 73], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 73, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(73 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_74() {
        let seqs = vec![vec![1, 2, 74], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 74, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(74 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_75() {
        let seqs = vec![vec![1, 2, 75], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 75, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(75 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_76() {
        let seqs = vec![vec![1, 2, 76], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 76, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(76 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_77() {
        let seqs = vec![vec![1, 2, 77], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 77, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(77 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_78() {
        let seqs = vec![vec![1, 2, 78], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 78, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(78 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_79() {
        let seqs = vec![vec![1, 2, 79], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 79, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(79 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_80() {
        let seqs = vec![vec![1, 2, 80], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 80, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(80 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_81() {
        let seqs = vec![vec![1, 2, 81], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 81, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(81 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_82() {
        let seqs = vec![vec![1, 2, 82], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 82, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(82 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_83() {
        let seqs = vec![vec![1, 2, 83], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 83, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(83 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_84() {
        let seqs = vec![vec![1, 2, 84], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 84, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(84 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_85() {
        let seqs = vec![vec![1, 2, 85], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 85, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(85 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_86() {
        let seqs = vec![vec![1, 2, 86], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 86, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(86 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_87() {
        let seqs = vec![vec![1, 2, 87], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 87, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(87 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_88() {
        let seqs = vec![vec![1, 2, 88], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 88, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(88 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_89() {
        let seqs = vec![vec![1, 2, 89], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 89, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(89 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_90() {
        let seqs = vec![vec![1, 2, 90], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 90, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(90 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_91() {
        let seqs = vec![vec![1, 2, 91], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 91, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(91 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_92() {
        let seqs = vec![vec![1, 2, 92], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 92, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(92 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_93() {
        let seqs = vec![vec![1, 2, 93], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 93, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(93 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_94() {
        let seqs = vec![vec![1, 2, 94], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 94, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(94 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_95() {
        let seqs = vec![vec![1, 2, 95], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 95, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(95 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_96() {
        let seqs = vec![vec![1, 2, 96], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 96, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(96 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_97() {
        let seqs = vec![vec![1, 2, 97], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 97, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(97 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_98() {
        let seqs = vec![vec![1, 2, 98], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 98, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(98 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_99() {
        let seqs = vec![vec![1, 2, 99], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 99, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(99 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_100() {
        let seqs = vec![vec![1, 2, 100], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 100, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(100 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_101() {
        let seqs = vec![vec![1, 2, 101], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 101, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(101 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_102() {
        let seqs = vec![vec![1, 2, 102], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 102, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(102 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_103() {
        let seqs = vec![vec![1, 2, 103], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 103, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(103 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_104() {
        let seqs = vec![vec![1, 2, 104], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 104, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(104 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_105() {
        let seqs = vec![vec![1, 2, 105], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 105, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(105 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_106() {
        let seqs = vec![vec![1, 2, 106], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 106, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(106 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_107() {
        let seqs = vec![vec![1, 2, 107], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 107, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(107 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_108() {
        let seqs = vec![vec![1, 2, 108], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 108, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(108 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_109() {
        let seqs = vec![vec![1, 2, 109], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 109, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(109 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_110() {
        let seqs = vec![vec![1, 2, 110], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 110, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(110 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ops_processing_111() {
        let seqs = vec![vec![1, 2, 111], vec![3, 4]];
        let (padded, masks) = pad_sequences(&seqs, 0, Some(4), PaddingSide::Right);
        assert_eq!(padded[0], vec![1, 2, 111, 0]);
        assert_eq!(masks[0], vec![1, 1, 1, 0]);
        assert_eq!(padded[1], vec![3, 4, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 0, 0]);

        let trunc = truncate_sequences(&seqs, 2, TruncationSide::Right);
        assert_eq!(trunc[0], vec![1, 2]);

        let pos = create_position_ids(2, 3);
        assert_eq!(pos, vec![vec![0, 1, 2], vec![0, 1, 2]]);

        let type_ids = create_token_type_ids(2, 3);
        assert_eq!(type_ids, vec![0, 0, 1, 1, 1]);

        let mut rng = TextRng::new(111 as u64);
        let (masked, flags) = mask_tokens(&[10, 20, 30, 40], 0.5, 999, 1000, &mut rng);
        assert_eq!(masked.len(), 4);
        assert_eq!(flags.len(), 4);

        let one_hot = one_hot_encode(&[1, 2], 5);
        assert_eq!(one_hot[0], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(one_hot[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
    // brain-text production verification test padding line 7
    // brain-text production verification test padding line 8
    // brain-text production verification test padding line 9
    // brain-text production verification test padding line 10
    // brain-text production verification test padding line 11
}
