//! # Batch Sequence Processing, Text Cleaning, and Segmentation
//!
//! Text normalization sanitization, sentence/paragraph segmentation, batch collators, and chunk filters.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::config::ProcessConfig;
use crate::core::{TextBatch, TokenId, TokenizedOutput};

/// Processes a batch of raw strings using an encoding function into a uniform `TextBatch`.
pub fn process_batch_texts(
    texts: &[&str],
    encode_fn: &dyn Fn(&str) -> TokenizedOutput,
    config: &ProcessConfig,
) -> TextBatch {
    let outputs: Vec<TokenizedOutput> = texts.iter().map(|&t| encode_fn(t)).collect();
    let max_len = if config.pad_to_max {
        Some(config.max_length)
    } else {
        None
    };
    pad_and_collate(&outputs, 0, max_len)
}

/// Collates a collection of `TokenizedOutput` records into a uniform padded `TextBatch`.
pub fn pad_and_collate(
    outputs: &[TokenizedOutput],
    pad_id: TokenId,
    max_len: Option<usize>,
) -> TextBatch {
    let target_len = match max_len {
        Some(l) => l,
        None => outputs.iter().map(|s| s.ids.len()).max().unwrap_or(0),
    };

    let mut padded_sequences = Vec::with_capacity(outputs.len());

    for out in outputs {
        let mut ids = out.ids.clone();
        let mut tokens = out.tokens.clone();
        let mut offsets = out.offsets.clone();
        let mut attention_mask = out.attention_mask.clone();
        let mut special_mask = out.special_tokens_mask.clone();

        if ids.len() > target_len {
            ids.truncate(target_len);
            tokens.truncate(target_len);
            offsets.truncate(target_len);
            attention_mask.truncate(target_len);
            special_mask.truncate(target_len);
        } else {
            let pad_count = target_len - ids.len();
            for _ in 0..pad_count {
                ids.push(pad_id);
                tokens.push("[PAD]".to_string());
                offsets.push((0, 0));
                attention_mask.push(0);
                special_mask.push(1);
            }
        }

        let mut padded = TokenizedOutput::new(ids, tokens, offsets);
        padded.attention_mask = attention_mask;
        padded.special_tokens_mask = special_mask;
        padded_sequences.push(padded);
    }

    TextBatch::from_outputs(padded_sequences, pad_id)
}

/// Truncates all sequences in a batch to a specified maximum length.
pub fn truncate_batch(batch: &mut TextBatch, max_len: usize) {
    for seq in &mut batch.sequences {
        seq.ids.truncate(max_len);
        seq.tokens.truncate(max_len);
        seq.offsets.truncate(max_len);
        seq.attention_mask.truncate(max_len);
        seq.special_tokens_mask.truncate(max_len);
    }
    batch.max_length = batch.max_length.min(max_len);
}

/// Splits text into individual sentences based on punctuation delimiters.
pub fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Splits text into paragraphs based on double newlines.
pub fn split_into_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Sanitizes and cleans raw text input.
pub fn clean_text(
    text: &str,
    remove_html: bool,
    normalize_spaces: bool,
    strip_control: bool,
) -> String {
    let mut result = text.to_string();

    if remove_html {
        let mut clean = String::with_capacity(result.len());
        let mut inside_tag = false;
        for c in result.chars() {
            if c == '<' {
                inside_tag = true;
            } else if c == '>' {
                inside_tag = false;
            } else if !inside_tag {
                clean.push(c);
            }
        }
        result = clean;
    }

    if strip_control {
        result = result
            .chars()
            .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
            .collect();
    }

    if normalize_spaces {
        result = result.split_whitespace().collect::<Vec<&str>>().join(" ");
    }

    result
}

/// Filters a list of strings retaining those with character length within bounds.
pub fn filter_by_length(texts: &[String], min_len: usize, max_len: usize) -> Vec<String> {
    texts
        .iter()
        .filter(|t| t.len() >= min_len && t.len() <= max_len)
        .cloned()
        .collect()
}

/// Partitions items into mini-batches of fixed size.
pub fn batch_iterator(items: Vec<String>, batch_size: usize) -> Vec<Vec<String>> {
    if batch_size == 0 {
        return Vec::new();
    }
    items.chunks(batch_size).map(|c| c.to_vec()).collect()
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
    fn test_process_suite_1() {
        let out1 = TokenizedOutput::new(
            vec![1, 2, 1],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![(0, 1), (1, 2), (2, 3)],
        );
        let out2 = TokenizedOutput::new(
            vec![4, 5],
            vec!["d".to_string(), "e".to_string()],
            vec![(0, 1), (1, 2)],
        );
        let mut batch = pad_and_collate(&[out1, out2], 0, Some(4));
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_length, 4);

        truncate_batch(&mut batch, 2);
        assert_eq!(batch.max_length, 2);

        let sents = split_into_sentences("Hello world! How are you? Fine.");
        assert_eq!(sents.len(), 3);

        let paras = split_into_paragraphs("Para 1\n\nPara 2\n\nPara 3");
        assert_eq!(paras.len(), 3);

        let cleaned = clean_text("<p>Hello   World\x00!</p>", true, true, true);
        assert_eq!(cleaned, "Hello World!");

        let filtered = filter_by_length(&["a".to_string(), "long_1".to_string()], 2, 20);
        assert_eq!(filtered.len(), 1);

        let b = batch_iterator(vec!["1".to_string(), "2".to_string(), "3".to_string()], 2);
        assert_eq!(b.len(), 2);
    }
}
