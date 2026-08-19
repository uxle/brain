# `brain-text`

Production-grade NLP framework — tokenizers, embeddings, text features, similarity metrics, and LM utilities — in 100% safe, zero-dependency Rust.

## Overview

`brain-text` implements the full natural-language stack for the Brain ecosystem: five tokenizer algorithms (BPE, SentencePiece Unigram, WordPiece, Char, Word) with byte-level encoding, normalizers, pre-tokenizers, post-processors, and trainers; trainable and pretrained embeddings (learned, sinusoidal, GloVe, Word2Vec, FastText); and text analysis utilities (TF-IDF, BM25, n-grams, readability scores, BLEU/ROUGE/CHRF metrics). A universal `Tokenizer` trait with batch encode/decode unifies every algorithm.

## Features

- **Tokenizers**: `BpeTokenizer`, `SentencePieceTokenizer`, `WordPieceTokenizer`, `CharTokenizer`, `WordTokenizer`, `ByteLevelEncoder` — all implementing the `Tokenizer` trait (`encode`, `decode`, `tokenize`, `vocab_size`, special-token IDs, batch variants).
- **Training**: `BpeTrainer`, `UnigramTrainer`, `WordPieceTrainer` with `TrainConfig`, plus `train_bpe_tokenizer`/`train_unigram_tokenizer` helpers.
- **Embeddings**: `WordEmbedding`, `PositionalEmbedding`, `PretrainedEmbedding` (`load_glove_str`, `load_word2vec_text_str`, `most_similar`, `analogy`), `FastTextEmbedding`.
- **Features & similarity**: `TfIdf`, `Bm25`, `BagOfWords`, `HashingVectorizer`, `TextSimilarity` (`SimilarityMetric`), `VocabTrie` for vocabulary pruning.
- **Metrics**: `bleu_score`, `corpus_bleu`, `rouge_n`, `rouge_l`, `chrf_score`, `perplexity`, `word_error_rate`.
- **Text processing**: `TextPipeline` (case/unicode/pattern transforms), n-grams/shingles/collocations, readability (`flesch_kincaid_grade`, `gunning_fog_index`), `TextAugmenter`, data collators for MLM/seq2seq, LM preprocessing (`LmPreprocessor`), and padding/masking ops.

## Modules

| Module | Contents |
|---|---|
| `tokenizer/` | `bpe`, `sentencepiece`, `wordpiece`, `char`, `bytelevel`, `normalizer`, `pretokenizer`, `post`, `trainer` |
| `embedding/` | `pretrained` (GloVe/Word2Vec), `fasttext`, plus `WordEmbedding`/`PositionalEmbedding` |
| `features.rs` | TF-IDF, BM25, Bag-of-Words, hashing vectorizer |
| `similarity.rs` | `TextSimilarity` + `SimilarityMetric` |
| `compute.rs` | BLEU, ROUGE, CHRF, perplexity, WER |
| `analyze.rs` | Readability scores, `TextStats` |
| `ops.rs`, `process.rs`, `transform.rs`, `lm.rs`, `helper.rs`, `optimize.rs`, `vocab.rs`, `utils.rs` | Token/sequence ops, cleaning, pipelines, LM preprocessing, collators, `Vocab`/`VocabBuilder`, distances |

## Quick Start

```rust
use brain_text::{SpecialKind, Vocab};

let mut vocab = Vocab::new();
vocab.add_special("[PAD]", SpecialKind::Pad);
vocab.insert("hello_1");
assert_eq!(vocab.pad_id(), Some(0));
assert!(vocab.contains("hello_1"));
```

For full pipelines, `TextBuilder` chains tokenizer/embedding builders: `.bpe()`, `.sentencepiece()`, `.wordpiece()`, `.char_tokenizer()`, `.word_tokenizer()`, `.embedding(vocab_size, dim)`.

## Testing

```bash
cargo test -p brain-text -j 2
```

## Workspace Role

Natural-language processing layer of the Brain stack. Depends on `brain-core` (tensors) and `brain-autograd` (gradients) — zero external dependencies, 100% safe Rust.