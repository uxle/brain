# `brain-transformer`

Production-grade Transformer architectures — attention, positional encodings, KV caching, and autoregressive generation — in 100% safe, zero-dependency Rust.

## Overview

`brain-transformer` implements the full Transformer stack for the Brain ecosystem: scaled dot-product, multi-head, grouped/multi-query, relative (T5-style), xFormers-lite, and FlashAttention-lite attention; RoPE, ALiBi, learned, and sinusoidal positional encodings; Pre-LN/Post-LN and RMSNorm normalization; and encoder/decoder stacks with SwiGLU/GEGLU FFNs. It ships four ready-to-run model families — `GptLite` (decoder-only), `BertLite` (encoder-only), `T5Lite` (encoder-decoder), and `LlamaLite` (RMSNorm + RoPE + SwiGLU + GQA) — plus a KV-cache-backed generation engine and high-level inference pipelines.

## Features

- **Attention mechanisms**: `MultiHeadAttention`, `GroupedQueryAttention` (with `MqaConfig` and `repeat_kv`), `CrossAttention`, `RelativeAttention` (T5 bucketed biases), `FlashAttentionLite` (block-tiled online softmax), `XformersAttentionLite` (memory-budget chunked queries), `scaled_dot_product_attention`, and the polymorphic `Attention`/`make_attention` factory.
- **Positional encodings**: `RotaryEmbedding` (RoPE 1D/2D, configurable theta and scaling), `AlibiPositionalBias`, `LearnedPositionalEmbedding`, `SinusoidalPositionalEmbedding`, via the `PositionalEncoding` trait.
- **Model families**: `GptLite`, `BertLite` (with `BertOutput`), `T5Lite`, and `LlamaLite`, all with config structs (`GptLiteConfig`, `BertLiteConfig`, `T5LiteConfig`, `LlamaLiteConfig`).
- **Inference**: `KvCache`/`LayerKvCache` multi-layer caching, `Generator` + `GenerateConfig` (greedy, temperature, top-k, top-p, min-p, repetition penalty), and `TextGenerationPipeline`/`SequenceClassificationPipeline` in the `impl` module.
- **Building blocks**: `TransformerBuilder` fluent config, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderBlock`, `TransformerDecoderLayer`, `TransformerEmbedding`, `FeedForwardNetwork`, `ClsHead`/`LmHead`, attention masking (`AttentionMask`), and normalization ops (`rms_norm`, `layer_norm`, GELU, SiLU, SwiGLU, etc.).

## Modules

| Module | Contents |
|---|---|
| `attention/` | `multi_head`, `multi_query`, `cross`, `relative`, `scaled`, `flash_lite`, `xformers_lite` |
| `models/` | `bert_lite`, `gpt_lite`, `t5_lite`, `llama_lite` |
| `position/` | `rope`, `alibi`, `learned` |
| `encoder/`, `decoder/` | Block/layer stacks (`block`, `layer`, `cross`) |
| `kv_cache.rs` | `KvCache`, `LayerKvCache`, `KvCacheConfig` |
| `generate.rs` | `GenerateConfig`, `Generator` decoding engine |
| `impl.rs` | `SequenceClassificationPipeline`, `TextGenerationPipeline` |
| `head.rs`, `ffn.rs`, `embedding_layers.rs` | `ClsHead`/`LmHead`, `FeedForwardNetwork`, `TransformerEmbedding` |
| `config.rs`, `core.rs`, `ops.rs`, `utils.rs`, `builder.rs` | Configs, masks/errors, activation ops, RNG/init helpers, `TransformerBuilder` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_transformer::{LlamaLite, LlamaLiteConfig};

let cfg = LlamaLiteConfig {
    vocab_size: 100,
    hidden_dim: 32,
    num_layers: 2,
    num_heads: 4,
    num_kv_heads: 4,
    head_dim: 8,
    intermediate_dim: 64,
    max_seq_len: 32,
    rope_theta: 10000.0,
    norm_eps: 1e-5,
};
let model = LlamaLite::new(cfg, 123);

// Token IDs for [batch=1, seq_len=4]
let logits = model.forward(&[1, 5, 12, 42], 1, 4).expect("LLaMA forward");
assert_eq!(logits.shape(), &[1, 4, 100]);
```

## Testing

```bash
cargo test -p brain-transformer -j 2
```

## Workspace Role

The Transformer backbone for the Brain framework. Depends only on `brain-core` (tensor primitives) — zero external dependencies, 100% safe Rust.