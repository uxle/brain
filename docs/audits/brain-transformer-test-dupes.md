# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-transformer/src`
- **Total Test Functions Scanned:** 5774
- **Duplicate / Template Groups:** 34
- **Total Padded / Duplicate Test Functions:** 5774 (100.0% of total tests)
- **Redundant Functions Removable:** 5740

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-transformer/src/attention/flash_lite.rs` | 171 | 171 | 100.0% |
| `crates/brain-transformer/src/attention/mod.rs` | 161 | 161 | 100.0% |
| `crates/brain-transformer/src/attention/multi_head.rs` | 173 | 173 | 100.0% |
| `crates/brain-transformer/src/attention/multi_query.rs` | 156 | 156 | 100.0% |
| `crates/brain-transformer/src/attention/relative.rs` | 174 | 174 | 100.0% |
| `crates/brain-transformer/src/attention/scaled.rs` | 198 | 198 | 100.0% |
| `crates/brain-transformer/src/attention/xformers_lite.rs` | 296 | 296 | 100.0% |
| `crates/brain-transformer/src/builder.rs` | 137 | 137 | 100.0% |
| `crates/brain-transformer/src/config/mod.rs` | 108 | 108 | 100.0% |
| `crates/brain-transformer/src/core.rs` | 135 | 135 | 100.0% |
| `crates/brain-transformer/src/decoder/cross.rs` | 188 | 188 | 100.0% |
| `crates/brain-transformer/src/decoder/layer.rs` | 172 | 172 | 100.0% |
| `crates/brain-transformer/src/decoder/mod.rs` | 143 | 143 | 100.0% |
| `crates/brain-transformer/src/embedding_layers.rs` | 147 | 147 | 100.0% |
| `crates/brain-transformer/src/encoder/block.rs` | 163 | 163 | 100.0% |
| `crates/brain-transformer/src/encoder/layer.rs` | 218 | 218 | 100.0% |
| `crates/brain-transformer/src/encoder/mod.rs` | 143 | 143 | 100.0% |
| `crates/brain-transformer/src/ffn/mod.rs` | 124 | 124 | 100.0% |
| `crates/brain-transformer/src/generate.rs` | 165 | 165 | 100.0% |
| `crates/brain-transformer/src/head/mod.rs` | 176 | 176 | 100.0% |
| `crates/brain-transformer/src/impl.rs` | 98 | 98 | 100.0% |
| `crates/brain-transformer/src/kv_cache.rs` | 100 | 100 | 100.0% |
| `crates/brain-transformer/src/lib.rs` | 244 | 244 | 100.0% |
| `crates/brain-transformer/src/models/bert_lite.rs` | 143 | 143 | 100.0% |
| `crates/brain-transformer/src/models/gpt_lite.rs` | 121 | 121 | 100.0% |
| `crates/brain-transformer/src/models/llama_lite.rs` | 112 | 112 | 100.0% |
| `crates/brain-transformer/src/models/mod.rs` | 219 | 219 | 100.0% |
| `crates/brain-transformer/src/models/t5_lite.rs` | 143 | 143 | 100.0% |
| `crates/brain-transformer/src/ops.rs` | 86 | 86 | 100.0% |
| `crates/brain-transformer/src/position/alibi.rs` | 228 | 228 | 100.0% |
| `crates/brain-transformer/src/position/learned.rs` | 247 | 247 | 100.0% |
| `crates/brain-transformer/src/position/mod.rs` | 363 | 363 | 100.0% |
| `crates/brain-transformer/src/position/rope.rs` | 194 | 194 | 100.0% |
| `crates/brain-transformer/src/utils.rs` | 128 | 128 | 100.0% |

## Top Duplicate Groups

### Group 1: 363 identical functions (e.g. `test_position_registry_1` in `crates/brain-transformer/src/position/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/position/mod.rs:80`):
```rust
fn test_position_registry_1() {
        let k1 = PositionKind::Rope;
        let k2 = PositionKind::Alibi;
        let k3 = PositionKind::Sinusoidal;
        assert_eq!(k1, PositionKind::Rope);
        assert_ne!(k2, k3);
    }
```

### Group 2: 296 identical functions (e.g. `test_xformers_lite_1` in `crates/brain-transformer/src/attention/xformers_lite.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/xformers_lite.rs:90`):
```rust
fn test_xformers_lite_1() {
        let cfg = XformersConfig { chunk_size: 4, scale: None, is_causal: false };
        let q = Tensor::from_vec(vec![1.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);
        let k_t = Tensor::from_vec(vec![1.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);
        let v = Tensor::from_vec(vec![2.0; 1 * 2 * 8 * 8], vec![1, 2, 8, 8]);

        let out = XformersAttentionLite::forward(&q, &k_t, &v, &cfg).unwrap();
        assert_eq!(out.shape(), &[1, 2, 8, 8]);
    }
```

### Group 3: 247 identical functions (e.g. `test_learned_sinusoidal_position_1` in `crates/brain-transformer/src/position/learned.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/position/learned.rs:136`):
```rust
fn test_learned_sinusoidal_position_1() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 1 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }
```

### Group 4: 244 identical functions (e.g. `test_prelude_integration_1` in `crates/brain-transformer/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/lib.rs:172`):
```rust
fn test_prelude_integration_1() {
        use crate::prelude::*;
        let cfg = TransformerConfig::default();
        assert!(cfg.validate().is_ok());

        let rng = TransformerRng::new(1 as u64);
        assert!(rng.clone().next_f64() < 1.0);

        let sin = SinusoidalPositionalEmbedding::generate(16, 8);
        assert_eq!(sin.shape(), &[16, 8]);
    }
```

### Group 5: 228 identical functions (e.g. `test_alibi_bias_1` in `crates/brain-transformer/src/position/alibi.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/position/alibi.rs:153`):
```rust
fn test_alibi_bias_1() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }
```

### Group 6: 219 identical functions (e.g. `test_models_registry_1` in `crates/brain-transformer/src/models/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/models/mod.rs:55`):
```rust
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
```

### Group 7: 218 identical functions (e.g. `test_encoder_layer_1` in `crates/brain-transformer/src/encoder/layer.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/encoder/layer.rs:71`):
```rust
fn test_encoder_layer_1() {
        let cfg = BlockConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            ..Default::default()
        };
        let layer = EncoderLayer::new(cfg, 0, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 1 * 2 * 16], vec![1, 2, 16]);
        let out = layer.forward(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[1, 2, 16]);
    }
```

### Group 8: 198 identical functions (e.g. `test_scaled_dot_product_1` in `crates/brain-transformer/src/attention/scaled.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/scaled.rs:171`):
```rust
fn test_scaled_dot_product_1() {
        let q = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);
        let k_t = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);
        let v = Tensor::from_vec(vec![2.0; 2 * 2 * 3 * 4], vec![2, 2, 3, 4]);

        let (out, weights) = scaled_dot_product_attention(&q, &k_t, &v, &AttentionMask::None, None).unwrap();
        assert_eq!(out.shape(), &[2, 2, 3, 4]);
        assert_eq!(weights.shape(), &[2, 2, 3, 3]);

        // Check row sum of softmax attention weights is exactly 1.0
        let w_data = weights.data();
        let row_sum: f64 = w_data[0..3].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
```

### Group 9: 194 identical functions (e.g. `test_rope_embedding_1` in `crates/brain-transformer/src/position/rope.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/position/rope.rs:234`):
```rust
fn test_rope_embedding_1() {
        let cfg = RopeConfig { dim: 16, max_seq_len: 64, theta: 10000.0, scaling_factor: 1.0, is_2d: false };
        let rope = RotaryEmbedding::new(cfg);
        assert_eq!(rope.cos_table.len(), 64);
        assert_eq!(rope.cos_table[0].len(), 8);

        let mut t = Tensor::from_vec(vec![1.0; 2 * 2 * 4 * 16], vec![2, 2, 4, 16]);
        rope.apply_rope_4d(&mut t, 0).unwrap();
        assert_eq!(t.shape(), &[2, 2, 4, 16]);

        let mut t2d = Tensor::from_vec(vec![1.0; 1 * 1 * 16 * 16], vec![1, 1, 16, 16]);
        rope.apply_rope_2d(&mut t2d, 4, 4).unwrap();
        assert_eq!(t2d.shape(), &[1, 1, 16, 16]);
    }
```

### Group 10: 188 identical functions (e.g. `test_cross_attention_1` in `crates/brain-transformer/src/decoder/cross.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/decoder/cross.rs:143`):
```rust
fn test_cross_attention_1() {
        let cfg = CrossAttnConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            bias: false,
            dropout: 0.0,
        };
        let cross = CrossAttention::new(cfg, 1 as u64);
        let dec_x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let enc_x = Tensor::from_vec(vec![2.0; 2 * 5 * 16], vec![2, 5, 16]);

        let out = cross.forward(&dec_x, &enc_x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
    }
```

### Group 11: 176 identical functions (e.g. `test_transformer_heads_1` in `crates/brain-transformer/src/head/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/head/mod.rs:180`):
```rust
fn test_transformer_heads_1() {
        let cfg = HeadConfig {
            hidden_dim: 16,
            vocab_size: 50,
            num_classes: Some(3),
            bias: false,
        };
        let lm = LmHead::new(cfg.clone(), 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let logits = lm.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 50]);

        let cls = ClsHead::new(cfg, 1 as u64);
        let class_logits = cls.forward(&x).unwrap();
        assert_eq!(class_logits.shape(), &[2, 3]);
// ... (truncated)
```

### Group 12: 174 identical functions (e.g. `test_relative_attention_1` in `crates/brain-transformer/src/attention/relative.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/relative.rs:216`):
```rust
fn test_relative_attention_1() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
// ... (truncated)
```

### Group 13: 173 identical functions (e.g. `test_multi_head_attention_1` in `crates/brain-transformer/src/attention/multi_head.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/multi_head.rs:229`):
```rust
fn test_multi_head_attention_1() {
        let cfg = MhaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            bias: true,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = mha.forward_mha(&x, None, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let causal_out = mha.forward_mha(&x, None, &AttentionMask::Causal).unwrap();
        assert_eq!(causal_out.shape(), &[2, 4, 32]);
// ... (truncated)
```

### Group 14: 172 identical functions (e.g. `test_decoder_layer_1` in `crates/brain-transformer/src/decoder/layer.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/decoder/layer.rs:247`):
```rust
fn test_decoder_layer_1() {
        let cfg = DecoderLayerConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            has_cross_attention: true,
            ..Default::default()
        };
        let layer = TransformerDecoderLayer::new(cfg, 1 as u64);
        let dec_x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let enc_x = Tensor::from_vec(vec![2.0; 2 * 4 * 16], vec![2, 4, 16]);

        let out = layer.forward(&dec_x, Some(&enc_x), &AttentionMask::Causal, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 16]);
// ... (truncated)
```

### Group 15: 171 identical functions (e.g. `test_flash_attention_lite_1` in `crates/brain-transformer/src/attention/flash_lite.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/flash_lite.rs:257`):
```rust
fn test_flash_attention_lite_1() {
        let cfg = FlashLiteConfig { block_m: 2, block_n: 2, is_causal: false, scale: None };
        let q = Tensor::from_vec(vec![1.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);
        let k_t = Tensor::from_vec(vec![1.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);
        let v = Tensor::from_vec(vec![2.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);

        let flash_out = FlashAttentionLite::forward(&q, &k_t, &v, &cfg).unwrap();
        assert_eq!(flash_out.shape(), &[1, 1, 4, 8]);

        // Check equivalence with standard reference SDPA
        let (ref_out, _) = crate::attention::scaled::scaled_dot_product_attention(
            &q, &k_t, &v, &crate::core::AttentionMask::None, None
        ).unwrap();

        assert!(crate::utils::all_close(flash_out.data(), ref_out.data(), 1e-4, 1e-4));
// ... (truncated)
```

### Group 16: 165 identical functions (e.g. `test_generation_pipeline_1` in `crates/brain-transformer/src/generate.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/generate.rs:202`):
```rust
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
// ... (truncated)
```

### Group 17: 163 identical functions (e.g. `test_encoder_block_1` in `crates/brain-transformer/src/encoder/block.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/encoder/block.rs:252`):
```rust
fn test_encoder_block_1() {
        let cfg = BlockConfig {
            hidden_dim: 16,
            num_heads: 2,
            head_dim: 8,
            intermediate_dim: 32,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::LayerNorm,
            activation: ActivationType::Gelu,
            norm_eps: 1e-5,
            layer_scale_init: None,
        };
        let block = TransformerEncoderBlock::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 16], vec![2, 3, 16]);
        let out = block.forward(&x, &AttentionMask::None).unwrap();
// ... (truncated)
```

### Group 18: 161 identical functions (e.g. `test_attention_registry_1` in `crates/brain-transformer/src/attention/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/mod.rs:129`):
```rust
fn test_attention_registry_1() {
        let cfg = AttentionConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            ..Default::default()
        };
        let attn = make_attention(AttentionKind::MultiHead, &cfg);
        assert_eq!(attn.kind(), AttentionKind::MultiHead);

        let q = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let k_t = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let v = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);

// ... (truncated)
```

### Group 19: 156 identical functions (e.g. `test_gqa_mqa_attention_1` in `crates/brain-transformer/src/attention/multi_query.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/attention/multi_query.rs:213`):
```rust
fn test_gqa_mqa_attention_1() {
        let cfg = GqaConfig {
            hidden_dim: 32,
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            bias: false,
            dropout: 0.0,
        };
        let gqa = GroupedQueryAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 3 * 32], vec![2, 3, 32]);
        let out = gqa.forward_gqa(&x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 32]);

        let kv = Tensor::from_vec(vec![1.0; 2 * 2 * 3 * 8], vec![2, 2, 3, 8]);
// ... (truncated)
```

### Group 20: 147 identical functions (e.g. `test_embedding_layers_1` in `crates/brain-transformer/src/embedding_layers.rs`)
- Files involved: 1
- Sample definition (`crates/brain-transformer/src/embedding_layers.rs:255`):
```rust
fn test_embedding_layers_1() {
        let cfg = EmbConfig {
            vocab_size: 100,
            hidden_dim: 32,
            max_position_embeddings: 64,
            type_vocab_size: Some(2),
            pos_encoding: PositionEncodingType::Learned,
            norm_type: Some(NormType::LayerNorm),
            ..Default::default()
        };
        let emb = TransformerEmbedding::new(cfg, 1 as u64);
        assert_eq!(emb.word_embeddings.shape(), &[100, 32]);
        assert_eq!(emb.position_embeddings.as_ref().unwrap().shape(), &[64, 32]);

        let ids = vec![1, 2, 3, 4];
// ... (truncated)
```
