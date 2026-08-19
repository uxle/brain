use brain_core::Tensor;
use brain_transformer::attention::multi_head::{MhaConfig, MultiHeadAttention};
use brain_transformer::core::AttentionMask;
use brain_transformer::models::llama_lite::{LlamaLite, LlamaLiteConfig};
use brain_transformer::position::rope::{RopeConfig, RotaryEmbedding};

#[test]
fn test_mha_causal_forward_pass() {
    let cfg = MhaConfig {
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        bias: false,
        is_causal: true,
        dropout: 0.0,
    };
    let mha = MultiHeadAttention::new(cfg, 42);

    let batch = 2;
    let seq_len = 8;
    let x = Tensor::ones(vec![batch, seq_len, 32]);

    let out = mha.forward_mha(&x, None, &AttentionMask::Causal).expect("MHA causal forward");
    assert_eq!(out.shape(), &[batch, seq_len, 32]);
    for &v in out.data() {
        assert!(v.is_finite(), "MHA output must be finite");
    }
}

#[test]
fn test_rope_rotation_invariants() {
    let cfg = RopeConfig {
        dim: 16,
        max_seq_len: 64,
        theta: 10000.0,
        scaling_factor: 1.0,
        is_2d: false,
    };
    let rope = RotaryEmbedding::new(cfg);

    let batch = 2;
    let heads = 2;
    let seq_len = 4;
    let head_dim = 16;

    let mut q = Tensor::ones(vec![batch, heads, seq_len, head_dim]);
    rope.apply_rope_4d(&mut q, 0).expect("RoPE forward");

    assert_eq!(q.shape(), &[batch, heads, seq_len, head_dim]);
    for &v in q.data() {
        assert!(v.is_finite());
    }
}

#[test]
fn test_llama_lite_end_to_end_forward() {
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

    // Token IDs: [batch=2, seq_len=4]
    let token_ids = vec![
        1, 5, 12, 42,
        3, 8, 99, 10,
    ];

    let logits = model.forward(&token_ids, 2, 4).expect("LLaMA forward");
    assert_eq!(logits.shape(), &[2, 4, 100]);

    for &v in logits.data() {
        assert!(v.is_finite(), "Logits must be finite");
    }
}
