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

#[test]
fn test_alibi_geometric_slopes_and_biases() {
    use brain_transformer::position::alibi::AlibiPositionalBias;

    let slopes = AlibiPositionalBias::compute_slopes(8);
    assert_eq!(slopes.len(), 8);

    // Assert strictly decreasing geometric progression
    for i in 0..7 {
        assert!(slopes[i] > slopes[i + 1], "Slopes must be monotonically decreasing");
    }
}

#[test]
fn test_kv_cache_layer_lifecycle() {
    use brain_transformer::kv_cache::LayerKvCache;

    let mut cache = LayerKvCache::new(1, 4, 8, 16);
    assert_eq!(cache.current_seq_len, 0);

    let k_step = Tensor::ones(vec![1, 4, 1, 8]);
    let v_step = Tensor::ones(vec![1, 4, 1, 8]);

    let (k_cached, v_cached) = cache.update(&k_step, &v_step).expect("Append KV step");
    assert_eq!(k_cached.shape(), &[1, 4, 1, 8]);
    assert_eq!(v_cached.shape(), &[1, 4, 1, 8]);
    assert_eq!(cache.current_seq_len, 1);
}
