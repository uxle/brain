//! # Neural Network Layers Verification Harness (Stage E, Phases 116-140)
//!
//! Tests Linear, Conv2d, ConvTranspose2d, BatchNorm2d, LayerNorm, RMSNorm,
//! Embedding, 20+ Activations, MultiheadAttention, LSTM, GRU, and Sequential containers.

use brain_core::Tensor;
use brain_autograd::Value;
use brain_nn::*;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -----------------------------------------------------------------------------
// Phase 116: Linear Layer
// -----------------------------------------------------------------------------
#[test]
fn test_linear_layer_forward() {
    let linear = Linear::new(4, 2, true);
    let x = Value::new(Tensor::ones(vec![3, 4]), true);
    let out = linear.forward(&x).unwrap();

    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(linear.parameters().len(), 2); // weight and bias
}

// -----------------------------------------------------------------------------
// Phase 117 & 118: Conv2d & ConvTranspose2d
// -----------------------------------------------------------------------------
#[test]
fn test_conv2d_and_conv_transpose2d() {
    let conv = Conv2d::new(1, 2, 3, true);
    let x = Value::new(Tensor::ones(vec![2, 1, 8, 8]), false);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2, 8, 8]);

    let deconv = ConvTranspose2d::new(2, 1, 2);
    let upsampled = deconv.forward(&out).unwrap();
    assert_eq!(upsampled.shape(), &[2, 1, 9, 9]);
}

// -----------------------------------------------------------------------------
// Phase 123, 124 & 125: Normalization Layers
// -----------------------------------------------------------------------------
#[test]
fn test_normalization_layers() {
    let bn = BatchNorm2d::new(4);
    let x_4d = Value::new(Tensor::from_slice(&[1.0; 2 * 4 * 4 * 4], vec![2, 4, 4, 4]), false);
    let bn_out = bn.forward(&x_4d).unwrap();
    assert_eq!(bn_out.shape(), &[2, 4, 4, 4]);

    let ln = LayerNorm::new(vec![8], 1e-5);
    let x_2d = Value::new(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 8]), false);
    let ln_out = Module::forward(&ln, &x_2d).unwrap();
    assert_eq!(ln_out.shape(), &[1, 8]);

    let rms = RMSNorm::new(8, 1e-5);
    let rms_out = Module::forward(&rms, &x_2d).unwrap();
    assert_eq!(rms_out.shape(), &[1, 8]);
}

// -----------------------------------------------------------------------------
// Phase 127: Embedding Layer
// -----------------------------------------------------------------------------
#[test]
fn test_embedding_lookup() {
    let emb = Embedding::new(100, 16);
    let out = emb.forward_indices(&[1, 5, 42]);
    assert_eq!(out.shape(), &[3, 16]);
}

// -----------------------------------------------------------------------------
// Phase 129-133: Neural Activation Functions
// -----------------------------------------------------------------------------
#[test]
fn test_neural_activations() {
    let x = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]);

    let y_relu = relu(&x);
    assert_eq!(y_relu.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);

    let y_gelu = gelu(&x);
    assert_eq!(y_gelu.shape(), &[1, 5]);
    assert!(approx(y_gelu.get(0), -0.0455, 0.05));

    let y_silu = silu(&x);
    assert_eq!(y_silu.shape(), &[1, 5]);

    let y_mish = mish(&x);
    assert_eq!(y_mish.shape(), &[1, 5]);

    let y_sig = sigmoid(&x);
    assert!(y_sig.data().iter().all(|&v| v >= 0.0 && v <= 1.0));

    let y_soft = softmax(&x);
    let sum_soft: f64 = y_soft.data().iter().sum();
    assert!(approx(sum_soft, 1.0, 1e-5));
}

// -----------------------------------------------------------------------------
// Phase 134: MultiheadAttention
// -----------------------------------------------------------------------------
#[test]
fn test_multihead_attention_forward() {
    let mha = MultiheadAttention::new(32, 4);
    let q = Tensor::ones(vec![2, 8, 32]);
    let k = Tensor::ones(vec![2, 8, 32]);
    let v = Tensor::ones(vec![2, 8, 32]);

    let out = mha.forward_mha(&q, &k, &v, None).unwrap();
    assert_eq!(out.shape(), &[2, 8, 32]);
}

// -----------------------------------------------------------------------------
// Phase 135 & 136: Recurrent Layers (LSTM & GRU)
// -----------------------------------------------------------------------------
#[test]
fn test_lstm_and_gru_sequential_forward() {
    let lstm = LSTM::new(8, 16, 1);
    let x_seq = Value::new(Tensor::ones(vec![2, 5, 8]), false); // batch=2, seq=5, hidden=8
    let out_lstm = lstm.forward(&x_seq).unwrap();
    assert_eq!(out_lstm.shape(), &[2, 5, 16]);

    let gru = GRU::new(8, 16);
    let out_gru = gru.forward(&x_seq).unwrap();
    assert_eq!(out_gru.shape(), &[2, 5, 16]);
}

// -----------------------------------------------------------------------------
// Phase 139: Sequential Container
// -----------------------------------------------------------------------------
#[test]
fn test_sequential_container_composition() {
    let mut seq = Sequential::new();
    seq.add(Linear::new(4, 8, true));
    seq.add(Linear::new(8, 2, true));

    let x = Value::new(Tensor::ones(vec![2, 4]), false);
    let out = seq.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(seq.parameters().len(), 4); // 2 weights + 2 biases
}

// -----------------------------------------------------------------------------
// Phase 140: Stage E Master Neural Architecture Integration Audit
// -----------------------------------------------------------------------------
#[test]
fn test_stage_e_master_neural_architecture_audit() {
    // Composite Vision Block: Conv2d -> BatchNorm2d -> GELU -> MaxPool2d
    let conv = Conv2d::new(3, 8, 3, true);
    let bn = BatchNorm2d::new(8);
    let pool = MaxPool2d::new(2, 2);

    let img = Value::new(Tensor::ones(vec![2, 3, 16, 16]), false);
    let h1 = conv.forward(&img).unwrap();
    let h2 = bn.forward(&h1).unwrap();
    let h3 = Value::new(gelu(h2.data()), false);
    let h4 = pool.forward(&h3);

    assert_eq!(h4.shape(), &[2, 8, 8, 8]);
    for &v in h4.data().data() {
        assert!(v.is_finite());
    }
}
