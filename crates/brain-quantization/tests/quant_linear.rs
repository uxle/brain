//! # Quantized Linear & Quantization Roundtrip Verification Tests

use brain_core::tensor::arithmetic::matmul;
use brain_core::Tensor;
use brain_quantization::{
    apply_magnitude_prune, dequantize_tensor, quantize_tensor, QuantConfig, QuantDType,
};

#[test]
fn test_linear_weight_quantize_dequantize_error() {
    let weight = Tensor::from_vec(
        vec![0.12, -0.45, 0.78, -0.23, 0.91, -0.05, 0.33, -0.88],
        vec![2, 4],
    );
    let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 2.0], vec![1, 4]);

    // Full fp32 Linear forward: y = x @ W^T
    let w_t = weight.transpose(0, 1);
    let y_fp32 = matmul(&x, &w_t);

    let cfg = QuantConfig {
        dtype: QuantDType::Int8,
        ..QuantConfig::default()
    };
    let qweight = quantize_tensor(&weight, &cfg).unwrap();
    let deq_weight = dequantize_tensor(&qweight).unwrap();

    let deq_w_t = deq_weight.transpose(0, 1);
    let y_quant = matmul(&x, &deq_w_t);

    assert_eq!(y_fp32.shape(), y_quant.shape());
    for (fp_val, q_val) in y_fp32.data().iter().zip(y_quant.data()) {
        let diff = (fp_val - q_val).abs();
        assert!(
            diff < 1e-2,
            "Quantized linear error too large: fp32={}, quant={}, diff={}",
            fp_val,
            q_val,
            diff
        );
    }
}

#[test]
fn test_magnitude_pruning_linear() {
    let mut weight = Tensor::from_vec(vec![0.01, -0.9, 0.8, 0.02], vec![2, 2]);
    let result = apply_magnitude_prune(&mut weight, 0.5).unwrap();
    assert_eq!(result.pruned_elements, 2);
    assert_eq!(result.total_elements, 4);

    let zeros = weight.data().iter().filter(|&&v| v == 0.0).count();
    assert_eq!(zeros, 2, "Expected 2 pruned weights to be zero");
}
