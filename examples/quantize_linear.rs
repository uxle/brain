//! # Dynamic 8-Bit Quantization Example
//!
//! Demonstrates quantizing floating-point layer weights to 8-bit integers.

use brain_core::tensor::arithmetic::matmul;
use brain_core::Tensor;
use brain_quantization::{
    apply_magnitude_prune, dequantize_tensor, quantize_tensor, QuantConfig, QuantDType,
};

fn main() {
    println!("=== Brain 1.0 Dynamic 8-bit Quantization Example ===");

    let weight = Tensor::from_vec(
        vec![0.12, -0.45, 0.78, -0.23, 0.91, -0.05, 0.33, -0.88],
        vec![2, 4],
    );
    let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 2.0], vec![1, 4]);

    // Full fp32 reference
    let w_t = weight.transpose(0, 1);
    let y_fp32 = matmul(&x, &w_t);

    // Quantize weights to Int8
    let cfg = QuantConfig {
        dtype: QuantDType::Int8,
        ..QuantConfig::default()
    };
    let qweight = quantize_tensor(&weight, &cfg).expect("Quantize");
    let deq_weight = dequantize_tensor(&qweight).expect("Dequantize");

    let deq_w_t = deq_weight.transpose(0, 1);
    let y_quant = matmul(&x, &deq_w_t);

    println!("FP32 output:     {:?}", y_fp32.to_vec());
    println!("Quantized output: {:?}", y_quant.to_vec());

    // Magnitude pruning
    let mut prunable_weight = weight.clone();
    let prune_info = apply_magnitude_prune(&mut prunable_weight, 0.5).expect("Prune");
    println!(
        "Pruning: {} of {} weights pruned ({:.0}% sparsity)",
        prune_info.pruned_elements,
        prune_info.total_elements,
        prune_info.actual_sparsity * 100.0
    );
}
