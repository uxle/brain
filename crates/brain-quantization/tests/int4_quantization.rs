use brain_core::Tensor;
use brain_quantization::{
    block_quant::BlockQuantizer, config::BlockQuantConfig, core::QuantDType, dequantize_tensor,
};

#[test]
fn test_int4_block_quantization() {
    let mut config = BlockQuantConfig::default();
    config.dtype = QuantDType::Int4;
    config.group_size = 4;
    config.symmetric = true;

    let bq = BlockQuantizer::new(config);
    let original = Tensor::from_slice(&[0.1, -0.4, 0.7, -0.2, 0.9, -0.05, 0.33, -0.88], vec![2, 4]);
    let q_tensor = bq.quantize_blocks(&original).unwrap();

    assert_eq!(q_tensor.params.dtype, QuantDType::Int4);
    assert_eq!(q_tensor.params.scales.len(), 2); // 8 elements / group_size 4 = 2 groups

    let dequant = dequantize_tensor(&q_tensor).unwrap();
    assert_eq!(dequant.shape(), original.shape());

    // Check that reconstruction error is bounded for 4-bit representation
    for i in 0..8 {
        let diff = (original.data()[i] - dequant.data()[i]).abs();
        assert!(diff < 0.25, "Int4 reconstruction error too high: {}", diff);
    }
}
