use brain_core::Tensor;
use brain_quantization::{config::FakeQuantConfig, core::QuantDType, fake_quant::FakeQuantize};

#[test]
fn test_fake_quantize_forward_and_ste() {
    let mut config = FakeQuantConfig::default();
    config.dtype = QuantDType::Int8;
    config.ste_grad_clip = true;

    let mut fq = FakeQuantize::new(config);
    let tensor = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    fq.init_from_tensor(&tensor).unwrap();

    let out = fq.forward(&tensor);
    assert_eq!(out.shape(), &[5]);

    // Test STE backward pass
    let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0], vec![5]);
    let grad_in = fq.backward_ste(&grad_out, &tensor);
    assert_eq!(grad_in.shape(), &[5]);
    for &g in grad_in.data() {
        assert!(g == 1.0 || g == 0.0);
    }
}
