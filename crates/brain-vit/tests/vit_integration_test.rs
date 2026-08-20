//! # Vision Transformer Integration Test
use brain_vit::config::VitConfig;
use brain_vit::r#impl::VitModel;

#[test]
fn test_vit_forward_pass_logits() {
    let mut cfg = VitConfig::tiny();
    cfg.depth = 1; // 1-layer forward test for speed

    let mut model = VitModel::new(cfg, 42).expect("VitModel creation");
    let pixels = vec![0.5f64; 3 * 224 * 224];
    let output = model.forward(&pixels, 1).expect("ViT forward pass");

    assert_eq!(output.logits.len(), 1);
    assert_eq!(output.logits[0].len(), 1000);

    for &val in &output.logits[0] {
        assert!(val.is_finite(), "ViT logits must be finite");
    }
}
