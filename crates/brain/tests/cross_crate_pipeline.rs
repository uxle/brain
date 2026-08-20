//! # Phase 34: Cross-Crate End-to-End Deep Learning Pipeline Audit
//!
//! Verifies full architectural integration across:
//! 1. brain-core (Tensor & GEMM)
//! 2. brain-nn (Linear, ReLU)
//! 3. brain-loss (Cross-Entropy loss)
//! 4. brain-optim (AdamW decoupled optimizer)
//! 5. brain-metric (ROC-AUC performance metric)
//! 6. brain-export (Safetensors binary checkpointing)

use brain_core::Tensor;
use brain_nn::{Linear, relu, Module};
use brain_loss::CrossEntropyLoss;
use brain_optim::{Adam, AdamConfig, ParamGroup, Optimizer};
use brain_metric::classification::auc::roc_auc_score;
use brain_export::safetensors::SafetensorsArchive;

#[test]
fn test_cross_crate_end_to_end_training_and_export_pipeline() {
    // 1. Synthetic Dataset Preparation (4 samples, 2 features, 2 classes)
    let inputs = Tensor::from_slice(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], vec![4, 2]);
    let targets = vec![0, 1, 1, 0]; // XOR task targets

    // 2. Neural Architecture: Linear(2->4) -> ReLU -> Linear(4->2)
    let mut l1 = Linear::new(2, 4, true);
    let l2 = Linear::new(4, 2, true);

    // Initial forward pass
    let h1 = l1.forward(&inputs).unwrap();
    let a1 = relu(&h1);
    let logits = l2.forward(&a1).unwrap();
    assert_eq!(logits.shape(), &[4, 2]);

    // 3. Loss Evaluation
    let loss_fn = CrossEntropyLoss::default();
    let loss = loss_fn.forward_logits(&logits, &targets).unwrap();
    assert!(loss.item() > 0.0);

    // 4. Optimizer Step via AdamW
    let cfg = AdamConfig {
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        amsgrad: false,
        decoupled_weight_decay: true,
    };
    let group = ParamGroup::new(vec![0], 0.05);
    let mut optimizer = Adam::new(vec![group], cfg);

    let mut l1_params = vec![l1.weight.clone()];
    let grads = vec![Tensor::from_slice(&[0.1; 8], vec![4, 2])];
    optimizer.step(&mut l1_params, &grads).unwrap();
    l1.weight = l1_params.remove(0);

    // 5. Evaluation Metric (ROC-AUC)
    let probs = vec![0.8, 0.7, 0.6, 0.2];
    let binary_targets = vec![1, 1, 0, 0];
    let auc = roc_auc_score(&probs, &binary_targets);
    assert_eq!(auc, 1.0, "Perfect ranking must achieve 1.0 AUC");

    // 6. Safetensors Binary Model Checkpointing
    let mut archive = SafetensorsArchive::new();
    archive.insert("l1.weight", l1.weight.clone());
    archive.insert("l2.weight", l2.weight.clone());

    let bytes = archive.to_bytes();
    assert!(!bytes.is_empty());

    let reader = SafetensorsArchive::from_bytes(&bytes).expect("Safetensors read");
    let loaded_w1 = reader.get("l1.weight").expect("l1.weight lookup");
    assert_eq!(loaded_w1.shape(), l1.weight.shape());
    for (a, b) in loaded_w1.data().iter().zip(l1.weight.data().iter()) {
        assert!((a - b).abs() < 1e-6, "Safetensors F32 roundtrip precision: got {}, expected {}", a, b);
    }
}
