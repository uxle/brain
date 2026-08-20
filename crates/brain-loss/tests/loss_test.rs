//! # Loss Functions Integration Tests

use brain_core::Tensor;
use brain_loss::classification::ce::{CrossEntropyConfig, CrossEntropyLoss};
use brain_loss::core::Reduction;
use brain_loss::regression::mse::{HuberLoss, MAELoss, MSELoss};
use brain_loss::regression::RegressionLoss;

#[test]
fn test_cross_entropy_loss_computation() {
    let ce = CrossEntropyLoss::new(CrossEntropyConfig {
        reduction: Reduction::Mean,
        label_smoothing: 0.0,
        ..Default::default()
    });

    // 2 samples, 3 classes
    let logits = Tensor::from_slice(&[2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3]);

    let targets = vec![0, 1]; // Correct classes
    let loss = ce.forward_logits(&logits, &targets).expect("CE forward");

    assert_eq!(loss.shape(), &[1]);
    let loss_val = loss.data()[0];
    assert!(
        loss_val > 0.0 && loss_val < 1.0,
        "Loss for confident correct prediction should be small"
    );
}

#[test]
fn test_mse_mae_huber_losses() {
    let pred = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
    let target = Tensor::from_slice(&[1.5, 2.0, 2.5, 5.0], vec![4]);

    let mse = MSELoss {
        reduction: Reduction::Mean,
    };
    let mse_val = mse.compute(&pred, &target).expect("MSE").data()[0];
    // diff = [-0.5, 0, 0.5, -1.0] -> sq = [0.25, 0, 0.25, 1.0] -> mean = 1.5 / 4 = 0.375
    assert!((mse_val - 0.375).abs() < 1e-6);

    let mae = MAELoss {
        reduction: Reduction::Mean,
    };
    let mae_val = mae.compute(&pred, &target).expect("MAE").data()[0];
    // abs = [0.5, 0, 0.5, 1.0] -> mean = 2.0 / 4 = 0.5
    assert!((mae_val - 0.5).abs() < 1e-6);

    let huber = HuberLoss {
        delta: 1.0,
        reduction: Reduction::Mean,
    };
    let huber_val = huber.compute(&pred, &target).expect("Huber").data()[0];
    assert!(huber_val > 0.0);
}

#[test]
fn test_focal_loss_and_infonce_loss() {
    use brain_loss::classification::focal::{FocalConfig, FocalLoss};
    use brain_loss::contrastive::infonce::{InfoNCELoss, InfoNceConfig};

    // Focal Loss
    let focal = FocalLoss::new(FocalConfig {
        gamma: 2.0,
        alpha: 0.25,
        reduction: Reduction::Mean,
    });
    let logits = Tensor::from_slice(&[5.0, -5.0], vec![1, 2]);
    let focal_loss = focal.forward_logits(&logits, &[0]).unwrap();
    assert!(focal_loss.data()[0] >= 0.0);

    // InfoNCE Loss
    let infonce = InfoNCELoss::new(InfoNceConfig {
        temperature: 0.5,
        reduction: Reduction::Mean,
    });
    let q = Tensor::from_slice(&[1.0, 0.0], vec![1, 2]);
    let pos = Tensor::from_slice(&[0.9, 0.1], vec![1, 2]);
    let neg1 = Tensor::from_slice(&[-0.8, 0.2], vec![1, 2]);
    let neg2 = Tensor::from_slice(&[0.0, -1.0], vec![1, 2]);

    let info_loss = infonce.compute(&q, &pos, &[neg1, neg2]).unwrap();
    assert!(info_loss.data()[0] >= 0.0);
}
