//! # End-to-End Trainer Regression Tests

use brain_core::Tensor;
use brain_train::{Batch, Conv2d, Flatten, Linear, MaxPool2d, ModelState, ReLU, Sequential, Trainer};

#[test]
fn test_mlp_regression_training() {
    let inputs = Tensor::from_vec(
        vec![
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ],
        vec![4, 2],
    );
    let targets = vec![0, 1, 1, 0]; // XOR task
    let batch = Batch::new(inputs, targets).unwrap();
    let batches = vec![batch];

    let model = Sequential::new()
        .add(Linear::new(2, 8, true))
        .add(ReLU::new())
        .add(Linear::new(8, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.1)
        .build()
        .unwrap();

    let before = trainer.evaluate(&batches).unwrap();
    let after = trainer.fit(&batches, 40).unwrap();

    assert!(
        after.loss < before.loss,
        "MLP loss should strictly decrease: before={}, after={}",
        before.loss,
        after.loss
    );
    assert!(after.accuracy >= 0.70, "Expected accuracy >= 0.70, got {}", after.accuracy);
}

#[test]
fn test_cnn_regression_training() {
    // 8 samples of 1x6x6 synthetic images
    let mut data_vec = Vec::with_capacity(8 * 36);
    let mut targets = Vec::with_capacity(8);

    for i in 0..4 {
        let mut img = vec![0.05; 36];
        img[0] = 3.0 + i as f64 * 0.1;
        img[1] = 3.0;
        img[6] = 3.0;
        data_vec.extend(img);
        targets.push(0);
    }
    for i in 0..4 {
        let mut img = vec![0.05; 36];
        img[35] = 3.0 + i as f64 * 0.1;
        img[34] = 3.0;
        img[29] = 3.0;
        data_vec.extend(img);
        targets.push(1);
    }

    let inputs = Tensor::from_vec(data_vec, vec![8, 1, 6, 6]);
    let batch = Batch::new(inputs, targets).unwrap();
    let batches = vec![batch];

    let model = Sequential::new()
        .add(Conv2d::new(1, 4, 3, true))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Flatten::new())
        .add(Linear::new(4 * 3 * 3, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.1)
        .build()
        .unwrap();

    let before = trainer.evaluate(&batches).unwrap();
    let after = trainer.fit(&batches, 20).unwrap();

    assert!(
        after.loss < before.loss,
        "CNN loss should strictly decrease: before={}, after={}",
        before.loss,
        after.loss
    );
    assert!(after.accuracy >= 0.85, "Expected CNN accuracy >= 0.85, got {}", after.accuracy);
}

#[test]
fn test_model_state_checkpoint_resume() {
    let inputs = Tensor::from_vec(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]);
    let targets = vec![0, 1];
    let batch = Batch::new(inputs, targets).unwrap();
    let batches = vec![batch];

    let model = Sequential::new()
        .add(Linear::new(2, 4, true))
        .add(ReLU::new())
        .add(Linear::new(4, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.05)
        .build()
        .unwrap();

    let phase1 = trainer.fit(&batches, 5).unwrap();
    let saved_state = trainer.state();
    let bytes = saved_state.to_bytes();

    let loaded_state = ModelState::from_bytes(&bytes).unwrap();
    let mut resumed_trainer = Trainer::builder()
        .model(
            Sequential::new()
                .add(Linear::new(2, 4, true))
                .add(ReLU::new())
                .add(Linear::new(4, 2, true)),
        )
        .learning_rate(0.05)
        .build()
        .unwrap();

    resumed_trainer.load_state(&loaded_state).unwrap();
    let phase2 = resumed_trainer.fit(&batches, 10).unwrap();

    assert!(
        phase2.loss <= phase1.loss,
        "Resumed training should maintain or improve loss: phase1={}, phase2={}",
        phase1.loss,
        phase2.loss
    );
}

#[test]
fn test_gradient_accumulation_training() {
    let b1 = Batch::new(Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]), vec![0]).unwrap();
    let b2 = Batch::new(Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]), vec![1]).unwrap();
    let batches = vec![b1, b2];

    let model = Sequential::new()
        .add(Linear::new(2, 4, true))
        .add(ReLU::new())
        .add(Linear::new(4, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.1)
        .build()
        .unwrap();

    let before = trainer.evaluate(&batches).unwrap();
    let after = trainer.fit_accumulated(&batches, 25, 2).unwrap();

    assert!(
        after.loss < before.loss,
        "Accumulated training loss should decrease: before={}, after={}",
        before.loss,
        after.loss
    );
}
