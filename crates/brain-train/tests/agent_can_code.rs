//! # Agent-Generated Training Script Verification Tests

use brain_core::Tensor;
use brain_train::{Batch, Linear, ReLU, Sequential, Trainer};

#[test]
fn test_autonomous_agent_model_synthesis_and_training() {
    // Synthetic dataset representing an agent classification task
    let inputs = Tensor::from_vec(
        vec![
            0.5, 0.5,
            0.8, 0.9,
            -0.5, -0.5,
            -0.8, -0.9,
        ],
        vec![4, 2],
    );
    let targets = vec![0, 0, 1, 1];
    let batch = Batch::new(inputs, targets).unwrap();
    let batches = vec![batch];

    // Synthesized architecture: Linear(2 -> 4) -> ReLU -> Linear(4 -> 2)
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
    let after = trainer.fit(&batches, 15).unwrap();

    assert!(
        after.loss < before.loss,
        "Autonomous training must strictly decrease loss: before={}, after={}",
        before.loss,
        after.loss
    );
    assert!(after.accuracy >= 0.75, "Expected accuracy >= 0.75, got {}", after.accuracy);
}
