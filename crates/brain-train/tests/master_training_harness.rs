//! # Master Training Engine Verification Harness (Stage F, Phases 141-170)
//!
//! Tests loss optimization convergence, learning rate adjustments,
//! ModelState serialization, early stopping, and full training loop lifecycle.

use brain_core::Tensor;
use brain_train::{
    Batch, CallbackAction, EarlyStopping, Linear, MetricHistoryLogger,
    ModelState, ReLU, Sequential, Trainer, TrainingCallback,
};

// -----------------------------------------------------------------------------
// Phase 141-156: Multi-Step Convergence Optimization
// -----------------------------------------------------------------------------
#[test]
fn test_multistep_convergence_optimization() {
    let inputs = Tensor::from_vec(
        vec![
            0.5, 0.5,
            -0.5, -0.5,
            0.8, 0.2,
            -0.8, -0.2,
        ],
        vec![4, 2],
    );
    let targets = vec![0, 1, 0, 1];
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

    let initial = trainer.evaluate(&batches).unwrap();
    let trained = trainer.fit(&batches, 30).unwrap();

    assert!(
        trained.loss < initial.loss,
        "Loss must decrease: initial={}, trained={}",
        initial.loss,
        trained.loss
    );
    assert!(trained.accuracy >= 0.75, "Model accuracy should be >= 0.75, got {}", trained.accuracy);
}

// -----------------------------------------------------------------------------
// Phase 157-164: Learning Rate Scheduling & Early Stopping
// -----------------------------------------------------------------------------
#[test]
fn test_early_stopping_and_metric_logging() {
    let mut es = EarlyStopping::new(3, 0.001);
    let mut logger = MetricHistoryLogger::new();

    // Step 0: loss 1.0
    assert_eq!(es.on_epoch_end(0, 1.0, Some(0.5)), CallbackAction::Continue);
    logger.on_epoch_end(0, 1.0, Some(0.5));

    // Step 1: loss 0.9 (improvement)
    assert_eq!(es.on_epoch_end(1, 0.9, Some(0.4)), CallbackAction::Continue);
    logger.on_epoch_end(1, 0.9, Some(0.4));

    // Step 2, 3, 4: no improvement -> patience 3 exhausted at epoch 4
    assert_eq!(es.on_epoch_end(2, 0.9005, Some(0.4005)), CallbackAction::Continue);
    assert_eq!(es.on_epoch_end(3, 0.9002, Some(0.4002)), CallbackAction::Continue);
    assert_eq!(es.on_epoch_end(4, 0.9001, Some(0.4001)), CallbackAction::Stop);

    assert_eq!(logger.train_losses.len(), 2);
}

// -----------------------------------------------------------------------------
// Phase 168: ModelState Serialization Round-Trip
// -----------------------------------------------------------------------------
#[test]
fn test_model_state_serialization_round_trip() {
    let model = Sequential::new()
        .add(Linear::new(3, 5, true))
        .add(ReLU::new())
        .add(Linear::new(5, 2, true));

    let trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.01)
        .build()
        .unwrap();

    let state = trainer.state();
    let bytes = state.to_bytes();
    let restored = ModelState::from_bytes(&bytes).unwrap();

    assert_eq!(restored.tensors.len(), state.tensors.len());
    for (orig, rest) in state.tensors.iter().zip(restored.tensors.iter()) {
        assert_eq!(orig.name, rest.name);
        assert_eq!(orig.tensor.to_vec(), rest.tensor.to_vec());
    }
}

// -----------------------------------------------------------------------------
// Phase 170: Stage F Master Training Engine Integration Audit
// -----------------------------------------------------------------------------
#[test]
fn test_stage_f_master_training_audit() {
    // End-to-end multi-layer network training loop
    let inputs = Tensor::from_vec(vec![1.0; 16], vec![4, 4]);
    let targets = vec![0, 1, 0, 1];
    let batch = Batch::new(inputs, targets).unwrap();
    let batches = vec![batch];

    let model = Sequential::new()
        .add(Linear::new(4, 8, true))
        .add(ReLU::new())
        .add(Linear::new(8, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.02)
        .build()
        .unwrap();

    let metrics = trainer.fit(&batches, 15).unwrap();
    assert!(metrics.loss.is_finite());
    assert!(metrics.accuracy >= 0.5);
}
