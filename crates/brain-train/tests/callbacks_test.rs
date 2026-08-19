//! # Training Callbacks Verification Test

use brain_train::{CallbackAction, EarlyStopping, MetricHistoryLogger, TrainingCallback};

#[test]
fn test_early_stopping_trigger() {
    let mut es = EarlyStopping::new(3, 0.01);
    assert!(!es.stopped);

    // Epoch 0: loss 1.0 (new best)
    let a0 = es.on_epoch_end(0, 1.0, Some(1.0));
    assert_eq!(a0, CallbackAction::Continue);
    assert_eq!(es.wait_count, 0);

    // Epoch 1: loss 0.999 (improvement < min_delta 0.01 -> wait = 1)
    let a1 = es.on_epoch_end(1, 0.999, Some(0.999));
    assert_eq!(a1, CallbackAction::Continue);
    assert_eq!(es.wait_count, 1);

    // Epoch 2: loss 1.05 (no improvement -> wait = 2)
    let a2 = es.on_epoch_end(2, 1.05, Some(1.05));
    assert_eq!(a2, CallbackAction::Continue);
    assert_eq!(es.wait_count, 2);

    // Epoch 3: loss 1.10 (no improvement -> wait = 3 >= patience -> STOP)
    let a3 = es.on_epoch_end(3, 1.10, Some(1.10));
    assert_eq!(a3, CallbackAction::Stop);
    assert!(es.stopped);
}

#[test]
fn test_metric_history_logger() {
    let mut logger = MetricHistoryLogger::new();
    logger.on_batch_end(0, 0.5);
    logger.on_batch_end(1, 0.4);
    logger.on_epoch_end(0, 0.45, Some(0.48));

    assert_eq!(logger.batch_losses.len(), 2);
    assert_eq!(logger.train_losses, vec![0.45]);
    assert_eq!(logger.val_losses, vec![Some(0.48)]);
}
