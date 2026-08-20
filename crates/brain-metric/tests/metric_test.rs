//! # Metric Integration Tests

use brain_metric::classification::auc::{pr_auc_score, roc_auc_score};

#[test]
fn test_roc_auc_score_perfect_ranking() {
    let probs = vec![0.9, 0.8, 0.4, 0.1];
    let targets = vec![1, 1, 0, 0];

    let auc = roc_auc_score(&probs, &targets);
    assert!(
        (auc - 1.0).abs() < 1e-6,
        "Perfect ranking must yield AUC = 1.0"
    );

    let pr_auc = pr_auc_score(&probs, &targets);
    assert!(pr_auc > 0.8);
}

#[test]
fn test_roc_auc_score_inverted_ranking() {
    let probs = vec![0.1, 0.2, 0.8, 0.9];
    let targets = vec![1, 1, 0, 0];

    let auc = roc_auc_score(&probs, &targets);
    assert!(
        (auc - 0.0).abs() < 1e-6,
        "Inverted ranking must yield AUC = 0.0"
    );
}

#[test]
fn test_multiclass_roc_auc_and_mcc_kappa() {
    use brain_metric::{cohen_kappa, matthews_corrcoef, multiclass_roc_auc};

    // 3 classes, 3 samples
    let probs = vec![
        vec![0.9, 0.05, 0.05],
        vec![0.1, 0.8, 0.1],
        vec![0.05, 0.05, 0.9],
    ];
    let targets = vec![0, 1, 2];

    let multi_auc = multiclass_roc_auc(&probs, &targets, 3);
    assert!((multi_auc - 1.0).abs() < 1e-6);

    let preds = vec![0, 1, 2, 0];
    let actual = vec![0, 1, 2, 1];
    let kappa = cohen_kappa(&preds, &actual, 3);
    assert!(kappa > 0.5);

    let bin_preds = vec![1, 0, 1, 0];
    let bin_actual = vec![1, 0, 1, 0];
    assert_eq!(matthews_corrcoef(&bin_preds, &bin_actual), 1.0);
}
