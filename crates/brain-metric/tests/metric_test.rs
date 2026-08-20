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
