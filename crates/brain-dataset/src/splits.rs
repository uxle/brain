//! # Dataset Splitting & Cross-Validation
//!
//! Provides train/val/test splits, stratified splitting, and k-fold cross-validation.

use std::collections::HashMap;

/// Container for partitioned index splits.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitResult {
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Splits dataset indices into train/val/test sets given fractions.
pub fn random_split_indices(total: usize, train_frac: f64, val_frac: f64) -> SplitResult {
    let train_count = ((total as f64) * train_frac) as usize;
    let val_count = ((total as f64) * val_frac) as usize;

    let train_indices = (0..train_count).collect();
    let val_indices = (train_count..train_count + val_count).collect();
    let test_indices = (train_count + val_count..total).collect();

    SplitResult {
        train_indices,
        val_indices,
        test_indices,
    }
}

/// Generates K-fold cross-validation splits over `total` items.
/// Returns a `Vec` of `(train_indices, val_indices)` tuples for each fold.
pub fn k_fold_split_indices(total: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert!(k >= 2, "K-fold cross validation requires k >= 2");
    assert!(total >= k, "Total elements must be >= k");

    let fold_size = total / k;
    let remainder = total % k;

    let mut folds = Vec::with_capacity(k);
    let mut current_idx = 0;

    for fold_i in 0..k {
        let current_fold_size = fold_size + if fold_i < remainder { 1 } else { 0 };
        let val_indices: Vec<usize> = (current_idx..current_idx + current_fold_size).collect();
        let train_indices: Vec<usize> = (0..current_idx)
            .chain(current_idx + current_fold_size..total)
            .collect();

        folds.push((train_indices, val_indices));
        current_idx += current_fold_size;
    }

    folds
}

/// Splits classification indices into train and test sets while preserving label class proportions.
pub fn stratified_split_indices(labels: &[usize], train_frac: f64) -> (Vec<usize>, Vec<usize>) {
    let mut class_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        class_map.entry(label).or_default().push(idx);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    for (_class, mut indices) in class_map {
        let n = indices.len();
        let train_count = ((n as f64) * train_frac).round() as usize;

        let (train_part, test_part) = indices.split_at_mut(train_count.min(n));
        train_indices.extend_from_slice(train_part);
        test_indices.extend_from_slice(test_part);
    }

    train_indices.sort();
    test_indices.sort();

    (train_indices, test_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_fold_splitting() {
        let folds = k_fold_split_indices(10, 5);
        assert_eq!(folds.len(), 5);

        for (train, val) in folds {
            assert_eq!(train.len(), 8);
            assert_eq!(val.len(), 2);
            assert_eq!(train.len() + val.len(), 10);
        }
    }

    #[test]
    fn test_stratified_splitting() {
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1]; // 4 of class 0, 4 of class 1
        let (train, test) = stratified_split_indices(&labels, 0.75);

        assert_eq!(train.len(), 6); // 3 of class 0, 3 of class 1
        assert_eq!(test.len(), 2); // 1 of class 0, 1 of class 1
    }
}
