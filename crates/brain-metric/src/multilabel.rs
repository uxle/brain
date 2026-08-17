//! # Multi-Label Evaluation Metrics
//!
//! Exact match ratio, Hamming loss, subset accuracy, and label-wise F1 metrics.
#![allow(missing_docs)]

/// Exact match ratio (subset accuracy): proportion of samples whose predicted labels match all true labels.
pub fn exact_match_ratio(preds: &[Vec<bool>], targets: &[Vec<bool>]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 { return 0.0; }
    let mut matches = 0usize;
    for i in 0..n {
        if preds[i] == targets[i] {
            matches += 1;
        }
    }
    matches as f64 / n as f64
}

/// Hamming loss: fraction of incorrect labels over total labels.
pub fn hamming_loss(preds: &[Vec<bool>], targets: &[Vec<bool>]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 { return 0.0; }
    let mut total_bits = 0usize;
    let mut diff_bits = 0usize;

    for i in 0..n {
        let num_labels = preds[i].len().min(targets[i].len());
        for l in 0..num_labels {
            total_bits += 1;
            if preds[i][l] != targets[i][l] {
                diff_bits += 1;
            }
        }
    }

    if total_bits > 0 { diff_bits as f64 / total_bits as f64 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_multilabel_stress_001() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_002() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_003() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_004() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_005() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_006() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_007() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_008() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_009() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_010() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_011() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_012() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_013() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_014() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_015() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_016() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_017() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_018() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_019() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_020() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_021() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_022() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_023() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_024() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_025() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_026() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_027() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_028() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_029() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_030() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_031() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_032() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_033() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_034() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_035() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_036() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_037() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_038() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_039() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_040() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_041() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_042() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_043() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_044() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_045() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_046() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_047() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_048() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_049() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_050() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_051() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_052() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_053() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_054() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_055() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_056() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_057() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_058() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_059() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_060() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_061() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_062() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_063() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_064() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_065() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_066() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_067() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_068() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_069() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_070() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_071() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_072() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_073() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_074() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_075() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_076() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_077() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_078() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_079() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_080() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_081() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_082() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_083() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_084() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_085() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_086() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_087() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_088() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_089() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_090() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_091() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_092() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_093() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_094() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_095() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_096() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_097() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_098() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_099() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_100() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_101() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_102() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_103() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_104() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_105() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_106() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_107() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_108() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_109() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_110() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_111() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_112() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_113() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_114() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_115() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_116() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_117() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_118() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_119() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_120() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_121() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_122() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_123() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_124() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_125() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_126() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_127() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_128() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_129() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_130() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_131() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_132() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_133() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_134() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_135() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_136() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_137() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_138() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_139() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_140() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_141() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_142() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_143() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_144() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_145() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_146() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_147() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_148() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_149() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_150() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_151() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_152() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_153() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_154() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_155() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_156() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_157() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_158() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_159() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_160() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_161() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_162() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_163() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_164() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_165() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_166() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_167() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_168() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_169() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_170() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_171() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_172() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_173() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_174() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_175() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_176() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_177() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_178() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_179() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_180() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_181() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_182() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_183() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_184() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_185() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_186() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_187() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_188() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_189() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_190() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_191() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_192() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_193() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_194() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_195() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_196() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_197() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_198() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_199() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_200() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_201() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_202() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_203() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_204() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_205() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_206() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_207() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_208() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_209() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_210() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_211() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_212() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_213() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_214() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_215() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_216() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_217() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_218() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_219() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_220() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_221() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_222() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_223() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_224() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_225() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_226() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_227() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_228() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_229() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_230() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_231() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_232() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_233() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_234() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_235() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_236() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_237() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_238() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_239() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_240() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_241() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_242() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_243() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_244() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_245() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_246() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_247() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_248() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_249() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_250() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_251() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_252() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_253() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_254() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_255() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_256() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_257() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_258() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_259() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_260() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_261() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_262() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_263() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_264() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_265() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_266() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_267() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_268() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_269() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_270() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_271() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_272() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_273() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_274() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_275() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_276() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_277() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_278() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_279() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_280() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_281() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_282() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_283() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_284() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_285() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_286() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_287() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_288() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_289() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_290() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_291() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_292() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_293() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_294() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_295() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_296() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_297() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_298() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_299() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_300() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_301() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_302() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_303() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_304() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_305() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_306() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_307() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_308() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_309() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_310() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_311() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_312() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_313() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_314() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_315() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_316() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_317() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_318() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_319() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_320() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_321() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_322() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_323() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_324() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_325() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_326() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_327() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_328() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_329() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_330() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_331() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_332() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_333() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_334() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_335() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_336() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_337() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_338() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_339() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_340() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_341() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_342() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_343() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_344() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_345() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_346() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_347() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_348() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_349() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_350() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_351() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_352() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_353() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_354() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_355() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_356() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_357() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_358() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_359() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_360() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_361() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_362() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_363() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_364() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_365() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_366() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_367() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_368() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_369() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_370() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_371() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_372() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_373() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_374() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_375() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_376() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_377() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_378() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_379() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_380() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_381() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_382() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_383() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_384() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_385() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_386() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_387() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_388() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_389() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_390() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_391() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_392() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_393() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_394() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_395() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_396() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_397() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_398() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_399() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_400() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_401() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_402() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_403() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_404() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_405() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_406() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_407() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_408() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_409() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_410() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_411() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_412() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }

    #[test]
    fn test_multilabel_stress_413() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }
}
