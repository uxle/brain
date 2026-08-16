//! Histogram calculations, bincounts, quantiles, and empirical distributions.
//!
//! This module provides statistical histograms, bincount frequencies, percentiles, and median computations.

use crate::tensor::Tensor;

/// Computes the histogram of tensor values over a specified range.
///
/// Returns (bin_counts, bin_edges).
pub fn histogram(input: &Tensor, bins: usize, range: (f64, f64)) -> (Tensor, Tensor) {
    assert!(bins > 0);
    let (min_val, max_val) = range;
    assert!(min_val < max_val);

    let bin_width = (max_val - min_val) / (bins as f64);
    let mut counts = vec![0.0; bins];
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(min_val + (i as f64) * bin_width);
    }

    for &x in input.data() {
        if x >= min_val && x <= max_val {
            let mut b = ((x - min_val) / bin_width) as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1.0;
        }
    }

    (
        Tensor::new(counts, vec![bins]),
        Tensor::new(edges, vec![bins + 1]),
    )
}

/// Counts the number of occurrences of each non-negative integer value in an array.
pub fn bincount(input: &Tensor, minlength: usize) -> Tensor {
    let max_val = input.data().iter().map(|&x| x as usize).fold(0, usize::max);
    let len = (max_val + 1).max(minlength);
    let mut counts = vec![0.0; len];
    for &x in input.data() {
        let idx = x as usize;
        if idx < len {
            counts[idx] += 1.0;
        }
    }
    Tensor::new(counts, vec![len])
}

/// Computes the q-th quantile (0.0 <= q <= 1.0) of tensor elements.
pub fn quantile(input: &Tensor, q: f64) -> f64 {
    assert!((0.0..=1.0).contains(&q));
    assert!(!input.is_empty());
    let mut sorted = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Computes the median of tensor elements.
pub fn median(input: &Tensor) -> f64 {
    quantile(input, 0.5)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let t = Tensor::from_slice(&[0.5, 1.5, 2.5, 3.5], vec![4]);
        let (counts, edges) = histogram(&t, 4, (0.0, 4.0));
        assert_eq!(counts.data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(edges.numel(), 5);
    }

    #[test]
    fn test_bincount_and_median() {
        let t = Tensor::from_slice(&[0.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![6]);
        let bc = bincount(&t, 0);
        assert_eq!(bc.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(median(&t), 2.0);
    }

    #[test]
    fn test_hist_stress_case_001() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let m = median(&t);
        assert!(m >= 1.0 && m <= 2.0);
    }

    #[test]
    fn test_hist_stress_case_002() {
        let t = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let m = median(&t);
        assert!(m >= 2.0 && m <= 3.0);
    }

    #[test]
    fn test_hist_stress_case_003() {
        let t = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let m = median(&t);
        assert!(m >= 3.0 && m <= 4.0);
    }

    #[test]
    fn test_hist_stress_case_004() {
        let t = Tensor::from_slice(&[4.0, 5.0], vec![2]);
        let m = median(&t);
        assert!(m >= 4.0 && m <= 5.0);
    }

    #[test]
    fn test_hist_stress_case_005() {
        let t = Tensor::from_slice(&[5.0, 6.0], vec![2]);
        let m = median(&t);
        assert!(m >= 5.0 && m <= 6.0);
    }

    #[test]
    fn test_hist_stress_case_006() {
        let t = Tensor::from_slice(&[6.0, 7.0], vec![2]);
        let m = median(&t);
        assert!(m >= 6.0 && m <= 7.0);
    }

    #[test]
    fn test_hist_stress_case_007() {
        let t = Tensor::from_slice(&[7.0, 8.0], vec![2]);
        let m = median(&t);
        assert!(m >= 7.0 && m <= 8.0);
    }

    #[test]
    fn test_hist_stress_case_008() {
        let t = Tensor::from_slice(&[8.0, 9.0], vec![2]);
        let m = median(&t);
        assert!(m >= 8.0 && m <= 9.0);
    }

    #[test]
    fn test_hist_stress_case_009() {
        let t = Tensor::from_slice(&[9.0, 10.0], vec![2]);
        let m = median(&t);
        assert!(m >= 9.0 && m <= 10.0);
    }

    #[test]
    fn test_hist_stress_case_010() {
        let t = Tensor::from_slice(&[10.0, 11.0], vec![2]);
        let m = median(&t);
        assert!(m >= 10.0 && m <= 11.0);
    }

    #[test]
    fn test_hist_stress_case_011() {
        let t = Tensor::from_slice(&[11.0, 12.0], vec![2]);
        let m = median(&t);
        assert!(m >= 11.0 && m <= 12.0);
    }

    #[test]
    fn test_hist_stress_case_012() {
        let t = Tensor::from_slice(&[12.0, 13.0], vec![2]);
        let m = median(&t);
        assert!(m >= 12.0 && m <= 13.0);
    }

    #[test]
    fn test_hist_stress_case_013() {
        let t = Tensor::from_slice(&[13.0, 14.0], vec![2]);
        let m = median(&t);
        assert!(m >= 13.0 && m <= 14.0);
    }

    #[test]
    fn test_hist_stress_case_014() {
        let t = Tensor::from_slice(&[14.0, 15.0], vec![2]);
        let m = median(&t);
        assert!(m >= 14.0 && m <= 15.0);
    }

    #[test]
    fn test_hist_stress_case_015() {
        let t = Tensor::from_slice(&[15.0, 16.0], vec![2]);
        let m = median(&t);
        assert!(m >= 15.0 && m <= 16.0);
    }

    #[test]
    fn test_hist_stress_case_016() {
        let t = Tensor::from_slice(&[16.0, 17.0], vec![2]);
        let m = median(&t);
        assert!(m >= 16.0 && m <= 17.0);
    }

    #[test]
    fn test_hist_stress_case_017() {
        let t = Tensor::from_slice(&[17.0, 18.0], vec![2]);
        let m = median(&t);
        assert!(m >= 17.0 && m <= 18.0);
    }

    #[test]
    fn test_hist_stress_case_018() {
        let t = Tensor::from_slice(&[18.0, 19.0], vec![2]);
        let m = median(&t);
        assert!(m >= 18.0 && m <= 19.0);
    }

    #[test]
    fn test_hist_stress_case_019() {
        let t = Tensor::from_slice(&[19.0, 20.0], vec![2]);
        let m = median(&t);
        assert!(m >= 19.0 && m <= 20.0);
    }

    #[test]
    fn test_hist_stress_case_020() {
        let t = Tensor::from_slice(&[20.0, 21.0], vec![2]);
        let m = median(&t);
        assert!(m >= 20.0 && m <= 21.0);
    }

    #[test]
    fn test_hist_stress_case_021() {
        let t = Tensor::from_slice(&[21.0, 22.0], vec![2]);
        let m = median(&t);
        assert!(m >= 21.0 && m <= 22.0);
    }

    #[test]
    fn test_hist_stress_case_022() {
        let t = Tensor::from_slice(&[22.0, 23.0], vec![2]);
        let m = median(&t);
        assert!(m >= 22.0 && m <= 23.0);
    }

    #[test]
    fn test_hist_stress_case_023() {
        let t = Tensor::from_slice(&[23.0, 24.0], vec![2]);
        let m = median(&t);
        assert!(m >= 23.0 && m <= 24.0);
    }

    #[test]
    fn test_hist_stress_case_024() {
        let t = Tensor::from_slice(&[24.0, 25.0], vec![2]);
        let m = median(&t);
        assert!(m >= 24.0 && m <= 25.0);
    }

    #[test]
    fn test_hist_stress_case_025() {
        let t = Tensor::from_slice(&[25.0, 26.0], vec![2]);
        let m = median(&t);
        assert!(m >= 25.0 && m <= 26.0);
    }

    #[test]
    fn test_hist_stress_case_026() {
        let t = Tensor::from_slice(&[26.0, 27.0], vec![2]);
        let m = median(&t);
        assert!(m >= 26.0 && m <= 27.0);
    }

    #[test]
    fn test_hist_stress_case_027() {
        let t = Tensor::from_slice(&[27.0, 28.0], vec![2]);
        let m = median(&t);
        assert!(m >= 27.0 && m <= 28.0);
    }

    #[test]
    fn test_hist_stress_case_028() {
        let t = Tensor::from_slice(&[28.0, 29.0], vec![2]);
        let m = median(&t);
        assert!(m >= 28.0 && m <= 29.0);
    }

    #[test]
    fn test_hist_stress_case_029() {
        let t = Tensor::from_slice(&[29.0, 30.0], vec![2]);
        let m = median(&t);
        assert!(m >= 29.0 && m <= 30.0);
    }

    #[test]
    fn test_hist_stress_case_030() {
        let t = Tensor::from_slice(&[30.0, 31.0], vec![2]);
        let m = median(&t);
        assert!(m >= 30.0 && m <= 31.0);
    }

    #[test]
    fn test_hist_stress_case_031() {
        let t = Tensor::from_slice(&[31.0, 32.0], vec![2]);
        let m = median(&t);
        assert!(m >= 31.0 && m <= 32.0);
    }

    #[test]
    fn test_hist_stress_case_032() {
        let t = Tensor::from_slice(&[32.0, 33.0], vec![2]);
        let m = median(&t);
        assert!(m >= 32.0 && m <= 33.0);
    }

    #[test]
    fn test_hist_stress_case_033() {
        let t = Tensor::from_slice(&[33.0, 34.0], vec![2]);
        let m = median(&t);
        assert!(m >= 33.0 && m <= 34.0);
    }

    #[test]
    fn test_hist_stress_case_034() {
        let t = Tensor::from_slice(&[34.0, 35.0], vec![2]);
        let m = median(&t);
        assert!(m >= 34.0 && m <= 35.0);
    }

    #[test]
    fn test_hist_stress_case_035() {
        let t = Tensor::from_slice(&[35.0, 36.0], vec![2]);
        let m = median(&t);
        assert!(m >= 35.0 && m <= 36.0);
    }

    #[test]
    fn test_hist_stress_case_036() {
        let t = Tensor::from_slice(&[36.0, 37.0], vec![2]);
        let m = median(&t);
        assert!(m >= 36.0 && m <= 37.0);
    }

    #[test]
    fn test_hist_stress_case_037() {
        let t = Tensor::from_slice(&[37.0, 38.0], vec![2]);
        let m = median(&t);
        assert!(m >= 37.0 && m <= 38.0);
    }

    #[test]
    fn test_hist_stress_case_038() {
        let t = Tensor::from_slice(&[38.0, 39.0], vec![2]);
        let m = median(&t);
        assert!(m >= 38.0 && m <= 39.0);
    }

    #[test]
    fn test_hist_stress_case_039() {
        let t = Tensor::from_slice(&[39.0, 40.0], vec![2]);
        let m = median(&t);
        assert!(m >= 39.0 && m <= 40.0);
    }

    #[test]
    fn test_hist_stress_case_040() {
        let t = Tensor::from_slice(&[40.0, 41.0], vec![2]);
        let m = median(&t);
        assert!(m >= 40.0 && m <= 41.0);
    }

    #[test]
    fn test_hist_stress_case_041() {
        let t = Tensor::from_slice(&[41.0, 42.0], vec![2]);
        let m = median(&t);
        assert!(m >= 41.0 && m <= 42.0);
    }

    #[test]
    fn test_hist_stress_case_042() {
        let t = Tensor::from_slice(&[42.0, 43.0], vec![2]);
        let m = median(&t);
        assert!(m >= 42.0 && m <= 43.0);
    }

    #[test]
    fn test_hist_stress_case_043() {
        let t = Tensor::from_slice(&[43.0, 44.0], vec![2]);
        let m = median(&t);
        assert!(m >= 43.0 && m <= 44.0);
    }

    #[test]
    fn test_hist_stress_case_044() {
        let t = Tensor::from_slice(&[44.0, 45.0], vec![2]);
        let m = median(&t);
        assert!(m >= 44.0 && m <= 45.0);
    }

    #[test]
    fn test_hist_stress_case_045() {
        let t = Tensor::from_slice(&[45.0, 46.0], vec![2]);
        let m = median(&t);
        assert!(m >= 45.0 && m <= 46.0);
    }

    #[test]
    fn test_hist_stress_case_046() {
        let t = Tensor::from_slice(&[46.0, 47.0], vec![2]);
        let m = median(&t);
        assert!(m >= 46.0 && m <= 47.0);
    }

    #[test]
    fn test_hist_stress_case_047() {
        let t = Tensor::from_slice(&[47.0, 48.0], vec![2]);
        let m = median(&t);
        assert!(m >= 47.0 && m <= 48.0);
    }

    #[test]
    fn test_hist_stress_case_048() {
        let t = Tensor::from_slice(&[48.0, 49.0], vec![2]);
        let m = median(&t);
        assert!(m >= 48.0 && m <= 49.0);
    }

    #[test]
    fn test_hist_stress_case_049() {
        let t = Tensor::from_slice(&[49.0, 50.0], vec![2]);
        let m = median(&t);
        assert!(m >= 49.0 && m <= 50.0);
    }

    #[test]
    fn test_hist_stress_case_050() {
        let t = Tensor::from_slice(&[50.0, 51.0], vec![2]);
        let m = median(&t);
        assert!(m >= 50.0 && m <= 51.0);
    }

    #[test]
    fn test_hist_stress_case_051() {
        let t = Tensor::from_slice(&[51.0, 52.0], vec![2]);
        let m = median(&t);
        assert!(m >= 51.0 && m <= 52.0);
    }

    #[test]
    fn test_hist_stress_case_052() {
        let t = Tensor::from_slice(&[52.0, 53.0], vec![2]);
        let m = median(&t);
        assert!(m >= 52.0 && m <= 53.0);
    }

    #[test]
    fn test_hist_stress_case_053() {
        let t = Tensor::from_slice(&[53.0, 54.0], vec![2]);
        let m = median(&t);
        assert!(m >= 53.0 && m <= 54.0);
    }

    #[test]
    fn test_hist_stress_case_054() {
        let t = Tensor::from_slice(&[54.0, 55.0], vec![2]);
        let m = median(&t);
        assert!(m >= 54.0 && m <= 55.0);
    }

    #[test]
    fn test_hist_stress_case_055() {
        let t = Tensor::from_slice(&[55.0, 56.0], vec![2]);
        let m = median(&t);
        assert!(m >= 55.0 && m <= 56.0);
    }

    #[test]
    fn test_hist_stress_case_056() {
        let t = Tensor::from_slice(&[56.0, 57.0], vec![2]);
        let m = median(&t);
        assert!(m >= 56.0 && m <= 57.0);
    }

    #[test]
    fn test_hist_stress_case_057() {
        let t = Tensor::from_slice(&[57.0, 58.0], vec![2]);
        let m = median(&t);
        assert!(m >= 57.0 && m <= 58.0);
    }

    #[test]
    fn test_hist_stress_case_058() {
        let t = Tensor::from_slice(&[58.0, 59.0], vec![2]);
        let m = median(&t);
        assert!(m >= 58.0 && m <= 59.0);
    }

    #[test]
    fn test_hist_stress_case_059() {
        let t = Tensor::from_slice(&[59.0, 60.0], vec![2]);
        let m = median(&t);
        assert!(m >= 59.0 && m <= 60.0);
    }

    #[test]
    fn test_hist_stress_case_060() {
        let t = Tensor::from_slice(&[60.0, 61.0], vec![2]);
        let m = median(&t);
        assert!(m >= 60.0 && m <= 61.0);
    }

    #[test]
    fn test_hist_stress_case_061() {
        let t = Tensor::from_slice(&[61.0, 62.0], vec![2]);
        let m = median(&t);
        assert!(m >= 61.0 && m <= 62.0);
    }

    #[test]
    fn test_hist_stress_case_062() {
        let t = Tensor::from_slice(&[62.0, 63.0], vec![2]);
        let m = median(&t);
        assert!(m >= 62.0 && m <= 63.0);
    }

    #[test]
    fn test_hist_stress_case_063() {
        let t = Tensor::from_slice(&[63.0, 64.0], vec![2]);
        let m = median(&t);
        assert!(m >= 63.0 && m <= 64.0);
    }

    #[test]
    fn test_hist_stress_case_064() {
        let t = Tensor::from_slice(&[64.0, 65.0], vec![2]);
        let m = median(&t);
        assert!(m >= 64.0 && m <= 65.0);
    }

    #[test]
    fn test_hist_stress_case_065() {
        let t = Tensor::from_slice(&[65.0, 66.0], vec![2]);
        let m = median(&t);
        assert!(m >= 65.0 && m <= 66.0);
    }

    #[test]
    fn test_hist_stress_case_066() {
        let t = Tensor::from_slice(&[66.0, 67.0], vec![2]);
        let m = median(&t);
        assert!(m >= 66.0 && m <= 67.0);
    }

    #[test]
    fn test_hist_stress_case_067() {
        let t = Tensor::from_slice(&[67.0, 68.0], vec![2]);
        let m = median(&t);
        assert!(m >= 67.0 && m <= 68.0);
    }

    #[test]
    fn test_hist_stress_case_068() {
        let t = Tensor::from_slice(&[68.0, 69.0], vec![2]);
        let m = median(&t);
        assert!(m >= 68.0 && m <= 69.0);
    }

    #[test]
    fn test_hist_stress_case_069() {
        let t = Tensor::from_slice(&[69.0, 70.0], vec![2]);
        let m = median(&t);
        assert!(m >= 69.0 && m <= 70.0);
    }

    #[test]
    fn test_hist_stress_case_070() {
        let t = Tensor::from_slice(&[70.0, 71.0], vec![2]);
        let m = median(&t);
        assert!(m >= 70.0 && m <= 71.0);
    }

    #[test]
    fn test_hist_stress_case_071() {
        let t = Tensor::from_slice(&[71.0, 72.0], vec![2]);
        let m = median(&t);
        assert!(m >= 71.0 && m <= 72.0);
    }

    #[test]
    fn test_hist_stress_case_072() {
        let t = Tensor::from_slice(&[72.0, 73.0], vec![2]);
        let m = median(&t);
        assert!(m >= 72.0 && m <= 73.0);
    }

    #[test]
    fn test_hist_stress_case_073() {
        let t = Tensor::from_slice(&[73.0, 74.0], vec![2]);
        let m = median(&t);
        assert!(m >= 73.0 && m <= 74.0);
    }

    #[test]
    fn test_hist_stress_case_074() {
        let t = Tensor::from_slice(&[74.0, 75.0], vec![2]);
        let m = median(&t);
        assert!(m >= 74.0 && m <= 75.0);
    }

    #[test]
    fn test_hist_stress_case_075() {
        let t = Tensor::from_slice(&[75.0, 76.0], vec![2]);
        let m = median(&t);
        assert!(m >= 75.0 && m <= 76.0);
    }

    #[test]
    fn test_hist_stress_case_076() {
        let t = Tensor::from_slice(&[76.0, 77.0], vec![2]);
        let m = median(&t);
        assert!(m >= 76.0 && m <= 77.0);
    }

    #[test]
    fn test_hist_stress_case_077() {
        let t = Tensor::from_slice(&[77.0, 78.0], vec![2]);
        let m = median(&t);
        assert!(m >= 77.0 && m <= 78.0);
    }

    #[test]
    fn test_hist_stress_case_078() {
        let t = Tensor::from_slice(&[78.0, 79.0], vec![2]);
        let m = median(&t);
        assert!(m >= 78.0 && m <= 79.0);
    }

    #[test]
    fn test_hist_stress_case_079() {
        let t = Tensor::from_slice(&[79.0, 80.0], vec![2]);
        let m = median(&t);
        assert!(m >= 79.0 && m <= 80.0);
    }

    #[test]
    fn test_hist_stress_case_080() {
        let t = Tensor::from_slice(&[80.0, 81.0], vec![2]);
        let m = median(&t);
        assert!(m >= 80.0 && m <= 81.0);
    }

    #[test]
    fn test_hist_stress_case_081() {
        let t = Tensor::from_slice(&[81.0, 82.0], vec![2]);
        let m = median(&t);
        assert!(m >= 81.0 && m <= 82.0);
    }

    #[test]
    fn test_hist_stress_case_082() {
        let t = Tensor::from_slice(&[82.0, 83.0], vec![2]);
        let m = median(&t);
        assert!(m >= 82.0 && m <= 83.0);
    }

    #[test]
    fn test_hist_stress_case_083() {
        let t = Tensor::from_slice(&[83.0, 84.0], vec![2]);
        let m = median(&t);
        assert!(m >= 83.0 && m <= 84.0);
    }

    #[test]
    fn test_hist_stress_case_084() {
        let t = Tensor::from_slice(&[84.0, 85.0], vec![2]);
        let m = median(&t);
        assert!(m >= 84.0 && m <= 85.0);
    }

    #[test]
    fn test_hist_stress_case_085() {
        let t = Tensor::from_slice(&[85.0, 86.0], vec![2]);
        let m = median(&t);
        assert!(m >= 85.0 && m <= 86.0);
    }

    #[test]
    fn test_hist_stress_case_086() {
        let t = Tensor::from_slice(&[86.0, 87.0], vec![2]);
        let m = median(&t);
        assert!(m >= 86.0 && m <= 87.0);
    }

    #[test]
    fn test_hist_stress_case_087() {
        let t = Tensor::from_slice(&[87.0, 88.0], vec![2]);
        let m = median(&t);
        assert!(m >= 87.0 && m <= 88.0);
    }

    #[test]
    fn test_hist_stress_case_088() {
        let t = Tensor::from_slice(&[88.0, 89.0], vec![2]);
        let m = median(&t);
        assert!(m >= 88.0 && m <= 89.0);
    }

    #[test]
    fn test_hist_stress_case_089() {
        let t = Tensor::from_slice(&[89.0, 90.0], vec![2]);
        let m = median(&t);
        assert!(m >= 89.0 && m <= 90.0);
    }

    #[test]
    fn test_hist_stress_case_090() {
        let t = Tensor::from_slice(&[90.0, 91.0], vec![2]);
        let m = median(&t);
        assert!(m >= 90.0 && m <= 91.0);
    }

    #[test]
    fn test_hist_stress_case_091() {
        let t = Tensor::from_slice(&[91.0, 92.0], vec![2]);
        let m = median(&t);
        assert!(m >= 91.0 && m <= 92.0);
    }

    #[test]
    fn test_hist_stress_case_092() {
        let t = Tensor::from_slice(&[92.0, 93.0], vec![2]);
        let m = median(&t);
        assert!(m >= 92.0 && m <= 93.0);
    }

    #[test]
    fn test_hist_stress_case_093() {
        let t = Tensor::from_slice(&[93.0, 94.0], vec![2]);
        let m = median(&t);
        assert!(m >= 93.0 && m <= 94.0);
    }

    #[test]
    fn test_hist_stress_case_094() {
        let t = Tensor::from_slice(&[94.0, 95.0], vec![2]);
        let m = median(&t);
        assert!(m >= 94.0 && m <= 95.0);
    }

    #[test]
    fn test_hist_stress_case_095() {
        let t = Tensor::from_slice(&[95.0, 96.0], vec![2]);
        let m = median(&t);
        assert!(m >= 95.0 && m <= 96.0);
    }

    #[test]
    fn test_hist_stress_case_096() {
        let t = Tensor::from_slice(&[96.0, 97.0], vec![2]);
        let m = median(&t);
        assert!(m >= 96.0 && m <= 97.0);
    }

    #[test]
    fn test_hist_stress_case_097() {
        let t = Tensor::from_slice(&[97.0, 98.0], vec![2]);
        let m = median(&t);
        assert!(m >= 97.0 && m <= 98.0);
    }

    #[test]
    fn test_hist_stress_case_098() {
        let t = Tensor::from_slice(&[98.0, 99.0], vec![2]);
        let m = median(&t);
        assert!(m >= 98.0 && m <= 99.0);
    }

    #[test]
    fn test_hist_stress_case_099() {
        let t = Tensor::from_slice(&[99.0, 100.0], vec![2]);
        let m = median(&t);
        assert!(m >= 99.0 && m <= 100.0);
    }

    #[test]
    fn test_hist_stress_case_100() {
        let t = Tensor::from_slice(&[100.0, 101.0], vec![2]);
        let m = median(&t);
        assert!(m >= 100.0 && m <= 101.0);
    }

    #[test]
    fn test_hist_stress_case_101() {
        let t = Tensor::from_slice(&[101.0, 102.0], vec![2]);
        let m = median(&t);
        assert!(m >= 101.0 && m <= 102.0);
    }

    #[test]
    fn test_hist_stress_case_102() {
        let t = Tensor::from_slice(&[102.0, 103.0], vec![2]);
        let m = median(&t);
        assert!(m >= 102.0 && m <= 103.0);
    }

    #[test]
    fn test_hist_stress_case_103() {
        let t = Tensor::from_slice(&[103.0, 104.0], vec![2]);
        let m = median(&t);
        assert!(m >= 103.0 && m <= 104.0);
    }

    #[test]
    fn test_hist_stress_case_104() {
        let t = Tensor::from_slice(&[104.0, 105.0], vec![2]);
        let m = median(&t);
        assert!(m >= 104.0 && m <= 105.0);
    }

    #[test]
    fn test_hist_stress_case_105() {
        let t = Tensor::from_slice(&[105.0, 106.0], vec![2]);
        let m = median(&t);
        assert!(m >= 105.0 && m <= 106.0);
    }

    #[test]
    fn test_hist_stress_case_106() {
        let t = Tensor::from_slice(&[106.0, 107.0], vec![2]);
        let m = median(&t);
        assert!(m >= 106.0 && m <= 107.0);
    }

    #[test]
    fn test_hist_stress_case_107() {
        let t = Tensor::from_slice(&[107.0, 108.0], vec![2]);
        let m = median(&t);
        assert!(m >= 107.0 && m <= 108.0);
    }

    #[test]
    fn test_hist_stress_case_108() {
        let t = Tensor::from_slice(&[108.0, 109.0], vec![2]);
        let m = median(&t);
        assert!(m >= 108.0 && m <= 109.0);
    }

    #[test]
    fn test_hist_stress_case_109() {
        let t = Tensor::from_slice(&[109.0, 110.0], vec![2]);
        let m = median(&t);
        assert!(m >= 109.0 && m <= 110.0);
    }

    #[test]
    fn test_hist_stress_case_110() {
        let t = Tensor::from_slice(&[110.0, 111.0], vec![2]);
        let m = median(&t);
        assert!(m >= 110.0 && m <= 111.0);
    }

    #[test]
    fn test_hist_stress_case_111() {
        let t = Tensor::from_slice(&[111.0, 112.0], vec![2]);
        let m = median(&t);
        assert!(m >= 111.0 && m <= 112.0);
    }

    #[test]
    fn test_hist_stress_case_112() {
        let t = Tensor::from_slice(&[112.0, 113.0], vec![2]);
        let m = median(&t);
        assert!(m >= 112.0 && m <= 113.0);
    }

    #[test]
    fn test_hist_stress_case_113() {
        let t = Tensor::from_slice(&[113.0, 114.0], vec![2]);
        let m = median(&t);
        assert!(m >= 113.0 && m <= 114.0);
    }

    #[test]
    fn test_hist_stress_case_114() {
        let t = Tensor::from_slice(&[114.0, 115.0], vec![2]);
        let m = median(&t);
        assert!(m >= 114.0 && m <= 115.0);
    }

    #[test]
    fn test_hist_stress_case_115() {
        let t = Tensor::from_slice(&[115.0, 116.0], vec![2]);
        let m = median(&t);
        assert!(m >= 115.0 && m <= 116.0);
    }

    #[test]
    fn test_hist_stress_case_116() {
        let t = Tensor::from_slice(&[116.0, 117.0], vec![2]);
        let m = median(&t);
        assert!(m >= 116.0 && m <= 117.0);
    }

    #[test]
    fn test_hist_stress_case_117() {
        let t = Tensor::from_slice(&[117.0, 118.0], vec![2]);
        let m = median(&t);
        assert!(m >= 117.0 && m <= 118.0);
    }

    #[test]
    fn test_hist_stress_case_118() {
        let t = Tensor::from_slice(&[118.0, 119.0], vec![2]);
        let m = median(&t);
        assert!(m >= 118.0 && m <= 119.0);
    }

    #[test]
    fn test_hist_stress_case_119() {
        let t = Tensor::from_slice(&[119.0, 120.0], vec![2]);
        let m = median(&t);
        assert!(m >= 119.0 && m <= 120.0);
    }

    #[test]
    fn test_hist_stress_case_120() {
        let t = Tensor::from_slice(&[120.0, 121.0], vec![2]);
        let m = median(&t);
        assert!(m >= 120.0 && m <= 121.0);
    }

    #[test]
    fn test_hist_stress_case_121() {
        let t = Tensor::from_slice(&[121.0, 122.0], vec![2]);
        let m = median(&t);
        assert!(m >= 121.0 && m <= 122.0);
    }

    #[test]
    fn test_hist_stress_case_122() {
        let t = Tensor::from_slice(&[122.0, 123.0], vec![2]);
        let m = median(&t);
        assert!(m >= 122.0 && m <= 123.0);
    }

    #[test]
    fn test_hist_stress_case_123() {
        let t = Tensor::from_slice(&[123.0, 124.0], vec![2]);
        let m = median(&t);
        assert!(m >= 123.0 && m <= 124.0);
    }

    #[test]
    fn test_hist_stress_case_124() {
        let t = Tensor::from_slice(&[124.0, 125.0], vec![2]);
        let m = median(&t);
        assert!(m >= 124.0 && m <= 125.0);
    }

    #[test]
    fn test_hist_stress_case_125() {
        let t = Tensor::from_slice(&[125.0, 126.0], vec![2]);
        let m = median(&t);
        assert!(m >= 125.0 && m <= 126.0);
    }

    #[test]
    fn test_hist_stress_case_126() {
        let t = Tensor::from_slice(&[126.0, 127.0], vec![2]);
        let m = median(&t);
        assert!(m >= 126.0 && m <= 127.0);
    }

    #[test]
    fn test_hist_stress_case_127() {
        let t = Tensor::from_slice(&[127.0, 128.0], vec![2]);
        let m = median(&t);
        assert!(m >= 127.0 && m <= 128.0);
    }

    #[test]
    fn test_hist_stress_case_128() {
        let t = Tensor::from_slice(&[128.0, 129.0], vec![2]);
        let m = median(&t);
        assert!(m >= 128.0 && m <= 129.0);
    }

    #[test]
    fn test_hist_stress_case_129() {
        let t = Tensor::from_slice(&[129.0, 130.0], vec![2]);
        let m = median(&t);
        assert!(m >= 129.0 && m <= 130.0);
    }

    #[test]
    fn test_hist_stress_case_130() {
        let t = Tensor::from_slice(&[130.0, 131.0], vec![2]);
        let m = median(&t);
        assert!(m >= 130.0 && m <= 131.0);
    }

    #[test]
    fn test_hist_stress_case_131() {
        let t = Tensor::from_slice(&[131.0, 132.0], vec![2]);
        let m = median(&t);
        assert!(m >= 131.0 && m <= 132.0);
    }

    #[test]
    fn test_hist_stress_case_132() {
        let t = Tensor::from_slice(&[132.0, 133.0], vec![2]);
        let m = median(&t);
        assert!(m >= 132.0 && m <= 133.0);
    }

    #[test]
    fn test_hist_stress_case_133() {
        let t = Tensor::from_slice(&[133.0, 134.0], vec![2]);
        let m = median(&t);
        assert!(m >= 133.0 && m <= 134.0);
    }

    #[test]
    fn test_hist_stress_case_134() {
        let t = Tensor::from_slice(&[134.0, 135.0], vec![2]);
        let m = median(&t);
        assert!(m >= 134.0 && m <= 135.0);
    }

    #[test]
    fn test_hist_stress_case_135() {
        let t = Tensor::from_slice(&[135.0, 136.0], vec![2]);
        let m = median(&t);
        assert!(m >= 135.0 && m <= 136.0);
    }

    #[test]
    fn test_hist_stress_case_136() {
        let t = Tensor::from_slice(&[136.0, 137.0], vec![2]);
        let m = median(&t);
        assert!(m >= 136.0 && m <= 137.0);
    }

    #[test]
    fn test_hist_stress_case_137() {
        let t = Tensor::from_slice(&[137.0, 138.0], vec![2]);
        let m = median(&t);
        assert!(m >= 137.0 && m <= 138.0);
    }

    #[test]
    fn test_hist_stress_case_138() {
        let t = Tensor::from_slice(&[138.0, 139.0], vec![2]);
        let m = median(&t);
        assert!(m >= 138.0 && m <= 139.0);
    }

    #[test]
    fn test_hist_stress_case_139() {
        let t = Tensor::from_slice(&[139.0, 140.0], vec![2]);
        let m = median(&t);
        assert!(m >= 139.0 && m <= 140.0);
    }

    #[test]
    fn test_hist_stress_case_140() {
        let t = Tensor::from_slice(&[140.0, 141.0], vec![2]);
        let m = median(&t);
        assert!(m >= 140.0 && m <= 141.0);
    }

    #[test]
    fn test_hist_stress_case_141() {
        let t = Tensor::from_slice(&[141.0, 142.0], vec![2]);
        let m = median(&t);
        assert!(m >= 141.0 && m <= 142.0);
    }

    #[test]
    fn test_hist_stress_case_142() {
        let t = Tensor::from_slice(&[142.0, 143.0], vec![2]);
        let m = median(&t);
        assert!(m >= 142.0 && m <= 143.0);
    }

    #[test]
    fn test_hist_stress_case_143() {
        let t = Tensor::from_slice(&[143.0, 144.0], vec![2]);
        let m = median(&t);
        assert!(m >= 143.0 && m <= 144.0);
    }

    #[test]
    fn test_hist_stress_case_144() {
        let t = Tensor::from_slice(&[144.0, 145.0], vec![2]);
        let m = median(&t);
        assert!(m >= 144.0 && m <= 145.0);
    }

    #[test]
    fn test_hist_stress_case_145() {
        let t = Tensor::from_slice(&[145.0, 146.0], vec![2]);
        let m = median(&t);
        assert!(m >= 145.0 && m <= 146.0);
    }

    #[test]
    fn test_hist_stress_case_146() {
        let t = Tensor::from_slice(&[146.0, 147.0], vec![2]);
        let m = median(&t);
        assert!(m >= 146.0 && m <= 147.0);
    }

    #[test]
    fn test_hist_stress_case_147() {
        let t = Tensor::from_slice(&[147.0, 148.0], vec![2]);
        let m = median(&t);
        assert!(m >= 147.0 && m <= 148.0);
    }

    #[test]
    fn test_hist_stress_case_148() {
        let t = Tensor::from_slice(&[148.0, 149.0], vec![2]);
        let m = median(&t);
        assert!(m >= 148.0 && m <= 149.0);
    }

    #[test]
    fn test_hist_stress_case_149() {
        let t = Tensor::from_slice(&[149.0, 150.0], vec![2]);
        let m = median(&t);
        assert!(m >= 149.0 && m <= 150.0);
    }

    #[test]
    fn test_hist_stress_case_150() {
        let t = Tensor::from_slice(&[150.0, 151.0], vec![2]);
        let m = median(&t);
        assert!(m >= 150.0 && m <= 151.0);
    }

    #[test]
    fn test_hist_stress_case_151() {
        let t = Tensor::from_slice(&[151.0, 152.0], vec![2]);
        let m = median(&t);
        assert!(m >= 151.0 && m <= 152.0);
    }

    #[test]
    fn test_hist_stress_case_152() {
        let t = Tensor::from_slice(&[152.0, 153.0], vec![2]);
        let m = median(&t);
        assert!(m >= 152.0 && m <= 153.0);
    }

    #[test]
    fn test_hist_stress_case_153() {
        let t = Tensor::from_slice(&[153.0, 154.0], vec![2]);
        let m = median(&t);
        assert!(m >= 153.0 && m <= 154.0);
    }

    #[test]
    fn test_hist_stress_case_154() {
        let t = Tensor::from_slice(&[154.0, 155.0], vec![2]);
        let m = median(&t);
        assert!(m >= 154.0 && m <= 155.0);
    }

    #[test]
    fn test_hist_stress_case_155() {
        let t = Tensor::from_slice(&[155.0, 156.0], vec![2]);
        let m = median(&t);
        assert!(m >= 155.0 && m <= 156.0);
    }

    #[test]
    fn test_hist_stress_case_156() {
        let t = Tensor::from_slice(&[156.0, 157.0], vec![2]);
        let m = median(&t);
        assert!(m >= 156.0 && m <= 157.0);
    }

    #[test]
    fn test_hist_stress_case_157() {
        let t = Tensor::from_slice(&[157.0, 158.0], vec![2]);
        let m = median(&t);
        assert!(m >= 157.0 && m <= 158.0);
    }

    #[test]
    fn test_hist_stress_case_158() {
        let t = Tensor::from_slice(&[158.0, 159.0], vec![2]);
        let m = median(&t);
        assert!(m >= 158.0 && m <= 159.0);
    }

    #[test]
    fn test_hist_stress_case_159() {
        let t = Tensor::from_slice(&[159.0, 160.0], vec![2]);
        let m = median(&t);
        assert!(m >= 159.0 && m <= 160.0);
    }

    #[test]
    fn test_hist_stress_case_160() {
        let t = Tensor::from_slice(&[160.0, 161.0], vec![2]);
        let m = median(&t);
        assert!(m >= 160.0 && m <= 161.0);
    }

    #[test]
    fn test_hist_stress_case_161() {
        let t = Tensor::from_slice(&[161.0, 162.0], vec![2]);
        let m = median(&t);
        assert!(m >= 161.0 && m <= 162.0);
    }

    #[test]
    fn test_hist_stress_case_162() {
        let t = Tensor::from_slice(&[162.0, 163.0], vec![2]);
        let m = median(&t);
        assert!(m >= 162.0 && m <= 163.0);
    }

    #[test]
    fn test_hist_stress_case_163() {
        let t = Tensor::from_slice(&[163.0, 164.0], vec![2]);
        let m = median(&t);
        assert!(m >= 163.0 && m <= 164.0);
    }

    #[test]
    fn test_hist_stress_case_164() {
        let t = Tensor::from_slice(&[164.0, 165.0], vec![2]);
        let m = median(&t);
        assert!(m >= 164.0 && m <= 165.0);
    }

    #[test]
    fn test_hist_stress_case_165() {
        let t = Tensor::from_slice(&[165.0, 166.0], vec![2]);
        let m = median(&t);
        assert!(m >= 165.0 && m <= 166.0);
    }

    #[test]
    fn test_hist_stress_case_166() {
        let t = Tensor::from_slice(&[166.0, 167.0], vec![2]);
        let m = median(&t);
        assert!(m >= 166.0 && m <= 167.0);
    }

    #[test]
    fn test_hist_stress_case_167() {
        let t = Tensor::from_slice(&[167.0, 168.0], vec![2]);
        let m = median(&t);
        assert!(m >= 167.0 && m <= 168.0);
    }

    #[test]
    fn test_hist_stress_case_168() {
        let t = Tensor::from_slice(&[168.0, 169.0], vec![2]);
        let m = median(&t);
        assert!(m >= 168.0 && m <= 169.0);
    }

    #[test]
    fn test_hist_stress_case_169() {
        let t = Tensor::from_slice(&[169.0, 170.0], vec![2]);
        let m = median(&t);
        assert!(m >= 169.0 && m <= 170.0);
    }

    #[test]
    fn test_hist_stress_case_170() {
        let t = Tensor::from_slice(&[170.0, 171.0], vec![2]);
        let m = median(&t);
        assert!(m >= 170.0 && m <= 171.0);
    }

    #[test]
    fn test_hist_stress_case_171() {
        let t = Tensor::from_slice(&[171.0, 172.0], vec![2]);
        let m = median(&t);
        assert!(m >= 171.0 && m <= 172.0);
    }

    #[test]
    fn test_hist_stress_case_172() {
        let t = Tensor::from_slice(&[172.0, 173.0], vec![2]);
        let m = median(&t);
        assert!(m >= 172.0 && m <= 173.0);
    }

    #[test]
    fn test_hist_stress_case_173() {
        let t = Tensor::from_slice(&[173.0, 174.0], vec![2]);
        let m = median(&t);
        assert!(m >= 173.0 && m <= 174.0);
    }

    #[test]
    fn test_hist_stress_case_174() {
        let t = Tensor::from_slice(&[174.0, 175.0], vec![2]);
        let m = median(&t);
        assert!(m >= 174.0 && m <= 175.0);
    }

    #[test]
    fn test_hist_stress_case_175() {
        let t = Tensor::from_slice(&[175.0, 176.0], vec![2]);
        let m = median(&t);
        assert!(m >= 175.0 && m <= 176.0);
    }

    #[test]
    fn test_hist_stress_case_176() {
        let t = Tensor::from_slice(&[176.0, 177.0], vec![2]);
        let m = median(&t);
        assert!(m >= 176.0 && m <= 177.0);
    }

    #[test]
    fn test_hist_stress_case_177() {
        let t = Tensor::from_slice(&[177.0, 178.0], vec![2]);
        let m = median(&t);
        assert!(m >= 177.0 && m <= 178.0);
    }

    #[test]
    fn test_hist_stress_case_178() {
        let t = Tensor::from_slice(&[178.0, 179.0], vec![2]);
        let m = median(&t);
        assert!(m >= 178.0 && m <= 179.0);
    }

    #[test]
    fn test_hist_stress_case_179() {
        let t = Tensor::from_slice(&[179.0, 180.0], vec![2]);
        let m = median(&t);
        assert!(m >= 179.0 && m <= 180.0);
    }

    #[test]
    fn test_hist_stress_case_180() {
        let t = Tensor::from_slice(&[180.0, 181.0], vec![2]);
        let m = median(&t);
        assert!(m >= 180.0 && m <= 181.0);
    }

    #[test]
    fn test_hist_stress_case_181() {
        let t = Tensor::from_slice(&[181.0, 182.0], vec![2]);
        let m = median(&t);
        assert!(m >= 181.0 && m <= 182.0);
    }

    #[test]
    fn test_hist_stress_case_182() {
        let t = Tensor::from_slice(&[182.0, 183.0], vec![2]);
        let m = median(&t);
        assert!(m >= 182.0 && m <= 183.0);
    }

    #[test]
    fn test_hist_stress_case_183() {
        let t = Tensor::from_slice(&[183.0, 184.0], vec![2]);
        let m = median(&t);
        assert!(m >= 183.0 && m <= 184.0);
    }

    #[test]
    fn test_hist_stress_case_184() {
        let t = Tensor::from_slice(&[184.0, 185.0], vec![2]);
        let m = median(&t);
        assert!(m >= 184.0 && m <= 185.0);
    }

    #[test]
    fn test_hist_stress_case_185() {
        let t = Tensor::from_slice(&[185.0, 186.0], vec![2]);
        let m = median(&t);
        assert!(m >= 185.0 && m <= 186.0);
    }

    #[test]
    fn test_hist_stress_case_186() {
        let t = Tensor::from_slice(&[186.0, 187.0], vec![2]);
        let m = median(&t);
        assert!(m >= 186.0 && m <= 187.0);
    }

    #[test]
    fn test_hist_stress_case_187() {
        let t = Tensor::from_slice(&[187.0, 188.0], vec![2]);
        let m = median(&t);
        assert!(m >= 187.0 && m <= 188.0);
    }

    #[test]
    fn test_hist_stress_case_188() {
        let t = Tensor::from_slice(&[188.0, 189.0], vec![2]);
        let m = median(&t);
        assert!(m >= 188.0 && m <= 189.0);
    }

    #[test]
    fn test_hist_stress_case_189() {
        let t = Tensor::from_slice(&[189.0, 190.0], vec![2]);
        let m = median(&t);
        assert!(m >= 189.0 && m <= 190.0);
    }

    #[test]
    fn test_hist_stress_case_190() {
        let t = Tensor::from_slice(&[190.0, 191.0], vec![2]);
        let m = median(&t);
        assert!(m >= 190.0 && m <= 191.0);
    }

    #[test]
    fn test_hist_stress_case_191() {
        let t = Tensor::from_slice(&[191.0, 192.0], vec![2]);
        let m = median(&t);
        assert!(m >= 191.0 && m <= 192.0);
    }

    #[test]
    fn test_hist_stress_case_192() {
        let t = Tensor::from_slice(&[192.0, 193.0], vec![2]);
        let m = median(&t);
        assert!(m >= 192.0 && m <= 193.0);
    }

    #[test]
    fn test_hist_stress_case_193() {
        let t = Tensor::from_slice(&[193.0, 194.0], vec![2]);
        let m = median(&t);
        assert!(m >= 193.0 && m <= 194.0);
    }

    #[test]
    fn test_hist_stress_case_194() {
        let t = Tensor::from_slice(&[194.0, 195.0], vec![2]);
        let m = median(&t);
        assert!(m >= 194.0 && m <= 195.0);
    }

    #[test]
    fn test_hist_stress_case_195() {
        let t = Tensor::from_slice(&[195.0, 196.0], vec![2]);
        let m = median(&t);
        assert!(m >= 195.0 && m <= 196.0);
    }

    #[test]
    fn test_hist_stress_case_196() {
        let t = Tensor::from_slice(&[196.0, 197.0], vec![2]);
        let m = median(&t);
        assert!(m >= 196.0 && m <= 197.0);
    }

    #[test]
    fn test_hist_stress_case_197() {
        let t = Tensor::from_slice(&[197.0, 198.0], vec![2]);
        let m = median(&t);
        assert!(m >= 197.0 && m <= 198.0);
    }

    #[test]
    fn test_hist_stress_case_198() {
        let t = Tensor::from_slice(&[198.0, 199.0], vec![2]);
        let m = median(&t);
        assert!(m >= 198.0 && m <= 199.0);
    }

    #[test]
    fn test_hist_stress_case_199() {
        let t = Tensor::from_slice(&[199.0, 200.0], vec![2]);
        let m = median(&t);
        assert!(m >= 199.0 && m <= 200.0);
    }

    #[test]
    fn test_hist_stress_case_200() {
        let t = Tensor::from_slice(&[200.0, 201.0], vec![2]);
        let m = median(&t);
        assert!(m >= 200.0 && m <= 201.0);
    }

    #[test]
    fn test_hist_stress_case_201() {
        let t = Tensor::from_slice(&[201.0, 202.0], vec![2]);
        let m = median(&t);
        assert!(m >= 201.0 && m <= 202.0);
    }

    #[test]
    fn test_hist_stress_case_202() {
        let t = Tensor::from_slice(&[202.0, 203.0], vec![2]);
        let m = median(&t);
        assert!(m >= 202.0 && m <= 203.0);
    }

    #[test]
    fn test_hist_stress_case_203() {
        let t = Tensor::from_slice(&[203.0, 204.0], vec![2]);
        let m = median(&t);
        assert!(m >= 203.0 && m <= 204.0);
    }

    #[test]
    fn test_hist_stress_case_204() {
        let t = Tensor::from_slice(&[204.0, 205.0], vec![2]);
        let m = median(&t);
        assert!(m >= 204.0 && m <= 205.0);
    }

    #[test]
    fn test_hist_stress_case_205() {
        let t = Tensor::from_slice(&[205.0, 206.0], vec![2]);
        let m = median(&t);
        assert!(m >= 205.0 && m <= 206.0);
    }

    #[test]
    fn test_hist_stress_case_206() {
        let t = Tensor::from_slice(&[206.0, 207.0], vec![2]);
        let m = median(&t);
        assert!(m >= 206.0 && m <= 207.0);
    }

    #[test]
    fn test_hist_stress_case_207() {
        let t = Tensor::from_slice(&[207.0, 208.0], vec![2]);
        let m = median(&t);
        assert!(m >= 207.0 && m <= 208.0);
    }

    #[test]
    fn test_hist_stress_case_208() {
        let t = Tensor::from_slice(&[208.0, 209.0], vec![2]);
        let m = median(&t);
        assert!(m >= 208.0 && m <= 209.0);
    }

    #[test]
    fn test_hist_stress_case_209() {
        let t = Tensor::from_slice(&[209.0, 210.0], vec![2]);
        let m = median(&t);
        assert!(m >= 209.0 && m <= 210.0);
    }

    #[test]
    fn test_hist_stress_case_210() {
        let t = Tensor::from_slice(&[210.0, 211.0], vec![2]);
        let m = median(&t);
        assert!(m >= 210.0 && m <= 211.0);
    }

    #[test]
    fn test_hist_stress_case_211() {
        let t = Tensor::from_slice(&[211.0, 212.0], vec![2]);
        let m = median(&t);
        assert!(m >= 211.0 && m <= 212.0);
    }

    #[test]
    fn test_hist_stress_case_212() {
        let t = Tensor::from_slice(&[212.0, 213.0], vec![2]);
        let m = median(&t);
        assert!(m >= 212.0 && m <= 213.0);
    }

    #[test]
    fn test_hist_stress_case_213() {
        let t = Tensor::from_slice(&[213.0, 214.0], vec![2]);
        let m = median(&t);
        assert!(m >= 213.0 && m <= 214.0);
    }

    #[test]
    fn test_hist_stress_case_214() {
        let t = Tensor::from_slice(&[214.0, 215.0], vec![2]);
        let m = median(&t);
        assert!(m >= 214.0 && m <= 215.0);
    }

    #[test]
    fn test_hist_stress_case_215() {
        let t = Tensor::from_slice(&[215.0, 216.0], vec![2]);
        let m = median(&t);
        assert!(m >= 215.0 && m <= 216.0);
    }

    #[test]
    fn test_hist_stress_case_216() {
        let t = Tensor::from_slice(&[216.0, 217.0], vec![2]);
        let m = median(&t);
        assert!(m >= 216.0 && m <= 217.0);
    }

    #[test]
    fn test_hist_stress_case_217() {
        let t = Tensor::from_slice(&[217.0, 218.0], vec![2]);
        let m = median(&t);
        assert!(m >= 217.0 && m <= 218.0);
    }

    #[test]
    fn test_hist_stress_case_218() {
        let t = Tensor::from_slice(&[218.0, 219.0], vec![2]);
        let m = median(&t);
        assert!(m >= 218.0 && m <= 219.0);
    }

    #[test]
    fn test_hist_stress_case_219() {
        let t = Tensor::from_slice(&[219.0, 220.0], vec![2]);
        let m = median(&t);
        assert!(m >= 219.0 && m <= 220.0);
    }

    #[test]
    fn test_hist_stress_case_220() {
        let t = Tensor::from_slice(&[220.0, 221.0], vec![2]);
        let m = median(&t);
        assert!(m >= 220.0 && m <= 221.0);
    }

    #[test]
    fn test_hist_stress_case_221() {
        let t = Tensor::from_slice(&[221.0, 222.0], vec![2]);
        let m = median(&t);
        assert!(m >= 221.0 && m <= 222.0);
    }

    #[test]
    fn test_hist_stress_case_222() {
        let t = Tensor::from_slice(&[222.0, 223.0], vec![2]);
        let m = median(&t);
        assert!(m >= 222.0 && m <= 223.0);
    }

    #[test]
    fn test_hist_stress_case_223() {
        let t = Tensor::from_slice(&[223.0, 224.0], vec![2]);
        let m = median(&t);
        assert!(m >= 223.0 && m <= 224.0);
    }

    #[test]
    fn test_hist_stress_case_224() {
        let t = Tensor::from_slice(&[224.0, 225.0], vec![2]);
        let m = median(&t);
        assert!(m >= 224.0 && m <= 225.0);
    }

    #[test]
    fn test_hist_stress_case_225() {
        let t = Tensor::from_slice(&[225.0, 226.0], vec![2]);
        let m = median(&t);
        assert!(m >= 225.0 && m <= 226.0);
    }

    #[test]
    fn test_hist_stress_case_226() {
        let t = Tensor::from_slice(&[226.0, 227.0], vec![2]);
        let m = median(&t);
        assert!(m >= 226.0 && m <= 227.0);
    }

    #[test]
    fn test_hist_stress_case_227() {
        let t = Tensor::from_slice(&[227.0, 228.0], vec![2]);
        let m = median(&t);
        assert!(m >= 227.0 && m <= 228.0);
    }

    #[test]
    fn test_hist_stress_case_228() {
        let t = Tensor::from_slice(&[228.0, 229.0], vec![2]);
        let m = median(&t);
        assert!(m >= 228.0 && m <= 229.0);
    }

    #[test]
    fn test_hist_stress_case_229() {
        let t = Tensor::from_slice(&[229.0, 230.0], vec![2]);
        let m = median(&t);
        assert!(m >= 229.0 && m <= 230.0);
    }

    #[test]
    fn test_hist_stress_case_230() {
        let t = Tensor::from_slice(&[230.0, 231.0], vec![2]);
        let m = median(&t);
        assert!(m >= 230.0 && m <= 231.0);
    }

    #[test]
    fn test_hist_stress_case_231() {
        let t = Tensor::from_slice(&[231.0, 232.0], vec![2]);
        let m = median(&t);
        assert!(m >= 231.0 && m <= 232.0);
    }

    #[test]
    fn test_hist_stress_case_232() {
        let t = Tensor::from_slice(&[232.0, 233.0], vec![2]);
        let m = median(&t);
        assert!(m >= 232.0 && m <= 233.0);
    }

    #[test]
    fn test_hist_stress_case_233() {
        let t = Tensor::from_slice(&[233.0, 234.0], vec![2]);
        let m = median(&t);
        assert!(m >= 233.0 && m <= 234.0);
    }

    #[test]
    fn test_hist_stress_case_234() {
        let t = Tensor::from_slice(&[234.0, 235.0], vec![2]);
        let m = median(&t);
        assert!(m >= 234.0 && m <= 235.0);
    }

    #[test]
    fn test_hist_stress_case_235() {
        let t = Tensor::from_slice(&[235.0, 236.0], vec![2]);
        let m = median(&t);
        assert!(m >= 235.0 && m <= 236.0);
    }

    #[test]
    fn test_hist_stress_case_236() {
        let t = Tensor::from_slice(&[236.0, 237.0], vec![2]);
        let m = median(&t);
        assert!(m >= 236.0 && m <= 237.0);
    }

    #[test]
    fn test_hist_stress_case_237() {
        let t = Tensor::from_slice(&[237.0, 238.0], vec![2]);
        let m = median(&t);
        assert!(m >= 237.0 && m <= 238.0);
    }

    #[test]
    fn test_hist_stress_case_238() {
        let t = Tensor::from_slice(&[238.0, 239.0], vec![2]);
        let m = median(&t);
        assert!(m >= 238.0 && m <= 239.0);
    }

    #[test]
    fn test_hist_stress_case_239() {
        let t = Tensor::from_slice(&[239.0, 240.0], vec![2]);
        let m = median(&t);
        assert!(m >= 239.0 && m <= 240.0);
    }

    #[test]
    fn test_hist_stress_case_240() {
        let t = Tensor::from_slice(&[240.0, 241.0], vec![2]);
        let m = median(&t);
        assert!(m >= 240.0 && m <= 241.0);
    }

    #[test]
    fn test_hist_stress_case_241() {
        let t = Tensor::from_slice(&[241.0, 242.0], vec![2]);
        let m = median(&t);
        assert!(m >= 241.0 && m <= 242.0);
    }

    #[test]
    fn test_hist_stress_case_242() {
        let t = Tensor::from_slice(&[242.0, 243.0], vec![2]);
        let m = median(&t);
        assert!(m >= 242.0 && m <= 243.0);
    }

    #[test]
    fn test_hist_stress_case_243() {
        let t = Tensor::from_slice(&[243.0, 244.0], vec![2]);
        let m = median(&t);
        assert!(m >= 243.0 && m <= 244.0);
    }

    #[test]
    fn test_hist_stress_case_244() {
        let t = Tensor::from_slice(&[244.0, 245.0], vec![2]);
        let m = median(&t);
        assert!(m >= 244.0 && m <= 245.0);
    }

    #[test]
    fn test_hist_stress_case_245() {
        let t = Tensor::from_slice(&[245.0, 246.0], vec![2]);
        let m = median(&t);
        assert!(m >= 245.0 && m <= 246.0);
    }

    #[test]
    fn test_hist_stress_case_246() {
        let t = Tensor::from_slice(&[246.0, 247.0], vec![2]);
        let m = median(&t);
        assert!(m >= 246.0 && m <= 247.0);
    }

    #[test]
    fn test_hist_stress_case_247() {
        let t = Tensor::from_slice(&[247.0, 248.0], vec![2]);
        let m = median(&t);
        assert!(m >= 247.0 && m <= 248.0);
    }

    #[test]
    fn test_hist_stress_case_248() {
        let t = Tensor::from_slice(&[248.0, 249.0], vec![2]);
        let m = median(&t);
        assert!(m >= 248.0 && m <= 249.0);
    }

    #[test]
    fn test_hist_stress_case_249() {
        let t = Tensor::from_slice(&[249.0, 250.0], vec![2]);
        let m = median(&t);
        assert!(m >= 249.0 && m <= 250.0);
    }

    #[test]
    fn test_hist_stress_case_250() {
        let t = Tensor::from_slice(&[250.0, 251.0], vec![2]);
        let m = median(&t);
        assert!(m >= 250.0 && m <= 251.0);
    }

    #[test]
    fn test_hist_stress_case_251() {
        let t = Tensor::from_slice(&[251.0, 252.0], vec![2]);
        let m = median(&t);
        assert!(m >= 251.0 && m <= 252.0);
    }

    #[test]
    fn test_hist_stress_case_252() {
        let t = Tensor::from_slice(&[252.0, 253.0], vec![2]);
        let m = median(&t);
        assert!(m >= 252.0 && m <= 253.0);
    }

    #[test]
    fn test_hist_stress_case_253() {
        let t = Tensor::from_slice(&[253.0, 254.0], vec![2]);
        let m = median(&t);
        assert!(m >= 253.0 && m <= 254.0);
    }

    #[test]
    fn test_hist_stress_case_254() {
        let t = Tensor::from_slice(&[254.0, 255.0], vec![2]);
        let m = median(&t);
        assert!(m >= 254.0 && m <= 255.0);
    }

    #[test]
    fn test_hist_stress_case_255() {
        let t = Tensor::from_slice(&[255.0, 256.0], vec![2]);
        let m = median(&t);
        assert!(m >= 255.0 && m <= 256.0);
    }

    #[test]
    fn test_hist_stress_case_256() {
        let t = Tensor::from_slice(&[256.0, 257.0], vec![2]);
        let m = median(&t);
        assert!(m >= 256.0 && m <= 257.0);
    }

    #[test]
    fn test_hist_stress_case_257() {
        let t = Tensor::from_slice(&[257.0, 258.0], vec![2]);
        let m = median(&t);
        assert!(m >= 257.0 && m <= 258.0);
    }

    #[test]
    fn test_hist_stress_case_258() {
        let t = Tensor::from_slice(&[258.0, 259.0], vec![2]);
        let m = median(&t);
        assert!(m >= 258.0 && m <= 259.0);
    }

    #[test]
    fn test_hist_stress_case_259() {
        let t = Tensor::from_slice(&[259.0, 260.0], vec![2]);
        let m = median(&t);
        assert!(m >= 259.0 && m <= 260.0);
    }

    #[test]
    fn test_hist_stress_case_260() {
        let t = Tensor::from_slice(&[260.0, 261.0], vec![2]);
        let m = median(&t);
        assert!(m >= 260.0 && m <= 261.0);
    }

    #[test]
    fn test_hist_stress_case_261() {
        let t = Tensor::from_slice(&[261.0, 262.0], vec![2]);
        let m = median(&t);
        assert!(m >= 261.0 && m <= 262.0);
    }

    #[test]
    fn test_hist_stress_case_262() {
        let t = Tensor::from_slice(&[262.0, 263.0], vec![2]);
        let m = median(&t);
        assert!(m >= 262.0 && m <= 263.0);
    }

    #[test]
    fn test_hist_stress_case_263() {
        let t = Tensor::from_slice(&[263.0, 264.0], vec![2]);
        let m = median(&t);
        assert!(m >= 263.0 && m <= 264.0);
    }

    #[test]
    fn test_hist_stress_case_264() {
        let t = Tensor::from_slice(&[264.0, 265.0], vec![2]);
        let m = median(&t);
        assert!(m >= 264.0 && m <= 265.0);
    }

    #[test]
    fn test_hist_stress_case_265() {
        let t = Tensor::from_slice(&[265.0, 266.0], vec![2]);
        let m = median(&t);
        assert!(m >= 265.0 && m <= 266.0);
    }

    #[test]
    fn test_hist_stress_case_266() {
        let t = Tensor::from_slice(&[266.0, 267.0], vec![2]);
        let m = median(&t);
        assert!(m >= 266.0 && m <= 267.0);
    }

    #[test]
    fn test_hist_stress_case_267() {
        let t = Tensor::from_slice(&[267.0, 268.0], vec![2]);
        let m = median(&t);
        assert!(m >= 267.0 && m <= 268.0);
    }

    #[test]
    fn test_hist_stress_case_268() {
        let t = Tensor::from_slice(&[268.0, 269.0], vec![2]);
        let m = median(&t);
        assert!(m >= 268.0 && m <= 269.0);
    }

    #[test]
    fn test_hist_stress_case_269() {
        let t = Tensor::from_slice(&[269.0, 270.0], vec![2]);
        let m = median(&t);
        assert!(m >= 269.0 && m <= 270.0);
    }

    #[test]
    fn test_hist_stress_case_270() {
        let t = Tensor::from_slice(&[270.0, 271.0], vec![2]);
        let m = median(&t);
        assert!(m >= 270.0 && m <= 271.0);
    }

    #[test]
    fn test_hist_stress_case_271() {
        let t = Tensor::from_slice(&[271.0, 272.0], vec![2]);
        let m = median(&t);
        assert!(m >= 271.0 && m <= 272.0);
    }

    #[test]
    fn test_hist_stress_case_272() {
        let t = Tensor::from_slice(&[272.0, 273.0], vec![2]);
        let m = median(&t);
        assert!(m >= 272.0 && m <= 273.0);
    }

    #[test]
    fn test_hist_stress_case_273() {
        let t = Tensor::from_slice(&[273.0, 274.0], vec![2]);
        let m = median(&t);
        assert!(m >= 273.0 && m <= 274.0);
    }

    #[test]
    fn test_hist_stress_case_274() {
        let t = Tensor::from_slice(&[274.0, 275.0], vec![2]);
        let m = median(&t);
        assert!(m >= 274.0 && m <= 275.0);
    }

    #[test]
    fn test_hist_stress_case_275() {
        let t = Tensor::from_slice(&[275.0, 276.0], vec![2]);
        let m = median(&t);
        assert!(m >= 275.0 && m <= 276.0);
    }

    #[test]
    fn test_hist_stress_case_276() {
        let t = Tensor::from_slice(&[276.0, 277.0], vec![2]);
        let m = median(&t);
        assert!(m >= 276.0 && m <= 277.0);
    }

    #[test]
    fn test_hist_stress_case_277() {
        let t = Tensor::from_slice(&[277.0, 278.0], vec![2]);
        let m = median(&t);
        assert!(m >= 277.0 && m <= 278.0);
    }

    #[test]
    fn test_hist_stress_case_278() {
        let t = Tensor::from_slice(&[278.0, 279.0], vec![2]);
        let m = median(&t);
        assert!(m >= 278.0 && m <= 279.0);
    }

    #[test]
    fn test_hist_stress_case_279() {
        let t = Tensor::from_slice(&[279.0, 280.0], vec![2]);
        let m = median(&t);
        assert!(m >= 279.0 && m <= 280.0);
    }

    #[test]
    fn test_hist_stress_case_280() {
        let t = Tensor::from_slice(&[280.0, 281.0], vec![2]);
        let m = median(&t);
        assert!(m >= 280.0 && m <= 281.0);
    }

    #[test]
    fn test_hist_stress_case_281() {
        let t = Tensor::from_slice(&[281.0, 282.0], vec![2]);
        let m = median(&t);
        assert!(m >= 281.0 && m <= 282.0);
    }

    #[test]
    fn test_hist_stress_case_282() {
        let t = Tensor::from_slice(&[282.0, 283.0], vec![2]);
        let m = median(&t);
        assert!(m >= 282.0 && m <= 283.0);
    }

    #[test]
    fn test_hist_stress_case_283() {
        let t = Tensor::from_slice(&[283.0, 284.0], vec![2]);
        let m = median(&t);
        assert!(m >= 283.0 && m <= 284.0);
    }

    #[test]
    fn test_hist_stress_case_284() {
        let t = Tensor::from_slice(&[284.0, 285.0], vec![2]);
        let m = median(&t);
        assert!(m >= 284.0 && m <= 285.0);
    }

    #[test]
    fn test_hist_stress_case_285() {
        let t = Tensor::from_slice(&[285.0, 286.0], vec![2]);
        let m = median(&t);
        assert!(m >= 285.0 && m <= 286.0);
    }

    #[test]
    fn test_hist_stress_case_286() {
        let t = Tensor::from_slice(&[286.0, 287.0], vec![2]);
        let m = median(&t);
        assert!(m >= 286.0 && m <= 287.0);
    }

    #[test]
    fn test_hist_stress_case_287() {
        let t = Tensor::from_slice(&[287.0, 288.0], vec![2]);
        let m = median(&t);
        assert!(m >= 287.0 && m <= 288.0);
    }

    #[test]
    fn test_hist_stress_case_288() {
        let t = Tensor::from_slice(&[288.0, 289.0], vec![2]);
        let m = median(&t);
        assert!(m >= 288.0 && m <= 289.0);
    }

    #[test]
    fn test_hist_stress_case_289() {
        let t = Tensor::from_slice(&[289.0, 290.0], vec![2]);
        let m = median(&t);
        assert!(m >= 289.0 && m <= 290.0);
    }

    #[test]
    fn test_hist_stress_case_290() {
        let t = Tensor::from_slice(&[290.0, 291.0], vec![2]);
        let m = median(&t);
        assert!(m >= 290.0 && m <= 291.0);
    }

    #[test]
    fn test_hist_stress_case_291() {
        let t = Tensor::from_slice(&[291.0, 292.0], vec![2]);
        let m = median(&t);
        assert!(m >= 291.0 && m <= 292.0);
    }

    #[test]
    fn test_hist_stress_case_292() {
        let t = Tensor::from_slice(&[292.0, 293.0], vec![2]);
        let m = median(&t);
        assert!(m >= 292.0 && m <= 293.0);
    }

    #[test]
    fn test_hist_stress_case_293() {
        let t = Tensor::from_slice(&[293.0, 294.0], vec![2]);
        let m = median(&t);
        assert!(m >= 293.0 && m <= 294.0);
    }

    #[test]
    fn test_hist_stress_case_294() {
        let t = Tensor::from_slice(&[294.0, 295.0], vec![2]);
        let m = median(&t);
        assert!(m >= 294.0 && m <= 295.0);
    }

    #[test]
    fn test_hist_stress_case_295() {
        let t = Tensor::from_slice(&[295.0, 296.0], vec![2]);
        let m = median(&t);
        assert!(m >= 295.0 && m <= 296.0);
    }

    #[test]
    fn test_hist_stress_case_296() {
        let t = Tensor::from_slice(&[296.0, 297.0], vec![2]);
        let m = median(&t);
        assert!(m >= 296.0 && m <= 297.0);
    }

    #[test]
    fn test_hist_stress_case_297() {
        let t = Tensor::from_slice(&[297.0, 298.0], vec![2]);
        let m = median(&t);
        assert!(m >= 297.0 && m <= 298.0);
    }

    #[test]
    fn test_hist_stress_case_298() {
        let t = Tensor::from_slice(&[298.0, 299.0], vec![2]);
        let m = median(&t);
        assert!(m >= 298.0 && m <= 299.0);
    }

    #[test]
    fn test_hist_stress_case_299() {
        let t = Tensor::from_slice(&[299.0, 300.0], vec![2]);
        let m = median(&t);
        assert!(m >= 299.0 && m <= 300.0);
    }

    #[test]
    fn test_hist_stress_case_300() {
        let t = Tensor::from_slice(&[300.0, 301.0], vec![2]);
        let m = median(&t);
        assert!(m >= 300.0 && m <= 301.0);
    }

    #[test]
    fn test_hist_stress_case_301() {
        let t = Tensor::from_slice(&[301.0, 302.0], vec![2]);
        let m = median(&t);
        assert!(m >= 301.0 && m <= 302.0);
    }

    #[test]
    fn test_hist_stress_case_302() {
        let t = Tensor::from_slice(&[302.0, 303.0], vec![2]);
        let m = median(&t);
        assert!(m >= 302.0 && m <= 303.0);
    }

    #[test]
    fn test_hist_stress_case_303() {
        let t = Tensor::from_slice(&[303.0, 304.0], vec![2]);
        let m = median(&t);
        assert!(m >= 303.0 && m <= 304.0);
    }

    #[test]
    fn test_hist_stress_case_304() {
        let t = Tensor::from_slice(&[304.0, 305.0], vec![2]);
        let m = median(&t);
        assert!(m >= 304.0 && m <= 305.0);
    }

    #[test]
    fn test_hist_stress_case_305() {
        let t = Tensor::from_slice(&[305.0, 306.0], vec![2]);
        let m = median(&t);
        assert!(m >= 305.0 && m <= 306.0);
    }

    #[test]
    fn test_hist_stress_case_306() {
        let t = Tensor::from_slice(&[306.0, 307.0], vec![2]);
        let m = median(&t);
        assert!(m >= 306.0 && m <= 307.0);
    }

    #[test]
    fn test_hist_stress_case_307() {
        let t = Tensor::from_slice(&[307.0, 308.0], vec![2]);
        let m = median(&t);
        assert!(m >= 307.0 && m <= 308.0);
    }

    #[test]
    fn test_hist_stress_case_308() {
        let t = Tensor::from_slice(&[308.0, 309.0], vec![2]);
        let m = median(&t);
        assert!(m >= 308.0 && m <= 309.0);
    }

    #[test]
    fn test_hist_stress_case_309() {
        let t = Tensor::from_slice(&[309.0, 310.0], vec![2]);
        let m = median(&t);
        assert!(m >= 309.0 && m <= 310.0);
    }

    #[test]
    fn test_hist_stress_case_310() {
        let t = Tensor::from_slice(&[310.0, 311.0], vec![2]);
        let m = median(&t);
        assert!(m >= 310.0 && m <= 311.0);
    }

    #[test]
    fn test_hist_stress_case_311() {
        let t = Tensor::from_slice(&[311.0, 312.0], vec![2]);
        let m = median(&t);
        assert!(m >= 311.0 && m <= 312.0);
    }

    #[test]
    fn test_hist_stress_case_312() {
        let t = Tensor::from_slice(&[312.0, 313.0], vec![2]);
        let m = median(&t);
        assert!(m >= 312.0 && m <= 313.0);
    }

    #[test]
    fn test_hist_stress_case_313() {
        let t = Tensor::from_slice(&[313.0, 314.0], vec![2]);
        let m = median(&t);
        assert!(m >= 313.0 && m <= 314.0);
    }

    #[test]
    fn test_hist_stress_case_314() {
        let t = Tensor::from_slice(&[314.0, 315.0], vec![2]);
        let m = median(&t);
        assert!(m >= 314.0 && m <= 315.0);
    }

    #[test]
    fn test_hist_stress_case_315() {
        let t = Tensor::from_slice(&[315.0, 316.0], vec![2]);
        let m = median(&t);
        assert!(m >= 315.0 && m <= 316.0);
    }

    #[test]
    fn test_hist_stress_case_316() {
        let t = Tensor::from_slice(&[316.0, 317.0], vec![2]);
        let m = median(&t);
        assert!(m >= 316.0 && m <= 317.0);
    }

    #[test]
    fn test_hist_stress_case_317() {
        let t = Tensor::from_slice(&[317.0, 318.0], vec![2]);
        let m = median(&t);
        assert!(m >= 317.0 && m <= 318.0);
    }

    #[test]
    fn test_hist_stress_case_318() {
        let t = Tensor::from_slice(&[318.0, 319.0], vec![2]);
        let m = median(&t);
        assert!(m >= 318.0 && m <= 319.0);
    }

    #[test]
    fn test_hist_stress_case_319() {
        let t = Tensor::from_slice(&[319.0, 320.0], vec![2]);
        let m = median(&t);
        assert!(m >= 319.0 && m <= 320.0);
    }

    #[test]
    fn test_hist_stress_case_320() {
        let t = Tensor::from_slice(&[320.0, 321.0], vec![2]);
        let m = median(&t);
        assert!(m >= 320.0 && m <= 321.0);
    }

    #[test]
    fn test_hist_stress_case_321() {
        let t = Tensor::from_slice(&[321.0, 322.0], vec![2]);
        let m = median(&t);
        assert!(m >= 321.0 && m <= 322.0);
    }

    #[test]
    fn test_hist_stress_case_322() {
        let t = Tensor::from_slice(&[322.0, 323.0], vec![2]);
        let m = median(&t);
        assert!(m >= 322.0 && m <= 323.0);
    }

    #[test]
    fn test_hist_stress_case_323() {
        let t = Tensor::from_slice(&[323.0, 324.0], vec![2]);
        let m = median(&t);
        assert!(m >= 323.0 && m <= 324.0);
    }

    #[test]
    fn test_hist_stress_case_324() {
        let t = Tensor::from_slice(&[324.0, 325.0], vec![2]);
        let m = median(&t);
        assert!(m >= 324.0 && m <= 325.0);
    }

    #[test]
    fn test_hist_stress_case_325() {
        let t = Tensor::from_slice(&[325.0, 326.0], vec![2]);
        let m = median(&t);
        assert!(m >= 325.0 && m <= 326.0);
    }

    #[test]
    fn test_hist_stress_case_326() {
        let t = Tensor::from_slice(&[326.0, 327.0], vec![2]);
        let m = median(&t);
        assert!(m >= 326.0 && m <= 327.0);
    }

    #[test]
    fn test_hist_stress_case_327() {
        let t = Tensor::from_slice(&[327.0, 328.0], vec![2]);
        let m = median(&t);
        assert!(m >= 327.0 && m <= 328.0);
    }

    #[test]
    fn test_hist_stress_case_328() {
        let t = Tensor::from_slice(&[328.0, 329.0], vec![2]);
        let m = median(&t);
        assert!(m >= 328.0 && m <= 329.0);
    }

    #[test]
    fn test_hist_stress_case_329() {
        let t = Tensor::from_slice(&[329.0, 330.0], vec![2]);
        let m = median(&t);
        assert!(m >= 329.0 && m <= 330.0);
    }

    #[test]
    fn test_hist_stress_case_330() {
        let t = Tensor::from_slice(&[330.0, 331.0], vec![2]);
        let m = median(&t);
        assert!(m >= 330.0 && m <= 331.0);
    }

    #[test]
    fn test_hist_stress_case_331() {
        let t = Tensor::from_slice(&[331.0, 332.0], vec![2]);
        let m = median(&t);
        assert!(m >= 331.0 && m <= 332.0);
    }

    #[test]
    fn test_hist_stress_case_332() {
        let t = Tensor::from_slice(&[332.0, 333.0], vec![2]);
        let m = median(&t);
        assert!(m >= 332.0 && m <= 333.0);
    }

    #[test]
    fn test_hist_stress_case_333() {
        let t = Tensor::from_slice(&[333.0, 334.0], vec![2]);
        let m = median(&t);
        assert!(m >= 333.0 && m <= 334.0);
    }

    #[test]
    fn test_hist_stress_case_334() {
        let t = Tensor::from_slice(&[334.0, 335.0], vec![2]);
        let m = median(&t);
        assert!(m >= 334.0 && m <= 335.0);
    }

    #[test]
    fn test_hist_stress_case_335() {
        let t = Tensor::from_slice(&[335.0, 336.0], vec![2]);
        let m = median(&t);
        assert!(m >= 335.0 && m <= 336.0);
    }

    #[test]
    fn test_hist_stress_case_336() {
        let t = Tensor::from_slice(&[336.0, 337.0], vec![2]);
        let m = median(&t);
        assert!(m >= 336.0 && m <= 337.0);
    }

    #[test]
    fn test_hist_stress_case_337() {
        let t = Tensor::from_slice(&[337.0, 338.0], vec![2]);
        let m = median(&t);
        assert!(m >= 337.0 && m <= 338.0);
    }

    #[test]
    fn test_hist_stress_case_338() {
        let t = Tensor::from_slice(&[338.0, 339.0], vec![2]);
        let m = median(&t);
        assert!(m >= 338.0 && m <= 339.0);
    }

    #[test]
    fn test_hist_stress_case_339() {
        let t = Tensor::from_slice(&[339.0, 340.0], vec![2]);
        let m = median(&t);
        assert!(m >= 339.0 && m <= 340.0);
    }

    #[test]
    fn test_hist_stress_case_340() {
        let t = Tensor::from_slice(&[340.0, 341.0], vec![2]);
        let m = median(&t);
        assert!(m >= 340.0 && m <= 341.0);
    }

    #[test]
    fn test_hist_stress_case_341() {
        let t = Tensor::from_slice(&[341.0, 342.0], vec![2]);
        let m = median(&t);
        assert!(m >= 341.0 && m <= 342.0);
    }

    #[test]
    fn test_hist_stress_case_342() {
        let t = Tensor::from_slice(&[342.0, 343.0], vec![2]);
        let m = median(&t);
        assert!(m >= 342.0 && m <= 343.0);
    }

    #[test]
    fn test_hist_stress_case_343() {
        let t = Tensor::from_slice(&[343.0, 344.0], vec![2]);
        let m = median(&t);
        assert!(m >= 343.0 && m <= 344.0);
    }

    #[test]
    fn test_hist_stress_case_344() {
        let t = Tensor::from_slice(&[344.0, 345.0], vec![2]);
        let m = median(&t);
        assert!(m >= 344.0 && m <= 345.0);
    }

    #[test]
    fn test_hist_stress_case_345() {
        let t = Tensor::from_slice(&[345.0, 346.0], vec![2]);
        let m = median(&t);
        assert!(m >= 345.0 && m <= 346.0);
    }

    #[test]
    fn test_hist_stress_case_346() {
        let t = Tensor::from_slice(&[346.0, 347.0], vec![2]);
        let m = median(&t);
        assert!(m >= 346.0 && m <= 347.0);
    }

    #[test]
    fn test_hist_stress_case_347() {
        let t = Tensor::from_slice(&[347.0, 348.0], vec![2]);
        let m = median(&t);
        assert!(m >= 347.0 && m <= 348.0);
    }

    #[test]
    fn test_hist_stress_case_348() {
        let t = Tensor::from_slice(&[348.0, 349.0], vec![2]);
        let m = median(&t);
        assert!(m >= 348.0 && m <= 349.0);
    }

    #[test]
    fn test_hist_stress_case_349() {
        let t = Tensor::from_slice(&[349.0, 350.0], vec![2]);
        let m = median(&t);
        assert!(m >= 349.0 && m <= 350.0);
    }

    #[test]
    fn test_hist_stress_case_350() {
        let t = Tensor::from_slice(&[350.0, 351.0], vec![2]);
        let m = median(&t);
        assert!(m >= 350.0 && m <= 351.0);
    }

    #[test]
    fn test_hist_stress_case_351() {
        let t = Tensor::from_slice(&[351.0, 352.0], vec![2]);
        let m = median(&t);
        assert!(m >= 351.0 && m <= 352.0);
    }

    #[test]
    fn test_hist_stress_case_352() {
        let t = Tensor::from_slice(&[352.0, 353.0], vec![2]);
        let m = median(&t);
        assert!(m >= 352.0 && m <= 353.0);
    }

    #[test]
    fn test_hist_stress_case_353() {
        let t = Tensor::from_slice(&[353.0, 354.0], vec![2]);
        let m = median(&t);
        assert!(m >= 353.0 && m <= 354.0);
    }

    #[test]
    fn test_hist_stress_case_354() {
        let t = Tensor::from_slice(&[354.0, 355.0], vec![2]);
        let m = median(&t);
        assert!(m >= 354.0 && m <= 355.0);
    }

    #[test]
    fn test_hist_stress_case_355() {
        let t = Tensor::from_slice(&[355.0, 356.0], vec![2]);
        let m = median(&t);
        assert!(m >= 355.0 && m <= 356.0);
    }

    #[test]
    fn test_hist_stress_case_356() {
        let t = Tensor::from_slice(&[356.0, 357.0], vec![2]);
        let m = median(&t);
        assert!(m >= 356.0 && m <= 357.0);
    }

    #[test]
    fn test_hist_stress_case_357() {
        let t = Tensor::from_slice(&[357.0, 358.0], vec![2]);
        let m = median(&t);
        assert!(m >= 357.0 && m <= 358.0);
    }

    #[test]
    fn test_hist_stress_case_358() {
        let t = Tensor::from_slice(&[358.0, 359.0], vec![2]);
        let m = median(&t);
        assert!(m >= 358.0 && m <= 359.0);
    }

    #[test]
    fn test_hist_stress_case_359() {
        let t = Tensor::from_slice(&[359.0, 360.0], vec![2]);
        let m = median(&t);
        assert!(m >= 359.0 && m <= 360.0);
    }

    #[test]
    fn test_hist_stress_case_360() {
        let t = Tensor::from_slice(&[360.0, 361.0], vec![2]);
        let m = median(&t);
        assert!(m >= 360.0 && m <= 361.0);
    }

    #[test]
    fn test_hist_stress_case_361() {
        let t = Tensor::from_slice(&[361.0, 362.0], vec![2]);
        let m = median(&t);
        assert!(m >= 361.0 && m <= 362.0);
    }

    #[test]
    fn test_hist_stress_case_362() {
        let t = Tensor::from_slice(&[362.0, 363.0], vec![2]);
        let m = median(&t);
        assert!(m >= 362.0 && m <= 363.0);
    }

    #[test]
    fn test_hist_stress_case_363() {
        let t = Tensor::from_slice(&[363.0, 364.0], vec![2]);
        let m = median(&t);
        assert!(m >= 363.0 && m <= 364.0);
    }

    #[test]
    fn test_hist_stress_case_364() {
        let t = Tensor::from_slice(&[364.0, 365.0], vec![2]);
        let m = median(&t);
        assert!(m >= 364.0 && m <= 365.0);
    }

    #[test]
    fn test_hist_stress_case_365() {
        let t = Tensor::from_slice(&[365.0, 366.0], vec![2]);
        let m = median(&t);
        assert!(m >= 365.0 && m <= 366.0);
    }

    #[test]
    fn test_hist_stress_case_366() {
        let t = Tensor::from_slice(&[366.0, 367.0], vec![2]);
        let m = median(&t);
        assert!(m >= 366.0 && m <= 367.0);
    }

    #[test]
    fn test_hist_stress_case_367() {
        let t = Tensor::from_slice(&[367.0, 368.0], vec![2]);
        let m = median(&t);
        assert!(m >= 367.0 && m <= 368.0);
    }

    #[test]
    fn test_hist_stress_case_368() {
        let t = Tensor::from_slice(&[368.0, 369.0], vec![2]);
        let m = median(&t);
        assert!(m >= 368.0 && m <= 369.0);
    }

    #[test]
    fn test_hist_stress_case_369() {
        let t = Tensor::from_slice(&[369.0, 370.0], vec![2]);
        let m = median(&t);
        assert!(m >= 369.0 && m <= 370.0);
    }

    #[test]
    fn test_hist_stress_case_370() {
        let t = Tensor::from_slice(&[370.0, 371.0], vec![2]);
        let m = median(&t);
        assert!(m >= 370.0 && m <= 371.0);
    }

    #[test]
    fn test_hist_stress_case_371() {
        let t = Tensor::from_slice(&[371.0, 372.0], vec![2]);
        let m = median(&t);
        assert!(m >= 371.0 && m <= 372.0);
    }

    #[test]
    fn test_hist_stress_case_372() {
        let t = Tensor::from_slice(&[372.0, 373.0], vec![2]);
        let m = median(&t);
        assert!(m >= 372.0 && m <= 373.0);
    }

    #[test]
    fn test_hist_stress_case_373() {
        let t = Tensor::from_slice(&[373.0, 374.0], vec![2]);
        let m = median(&t);
        assert!(m >= 373.0 && m <= 374.0);
    }

    #[test]
    fn test_hist_stress_case_374() {
        let t = Tensor::from_slice(&[374.0, 375.0], vec![2]);
        let m = median(&t);
        assert!(m >= 374.0 && m <= 375.0);
    }

    #[test]
    fn test_hist_stress_case_375() {
        let t = Tensor::from_slice(&[375.0, 376.0], vec![2]);
        let m = median(&t);
        assert!(m >= 375.0 && m <= 376.0);
    }

    #[test]
    fn test_hist_stress_case_376() {
        let t = Tensor::from_slice(&[376.0, 377.0], vec![2]);
        let m = median(&t);
        assert!(m >= 376.0 && m <= 377.0);
    }

    #[test]
    fn test_hist_stress_case_377() {
        let t = Tensor::from_slice(&[377.0, 378.0], vec![2]);
        let m = median(&t);
        assert!(m >= 377.0 && m <= 378.0);
    }

    #[test]
    fn test_hist_stress_case_378() {
        let t = Tensor::from_slice(&[378.0, 379.0], vec![2]);
        let m = median(&t);
        assert!(m >= 378.0 && m <= 379.0);
    }

    #[test]
    fn test_hist_stress_case_379() {
        let t = Tensor::from_slice(&[379.0, 380.0], vec![2]);
        let m = median(&t);
        assert!(m >= 379.0 && m <= 380.0);
    }

    #[test]
    fn test_hist_stress_case_380() {
        let t = Tensor::from_slice(&[380.0, 381.0], vec![2]);
        let m = median(&t);
        assert!(m >= 380.0 && m <= 381.0);
    }

    #[test]
    fn test_hist_stress_case_381() {
        let t = Tensor::from_slice(&[381.0, 382.0], vec![2]);
        let m = median(&t);
        assert!(m >= 381.0 && m <= 382.0);
    }

    #[test]
    fn test_hist_stress_case_382() {
        let t = Tensor::from_slice(&[382.0, 383.0], vec![2]);
        let m = median(&t);
        assert!(m >= 382.0 && m <= 383.0);
    }

    #[test]
    fn test_hist_stress_case_383() {
        let t = Tensor::from_slice(&[383.0, 384.0], vec![2]);
        let m = median(&t);
        assert!(m >= 383.0 && m <= 384.0);
    }

    #[test]
    fn test_hist_stress_case_384() {
        let t = Tensor::from_slice(&[384.0, 385.0], vec![2]);
        let m = median(&t);
        assert!(m >= 384.0 && m <= 385.0);
    }

    #[test]
    fn test_hist_stress_case_385() {
        let t = Tensor::from_slice(&[385.0, 386.0], vec![2]);
        let m = median(&t);
        assert!(m >= 385.0 && m <= 386.0);
    }

    #[test]
    fn test_hist_stress_case_386() {
        let t = Tensor::from_slice(&[386.0, 387.0], vec![2]);
        let m = median(&t);
        assert!(m >= 386.0 && m <= 387.0);
    }

    #[test]
    fn test_hist_stress_case_387() {
        let t = Tensor::from_slice(&[387.0, 388.0], vec![2]);
        let m = median(&t);
        assert!(m >= 387.0 && m <= 388.0);
    }

    #[test]
    fn test_hist_stress_case_388() {
        let t = Tensor::from_slice(&[388.0, 389.0], vec![2]);
        let m = median(&t);
        assert!(m >= 388.0 && m <= 389.0);
    }

    #[test]
    fn test_hist_stress_case_389() {
        let t = Tensor::from_slice(&[389.0, 390.0], vec![2]);
        let m = median(&t);
        assert!(m >= 389.0 && m <= 390.0);
    }

    #[test]
    fn test_hist_stress_case_390() {
        let t = Tensor::from_slice(&[390.0, 391.0], vec![2]);
        let m = median(&t);
        assert!(m >= 390.0 && m <= 391.0);
    }

    #[test]
    fn test_hist_stress_case_391() {
        let t = Tensor::from_slice(&[391.0, 392.0], vec![2]);
        let m = median(&t);
        assert!(m >= 391.0 && m <= 392.0);
    }

    #[test]
    fn test_hist_stress_case_392() {
        let t = Tensor::from_slice(&[392.0, 393.0], vec![2]);
        let m = median(&t);
        assert!(m >= 392.0 && m <= 393.0);
    }

    #[test]
    fn test_hist_stress_case_393() {
        let t = Tensor::from_slice(&[393.0, 394.0], vec![2]);
        let m = median(&t);
        assert!(m >= 393.0 && m <= 394.0);
    }

    #[test]
    fn test_hist_stress_case_394() {
        let t = Tensor::from_slice(&[394.0, 395.0], vec![2]);
        let m = median(&t);
        assert!(m >= 394.0 && m <= 395.0);
    }

    #[test]
    fn test_hist_stress_case_395() {
        let t = Tensor::from_slice(&[395.0, 396.0], vec![2]);
        let m = median(&t);
        assert!(m >= 395.0 && m <= 396.0);
    }

    #[test]
    fn test_hist_stress_case_396() {
        let t = Tensor::from_slice(&[396.0, 397.0], vec![2]);
        let m = median(&t);
        assert!(m >= 396.0 && m <= 397.0);
    }

    #[test]
    fn test_hist_stress_case_397() {
        let t = Tensor::from_slice(&[397.0, 398.0], vec![2]);
        let m = median(&t);
        assert!(m >= 397.0 && m <= 398.0);
    }

    #[test]
    fn test_hist_stress_case_398() {
        let t = Tensor::from_slice(&[398.0, 399.0], vec![2]);
        let m = median(&t);
        assert!(m >= 398.0 && m <= 399.0);
    }

    #[test]
    fn test_hist_stress_case_399() {
        let t = Tensor::from_slice(&[399.0, 400.0], vec![2]);
        let m = median(&t);
        assert!(m >= 399.0 && m <= 400.0);
    }

    #[test]
    fn test_hist_stress_case_400() {
        let t = Tensor::from_slice(&[400.0, 401.0], vec![2]);
        let m = median(&t);
        assert!(m >= 400.0 && m <= 401.0);
    }

    #[test]
    fn test_hist_stress_case_401() {
        let t = Tensor::from_slice(&[401.0, 402.0], vec![2]);
        let m = median(&t);
        assert!(m >= 401.0 && m <= 402.0);
    }

    #[test]
    fn test_hist_stress_case_402() {
        let t = Tensor::from_slice(&[402.0, 403.0], vec![2]);
        let m = median(&t);
        assert!(m >= 402.0 && m <= 403.0);
    }

    #[test]
    fn test_hist_stress_case_403() {
        let t = Tensor::from_slice(&[403.0, 404.0], vec![2]);
        let m = median(&t);
        assert!(m >= 403.0 && m <= 404.0);
    }

    #[test]
    fn test_hist_stress_case_404() {
        let t = Tensor::from_slice(&[404.0, 405.0], vec![2]);
        let m = median(&t);
        assert!(m >= 404.0 && m <= 405.0);
    }

    #[test]
    fn test_hist_stress_case_405() {
        let t = Tensor::from_slice(&[405.0, 406.0], vec![2]);
        let m = median(&t);
        assert!(m >= 405.0 && m <= 406.0);
    }

    #[test]
    fn test_hist_stress_case_406() {
        let t = Tensor::from_slice(&[406.0, 407.0], vec![2]);
        let m = median(&t);
        assert!(m >= 406.0 && m <= 407.0);
    }

    #[test]
    fn test_hist_stress_case_407() {
        let t = Tensor::from_slice(&[407.0, 408.0], vec![2]);
        let m = median(&t);
        assert!(m >= 407.0 && m <= 408.0);
    }

    #[test]
    fn test_hist_stress_case_408() {
        let t = Tensor::from_slice(&[408.0, 409.0], vec![2]);
        let m = median(&t);
        assert!(m >= 408.0 && m <= 409.0);
    }

    #[test]
    fn test_hist_stress_case_409() {
        let t = Tensor::from_slice(&[409.0, 410.0], vec![2]);
        let m = median(&t);
        assert!(m >= 409.0 && m <= 410.0);
    }

    #[test]
    fn test_hist_stress_case_410() {
        let t = Tensor::from_slice(&[410.0, 411.0], vec![2]);
        let m = median(&t);
        assert!(m >= 410.0 && m <= 411.0);
    }

    #[test]
    fn test_hist_stress_case_411() {
        let t = Tensor::from_slice(&[411.0, 412.0], vec![2]);
        let m = median(&t);
        assert!(m >= 411.0 && m <= 412.0);
    }

    #[test]
    fn test_hist_stress_case_412() {
        let t = Tensor::from_slice(&[412.0, 413.0], vec![2]);
        let m = median(&t);
        assert!(m >= 412.0 && m <= 413.0);
    }

    #[test]
    fn test_hist_stress_case_413() {
        let t = Tensor::from_slice(&[413.0, 414.0], vec![2]);
        let m = median(&t);
        assert!(m >= 413.0 && m <= 414.0);
    }

    #[test]
    fn test_hist_stress_case_414() {
        let t = Tensor::from_slice(&[414.0, 415.0], vec![2]);
        let m = median(&t);
        assert!(m >= 414.0 && m <= 415.0);
    }

    #[test]
    fn test_hist_stress_case_415() {
        let t = Tensor::from_slice(&[415.0, 416.0], vec![2]);
        let m = median(&t);
        assert!(m >= 415.0 && m <= 416.0);
    }

    #[test]
    fn test_hist_stress_case_416() {
        let t = Tensor::from_slice(&[416.0, 417.0], vec![2]);
        let m = median(&t);
        assert!(m >= 416.0 && m <= 417.0);
    }

    #[test]
    fn test_hist_stress_case_417() {
        let t = Tensor::from_slice(&[417.0, 418.0], vec![2]);
        let m = median(&t);
        assert!(m >= 417.0 && m <= 418.0);
    }

    #[test]
    fn test_hist_stress_case_418() {
        let t = Tensor::from_slice(&[418.0, 419.0], vec![2]);
        let m = median(&t);
        assert!(m >= 418.0 && m <= 419.0);
    }

    #[test]
    fn test_hist_stress_case_419() {
        let t = Tensor::from_slice(&[419.0, 420.0], vec![2]);
        let m = median(&t);
        assert!(m >= 419.0 && m <= 420.0);
    }

    #[test]
    fn test_hist_stress_case_420() {
        let t = Tensor::from_slice(&[420.0, 421.0], vec![2]);
        let m = median(&t);
        assert!(m >= 420.0 && m <= 421.0);
    }

    #[test]
    fn test_hist_stress_case_421() {
        let t = Tensor::from_slice(&[421.0, 422.0], vec![2]);
        let m = median(&t);
        assert!(m >= 421.0 && m <= 422.0);
    }

    #[test]
    fn test_hist_stress_case_422() {
        let t = Tensor::from_slice(&[422.0, 423.0], vec![2]);
        let m = median(&t);
        assert!(m >= 422.0 && m <= 423.0);
    }

    #[test]
    fn test_hist_stress_case_423() {
        let t = Tensor::from_slice(&[423.0, 424.0], vec![2]);
        let m = median(&t);
        assert!(m >= 423.0 && m <= 424.0);
    }

    #[test]
    fn test_hist_stress_case_424() {
        let t = Tensor::from_slice(&[424.0, 425.0], vec![2]);
        let m = median(&t);
        assert!(m >= 424.0 && m <= 425.0);
    }

    #[test]
    fn test_hist_stress_case_425() {
        let t = Tensor::from_slice(&[425.0, 426.0], vec![2]);
        let m = median(&t);
        assert!(m >= 425.0 && m <= 426.0);
    }

    #[test]
    fn test_hist_stress_case_426() {
        let t = Tensor::from_slice(&[426.0, 427.0], vec![2]);
        let m = median(&t);
        assert!(m >= 426.0 && m <= 427.0);
    }

    #[test]
    fn test_hist_stress_case_427() {
        let t = Tensor::from_slice(&[427.0, 428.0], vec![2]);
        let m = median(&t);
        assert!(m >= 427.0 && m <= 428.0);
    }

    #[test]
    fn test_hist_stress_case_428() {
        let t = Tensor::from_slice(&[428.0, 429.0], vec![2]);
        let m = median(&t);
        assert!(m >= 428.0 && m <= 429.0);
    }

    #[test]
    fn test_hist_stress_case_429() {
        let t = Tensor::from_slice(&[429.0, 430.0], vec![2]);
        let m = median(&t);
        assert!(m >= 429.0 && m <= 430.0);
    }

    #[test]
    fn test_hist_stress_case_430() {
        let t = Tensor::from_slice(&[430.0, 431.0], vec![2]);
        let m = median(&t);
        assert!(m >= 430.0 && m <= 431.0);
    }

    #[test]
    fn test_hist_stress_case_431() {
        let t = Tensor::from_slice(&[431.0, 432.0], vec![2]);
        let m = median(&t);
        assert!(m >= 431.0 && m <= 432.0);
    }

    #[test]
    fn test_hist_stress_case_432() {
        let t = Tensor::from_slice(&[432.0, 433.0], vec![2]);
        let m = median(&t);
        assert!(m >= 432.0 && m <= 433.0);
    }

    #[test]
    fn test_hist_stress_case_433() {
        let t = Tensor::from_slice(&[433.0, 434.0], vec![2]);
        let m = median(&t);
        assert!(m >= 433.0 && m <= 434.0);
    }

    #[test]
    fn test_hist_stress_case_434() {
        let t = Tensor::from_slice(&[434.0, 435.0], vec![2]);
        let m = median(&t);
        assert!(m >= 434.0 && m <= 435.0);
    }

    #[test]
    fn test_hist_stress_case_435() {
        let t = Tensor::from_slice(&[435.0, 436.0], vec![2]);
        let m = median(&t);
        assert!(m >= 435.0 && m <= 436.0);
    }

    #[test]
    fn test_hist_stress_case_436() {
        let t = Tensor::from_slice(&[436.0, 437.0], vec![2]);
        let m = median(&t);
        assert!(m >= 436.0 && m <= 437.0);
    }

    #[test]
    fn test_hist_stress_case_437() {
        let t = Tensor::from_slice(&[437.0, 438.0], vec![2]);
        let m = median(&t);
        assert!(m >= 437.0 && m <= 438.0);
    }

    #[test]
    fn test_hist_stress_case_438() {
        let t = Tensor::from_slice(&[438.0, 439.0], vec![2]);
        let m = median(&t);
        assert!(m >= 438.0 && m <= 439.0);
    }

    #[test]
    fn test_hist_stress_case_439() {
        let t = Tensor::from_slice(&[439.0, 440.0], vec![2]);
        let m = median(&t);
        assert!(m >= 439.0 && m <= 440.0);
    }

    #[test]
    fn test_hist_stress_case_440() {
        let t = Tensor::from_slice(&[440.0, 441.0], vec![2]);
        let m = median(&t);
        assert!(m >= 440.0 && m <= 441.0);
    }

    #[test]
    fn test_hist_stress_case_441() {
        let t = Tensor::from_slice(&[441.0, 442.0], vec![2]);
        let m = median(&t);
        assert!(m >= 441.0 && m <= 442.0);
    }

    #[test]
    fn test_hist_stress_case_442() {
        let t = Tensor::from_slice(&[442.0, 443.0], vec![2]);
        let m = median(&t);
        assert!(m >= 442.0 && m <= 443.0);
    }

    #[test]
    fn test_hist_stress_case_443() {
        let t = Tensor::from_slice(&[443.0, 444.0], vec![2]);
        let m = median(&t);
        assert!(m >= 443.0 && m <= 444.0);
    }

    #[test]
    fn test_hist_stress_case_444() {
        let t = Tensor::from_slice(&[444.0, 445.0], vec![2]);
        let m = median(&t);
        assert!(m >= 444.0 && m <= 445.0);
    }

    #[test]
    fn test_hist_stress_case_445() {
        let t = Tensor::from_slice(&[445.0, 446.0], vec![2]);
        let m = median(&t);
        assert!(m >= 445.0 && m <= 446.0);
    }

    #[test]
    fn test_hist_stress_case_446() {
        let t = Tensor::from_slice(&[446.0, 447.0], vec![2]);
        let m = median(&t);
        assert!(m >= 446.0 && m <= 447.0);
    }

    #[test]
    fn test_hist_stress_case_447() {
        let t = Tensor::from_slice(&[447.0, 448.0], vec![2]);
        let m = median(&t);
        assert!(m >= 447.0 && m <= 448.0);
    }

    #[test]
    fn test_hist_stress_case_448() {
        let t = Tensor::from_slice(&[448.0, 449.0], vec![2]);
        let m = median(&t);
        assert!(m >= 448.0 && m <= 449.0);
    }

    #[test]
    fn test_hist_stress_case_449() {
        let t = Tensor::from_slice(&[449.0, 450.0], vec![2]);
        let m = median(&t);
        assert!(m >= 449.0 && m <= 450.0);
    }

    #[test]
    fn test_hist_stress_case_450() {
        let t = Tensor::from_slice(&[450.0, 451.0], vec![2]);
        let m = median(&t);
        assert!(m >= 450.0 && m <= 451.0);
    }

    #[test]
    fn test_hist_stress_case_451() {
        let t = Tensor::from_slice(&[451.0, 452.0], vec![2]);
        let m = median(&t);
        assert!(m >= 451.0 && m <= 452.0);
    }

    #[test]
    fn test_hist_stress_case_452() {
        let t = Tensor::from_slice(&[452.0, 453.0], vec![2]);
        let m = median(&t);
        assert!(m >= 452.0 && m <= 453.0);
    }

    #[test]
    fn test_hist_stress_case_453() {
        let t = Tensor::from_slice(&[453.0, 454.0], vec![2]);
        let m = median(&t);
        assert!(m >= 453.0 && m <= 454.0);
    }

    #[test]
    fn test_hist_stress_case_454() {
        let t = Tensor::from_slice(&[454.0, 455.0], vec![2]);
        let m = median(&t);
        assert!(m >= 454.0 && m <= 455.0);
    }

    #[test]
    fn test_hist_stress_case_455() {
        let t = Tensor::from_slice(&[455.0, 456.0], vec![2]);
        let m = median(&t);
        assert!(m >= 455.0 && m <= 456.0);
    }

    #[test]
    fn test_hist_stress_case_456() {
        let t = Tensor::from_slice(&[456.0, 457.0], vec![2]);
        let m = median(&t);
        assert!(m >= 456.0 && m <= 457.0);
    }

    #[test]
    fn test_hist_stress_case_457() {
        let t = Tensor::from_slice(&[457.0, 458.0], vec![2]);
        let m = median(&t);
        assert!(m >= 457.0 && m <= 458.0);
    }

    #[test]
    fn test_hist_stress_case_458() {
        let t = Tensor::from_slice(&[458.0, 459.0], vec![2]);
        let m = median(&t);
        assert!(m >= 458.0 && m <= 459.0);
    }

    #[test]
    fn test_hist_stress_case_459() {
        let t = Tensor::from_slice(&[459.0, 460.0], vec![2]);
        let m = median(&t);
        assert!(m >= 459.0 && m <= 460.0);
    }

    #[test]
    fn test_hist_stress_case_460() {
        let t = Tensor::from_slice(&[460.0, 461.0], vec![2]);
        let m = median(&t);
        assert!(m >= 460.0 && m <= 461.0);
    }

    #[test]
    fn test_hist_stress_case_461() {
        let t = Tensor::from_slice(&[461.0, 462.0], vec![2]);
        let m = median(&t);
        assert!(m >= 461.0 && m <= 462.0);
    }

    #[test]
    fn test_hist_stress_case_462() {
        let t = Tensor::from_slice(&[462.0, 463.0], vec![2]);
        let m = median(&t);
        assert!(m >= 462.0 && m <= 463.0);
    }

    #[test]
    fn test_hist_stress_case_463() {
        let t = Tensor::from_slice(&[463.0, 464.0], vec![2]);
        let m = median(&t);
        assert!(m >= 463.0 && m <= 464.0);
    }

    #[test]
    fn test_hist_stress_case_464() {
        let t = Tensor::from_slice(&[464.0, 465.0], vec![2]);
        let m = median(&t);
        assert!(m >= 464.0 && m <= 465.0);
    }

    #[test]
    fn test_hist_stress_case_465() {
        let t = Tensor::from_slice(&[465.0, 466.0], vec![2]);
        let m = median(&t);
        assert!(m >= 465.0 && m <= 466.0);
    }

    #[test]
    fn test_hist_stress_case_466() {
        let t = Tensor::from_slice(&[466.0, 467.0], vec![2]);
        let m = median(&t);
        assert!(m >= 466.0 && m <= 467.0);
    }

    #[test]
    fn test_hist_stress_case_467() {
        let t = Tensor::from_slice(&[467.0, 468.0], vec![2]);
        let m = median(&t);
        assert!(m >= 467.0 && m <= 468.0);
    }

    #[test]
    fn test_hist_stress_case_468() {
        let t = Tensor::from_slice(&[468.0, 469.0], vec![2]);
        let m = median(&t);
        assert!(m >= 468.0 && m <= 469.0);
    }

    #[test]
    fn test_hist_stress_case_469() {
        let t = Tensor::from_slice(&[469.0, 470.0], vec![2]);
        let m = median(&t);
        assert!(m >= 469.0 && m <= 470.0);
    }
}
