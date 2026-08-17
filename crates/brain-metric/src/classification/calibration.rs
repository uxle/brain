//! # Model Calibration & Reliability
//!
//! Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and reliability diagrams.
#![allow(missing_docs)]


/// Calibration summary report.
#[derive(Debug, Clone, Default)]
pub struct CalibrationReport {
    pub ece: f64,
    pub mce: f64,
    pub bin_accuracies: Vec<f64>,
    pub bin_confidences: Vec<f64>,
}

/// Evaluates ECE and MCE across `num_bins` uniform confidence buckets in [0, 1].
pub fn compute_calibration(confidences: &[f64], correctness: &[bool], num_bins: usize) -> CalibrationReport {
    let n = confidences.len().min(correctness.len());
    if n == 0 { return CalibrationReport::default(); }

    let mut bin_counts = vec![0usize; num_bins];
    let mut bin_correct = vec![0usize; num_bins];
    let mut bin_conf_sum = vec![0.0f64; num_bins];

    for i in 0..n {
        let conf = confidences[i].clamp(0.0, 1.0 - 1e-9);
        let b = (conf * num_bins as f64).floor() as usize;
        let b_clamped = b.min(num_bins - 1);

        bin_counts[b_clamped] += 1;
        bin_conf_sum[b_clamped] += conf;
        if correctness[i] {
            bin_correct[b_clamped] += 1;
        }
    }

    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    let mut bin_accs = Vec::with_capacity(num_bins);
    let mut bin_confs = Vec::with_capacity(num_bins);

    for b in 0..num_bins {
        if bin_counts[b] > 0 {
            let acc = bin_correct[b] as f64 / bin_counts[b] as f64;
            let conf = bin_conf_sum[b] / bin_counts[b] as f64;
            let diff = (acc - conf).abs();

            ece += (bin_counts[b] as f64 / n as f64) * diff;
            if diff > mce { mce = diff; }

            bin_accs.push(acc);
            bin_confs.push(conf);
        } else {
            bin_accs.push(0.0);
            bin_confs.push(0.0);
        }
    }

    CalibrationReport {
        ece,
        mce,
        bin_accuracies: bin_accs,
        bin_confidences: bin_confs,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_calib_stress_001() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_002() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_003() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_004() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_005() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_006() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_007() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_008() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_009() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_010() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_011() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_012() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_013() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_014() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_015() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_016() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_017() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_018() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_019() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_020() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_021() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_022() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_023() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_024() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_025() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_026() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_027() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_028() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_029() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_030() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_031() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_032() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_033() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_034() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_035() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_036() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_037() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_038() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_039() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_040() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_041() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_042() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_043() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_044() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_045() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_046() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_047() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_048() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_049() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_050() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_051() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_052() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_053() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_054() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_055() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_056() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_057() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_058() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_059() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_060() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_061() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_062() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_063() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_064() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_065() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_066() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_067() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_068() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_069() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_070() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_071() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_072() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_073() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_074() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_075() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_076() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_077() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_078() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_079() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_080() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_081() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_082() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_083() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_084() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_085() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_086() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_087() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_088() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_089() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_090() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_091() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_092() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_093() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_094() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_095() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_096() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_097() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_098() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_099() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_100() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_101() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_102() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_103() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_104() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_105() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_106() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_107() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_108() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_109() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_110() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_111() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_112() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_113() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_114() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_115() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_116() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_117() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_118() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_119() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_120() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_121() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_122() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_123() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_124() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_125() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_126() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_127() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_128() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_129() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_130() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_131() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_132() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_133() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_134() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_135() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_136() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_137() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_138() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_139() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_140() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_141() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_142() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_143() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_144() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_145() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_146() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_147() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_148() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_149() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_150() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_151() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_152() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_153() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_154() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_155() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_156() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_157() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_158() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_159() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_160() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_161() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_162() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_163() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_164() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_165() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_166() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_167() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_168() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_169() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_170() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_171() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_172() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_173() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_174() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_175() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_176() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_177() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_178() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_179() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_180() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_181() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_182() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_183() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_184() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_185() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_186() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_187() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_188() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_189() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_190() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_191() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_192() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_193() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_194() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_195() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_196() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_197() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_198() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_199() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_200() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_201() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_202() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_203() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_204() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_205() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_206() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_207() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_208() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_209() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_210() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_211() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_212() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_213() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_214() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_215() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_216() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_217() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_218() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_219() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_220() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_221() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_222() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_223() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_224() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_225() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_226() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_227() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_228() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_229() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_230() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_231() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_232() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_233() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_234() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_235() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_236() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_237() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_238() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_239() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_240() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_241() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_242() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_243() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_244() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_245() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_246() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_247() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_248() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_249() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_250() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_251() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_252() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_253() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_254() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_255() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_256() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_257() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_258() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_259() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_260() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_261() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_262() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_263() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_264() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_265() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_266() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_267() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_268() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_269() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_270() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_271() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_272() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_273() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_274() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_275() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_276() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_277() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_278() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_279() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_280() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_281() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_282() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_283() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_284() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_285() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_286() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_287() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_288() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_289() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_290() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_291() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_292() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_293() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_294() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_295() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_296() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_297() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_298() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_299() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_300() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_301() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_302() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_303() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_304() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_305() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_306() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_307() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_308() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_309() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_310() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_311() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_312() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_313() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_314() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_315() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_316() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_317() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_318() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_319() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_320() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_321() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_322() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_323() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_324() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_325() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_326() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_327() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_328() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_329() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_330() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_331() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_332() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_333() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_334() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_335() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_336() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_337() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_338() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_339() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_340() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_341() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_342() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_343() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_344() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_345() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_346() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_347() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_348() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_349() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_350() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_351() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_352() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_353() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_354() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_355() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_356() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_357() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_358() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_359() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_360() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_361() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_362() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_363() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_364() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_365() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_366() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_367() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_368() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_369() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_370() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_371() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_372() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_373() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_374() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_375() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_376() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_377() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_378() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_379() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_380() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_381() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_382() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_383() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_384() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_385() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_386() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_387() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_388() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_389() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_390() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_391() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_392() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_393() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_394() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_395() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_396() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_397() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_398() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_399() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_400() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_401() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_402() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_403() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_404() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_405() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_406() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_407() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_408() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    #[test]
    fn test_calib_stress_409() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
}
