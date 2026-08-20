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
pub fn compute_calibration(
    confidences: &[f64],
    correctness: &[bool],
    num_bins: usize,
) -> CalibrationReport {
    let n = confidences.len().min(correctness.len());
    if n == 0 {
        return CalibrationReport::default();
    }

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
            if diff > mce {
                mce = diff;
            }

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
