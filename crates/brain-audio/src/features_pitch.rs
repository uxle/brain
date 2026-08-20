//! # Fundamental Frequency ($F_0$) and Pitch Tracking
//!
//! Advanced pitch tracking algorithms:
//! * Autocorrelation-based pitch detector with parabolic interpolation
//! * Real Cepstrum peak pitch detector
//! * Continuous pitch contour smoothing and voicing probability

/// Estimates fundamental pitch ($F_0$) via normalized autocorrelation method with parabolic refinement.
pub fn pitch_autocorr(frame: &[f64], sample_rate: u32, min_f0: f64, max_f0: f64) -> Option<f64> {
    let n = frame.len();
    if n < 64 {
        return None;
    }
    let min_lag = (sample_rate as f64 / max_f0).floor() as usize;
    let max_lag = (sample_rate as f64 / min_f0).ceil() as usize;

    let mut corrs = vec![0.0; max_lag + 2];
    let mut best_lag = 0;
    let mut max_corr = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag.min(n / 2) {
        let mut corr = 0.0;
        let mut norm_x = 0.0;
        let mut norm_y = 0.0;
        for i in 0..n - lag {
            corr += frame[i] * frame[i + lag];
            norm_x += frame[i] * frame[i];
            norm_y += frame[i + lag] * frame[i + lag];
        }
        let norm = (norm_x * norm_y).sqrt();
        let norm_corr = if norm > 1e-10 { corr / norm } else { 0.0 };
        corrs[lag] = norm_corr;

        if norm_corr > max_corr {
            max_corr = norm_corr;
            best_lag = lag;
        }
    }

    if max_corr > 0.3 && best_lag > min_lag && best_lag < max_lag {
        // Parabolic interpolation for sub-sample lag accuracy
        let alpha = corrs[best_lag - 1];
        let beta = corrs[best_lag];
        let gamma = corrs[best_lag + 1];
        let denom = 2.0 * (2.0 * beta - alpha - gamma);
        let delta = if denom.abs() > 1e-9 {
            (gamma - alpha) / denom
        } else {
            0.0
        };
        let refined_lag = best_lag as f64 + delta.clamp(-0.5, 0.5);
        Some(sample_rate as f64 / refined_lag)
    } else if max_corr > 0.3 && best_lag > 0 {
        Some(sample_rate as f64 / best_lag as f64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
