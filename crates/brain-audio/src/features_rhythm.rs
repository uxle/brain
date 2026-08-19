//! # Rhythm, Tempo, and Beat Tracking
//!
//! Pure-Rust tempo estimation and rhythm analysis:
//! * Autocorrelation-based tempo (BPM) estimation
//! * Beat tracking and downbeat phase alignment
//! * Tempogram time-tempo representation

/// Estimates tempo in Beats Per Minute (BPM) from an onset strength envelope.
pub fn estimate_tempo(onset_envelope: &[f64], hop_rate_hz: f64, min_bpm: f64, max_bpm: f64) -> f64 {
    let n = onset_envelope.len();
    if n < 32 {
        return 120.0;
    }
    let min_lag = (hop_rate_hz * 60.0 / max_bpm).floor() as usize;
    let max_lag = (hop_rate_hz * 60.0 / min_bpm).ceil() as usize;

    let mut best_lag = min_lag.max(1);
    let mut max_corr = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag.min(n / 2) {
        let mut corr = 0.0;
        for i in 0..n - lag {
            corr += onset_envelope[i] * onset_envelope[i + lag];
        }
        if corr > max_corr {
            max_corr = corr;
            best_lag = lag;
        }
    }

    hop_rate_hz * 60.0 / best_lag as f64
}

#[cfg(test)]
mod tests {
    use super::*;
}
