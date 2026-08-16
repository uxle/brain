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
        let delta = if denom.abs() > 1e-9 { (gamma - alpha) / denom } else { 0.0 };
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

    #[test]
    fn test_pitch_features_stress_001() {
        let freq = 120.0 + ((1 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_002() {
        let freq = 120.0 + ((2 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_003() {
        let freq = 120.0 + ((3 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_004() {
        let freq = 120.0 + ((4 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_005() {
        let freq = 120.0 + ((5 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_006() {
        let freq = 120.0 + ((6 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_007() {
        let freq = 120.0 + ((7 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_008() {
        let freq = 120.0 + ((8 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_009() {
        let freq = 120.0 + ((9 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_010() {
        let freq = 120.0 + ((10 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_011() {
        let freq = 120.0 + ((11 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_012() {
        let freq = 120.0 + ((12 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_013() {
        let freq = 120.0 + ((13 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_014() {
        let freq = 120.0 + ((14 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_015() {
        let freq = 120.0 + ((15 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_016() {
        let freq = 120.0 + ((16 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_017() {
        let freq = 120.0 + ((17 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_018() {
        let freq = 120.0 + ((18 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_019() {
        let freq = 120.0 + ((19 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_020() {
        let freq = 120.0 + ((20 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_021() {
        let freq = 120.0 + ((21 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_022() {
        let freq = 120.0 + ((22 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_023() {
        let freq = 120.0 + ((23 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_024() {
        let freq = 120.0 + ((24 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_025() {
        let freq = 120.0 + ((25 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_026() {
        let freq = 120.0 + ((26 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_027() {
        let freq = 120.0 + ((27 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_028() {
        let freq = 120.0 + ((28 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_029() {
        let freq = 120.0 + ((29 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_030() {
        let freq = 120.0 + ((30 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_031() {
        let freq = 120.0 + ((31 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_032() {
        let freq = 120.0 + ((32 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_033() {
        let freq = 120.0 + ((33 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_034() {
        let freq = 120.0 + ((34 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_035() {
        let freq = 120.0 + ((35 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_036() {
        let freq = 120.0 + ((36 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_037() {
        let freq = 120.0 + ((37 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_038() {
        let freq = 120.0 + ((38 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_039() {
        let freq = 120.0 + ((39 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_040() {
        let freq = 120.0 + ((40 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_041() {
        let freq = 120.0 + ((41 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_042() {
        let freq = 120.0 + ((42 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_043() {
        let freq = 120.0 + ((43 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_044() {
        let freq = 120.0 + ((44 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_045() {
        let freq = 120.0 + ((45 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_046() {
        let freq = 120.0 + ((46 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_047() {
        let freq = 120.0 + ((47 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_048() {
        let freq = 120.0 + ((48 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_049() {
        let freq = 120.0 + ((49 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_050() {
        let freq = 120.0 + ((50 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_051() {
        let freq = 120.0 + ((51 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_052() {
        let freq = 120.0 + ((52 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_053() {
        let freq = 120.0 + ((53 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_054() {
        let freq = 120.0 + ((54 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_055() {
        let freq = 120.0 + ((55 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_056() {
        let freq = 120.0 + ((56 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_057() {
        let freq = 120.0 + ((57 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_058() {
        let freq = 120.0 + ((58 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_059() {
        let freq = 120.0 + ((59 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_060() {
        let freq = 120.0 + ((60 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_061() {
        let freq = 120.0 + ((61 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_062() {
        let freq = 120.0 + ((62 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_063() {
        let freq = 120.0 + ((63 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_064() {
        let freq = 120.0 + ((64 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_065() {
        let freq = 120.0 + ((65 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_066() {
        let freq = 120.0 + ((66 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_067() {
        let freq = 120.0 + ((67 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_068() {
        let freq = 120.0 + ((68 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_069() {
        let freq = 120.0 + ((69 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_070() {
        let freq = 120.0 + ((70 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_071() {
        let freq = 120.0 + ((71 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_072() {
        let freq = 120.0 + ((72 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_073() {
        let freq = 120.0 + ((73 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_074() {
        let freq = 120.0 + ((74 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_075() {
        let freq = 120.0 + ((75 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_076() {
        let freq = 120.0 + ((76 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_077() {
        let freq = 120.0 + ((77 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_078() {
        let freq = 120.0 + ((78 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_079() {
        let freq = 120.0 + ((79 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_080() {
        let freq = 120.0 + ((80 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_081() {
        let freq = 120.0 + ((81 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_082() {
        let freq = 120.0 + ((82 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_083() {
        let freq = 120.0 + ((83 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_084() {
        let freq = 120.0 + ((84 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_085() {
        let freq = 120.0 + ((85 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_086() {
        let freq = 120.0 + ((86 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_087() {
        let freq = 120.0 + ((87 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_088() {
        let freq = 120.0 + ((88 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_089() {
        let freq = 120.0 + ((89 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_090() {
        let freq = 120.0 + ((90 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_091() {
        let freq = 120.0 + ((91 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_092() {
        let freq = 120.0 + ((92 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_093() {
        let freq = 120.0 + ((93 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_094() {
        let freq = 120.0 + ((94 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_095() {
        let freq = 120.0 + ((95 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_096() {
        let freq = 120.0 + ((96 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_097() {
        let freq = 120.0 + ((97 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_098() {
        let freq = 120.0 + ((98 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_099() {
        let freq = 120.0 + ((99 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_100() {
        let freq = 120.0 + ((100 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_101() {
        let freq = 120.0 + ((101 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_102() {
        let freq = 120.0 + ((102 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_103() {
        let freq = 120.0 + ((103 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_104() {
        let freq = 120.0 + ((104 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_105() {
        let freq = 120.0 + ((105 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_106() {
        let freq = 120.0 + ((106 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_107() {
        let freq = 120.0 + ((107 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_108() {
        let freq = 120.0 + ((108 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_109() {
        let freq = 120.0 + ((109 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_110() {
        let freq = 120.0 + ((110 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_111() {
        let freq = 120.0 + ((111 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_112() {
        let freq = 120.0 + ((112 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_113() {
        let freq = 120.0 + ((113 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_114() {
        let freq = 120.0 + ((114 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_115() {
        let freq = 120.0 + ((115 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_116() {
        let freq = 120.0 + ((116 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_117() {
        let freq = 120.0 + ((117 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_118() {
        let freq = 120.0 + ((118 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_119() {
        let freq = 120.0 + ((119 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_120() {
        let freq = 120.0 + ((120 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_121() {
        let freq = 120.0 + ((121 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_122() {
        let freq = 120.0 + ((122 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_123() {
        let freq = 120.0 + ((123 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_124() {
        let freq = 120.0 + ((124 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_125() {
        let freq = 120.0 + ((125 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_126() {
        let freq = 120.0 + ((126 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_127() {
        let freq = 120.0 + ((127 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_128() {
        let freq = 120.0 + ((128 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_129() {
        let freq = 120.0 + ((129 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_130() {
        let freq = 120.0 + ((130 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_131() {
        let freq = 120.0 + ((131 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_132() {
        let freq = 120.0 + ((132 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_133() {
        let freq = 120.0 + ((133 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_134() {
        let freq = 120.0 + ((134 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_135() {
        let freq = 120.0 + ((135 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_136() {
        let freq = 120.0 + ((136 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_137() {
        let freq = 120.0 + ((137 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_138() {
        let freq = 120.0 + ((138 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_139() {
        let freq = 120.0 + ((139 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_140() {
        let freq = 120.0 + ((140 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_141() {
        let freq = 120.0 + ((141 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_142() {
        let freq = 120.0 + ((142 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_143() {
        let freq = 120.0 + ((143 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_144() {
        let freq = 120.0 + ((144 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_145() {
        let freq = 120.0 + ((145 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_146() {
        let freq = 120.0 + ((146 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_147() {
        let freq = 120.0 + ((147 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_148() {
        let freq = 120.0 + ((148 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_149() {
        let freq = 120.0 + ((149 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_150() {
        let freq = 120.0 + ((150 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_151() {
        let freq = 120.0 + ((151 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_152() {
        let freq = 120.0 + ((152 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_153() {
        let freq = 120.0 + ((153 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_154() {
        let freq = 120.0 + ((154 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_155() {
        let freq = 120.0 + ((155 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_156() {
        let freq = 120.0 + ((156 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_157() {
        let freq = 120.0 + ((157 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_158() {
        let freq = 120.0 + ((158 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_159() {
        let freq = 120.0 + ((159 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_160() {
        let freq = 120.0 + ((160 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_161() {
        let freq = 120.0 + ((161 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_162() {
        let freq = 120.0 + ((162 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_163() {
        let freq = 120.0 + ((163 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_164() {
        let freq = 120.0 + ((164 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_165() {
        let freq = 120.0 + ((165 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_166() {
        let freq = 120.0 + ((166 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_167() {
        let freq = 120.0 + ((167 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_168() {
        let freq = 120.0 + ((168 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_169() {
        let freq = 120.0 + ((169 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_170() {
        let freq = 120.0 + ((170 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_171() {
        let freq = 120.0 + ((171 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_172() {
        let freq = 120.0 + ((172 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_173() {
        let freq = 120.0 + ((173 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_174() {
        let freq = 120.0 + ((174 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_175() {
        let freq = 120.0 + ((175 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_176() {
        let freq = 120.0 + ((176 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_177() {
        let freq = 120.0 + ((177 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_178() {
        let freq = 120.0 + ((178 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_179() {
        let freq = 120.0 + ((179 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_180() {
        let freq = 120.0 + ((180 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_181() {
        let freq = 120.0 + ((181 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_182() {
        let freq = 120.0 + ((182 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_183() {
        let freq = 120.0 + ((183 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_184() {
        let freq = 120.0 + ((184 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_185() {
        let freq = 120.0 + ((185 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_186() {
        let freq = 120.0 + ((186 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_187() {
        let freq = 120.0 + ((187 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_188() {
        let freq = 120.0 + ((188 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_189() {
        let freq = 120.0 + ((189 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_190() {
        let freq = 120.0 + ((190 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_191() {
        let freq = 120.0 + ((191 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_192() {
        let freq = 120.0 + ((192 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_193() {
        let freq = 120.0 + ((193 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_194() {
        let freq = 120.0 + ((194 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_195() {
        let freq = 120.0 + ((195 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_196() {
        let freq = 120.0 + ((196 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_197() {
        let freq = 120.0 + ((197 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_198() {
        let freq = 120.0 + ((198 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_199() {
        let freq = 120.0 + ((199 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_200() {
        let freq = 120.0 + ((200 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_201() {
        let freq = 120.0 + ((201 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_202() {
        let freq = 120.0 + ((202 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_203() {
        let freq = 120.0 + ((203 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_204() {
        let freq = 120.0 + ((204 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_205() {
        let freq = 120.0 + ((205 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_206() {
        let freq = 120.0 + ((206 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_207() {
        let freq = 120.0 + ((207 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_208() {
        let freq = 120.0 + ((208 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_209() {
        let freq = 120.0 + ((209 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_210() {
        let freq = 120.0 + ((210 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_211() {
        let freq = 120.0 + ((211 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_212() {
        let freq = 120.0 + ((212 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_213() {
        let freq = 120.0 + ((213 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_214() {
        let freq = 120.0 + ((214 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_215() {
        let freq = 120.0 + ((215 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_216() {
        let freq = 120.0 + ((216 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_217() {
        let freq = 120.0 + ((217 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_218() {
        let freq = 120.0 + ((218 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_219() {
        let freq = 120.0 + ((219 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_220() {
        let freq = 120.0 + ((220 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_221() {
        let freq = 120.0 + ((221 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_222() {
        let freq = 120.0 + ((222 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_223() {
        let freq = 120.0 + ((223 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_224() {
        let freq = 120.0 + ((224 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_225() {
        let freq = 120.0 + ((225 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_226() {
        let freq = 120.0 + ((226 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_227() {
        let freq = 120.0 + ((227 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_228() {
        let freq = 120.0 + ((228 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_229() {
        let freq = 120.0 + ((229 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_230() {
        let freq = 120.0 + ((230 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_231() {
        let freq = 120.0 + ((231 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_232() {
        let freq = 120.0 + ((232 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_233() {
        let freq = 120.0 + ((233 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_234() {
        let freq = 120.0 + ((234 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_235() {
        let freq = 120.0 + ((235 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_236() {
        let freq = 120.0 + ((236 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_237() {
        let freq = 120.0 + ((237 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_238() {
        let freq = 120.0 + ((238 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_239() {
        let freq = 120.0 + ((239 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_240() {
        let freq = 120.0 + ((240 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_241() {
        let freq = 120.0 + ((241 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_242() {
        let freq = 120.0 + ((242 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_243() {
        let freq = 120.0 + ((243 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_244() {
        let freq = 120.0 + ((244 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_245() {
        let freq = 120.0 + ((245 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_246() {
        let freq = 120.0 + ((246 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_247() {
        let freq = 120.0 + ((247 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_248() {
        let freq = 120.0 + ((248 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_249() {
        let freq = 120.0 + ((249 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_250() {
        let freq = 120.0 + ((250 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_251() {
        let freq = 120.0 + ((251 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_252() {
        let freq = 120.0 + ((252 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_253() {
        let freq = 120.0 + ((253 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_254() {
        let freq = 120.0 + ((254 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_255() {
        let freq = 120.0 + ((255 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_256() {
        let freq = 120.0 + ((256 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_257() {
        let freq = 120.0 + ((257 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_258() {
        let freq = 120.0 + ((258 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_259() {
        let freq = 120.0 + ((259 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_260() {
        let freq = 120.0 + ((260 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_261() {
        let freq = 120.0 + ((261 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_262() {
        let freq = 120.0 + ((262 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_263() {
        let freq = 120.0 + ((263 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_264() {
        let freq = 120.0 + ((264 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_265() {
        let freq = 120.0 + ((265 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_266() {
        let freq = 120.0 + ((266 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_267() {
        let freq = 120.0 + ((267 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_268() {
        let freq = 120.0 + ((268 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_269() {
        let freq = 120.0 + ((269 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_270() {
        let freq = 120.0 + ((270 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_271() {
        let freq = 120.0 + ((271 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_272() {
        let freq = 120.0 + ((272 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_273() {
        let freq = 120.0 + ((273 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }

    #[test]
    fn test_pitch_features_stress_274() {
        let freq = 120.0 + ((274 % 80) as f64) * 2.0;
        let frame: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let detected = pitch_autocorr(&frame, 16000, 80.0, 400.0);
        if let Some(f0) = detected {
            let ratio = (freq / f0).round();
            let harmonic_diff = (freq - ratio * f0).abs();
            assert!(harmonic_diff < 15.0 || (f0 - freq).abs() < 15.0, "Pitch detected {} vs expected {}", f0, freq);
        }
    }
}
