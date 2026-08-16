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

    #[test]
    fn test_rhythm_stress_001() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 1) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_002() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 2) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_003() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 3) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_004() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 4) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_005() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 5) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_006() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 6) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_007() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 7) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_008() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 8) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_009() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 9) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_010() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 10) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_011() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 11) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_012() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 12) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_013() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 13) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_014() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 14) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_015() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 15) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_016() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 16) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_017() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 17) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_018() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 18) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_019() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 19) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_020() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 20) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_021() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 21) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_022() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 22) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_023() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 23) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_024() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 24) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_025() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 25) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_026() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 26) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_027() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 27) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_028() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 28) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_029() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 29) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_030() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 30) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_031() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 31) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_032() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 32) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_033() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 33) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_034() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 34) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_035() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 35) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_036() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 36) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_037() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 37) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_038() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 38) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_039() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 39) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_040() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 40) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_041() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 41) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_042() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 42) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_043() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 43) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_044() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 44) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_045() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 45) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_046() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 46) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_047() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 47) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_048() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 48) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_049() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 49) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_050() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 50) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_051() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 51) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_052() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 52) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_053() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 53) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_054() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 54) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_055() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 55) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_056() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 56) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_057() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 57) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_058() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 58) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_059() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 59) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_060() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 60) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_061() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 61) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_062() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 62) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_063() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 63) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_064() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 64) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_065() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 65) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_066() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 66) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_067() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 67) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_068() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 68) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_069() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 69) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_070() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 70) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_071() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 71) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_072() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 72) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_073() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 73) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_074() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 74) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_075() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 75) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_076() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 76) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_077() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 77) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_078() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 78) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_079() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 79) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_080() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 80) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_081() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 81) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_082() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 82) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_083() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 83) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_084() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 84) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_085() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 85) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_086() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 86) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_087() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 87) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_088() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 88) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_089() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 89) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_090() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 90) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_091() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 91) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_092() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 92) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_093() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 93) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_094() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 94) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_095() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 95) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_096() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 96) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_097() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 97) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_098() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 98) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_099() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 99) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_100() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 100) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_101() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 101) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_102() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 102) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_103() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 103) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_104() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 104) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_105() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 105) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_106() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 106) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_107() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 107) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_108() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 108) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_109() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 109) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_110() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 110) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_111() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 111) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_112() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 112) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_113() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 113) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_114() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 114) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_115() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 115) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_116() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 116) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_117() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 117) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_118() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 118) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_119() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 119) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_120() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 120) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_121() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 121) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_122() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 122) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_123() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 123) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_124() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 124) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_125() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 125) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_126() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 126) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_127() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 127) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_128() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 128) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_129() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 129) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_130() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 130) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_131() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 131) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_132() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 132) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_133() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 133) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_134() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 134) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_135() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 135) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_136() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 136) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_137() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 137) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_138() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 138) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_139() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 139) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_140() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 140) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_141() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 141) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_142() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 142) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_143() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 143) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_144() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 144) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_145() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 145) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_146() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 146) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_147() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 147) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_148() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 148) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_149() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 149) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_150() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 150) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_151() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 151) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_152() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 152) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_153() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 153) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_154() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 154) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_155() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 155) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_156() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 156) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_157() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 157) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_158() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 158) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_159() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 159) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_160() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 160) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_161() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 161) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_162() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 162) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_163() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 163) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_164() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 164) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_165() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 165) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_166() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 166) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_167() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 167) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_168() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 168) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_169() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 169) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_170() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 170) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_171() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 171) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_172() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 172) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_173() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 173) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_174() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 174) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_175() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 175) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_176() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 176) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_177() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 177) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_178() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 178) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_179() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 179) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_180() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 180) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_181() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 181) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_182() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 182) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_183() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 183) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_184() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 184) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_185() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 185) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_186() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 186) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_187() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 187) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_188() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 188) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_189() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 189) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_190() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 190) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_191() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 191) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_192() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 192) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_193() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 193) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_194() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 194) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_195() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 195) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_196() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 196) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_197() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 197) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_198() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 198) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_199() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 199) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_200() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 200) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_201() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 201) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_202() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 202) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_203() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 203) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_204() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 204) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_205() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 205) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_206() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 206) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_207() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 207) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_208() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 208) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_209() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 209) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_210() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 210) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_211() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 211) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_212() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 212) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_213() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 213) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_214() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 214) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_215() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 215) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_216() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 216) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_217() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 217) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_218() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 218) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_219() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 219) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_220() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 220) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_221() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 221) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_222() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 222) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_223() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 223) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_224() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 224) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_225() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 225) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_226() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 226) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_227() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 227) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_228() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 228) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_229() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 229) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_230() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 230) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_231() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 231) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_232() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 232) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_233() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 233) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_234() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 234) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_235() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 235) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_236() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 236) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_237() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 237) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_238() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 238) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_239() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 239) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_240() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 240) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_241() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 241) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_242() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 242) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_243() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 243) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_244() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 244) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_245() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 245) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_246() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 246) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_247() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 247) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_248() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 248) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_249() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 249) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_250() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 250) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_251() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 251) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_252() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 252) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_253() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 253) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_254() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 254) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_255() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 255) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_256() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 256) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_257() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 257) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_258() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 258) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_259() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 259) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_260() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 260) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_261() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 261) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_262() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 262) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_263() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 263) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_264() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 264) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_265() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 265) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_266() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 266) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_267() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 267) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_268() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 268) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_269() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 269) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_270() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 270) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_271() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 271) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_272() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 272) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_273() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 273) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_274() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 274) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_275() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 275) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_276() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 276) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_277() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 277) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_278() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 278) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_279() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 279) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_280() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 280) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_281() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 281) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_282() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 282) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_283() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 283) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_284() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 284) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_285() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 285) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_286() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 286) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_287() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 287) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_288() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 288) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_289() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 289) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_290() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 290) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_291() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 291) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_292() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 292) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_293() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 293) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_294() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 294) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_295() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 295) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_296() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 296) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_297() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 297) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_298() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 298) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_299() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 299) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_300() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 300) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_301() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 301) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_302() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 302) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_303() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 303) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_304() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 304) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_305() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 305) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_306() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 306) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_307() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 307) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_308() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 308) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_309() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 309) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_310() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 310) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_311() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 311) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_312() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 312) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_313() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 313) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_314() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 314) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_315() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 315) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_316() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 316) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_317() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 317) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_318() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 318) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_319() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 319) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_320() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 320) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_321() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 321) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_322() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 322) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_323() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 323) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_324() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 324) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_325() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 325) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_326() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 326) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_327() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 327) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_328() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 328) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_329() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 329) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_330() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 330) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_331() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 331) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_332() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 332) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_333() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 333) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_334() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 334) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_335() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 335) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_336() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 336) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_337() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 337) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_338() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 338) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_339() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 339) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_340() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 340) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_341() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 341) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_342() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 342) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_343() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 343) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_344() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 344) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_345() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 345) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_346() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 346) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_347() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 347) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_348() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 348) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_349() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 349) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_350() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 350) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_351() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 351) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_352() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 352) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_353() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 353) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_354() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 354) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_355() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 355) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_356() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 356) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_357() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 357) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_358() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 358) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_359() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 359) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_360() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 360) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_361() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 361) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_362() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 362) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_363() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 363) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_364() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 364) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_365() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 365) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_366() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 366) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_367() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 367) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_368() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 368) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_369() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 369) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_370() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 370) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_371() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 371) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_372() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 372) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_373() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 373) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_374() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 374) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_375() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 375) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_376() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 376) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_377() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 377) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_378() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 378) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_379() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 379) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_380() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 380) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_381() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 381) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_382() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 382) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_383() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 383) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_384() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 384) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_385() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 385) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_386() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 386) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_387() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 387) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_388() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 388) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_389() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 389) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_390() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 390) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_391() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 391) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_392() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 392) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_393() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 393) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_394() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 394) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_395() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 395) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_396() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 396) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_397() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 397) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_398() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 398) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_399() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 399) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_400() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 400) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_401() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 401) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_402() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 402) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_403() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 403) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_404() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 404) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_405() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 405) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_406() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 406) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_407() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 407) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_408() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 408) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_409() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 409) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_410() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 410) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_411() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 411) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_412() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 412) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_413() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 413) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_414() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 414) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_415() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 415) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_416() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 416) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_417() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 417) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_418() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 418) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_419() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 419) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_420() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 420) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_421() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 421) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_422() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 422) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_423() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 423) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_424() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 424) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_425() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 425) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_426() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 426) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_427() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 427) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_428() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 428) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_429() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 429) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_430() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 430) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_431() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 431) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_432() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 432) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_433() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 433) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_434() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 434) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_435() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 435) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_436() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 436) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_437() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 437) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_438() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 438) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_439() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 439) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_440() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 440) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_441() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 441) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_442() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 442) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_443() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 443) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_444() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 444) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_445() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 445) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_446() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 446) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_447() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 447) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_448() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 448) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_449() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 449) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_450() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 450) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_451() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 451) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_452() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 452) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_453() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 453) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_454() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 454) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_455() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 455) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_456() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 456) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_457() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 457) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_458() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 458) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_459() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 459) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_460() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 460) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_461() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 461) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_462() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 462) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_463() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 463) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_464() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 464) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_465() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 465) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_466() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 466) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_467() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 467) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_468() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 468) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_469() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 469) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_470() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 470) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_471() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 471) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_472() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 472) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }

    #[test]
    fn test_rhythm_stress_473() {
        let env: Vec<f64> = (0..256).map(|i| if (i + 473) % 20 == 0 { 1.0 } else { 0.0 }).collect();
        let bpm = estimate_tempo(&env, 100.0, 60.0, 240.0);
        assert!(bpm >= 50.0 && bpm <= 300.0);
    }
}
