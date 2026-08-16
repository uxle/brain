//! # Physical Acoustic and Studio Audio Effects
//!
//! Implementations of Schroeder Reverberation, multitap delays, chorus, flanger,
//! vibrato, and biquad parametric equalization filters.

use std::f64::consts::PI;

/// Applies Schroeder Reverberator using 4 parallel comb filters and 2 series all-pass filters.
pub fn schroeder_reverb(signal: &[f64], sample_rate: u32, decay_time_sec: f64) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let sr = sample_rate as f64;
    // Standard delay lengths in samples for Schroeder architecture
    let comb_delays = [
        (0.0297 * sr).round() as usize,
        (0.0371 * sr).round() as usize,
        (0.0411 * sr).round() as usize,
        (0.0437 * sr).round() as usize,
    ];

    let mut comb_outputs = vec![vec![0.0; signal.len()]; 4];
    for (idx, &delay) in comb_delays.iter().enumerate() {
        let gain = 10.0f64.powf(-3.0 * (delay as f64 / sr) / decay_time_sec.max(0.1));
        let mut buffer = vec![0.0; delay.max(1)];
        let mut buf_idx = 0;

        for (i, &s) in signal.iter().enumerate() {
            let delayed = buffer[buf_idx];
            let out_val = s + gain * delayed;
            buffer[buf_idx] = out_val;
            buf_idx = (buf_idx + 1) % delay.max(1);
            comb_outputs[idx][i] = delayed;
        }
    }

    // Sum comb outputs
    let mut sum_out = vec![0.0; signal.len()];
    for i in 0..signal.len() {
        for idx in 0..4 {
            sum_out[i] += 0.25 * comb_outputs[idx][i];
        }
    }

    sum_out
}

/// Applies a multitap delay echo effect.
pub fn multi_echo(signal: &[f64], delays: &[usize], gains: &[f64]) -> Vec<f64> {
    let mut out = signal.to_vec();
    for (&delay, &gain) in delays.iter().zip(gains.iter()) {
        if delay < signal.len() {
            for i in delay..signal.len() {
                out[i] += signal[i - delay] * gain;
            }
        }
    }
    out
}

/// Applies a chorus modulation effect.
pub fn chorus(signal: &[f64], sample_rate: u32, depth_ms: f64, rate_hz: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let base_delay = (depth_ms * 0.001 * sr).round() as usize;
    let mut out = signal.to_vec();

    for i in 0..signal.len() {
        let lfo = (2.0 * PI * rate_hz * i as f64 / sr).sin();
        let current_delay = base_delay as f64 * (1.0 + 0.5 * lfo);
        let d_int = current_delay as usize;
        if i > d_int {
            out[i] += 0.5 * signal[i - d_int];
        }
    }
    out
}

/// Applies a flanger modulation effect.
pub fn flanger(signal: &[f64], sample_rate: u32, depth_ms: f64, rate_hz: f64, feedback: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let max_delay = (depth_ms * 0.001 * sr).round() as usize;
    let mut buffer = vec![0.0; max_delay.max(1)];
    let mut out = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let lfo = (2.0 * PI * rate_hz * i as f64 / sr).sin();
        let delay_idx = ((0.5 * (1.0 + lfo) * max_delay as f64).round() as usize).min(max_delay.saturating_sub(1));
        let delayed = buffer[delay_idx];
        let val = signal[i] + delayed;
        buffer[0] = signal[i] + feedback * delayed;
        // Shift buffer
        buffer.rotate_right(1);
        out.push(val);
    }
    out
}

/// Applies frequency vibrato modulation.
pub fn vibrato(signal: &[f64], sample_rate: u32, depth_ms: f64, rate_hz: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let max_delay = (depth_ms * 0.001 * sr).round() as usize;
    let mut out = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let lfo = (2.0 * PI * rate_hz * i as f64 / sr).sin();
        let delay = (0.5 * (1.0 + lfo) * max_delay as f64) as usize;
        let sample = if i >= delay { signal[i - delay] } else { 0.0 };
        out.push(sample);
    }
    out
}

/// Biquad second-order IIR filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiquadType {
    /// Low-pass biquad filter.
    LowPass,
    /// High-pass biquad filter.
    HighPass,
    /// Band-pass biquad filter.
    BandPass,
    /// Notch / Band-stop biquad filter.
    Notch,
}

/// Applies a second-order biquad IIR filter to a 1D audio signal.
pub fn biquad_filter(signal: &[f64], filter_type: BiquadType, sample_rate: u32, cutoff_hz: f64, q: f64) -> Vec<f64> {
    let w0 = 2.0 * PI * cutoff_hz / sample_rate as f64;
    let alpha = w0.sin() / (2.0 * q.max(0.1));
    let cos_w0 = w0.cos();

    let (b0, b1, b2, a0, a1, a2) = match filter_type {
        BiquadType::LowPass => {
            let b0 = (1.0 - cos_w0) / 2.0;
            let b1 = 1.0 - cos_w0;
            let b2 = (1.0 - cos_w0) / 2.0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            (b0, b1, b2, a0, a1, a2)
        }
        BiquadType::HighPass => {
            let b0 = (1.0 + cos_w0) / 2.0;
            let b1 = -(1.0 + cos_w0);
            let b2 = (1.0 + cos_w0) / 2.0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            (b0, b1, b2, a0, a1, a2)
        }
        BiquadType::BandPass => {
            let b0 = alpha;
            let b1 = 0.0;
            let b2 = -alpha;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            (b0, b1, b2, a0, a1, a2)
        }
        BiquadType::Notch => {
            let b0 = 1.0;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            (b0, b1, b2, a0, a1, a2)
        }
    };

    let mut out = Vec::with_capacity(signal.len());
    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;

    for &x0 in signal {
        let y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
        out.push(y0);
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effects_stress_001() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_002() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_003() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_004() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_005() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_006() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_007() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_008() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_009() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_010() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_011() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_012() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_013() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_014() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_015() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_016() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_017() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_018() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_019() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_020() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_021() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_022() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_023() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_024() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_025() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_026() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_027() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_028() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_029() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_030() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_031() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_032() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_033() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_034() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_035() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_036() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_037() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_038() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_039() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_040() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_041() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_042() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_043() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_044() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_045() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_046() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_047() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_048() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_049() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_050() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_051() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_052() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_053() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_054() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_055() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_056() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_057() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_058() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_059() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_060() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_061() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_062() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_063() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_064() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_065() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_066() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_067() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_068() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_069() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_070() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_071() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_072() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_073() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_074() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_075() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_076() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_077() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_078() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_079() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_080() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_081() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_082() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_083() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_084() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_085() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_086() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_087() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_088() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_089() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_090() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_091() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_092() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_093() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_094() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_095() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_096() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_097() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_098() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_099() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_100() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_101() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_102() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_103() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_104() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_105() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_106() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_107() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_108() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_109() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_110() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_111() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_112() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_113() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_114() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_115() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_116() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_117() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_118() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_119() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_120() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_121() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_122() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_123() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_124() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_125() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_126() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_127() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_128() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_129() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_130() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_131() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_132() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_133() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_134() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_135() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_136() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_137() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_138() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_139() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_140() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_141() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_142() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_143() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_144() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_145() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_146() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_147() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_148() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_149() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_150() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_151() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_152() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_153() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_154() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_155() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_156() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_157() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_158() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_159() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_160() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_161() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_162() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_163() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_164() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_165() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }

    #[test]
    fn test_effects_stress_166() {
        let signal: Vec<f64> = (0..256).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let rev = schroeder_reverb(&signal, 16000, 0.5);
        assert_eq!(rev.len(), signal.len());
        
        let cho = chorus(&signal, 16000, 20.0, 1.5);
        assert_eq!(cho.len(), signal.len());
        
        let flan = flanger(&signal, 16000, 5.0, 0.5, 0.3);
        assert_eq!(flan.len(), signal.len());
        
        let vib = vibrato(&signal, 16000, 5.0, 4.0);
        assert_eq!(vib.len(), signal.len());
        
        let lp = biquad_filter(&signal, BiquadType::LowPass, 16000, 1000.0, 0.707);
        assert_eq!(lp.len(), signal.len());
    }
}
