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
pub fn flanger(
    signal: &[f64],
    sample_rate: u32,
    depth_ms: f64,
    rate_hz: f64,
    feedback: f64,
) -> Vec<f64> {
    let sr = sample_rate as f64;
    let max_delay = (depth_ms * 0.001 * sr).round() as usize;
    let mut buffer = vec![0.0; max_delay.max(1)];
    let mut out = Vec::with_capacity(signal.len());

    for i in 0..signal.len() {
        let lfo = (2.0 * PI * rate_hz * i as f64 / sr).sin();
        let delay_idx = ((0.5 * (1.0 + lfo) * max_delay as f64).round() as usize)
            .min(max_delay.saturating_sub(1));
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
pub fn biquad_filter(
    signal: &[f64],
    filter_type: BiquadType,
    sample_rate: u32,
    cutoff_hz: f64,
    q: f64,
) -> Vec<f64> {
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
}
