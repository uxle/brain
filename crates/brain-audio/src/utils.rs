//! # Audio Signal Processing Utilities and Window Functions
//!
//! Pure-Rust implementations of psychoacoustic frequency mappings, window functions,
//! decibel conversions, and framing utilities.

use std::f64::consts::PI;

/// Converts frequency in Hertz to Mel scale using the Slaney formulation.
///
/// # Examples
///
/// ```
/// use brain_audio::utils::{hz_to_mel_slaney, mel_to_hz_slaney};
/// let mel = hz_to_mel_slaney(1000.0);
/// let hz = mel_to_hz_slaney(mel);
/// assert!((hz - 1000.0).abs() < 1e-4);
/// ```
#[inline]
pub fn hz_to_mel_slaney(hz: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0; // 1000.0 / (200.0 / 3.0)
    const LOGSTEP: f64 = 0.06875177742094912; // ln(6.4) / 27.0
    if hz < MIN_LOG_HZ {
        3.0 * hz / 200.0
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Converts Mel scale value to frequency in Hertz using the Slaney formulation.
#[inline]
pub fn mel_to_hz_slaney(mel: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875177742094912;
    if mel < MIN_LOG_MEL {
        200.0 * mel / 3.0
    } else {
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    }
}

/// Converts frequency in Hertz to Mel scale using the HTK formula.
#[inline]
pub fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts Mel scale value to frequency in Hertz using the HTK formula.
#[inline]
pub fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Converts frequency in Hertz to Bark scale (Traunmuller 1990).
#[inline]
pub fn hz_to_bark(hz: f64) -> f64 {
    ((26.81 * hz) / (1960.0 + hz)) - 0.53
}

/// Converts Bark scale value to frequency in Hertz.
#[inline]
pub fn bark_to_hz(bark: f64) -> f64 {
    let b = bark + 0.53;
    1960.0 * b / (26.81 - b)
}

/// Converts frequency in Hertz to Equivalent Rectangular Bandwidth (ERB) rate.
#[inline]
pub fn hz_to_erb(hz: f64) -> f64 {
    21.4 * (0.00437 * hz + 1.0).log10()
}

/// Converts ERB rate to frequency in Hertz.
#[inline]
pub fn erb_to_hz(erb: f64) -> f64 {
    (10.0f64.powf(erb / 21.4) - 1.0) / 0.00437
}

/// Computes center frequencies for FFT bins up to Nyquist frequency.
pub fn fft_freqs(sample_rate: u32, n_fft: usize) -> Vec<f64> {
    let num_bins = n_fft / 2 + 1;
    let delta = sample_rate as f64 / n_fft as f64;
    (0..num_bins).map(|i| i as f64 * delta).collect()
}

/// Converts amplitude to decibels: `20 * log10(max(amplitude, amin)) - top_db`.
///
/// # Examples
///
/// ```
/// use brain_audio::utils::amplitude_to_db;
/// let db = amplitude_to_db(&[1.0, 0.1], 1e-10, None);
/// assert!((db[0] - 0.0).abs() < 1e-4);
/// assert!((db[1] - (-20.0)).abs() < 1e-4);
/// ```
pub fn amplitude_to_db(amplitude: &[f64], amin: f64, top_db: Option<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = amplitude
        .iter()
        .map(|&x| 20.0 * x.max(amin).log10())
        .collect();
    if let Some(top) = top_db {
        let max_db = out.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let floor_db = max_db - top;
        for v in &mut out {
            *v = v.max(floor_db);
        }
    }
    out
}

/// Converts decibels back to linear amplitude: `10^(db / 20)`.
pub fn db_to_amplitude(db: &[f64]) -> Vec<f64> {
    db.iter().map(|&x| 10.0f64.powf(x / 20.0)).collect()
}

/// Converts power to decibels: `10 * log10(max(power, amin)) - top_db`.
pub fn power_to_db(power: &[f64], amin: f64, top_db: Option<f64>) -> Vec<f64> {
    let mut out: Vec<f64> = power.iter().map(|&x| 10.0 * x.max(amin).log10()).collect();
    if let Some(top) = top_db {
        let max_db = out.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let floor_db = max_db - top;
        for v in &mut out {
            *v = v.max(floor_db);
        }
    }
    out
}

/// Converts decibels back to linear power: `10^(db / 10)`.
pub fn db_to_power(db: &[f64]) -> Vec<f64> {
    db.iter().map(|&x| 10.0f64.powf(x / 10.0)).collect()
}

/// Generates a Hann raised cosine analysis window of length `size`.
pub fn hann_window(size: usize, periodic: bool) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let denom = if periodic {
        size as f64
    } else {
        (size - 1) as f64
    };
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / denom).cos()))
        .collect()
}

/// Generates a Hamming window of length `size`.
pub fn hamming_window(size: usize, periodic: bool) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let denom = if periodic {
        size as f64
    } else {
        (size - 1) as f64
    };
    const ALPHA: f64 = 0.54;
    const BETA: f64 = 0.46;
    (0..size)
        .map(|i| ALPHA - BETA * (2.0 * PI * i as f64 / denom).cos())
        .collect()
}

/// Generates a Blackman window of length `size`.
pub fn blackman_window(size: usize, periodic: bool) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let denom = if periodic {
        size as f64
    } else {
        (size - 1) as f64
    };
    const A0: f64 = 0.42;
    const A1: f64 = 0.5;
    const A2: f64 = 0.08;
    (0..size)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / denom;
            A0 - A1 * angle.cos() + A2 * (2.0 * angle).cos()
        })
        .collect()
}

/// Generates a Bartlett triangular window of length `size`.
pub fn bartlett_window(size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = (size - 1) as f64;
    (0..size)
        .map(|i| 1.0 - (2.0 * i as f64 - n).abs() / n)
        .collect()
}

/// Zero-pads or truncates a 1D audio slice to match target length `target_len`.
pub fn ensure_length(signal: &[f64], target_len: usize) -> Vec<f64> {
    if signal.len() >= target_len {
        signal[..target_len].to_vec()
    } else {
        let mut out = Vec::with_capacity(target_len);
        out.extend_from_slice(signal);
        out.resize(target_len, 0.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
