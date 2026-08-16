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
    let mut out: Vec<f64> = power
        .iter()
        .map(|&x| 10.0 * x.max(amin).log10())
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
    let denom = if periodic { size as f64 } else { (size - 1) as f64 };
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
    let denom = if periodic { size as f64 } else { (size - 1) as f64 };
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
    let denom = if periodic { size as f64 } else { (size - 1) as f64 };
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

    #[test]
    fn test_audio_utils_stress_001() {
        let hz = (1 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (1 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_002() {
        let hz = (2 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (2 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_003() {
        let hz = (3 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (3 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_004() {
        let hz = (4 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (4 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_005() {
        let hz = (5 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (5 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_006() {
        let hz = (6 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (6 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_007() {
        let hz = (7 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (7 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_008() {
        let hz = (8 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (8 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_009() {
        let hz = (9 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (9 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_010() {
        let hz = (10 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (10 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_011() {
        let hz = (11 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (11 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_012() {
        let hz = (12 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (12 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_013() {
        let hz = (13 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (13 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_014() {
        let hz = (14 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (14 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_015() {
        let hz = (15 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (15 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_016() {
        let hz = (16 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (16 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_017() {
        let hz = (17 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (17 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_018() {
        let hz = (18 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (18 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_019() {
        let hz = (19 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (19 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_020() {
        let hz = (20 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (20 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_021() {
        let hz = (21 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (21 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_022() {
        let hz = (22 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (22 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_023() {
        let hz = (23 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (23 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_024() {
        let hz = (24 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (24 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_025() {
        let hz = (25 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (25 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_026() {
        let hz = (26 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (26 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_027() {
        let hz = (27 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (27 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_028() {
        let hz = (28 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (28 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_029() {
        let hz = (29 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (29 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_030() {
        let hz = (30 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (30 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_031() {
        let hz = (31 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (31 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_032() {
        let hz = (32 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (32 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_033() {
        let hz = (33 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (33 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_034() {
        let hz = (34 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (34 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_035() {
        let hz = (35 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (35 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_036() {
        let hz = (36 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (36 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_037() {
        let hz = (37 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (37 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_038() {
        let hz = (38 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (38 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_039() {
        let hz = (39 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (39 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_040() {
        let hz = (40 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (40 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_041() {
        let hz = (41 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (41 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_042() {
        let hz = (42 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (42 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_043() {
        let hz = (43 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (43 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_044() {
        let hz = (44 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (44 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_045() {
        let hz = (45 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (45 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_046() {
        let hz = (46 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (46 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_047() {
        let hz = (47 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (47 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_048() {
        let hz = (48 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (48 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_049() {
        let hz = (49 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (49 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_050() {
        let hz = (50 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (50 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_051() {
        let hz = (51 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (51 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_052() {
        let hz = (52 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (52 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_053() {
        let hz = (53 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (53 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_054() {
        let hz = (54 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (54 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_055() {
        let hz = (55 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (55 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_056() {
        let hz = (56 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (56 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_057() {
        let hz = (57 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (57 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_058() {
        let hz = (58 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (58 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_059() {
        let hz = (59 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (59 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_060() {
        let hz = (60 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (60 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_061() {
        let hz = (61 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (61 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_062() {
        let hz = (62 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (62 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_063() {
        let hz = (63 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (63 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_064() {
        let hz = (64 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (64 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_065() {
        let hz = (65 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (65 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_066() {
        let hz = (66 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (66 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_067() {
        let hz = (67 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (67 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_068() {
        let hz = (68 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (68 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_069() {
        let hz = (69 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (69 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_070() {
        let hz = (70 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (70 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_071() {
        let hz = (71 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (71 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_072() {
        let hz = (72 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (72 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_073() {
        let hz = (73 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (73 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_074() {
        let hz = (74 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (74 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_075() {
        let hz = (75 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (75 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_076() {
        let hz = (76 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (76 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_077() {
        let hz = (77 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (77 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_078() {
        let hz = (78 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (78 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_079() {
        let hz = (79 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (79 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_080() {
        let hz = (80 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (80 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_081() {
        let hz = (81 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (81 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_082() {
        let hz = (82 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (82 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_083() {
        let hz = (83 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (83 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_084() {
        let hz = (84 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (84 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_085() {
        let hz = (85 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (85 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_086() {
        let hz = (86 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (86 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_087() {
        let hz = (87 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (87 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_088() {
        let hz = (88 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (88 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_089() {
        let hz = (89 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (89 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_090() {
        let hz = (90 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (90 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_091() {
        let hz = (91 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (91 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_092() {
        let hz = (92 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (92 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_093() {
        let hz = (93 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (93 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_094() {
        let hz = (94 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (94 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_095() {
        let hz = (95 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (95 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_096() {
        let hz = (96 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (96 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_097() {
        let hz = (97 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (97 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_098() {
        let hz = (98 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (98 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_099() {
        let hz = (99 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (99 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_100() {
        let hz = (100 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (100 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_101() {
        let hz = (101 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (101 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_102() {
        let hz = (102 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (102 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_103() {
        let hz = (103 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (103 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_104() {
        let hz = (104 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (104 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_105() {
        let hz = (105 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (105 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_106() {
        let hz = (106 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (106 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_107() {
        let hz = (107 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (107 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_108() {
        let hz = (108 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (108 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_109() {
        let hz = (109 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (109 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_110() {
        let hz = (110 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (110 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_111() {
        let hz = (111 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (111 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_112() {
        let hz = (112 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (112 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_113() {
        let hz = (113 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (113 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_114() {
        let hz = (114 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (114 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_115() {
        let hz = (115 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (115 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn test_audio_utils_stress_116() {
        let hz = (116 as f64) * 45.0 + 20.0;
        let mel_s = hz_to_mel_slaney(hz);
        let hz_s = mel_to_hz_slaney(mel_s);
        assert!((hz - hz_s).abs() < 1e-4, "Slaney Mel roundtrip mismatch: {} vs {}", hz, hz_s);
        
        let mel_h = hz_to_mel_htk(hz);
        let hz_h = mel_to_hz_htk(mel_h);
        assert!((hz - hz_h).abs() < 1e-4, "HTK Mel roundtrip mismatch: {} vs {}", hz, hz_h);
        
        let bark = hz_to_bark(hz);
        let hz_b = bark_to_hz(bark);
        assert!((hz - hz_b).abs() < 1e-4, "Bark roundtrip mismatch: {} vs {}", hz, hz_b);
        
        let size = 16 + (116 % 64);
        let w_hann = hann_window(size, false);
        assert_eq!(w_hann.len(), size);
        assert!((w_hann[0]).abs() < 1e-6);
        assert!((w_hann[size - 1]).abs() < 1e-6);
        
        let db = amplitude_to_db(&[1.0, 0.1, 0.01], 1e-10, None);
        assert!((db[0] - 0.0).abs() < 1e-4);
        assert!((db[1] - (-20.0)).abs() < 1e-4);
        assert!((db[2] - (-40.0)).abs() < 1e-4);
    }
}
