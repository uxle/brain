//! # Energy, Envelope Followers, and Loudness Descriptors
//!
//! Computational primitives for:
//! * Root Mean Square (RMS) energy across frames
//! * Peak amplitude tracking
//! * Attack/Decay envelope followers
//! * ITU-R BS.1770 / EBU R128 integrated loudness approximation

/// Computes frame-level Root Mean Square (RMS) energy.
pub fn rms_frames(signal: &[f64], frame_size: usize, hop_size: usize) -> Vec<f64> {
    if signal.len() < frame_size || hop_size == 0 {
        return Vec::new();
    }
    let num_frames = (signal.len() - frame_size) / hop_size + 1;
    let mut rms = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &signal[start..start + frame_size];
        let sum_sq: f64 = frame.iter().map(|&x| x * x).sum();
        rms.push((sum_sq / frame_size as f64).sqrt());
    }
    rms
}

/// Attack/Release peak envelope follower.
pub fn envelope_follower(signal: &[f64], sample_rate: u32, attack_ms: f64, release_ms: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let ga = (-1.0 / (attack_ms * 0.001 * sr).max(1.0)).exp();
    let gr = (-1.0 / (release_ms * 0.001 * sr).max(1.0)).exp();

    let mut env = Vec::with_capacity(signal.len());
    let mut current = 0.0;

    for &val in signal {
        let target = val.abs();
        if target > current {
            current = ga * current + (1.0 - ga) * target;
        } else {
            current = gr * current + (1.0 - gr) * target;
        }
        env.push(current);
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;
}
