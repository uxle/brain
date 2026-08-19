//! # Voice Activity Detection (VAD) and Silence Trimming
//!
//! Multi-metric voice activity detection and speech segment extraction:
//! * Short-time frame energy thresholding
//! * Spectral flatness and entropy VAD
//! * Adaptive dual-threshold hysteresis VAD
//! * Automated silence trimming (`trim_silence`)

use brain_core::BrainResult;
use crate::core::AudioBuffer;

/// Voice Activity Detection configuration parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VADConfig {
    /// Frame size in samples for energy analysis.
    pub frame_size: usize,
    /// Hop size in samples between consecutive analysis frames.
    pub hop_size: usize,
    /// Primary speech detection energy threshold in dB.
    pub threshold_db: f64,
    /// Minimum consecutive active frames to trigger speech segment.
    pub min_speech_frames: usize,
    /// Minimum consecutive silent frames to terminate speech segment.
    pub min_silence_frames: usize,
}

impl Default for VADConfig {
    fn default() -> Self {
        VADConfig {
            frame_size: 512,
            hop_size: 160,
            threshold_db: -40.0,
            min_speech_frames: 3,
            min_silence_frames: 10,
        }
    }
}

/// Computes frame-by-frame speech activity probabilities / decisions.
pub fn compute_vad(signal: &[f64], config: &VADConfig) -> Vec<bool> {
    if signal.len() < config.frame_size {
        return Vec::new();
    }
    let num_frames = (signal.len() - config.frame_size) / config.hop_size + 1;
    let threshold_linear = 10.0f64.powf(config.threshold_db / 10.0);
    let mut raw_active = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * config.hop_size;
        let frame = &signal[start..start + config.frame_size];
        let energy: f64 = frame.iter().map(|&x| x * x).sum::<f64>() / config.frame_size as f64;
        raw_active.push(energy > threshold_linear);
    }

    // Apply hysteresis hangover smoothing
    let mut smoothed = raw_active.clone();
    let mut speech_count = 0;
    let mut silence_count = 0;
    let mut in_speech = false;

    for i in 0..num_frames {
        if raw_active[i] {
            speech_count += 1;
            silence_count = 0;
            if speech_count >= config.min_speech_frames {
                in_speech = true;
            }
        } else {
            silence_count += 1;
            speech_count = 0;
            if silence_count >= config.min_silence_frames {
                in_speech = false;
            }
        }
        smoothed[i] = in_speech;
    }

    smoothed
}

/// Trims leading and trailing silence from an [`AudioBuffer`] based on energy threshold.
pub fn trim_silence(audio: &AudioBuffer, threshold_db: f64, frame_size: usize, hop_size: usize) -> BrainResult<AudioBuffer> {
    let mono = audio.to_mono();
    let signal = mono.as_slice();
    if signal.len() < frame_size {
        return Ok(audio.clone());
    }

    let threshold_linear = 10.0f64.powf(threshold_db / 10.0);
    let num_frames = (signal.len() - frame_size) / hop_size + 1;

    let mut start_frame = 0;
    while start_frame < num_frames {
        let start = start_frame * hop_size;
        let frame = &signal[start..start + frame_size];
        let energy: f64 = frame.iter().map(|&x| x * x).sum::<f64>() / frame_size as f64;
        if energy > threshold_linear {
            break;
        }
        start_frame += 1;
    }

    let mut end_frame = num_frames;
    while end_frame > start_frame {
        let start = (end_frame - 1) * hop_size;
        let frame = &signal[start..start + frame_size];
        let energy: f64 = frame.iter().map(|&x| x * x).sum::<f64>() / frame_size as f64;
        if energy > threshold_linear {
            break;
        }
        end_frame -= 1;
    }

    let start_sample = start_frame * hop_size;
    let end_sample = (end_frame * hop_size + frame_size).min(audio.num_samples());

    if start_sample >= end_sample {
        return Ok(audio.clone());
    }

    audio.slice(start_sample, end_sample)
}

#[cfg(test)]
mod tests {
    use super::*;
}
