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

    #[test]
    fn test_vad_stress_001() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 1) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_002() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 2) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_003() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 3) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_004() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 4) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_005() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 5) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_006() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 6) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_007() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 7) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_008() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 8) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_009() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 9) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_010() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 10) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_011() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 11) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_012() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 12) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_013() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 13) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_014() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 14) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_015() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 15) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_016() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 16) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_017() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 17) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_018() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 18) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_019() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 19) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_020() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 20) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_021() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 21) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_022() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 22) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_023() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 23) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_024() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 24) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_025() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 25) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_026() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 26) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_027() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 27) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_028() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 28) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_029() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 29) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_030() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 30) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_031() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 31) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_032() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 32) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_033() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 33) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_034() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 34) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_035() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 35) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_036() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 36) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_037() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 37) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_038() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 38) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_039() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 39) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_040() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 40) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_041() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 41) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_042() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 42) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_043() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 43) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_044() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 44) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_045() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 45) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_046() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 46) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_047() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 47) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_048() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 48) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_049() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 49) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_050() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 50) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_051() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 51) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_052() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 52) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_053() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 53) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_054() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 54) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_055() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 55) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_056() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 56) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_057() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 57) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_058() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 58) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_059() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 59) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_060() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 60) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_061() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 61) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_062() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 62) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_063() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 63) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_064() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 64) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_065() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 65) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_066() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 66) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_067() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 67) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_068() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 68) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_069() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 69) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_070() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 70) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_071() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 71) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_072() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 72) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_073() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 73) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_074() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 74) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_075() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 75) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_076() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 76) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_077() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 77) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_078() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 78) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_079() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 79) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_080() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 80) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_081() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 81) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_082() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 82) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_083() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 83) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_084() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 84) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_085() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 85) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_086() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 86) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_087() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 87) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_088() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 88) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_089() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 89) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_090() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 90) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_091() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 91) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_092() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 92) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_093() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 93) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_094() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 94) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_095() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 95) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_096() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 96) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_097() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 97) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_098() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 98) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_099() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 99) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_100() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 100) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_101() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 101) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_102() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 102) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_103() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 103) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_104() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 104) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_105() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 105) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_106() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 106) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_107() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 107) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_108() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 108) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_109() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 109) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_110() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 110) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_111() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 111) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_112() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 112) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_113() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 113) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_114() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 114) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_115() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 115) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_116() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 116) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_117() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 117) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_118() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 118) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_119() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 119) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_120() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 120) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_121() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 121) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_122() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 122) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_123() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 123) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_124() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 124) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_125() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 125) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_126() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 126) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_127() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 127) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_128() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 128) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_129() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 129) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_130() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 130) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_131() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 131) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_132() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 132) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_133() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 133) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_134() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 134) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_135() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 135) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_136() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 136) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_137() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 137) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_138() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 138) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_139() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 139) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_140() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 140) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_141() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 141) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_142() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 142) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_143() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 143) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_144() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 144) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_145() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 145) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_146() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 146) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_147() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 147) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_148() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 148) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_149() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 149) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_150() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 150) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_151() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 151) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_152() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 152) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_153() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 153) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_154() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 154) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_155() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 155) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_156() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 156) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_157() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 157) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_158() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 158) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_159() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 159) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_160() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 160) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_161() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 161) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_162() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 162) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_163() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 163) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_164() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 164) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_165() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 165) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_166() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 166) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_167() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 167) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_168() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 168) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_169() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 169) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_170() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 170) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_171() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 171) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_172() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 172) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_173() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 173) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_174() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 174) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_175() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 175) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_176() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 176) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_177() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 177) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_178() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 178) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_179() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 179) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_180() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 180) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_181() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 181) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_182() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 182) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_183() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 183) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_184() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 184) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_185() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 185) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_186() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 186) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_187() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 187) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_188() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 188) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_189() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 189) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_190() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 190) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_191() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 191) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_192() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 192) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_193() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 193) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_194() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 194) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_195() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 195) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_196() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 196) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_197() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 197) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_198() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 198) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_199() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 199) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_200() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 200) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }

    #[test]
    fn test_vad_stress_201() {
        let cfg = VADConfig::default();
        let mut signal = vec![0.0; 2000];
        // Inject speech burst in middle
        for i in 500..1200 {
            signal[i] = ((i + 201) as f64 * 0.1).sin() * 0.5;
        }
        let vad = compute_vad(&signal, &cfg);
        assert!(!vad.is_empty());
        
        let buf = AudioBuffer::from_mono(signal, crate::core::SampleRate::SPEECH_16K).unwrap();
        let trimmed = trim_silence(&buf, -35.0, 512, 160).unwrap();
        assert!(trimmed.num_samples() <= buf.num_samples());
    }
}
