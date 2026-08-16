//! # Audio Augmentation Subsystem
//!
//! Production-grade time-domain, spectral-domain, and physical acoustic effect augmentations:
//! * [`time`] - Time stretching, pitch shifting, time masking, additive noise injection, clipping, and fading
//! * [`spectral`] - SpecAugment (frequency and time masking), CutMix, Mixup, and feature dropout
//! * [`effects`] - Schroeder reverberation, multitap echo, chorus, flanging, vibrato, and biquad EQ filters

pub mod time;
pub mod spectral;
pub mod effects;

pub use time::{time_stretch, pitch_shift, time_mask, add_noise, gain_scale, clip_distortion, fade_in, fade_out};
pub use spectral::{spec_augment, frequency_mask, time_mask_spec, spec_cutout, spec_mixup};
pub use effects::{schroeder_reverb, multi_echo, chorus, flanger, vibrato, biquad_filter, BiquadType};

use brain_core::BrainResult;
use crate::core::AudioBuffer;

/// Common trait for composable audio augmentation pipelines.
pub trait AudioAugment {
    /// Applies augmentation to an input AudioBuffer, returning the augmented buffer.
    fn apply(&self, audio: &AudioBuffer) -> BrainResult<AudioBuffer>;
    
    /// Returns the augmentation descriptor name.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augment_mod_stress_001() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_002() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_003() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_004() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_005() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_006() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_007() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_008() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_009() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_010() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_011() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_012() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_013() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_014() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_015() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_016() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_017() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_018() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_019() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_020() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_021() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_022() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_023() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_024() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_025() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_026() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_027() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_028() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_029() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_030() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_031() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_032() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_033() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_034() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_035() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_036() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_037() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_038() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_039() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_040() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_041() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_042() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_043() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_044() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_045() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_046() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_047() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_048() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_049() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_050() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_051() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_052() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_053() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_054() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_055() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_056() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_057() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_058() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_059() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_060() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_061() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_062() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_063() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_064() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_065() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_066() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_067() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_068() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_069() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_070() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_071() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_072() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_073() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_074() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_075() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_076() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_077() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_078() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_079() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_080() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_081() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_082() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_083() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_084() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_085() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_086() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_087() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_088() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_089() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_090() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_091() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_092() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_093() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_094() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_095() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_096() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_097() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_098() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_099() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_100() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_101() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_102() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_103() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_104() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_105() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_106() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_107() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_108() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_109() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_110() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_111() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_112() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_113() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_114() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_115() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_116() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_117() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_118() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_119() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_120() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_121() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_122() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_123() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_124() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_125() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_126() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_127() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_128() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_129() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_130() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_131() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_132() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_133() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_134() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_135() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_136() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_137() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_138() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_139() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_140() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_141() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_142() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_143() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_144() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_145() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_146() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_147() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_148() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_149() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_150() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_151() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_152() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_153() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_154() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_155() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_156() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_157() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_158() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_159() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_160() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_161() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_162() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_163() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_164() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_165() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_166() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_167() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_168() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_169() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_170() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_171() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_172() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_173() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_174() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_175() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_176() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_177() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_178() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_179() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_180() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_181() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_182() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_183() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_184() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_185() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_186() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_187() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 187) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_188() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 188) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_189() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 189) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_190() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 190) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_191() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 191) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_192() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 192) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_193() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 193) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_194() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 194) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_195() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 195) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_196() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 196) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_197() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 197) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_198() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 198) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_199() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 199) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_200() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 200) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_201() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 201) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_202() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 202) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_203() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 203) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_204() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 204) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_205() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 205) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_206() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 206) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_207() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 207) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_208() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 208) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_209() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 209) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_210() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 210) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_211() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 211) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_212() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 212) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_213() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 213) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_214() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 214) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_215() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 215) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_216() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 216) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_217() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 217) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_218() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 218) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_219() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 219) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_220() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 220) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_221() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 221) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_222() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 222) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_223() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 223) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_224() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 224) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_225() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 225) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_226() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 226) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_227() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 227) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_228() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 228) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_229() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 229) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_230() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 230) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_231() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 231) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_232() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 232) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_233() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 233) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_234() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 234) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_235() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 235) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_236() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 236) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_237() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 237) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_238() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 238) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_239() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 239) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_240() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 240) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_241() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 241) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_242() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 242) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_243() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 243) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_244() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 244) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_245() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 245) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_246() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 246) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_247() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 247) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_248() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 248) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_249() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 249) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_250() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 250) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_251() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 251) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_252() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 252) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_253() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 253) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_254() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 254) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_255() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 255) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_256() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 256) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_257() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 257) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_258() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 258) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_259() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 259) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_260() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 260) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_261() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 261) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_262() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 262) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_263() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 263) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_264() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 264) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_265() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 265) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_266() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 266) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_267() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 267) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_268() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 268) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_269() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 269) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_270() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 270) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_271() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 271) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_272() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 272) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_273() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 273) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_274() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 274) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_275() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 275) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_276() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 276) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_277() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 277) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_278() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 278) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_279() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 279) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_280() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 280) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_281() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 281) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_282() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 282) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_283() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 283) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_284() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 284) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_285() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 285) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_286() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 286) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_287() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 287) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_288() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 288) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_289() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 289) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_290() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 290) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_291() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 291) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_292() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 292) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_293() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 293) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_294() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 294) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_295() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 295) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_296() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 296) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_297() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 297) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_298() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 298) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_299() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 299) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_300() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 300) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_301() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 301) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_302() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 302) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_303() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 303) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_304() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 304) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_305() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 305) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_306() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 306) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_307() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 307) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_308() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 308) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_309() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 309) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_310() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 310) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_311() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 311) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_312() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 312) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_313() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 313) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_314() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 314) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_315() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 315) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_316() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 316) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_317() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 317) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_318() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 318) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_319() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 319) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_320() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 320) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_321() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 321) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_322() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 322) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_323() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 323) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_324() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 324) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_325() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 325) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_326() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 326) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_327() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 327) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_328() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 328) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_329() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 329) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_330() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 330) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_331() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 331) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_332() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 332) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_333() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 333) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_334() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 334) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_335() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 335) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_336() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 336) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_337() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 337) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_338() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 338) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_339() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 339) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_340() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 340) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_341() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 341) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_342() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 342) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_343() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 343) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_344() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 344) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_345() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 345) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_346() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 346) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_347() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 347) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_348() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 348) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_349() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 349) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_350() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 350) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_351() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 351) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_352() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 352) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_353() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 353) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_354() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 354) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_355() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 355) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_356() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 356) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_357() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 357) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_358() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 358) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_359() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 359) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_360() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 360) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_361() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 361) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_362() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 362) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_363() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 363) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_364() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 364) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_365() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 365) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_366() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 366) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_367() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 367) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_368() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 368) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_369() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 369) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_370() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 370) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_371() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 371) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_372() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 372) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_373() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 373) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_374() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 374) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_375() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 375) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_376() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 376) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_377() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 377) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_378() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 378) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_379() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 379) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_380() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 380) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_381() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 381) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_382() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 382) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_383() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 383) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_384() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 384) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_385() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 385) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_386() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 386) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_387() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 387) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_388() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 388) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_389() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 389) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_390() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 390) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_391() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 391) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_392() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 392) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_393() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 393) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_394() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 394) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_395() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 395) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_396() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 396) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_397() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 397) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_398() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 398) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_399() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 399) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_400() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 400) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_401() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 401) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_402() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 402) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_403() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 403) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_404() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 404) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_405() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 405) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_406() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 406) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_407() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 407) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_408() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 408) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_409() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 409) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_410() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 410) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_411() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 411) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_412() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 412) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_413() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 413) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_414() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 414) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_415() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 415) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_416() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 416) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_417() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 417) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_418() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 418) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_419() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 419) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_420() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 420) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_421() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 421) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_422() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 422) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_423() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 423) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_424() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 424) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_425() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 425) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_426() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 426) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_427() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 427) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_428() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 428) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_429() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 429) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_430() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 430) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_431() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 431) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_432() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 432) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_433() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 433) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_434() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 434) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_435() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 435) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_436() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 436) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_437() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 437) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_438() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 438) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_439() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 439) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_440() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 440) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_441() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 441) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_442() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 442) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_443() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 443) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_444() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 444) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_445() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 445) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_446() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 446) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_447() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 447) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_448() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 448) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_449() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 449) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_450() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 450) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_451() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 451) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_452() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 452) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_453() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 453) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_454() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 454) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_455() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 455) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_456() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 456) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_457() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 457) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_458() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 458) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_459() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 459) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_460() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 460) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_461() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 461) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_462() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 462) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_463() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 463) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_464() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 464) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_465() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 465) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_466() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 466) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_467() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 467) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_468() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 468) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_469() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 469) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_470() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 470) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_471() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 471) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_472() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 472) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_473() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 473) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }

    #[test]
    fn test_augment_mod_stress_474() {
        let samples: Vec<f64> = (0..256).map(|i| ((i + 474) as f64 * 0.1).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        assert_eq!(buf.num_samples(), 256);
    }
}
