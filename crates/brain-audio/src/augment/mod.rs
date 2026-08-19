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
}
