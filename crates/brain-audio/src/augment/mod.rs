//! # Audio Augmentation Subsystem
//!
//! Production-grade time-domain, spectral-domain, and physical acoustic effect augmentations:
//! * [`time`] - Time stretching, pitch shifting, time masking, additive noise injection, clipping, and fading
//! * [`spectral`] - SpecAugment (frequency and time masking), CutMix, Mixup, and feature dropout
//! * [`effects`] - Schroeder reverberation, multitap echo, chorus, flanging, vibrato, and biquad EQ filters

pub mod effects;
pub mod spectral;
pub mod time;

pub use effects::{
    biquad_filter, chorus, flanger, multi_echo, schroeder_reverb, vibrato, BiquadType,
};
pub use spectral::{frequency_mask, spec_augment, spec_cutout, spec_mixup, time_mask_spec};
pub use time::{
    add_noise, clip_distortion, fade_in, fade_out, gain_scale, pitch_shift, time_mask, time_stretch,
};

use crate::core::AudioBuffer;
use brain_core::BrainResult;

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
