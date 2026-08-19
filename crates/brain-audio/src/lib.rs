//! # Brain Audio — Pure Rust High-Performance Audio Processing Framework
//!
//! Production-grade audio feature extraction, augmentation, DSP primitives,
//! and batch pipelines for deep learning models.
//!
//! ## Subsystems
//!
//! * [`core`] - Multi-channel `AudioBuffer`, sample rate, and channel primitives
//! * [`config`] - Strongly-typed STFT, Mel, and MFCC configurations
//! * [`ops`] - Signal transforms, pre-emphasis, Hilbert envelopes, Griffin-Lim
//! * [`utils`] - Window functions (Hann, Hamming, Blackman), frequency scales (Mel, Bark, ERB)
//! * [`feature`] - STFT, spectrograms, log-Mel filter banks, MFCC, Chroma, Wavelets
//! * [`augment`] - Time-domain, SpecAugment, and acoustic effect augmentations
//! * [`io`] - WAV, MP3, and FLAC decoding / encoding
//! * [`resample`] - Sinc, cubic, and linear sample rate conversion
//! * [`vad`] - Energy and spectral Voice Activity Detection
//! * [`align`] - Dynamic Time Warping (DTW) sequence alignment
//! * [`denoise`] - Spectral subtraction and Wiener filtering
//! * [`batch`] - Audio batch collation and padding

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod core;
pub mod config;
pub mod ops;
pub mod utils;
pub mod r#impl;
pub mod feature;
pub mod augment;
pub mod io;
pub mod resample;
pub mod vad;
pub mod align;
pub mod encoding;
pub mod denoise;
pub mod features_pitch;
pub mod features_energy;
pub mod features_rhythm;
pub mod batch;

/// Audio prelude for convenient imports.
pub mod prelude {
    pub use crate::core::{AudioBuffer, Channels, SampleRate, AudioFormat};
    pub use crate::config::{STFTConfig, MelConfig, MFCCConfig, WindowType};
    pub use crate::feature::{stft, istft, spectrogram, mel_spectrogram, mfcc};
    pub use crate::augment::{time_stretch, pitch_shift, spec_augment};
    pub use crate::io::{read_wav, write_wav};
    pub use crate::resample::{resample_audio, ResampleMethod};
}

pub use prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
}
