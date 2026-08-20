//! # Feature Extraction Subsystem
//!
//! Audio feature extraction modules:
//! * [`stft`] - Short-Time Fourier Transform (STFT), Inverse STFT (iSTFT), and Phase Vocoder
//! * [`spectral`] - Magnitude, Power, and Mel spectrograms, delta features, and spectral descriptors
//! * [`mfcc`] - Mel-Frequency Cepstral Coefficients (MFCC) and Cepstral Normalization
//! * [`tonal`] - Chroma representations, pitch detection (YIN/Autocorrelation), and harmonicity
//! * [`wavelet`] - Discrete Wavelet Transform (DWT), wavelet packets, and multi-scale denoising

pub mod mfcc;
pub mod spectral;
pub mod stft;
pub mod tonal;
pub mod wavelet;

pub use mfcc::{compute_mfcc, mfcc, MFCCProcessor};
pub use spectral::{compute_deltas, mel_spectrogram, spectrogram, SpectralDescriptors};
pub use stft::{istft, stft, PhaseVocoder, STFTProcessor};
pub use tonal::{chroma_cens, chroma_stft, detect_pitch_yin, spectral_flux, zero_crossing_rate};
pub use wavelet::{dwt, idwt, WaveletDenoise, WaveletType};

use crate::core::AudioBuffer;
use brain_core::{BrainResult, Tensor};

/// Common trait implemented by all audio feature extractors.
pub trait AudioFeature {
    /// Extracts features from an input audio buffer, returning a `Tensor`.
    fn extract(&self, audio: &AudioBuffer) -> BrainResult<Tensor>;

    /// Returns the feature representation name.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
}
