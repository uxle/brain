//! # Feature Extraction Subsystem
//!
//! Audio feature extraction modules:
//! * [`stft`] - Short-Time Fourier Transform (STFT), Inverse STFT (iSTFT), and Phase Vocoder
//! * [`spectral`] - Magnitude, Power, and Mel spectrograms, delta features, and spectral descriptors
//! * [`mfcc`] - Mel-Frequency Cepstral Coefficients (MFCC) and Cepstral Normalization
//! * [`tonal`] - Chroma representations, pitch detection (YIN/Autocorrelation), and harmonicity
//! * [`wavelet`] - Discrete Wavelet Transform (DWT), wavelet packets, and multi-scale denoising

pub mod stft;
pub mod spectral;
pub mod mfcc;
pub mod tonal;
pub mod wavelet;

pub use stft::{stft, istft, STFTProcessor, PhaseVocoder};
pub use spectral::{spectrogram, mel_spectrogram, compute_deltas, SpectralDescriptors};
pub use mfcc::{mfcc, compute_mfcc, MFCCProcessor};
pub use tonal::{chroma_stft, chroma_cens, detect_pitch_yin, zero_crossing_rate, spectral_flux};
pub use wavelet::{dwt, idwt, WaveletType, WaveletDenoise};

use brain_core::{BrainResult, Tensor};
use crate::core::AudioBuffer;

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
