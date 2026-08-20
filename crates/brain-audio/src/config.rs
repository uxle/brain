//! # Audio Configuration System
//!
//! Strongly-typed configurations for audio feature extraction, augmentation,
//! resampling, and voice activity detection pipelines.

use brain_core::{BrainError, BrainResult};

/// Window function types supported across STFT and filter designs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WindowType {
    /// Standard Hann raised cosine window (most common for speech/music).
    #[default]
    Hann,
    /// Hamming window with slight pedestal to cancel first side lobe.
    Hamming,
    /// Blackman window with -58 dB side lobe rejection.
    Blackman,
    /// Bartlett triangular window.
    Bartlett,
    /// Rectangular / Dirichlet boxcar window.
    Rectangular,
    /// Flat-top window for calibration and precise amplitude measurements.
    FlatTop,
    /// Gaussian window with parameterized standard deviation.
    Gaussian,
    /// Kaiser-Bessel window with tunable beta trade-off parameter.
    Kaiser,
}

/// Mel filterbank frequency scale standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MelScale {
    /// HTK formula: `mel = 2595 * log10(1 + hz / 700)`.
    Htk,
    /// Slaney formula: linear below 1 kHz, logarithmic above 1 kHz.
    #[default]
    Slaney,
}

/// Normalization method applied to Mel filter banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MelNorm {
    /// No area normalization (peak of all triangular filters = 1.0).
    None,
    /// Slaney area normalization (divides each triangular filter by bandwidth in Hz).
    #[default]
    Slaney,
}

/// Short-Time Fourier Transform configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct STFTConfig {
    /// Sample rate of the incoming audio signal in Hz.
    pub sample_rate: u32,
    /// Size of the Fourier transform (number of FFT frequency bins).
    pub n_fft: usize,
    /// Window size in samples. Defaults to `n_fft`.
    pub win_length: usize,
    /// Hop length (stride) in samples between consecutive analysis windows.
    pub hop_length: usize,
    /// Window weighting function to apply prior to FFT.
    pub window_type: WindowType,
    /// Whether to center analysis windows by reflection padding `n_fft / 2`.
    pub center: bool,
    /// Whether to normalize STFT output by window energy sum.
    pub normalized: bool,
    /// Power factor for spectrogram: `Some(1.0)` for magnitude, `Some(2.0)` for power, `None` for complex.
    pub power: Option<f64>,
}

impl Default for STFTConfig {
    fn default() -> Self {
        Self::default_speech()
    }
}

impl STFTConfig {
    /// Standard speech configuration (16 kHz, 25 ms window, 10 ms hop).
    pub fn default_speech() -> Self {
        STFTConfig {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400, // 25 ms at 16 kHz
            hop_length: 160, // 10 ms at 16 kHz
            window_type: WindowType::Hann,
            center: true,
            normalized: false,
            power: Some(2.0),
        }
    }

    /// Standard music analysis configuration (44.1 kHz, 2048 FFT, 512 hop).
    pub fn default_music() -> Self {
        STFTConfig {
            sample_rate: 44100,
            n_fft: 2048,
            win_length: 2048,
            hop_length: 512,
            window_type: WindowType::Hann,
            center: true,
            normalized: false,
            power: Some(2.0),
        }
    }

    /// Validates parameter consistency.
    pub fn validate(&self) -> BrainResult<()> {
        if self.sample_rate == 0 {
            return Err(BrainError::invalid_value("sample_rate must be > 0"));
        }
        if self.n_fft == 0 || (self.n_fft & (self.n_fft - 1)) != 0 {
            return Err(BrainError::invalid_value(format!(
                "n_fft must be a power of two, got {}",
                self.n_fft
            )));
        }
        if self.win_length == 0 || self.win_length > self.n_fft {
            return Err(BrainError::invalid_value(format!(
                "win_length ({}) must be between 1 and n_fft ({})",
                self.win_length, self.n_fft
            )));
        }
        if self.hop_length == 0 {
            return Err(BrainError::invalid_value("hop_length must be > 0"));
        }
        Ok(())
    }
}

/// Mel Spectrogram Extraction Configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    /// Base STFT analysis parameters.
    pub stft: STFTConfig,
    /// Number of Mel frequency filter banks.
    pub n_mels: usize,
    /// Lowest frequency in Hz for Mel filter bank analysis.
    pub f_min: f64,
    /// Highest frequency in Hz for Mel filter bank analysis (`None` = Nyquist frequency).
    pub f_max: Option<f64>,
    /// Mel scale formulation (Slaney vs HTK).
    pub mel_scale: MelScale,
    /// Mel filter area normalization policy.
    pub mel_norm: MelNorm,
    /// Minimum epsilon floor for logarithm computation.
    pub eps: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self::default_speech_80()
    }
}

impl MelConfig {
    /// Standard 80-channel log-Mel configuration for speech synthesis / Whisper models.
    pub fn default_speech_80() -> Self {
        MelConfig {
            stft: STFTConfig::default_speech(),
            n_mels: 80,
            f_min: 0.0,
            f_max: Some(8000.0),
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
            eps: 1e-10,
        }
    }

    /// 128-channel Mel configuration for high-fidelity music tagging and audio representation.
    pub fn default_music_128() -> Self {
        MelConfig {
            stft: STFTConfig::default_music(),
            n_mels: 128,
            f_min: 20.0,
            f_max: Some(20000.0),
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
            eps: 1e-10,
        }
    }

    /// Validates Mel configuration consistency.
    pub fn validate(&self) -> BrainResult<()> {
        self.stft.validate()?;
        if self.n_mels == 0 {
            return Err(BrainError::invalid_value("n_mels must be > 0"));
        }
        if self.f_min < 0.0 {
            return Err(BrainError::invalid_value("f_min must be >= 0.0"));
        }
        let nyquist = self.stft.sample_rate as f64 / 2.0;
        if let Some(f_max) = self.f_max {
            if f_max <= self.f_min || f_max > nyquist {
                return Err(BrainError::invalid_value(format!(
                    "f_max ({}) must be between f_min ({}) and Nyquist ({})",
                    f_max, self.f_min, nyquist
                )));
            }
        }
        Ok(())
    }
}

/// Mel-Frequency Cepstral Coefficients (MFCC) configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MFCCConfig {
    /// Base Mel spectrogram configuration.
    pub mel: MelConfig,
    /// Number of output cepstral coefficients to keep (e.g. 13, 20, 40).
    pub n_mfcc: usize,
    /// Sine lifter parameter (0.0 = no liftering, standard value is 22.0).
    pub lifter: f64,
    /// Whether to apply Cepstral Mean Normalization across time frames.
    pub cepstral_mean_norm: bool,
    /// Whether to apply Cepstral Variance Normalization across time frames.
    pub cepstral_var_norm: bool,
}

impl Default for MFCCConfig {
    fn default() -> Self {
        MFCCConfig {
            mel: MelConfig::default_speech_80(),
            n_mfcc: 13,
            lifter: 22.0,
            cepstral_mean_norm: false,
            cepstral_var_norm: false,
        }
    }
}

impl MFCCConfig {
    /// Validates MFCC configuration parameters.
    pub fn validate(&self) -> BrainResult<()> {
        self.mel.validate()?;
        if self.n_mfcc == 0 || self.n_mfcc > self.mel.n_mels {
            return Err(BrainError::invalid_value(format!(
                "n_mfcc ({}) must be between 1 and n_mels ({})",
                self.n_mfcc, self.mel.n_mels
            )));
        }
        if self.lifter < 0.0 {
            return Err(BrainError::invalid_value("lifter parameter must be >= 0.0"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
