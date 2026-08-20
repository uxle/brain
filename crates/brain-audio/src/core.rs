//! # Core Audio Primitives and Buffer Types
//!
//! Provides fundamental audio representations:
//! * [`AudioBuffer`] - Multi-channel audio sample buffer (channels × samples)
//! * [`SampleRate`] - Strongly-typed sample rate with standard constants
//! * [`Channels`] - Channel layout and channel count representations
//! * [`AudioFormat`] - Audio sample bit-depth and representation formats

use brain_core::{BrainError, BrainResult, Tensor};
use std::fmt;

/// Strongly-typed audio sample rate in Hertz (Hz).
///
/// # Examples
///
/// ```
/// use brain_audio::core::SampleRate;
/// let sr = SampleRate::new(16000).unwrap();
/// assert_eq!(sr.hz(), 16000);
/// assert_eq!(sr.duration_seconds(32000), 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleRate(pub u32);

impl SampleRate {
    /// 8 kHz standard telephony rate.
    pub const CD_8K: Self = SampleRate(8000);
    /// 16 kHz standard wideband speech / ASR rate.
    pub const SPEECH_16K: Self = SampleRate(16000);
    /// 22.05 kHz half-CD sample rate.
    pub const HALF_CD_22K: Self = SampleRate(22050);
    /// 24 kHz speech synthesis / high-quality codec rate.
    pub const AUDIO_24K: Self = SampleRate(24000);
    /// 32 kHz broadcast audio rate.
    pub const BROADCAST_32K: Self = SampleRate(32000);
    /// 44.1 kHz Red Book CD standard rate.
    pub const CD_44K: Self = SampleRate(44100);
    /// 48 kHz DVD / professional standard video audio rate.
    pub const STUDIO_48K: Self = SampleRate(48000);
    /// 88.2 kHz high-resolution audio rate.
    pub const HIRES_88K: Self = SampleRate(88200);
    /// 96 kHz studio master high-resolution rate.
    pub const HIRES_96K: Self = SampleRate(96000);
    /// 192 kHz ultra-high-resolution studio rate.
    pub const HIRES_192K: Self = SampleRate(192000);

    /// Creates a new SampleRate instance with validity check.
    pub fn new(hz: u32) -> BrainResult<Self> {
        if hz == 0 {
            return Err(BrainError::invalid_value("sample rate must be non-zero"));
        }
        if hz > 1_000_000 {
            return Err(BrainError::invalid_value("sample rate exceeds 1 MHz limit"));
        }
        Ok(SampleRate(hz))
    }

    /// Returns sample rate as u32 integer.
    #[inline]
    pub fn hz(&self) -> u32 {
        self.0
    }

    /// Returns sample rate as f64 float.
    #[inline]
    pub fn as_f64(&self) -> f64 {
        self.0 as f64
    }

    /// Calculates duration in seconds for a given number of samples.
    #[inline]
    pub fn duration_seconds(&self, num_samples: usize) -> f64 {
        num_samples as f64 / self.as_f64()
    }

    /// Calculates number of samples corresponding to a duration in seconds.
    #[inline]
    pub fn samples_from_duration(&self, duration_sec: f64) -> usize {
        (duration_sec * self.as_f64()).round() as usize
    }

    /// Calculates the Nyquist frequency (half the sample rate).
    #[inline]
    pub fn nyquist_hz(&self) -> f64 {
        self.as_f64() / 2.0
    }
}

impl fmt::Display for SampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.0)
    }
}

/// Channel count and layout specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Channels(pub u16);

impl Channels {
    /// Mono channel layout (1 channel).
    pub const MONO: Self = Channels(1);
    /// Stereo channel layout (2 channels).
    pub const STEREO: Self = Channels(2);
    /// 2.1 channel layout (3 channels).
    pub const STEREO_SUB: Self = Channels(3);
    /// Quadraphonic channel layout (4 channels).
    pub const QUAD: Self = Channels(4);
    /// 5.1 surround sound layout (6 channels).
    pub const SURROUND_5_1: Self = Channels(6);
    /// 7.1 surround sound layout (8 channels).
    pub const SURROUND_7_1: Self = Channels(8);

    /// Creates a Channels instance with validation.
    pub fn new(count: u16) -> BrainResult<Self> {
        if count == 0 {
            return Err(BrainError::invalid_value(
                "channel count must be at least 1",
            ));
        }
        if count > 256 {
            return Err(BrainError::invalid_value(
                "channel count exceeds maximum 256",
            ));
        }
        Ok(Channels(count))
    }

    /// Returns the number of channels as usize.
    #[inline]
    pub fn count(&self) -> usize {
        self.0 as usize
    }

    /// Returns whether this represents mono audio.
    #[inline]
    pub fn is_mono(&self) -> bool {
        self.0 == 1
    }

    /// Returns whether this represents stereo audio.
    #[inline]
    pub fn is_stereo(&self) -> bool {
        self.0 == 2
    }
}

impl fmt::Display for Channels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            1 => write!(f, "Mono (1 ch)"),
            2 => write!(f, "Stereo (2 ch)"),
            n => write!(f, "Multi-channel ({} ch)", n),
        }
    }
}

/// Audio sample bit-depth and representation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    /// 8-bit unsigned integer PCM (0..255, 128 is center).
    PcmU8,
    /// 16-bit signed integer PCM (-32768..32767).
    PcmI16,
    /// 24-bit signed integer PCM packed or aligned.
    PcmI24,
    /// 32-bit signed integer PCM.
    PcmI32,
    /// 32-bit IEEE 754 floating point (-1.0..1.0 nominal).
    Float32,
    /// 64-bit IEEE 754 double precision floating point.
    Float64,
}

impl AudioFormat {
    /// Returns the byte width per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            AudioFormat::PcmU8 => 1,
            AudioFormat::PcmI16 => 2,
            AudioFormat::PcmI24 => 3,
            AudioFormat::PcmI32 | AudioFormat::Float32 => 4,
            AudioFormat::Float64 => 8,
        }
    }

    /// Returns bit depth in bits per sample.
    pub fn bit_depth(&self) -> usize {
        self.bytes_per_sample() * 8
    }

    /// Returns whether this is a floating point format.
    pub fn is_float(&self) -> bool {
        matches!(self, AudioFormat::Float32 | AudioFormat::Float64)
    }
}

/// Multi-channel planar 64-bit floating point audio buffer.
///
/// # Examples
///
/// ```
/// use brain_audio::core::{AudioBuffer, SampleRate};
/// let buf = AudioBuffer::zeros(1, 1600, SampleRate::SPEECH_16K).unwrap();
/// assert_eq!(buf.channels(), 1);
/// assert_eq!(buf.num_samples(), 1600);
/// assert_eq!(buf.duration(), 0.1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    data: Vec<f64>,
    channels: usize,
    num_samples: usize,
    sample_rate: SampleRate,
}

impl AudioBuffer {
    /// Creates a new zero-initialized audio buffer.
    pub fn zeros(
        channels: usize,
        num_samples: usize,
        sample_rate: SampleRate,
    ) -> BrainResult<Self> {
        if channels == 0 || num_samples == 0 {
            return Err(BrainError::invalid_value(
                "channels and num_samples must be non-zero",
            ));
        }
        let total = channels.checked_mul(num_samples).ok_or_else(|| {
            BrainError::overflow("channels * samples", "usize", "AudioBuffer::zeros")
        })?;
        Ok(AudioBuffer {
            data: vec![0.0; total],
            channels,
            num_samples,
            sample_rate,
        })
    }

    /// Creates an audio buffer from planar slice data.
    pub fn from_slice(
        data: &[f64],
        channels: usize,
        num_samples: usize,
        sample_rate: SampleRate,
    ) -> BrainResult<Self> {
        if channels == 0 || num_samples == 0 {
            return Err(BrainError::invalid_value(
                "channels and num_samples must be non-zero",
            ));
        }
        if data.len() != channels * num_samples {
            return Err(BrainError::shape_mismatch(
                format!(
                    "channels({}) * samples({}) = {}",
                    channels,
                    num_samples,
                    channels * num_samples
                ),
                data.len().to_string(),
                "AudioBuffer::from_slice",
            ));
        }
        Ok(AudioBuffer {
            data: data.to_vec(),
            channels,
            num_samples,
            sample_rate,
        })
    }

    /// Creates a mono AudioBuffer from a 1D vector of samples.
    pub fn from_mono(samples: Vec<f64>, sample_rate: SampleRate) -> BrainResult<Self> {
        let num_samples = samples.len();
        if num_samples == 0 {
            return Err(BrainError::invalid_value("samples cannot be empty"));
        }
        Ok(AudioBuffer {
            data: samples,
            channels: 1,
            num_samples,
            sample_rate,
        })
    }

    /// Creates an AudioBuffer from a 1D or 2D `brain_core::Tensor`.
    pub fn from_tensor(tensor: &Tensor, sample_rate: SampleRate) -> BrainResult<Self> {
        let ndim = tensor.ndim();
        let (channels, num_samples) = match ndim {
            1 => (1, tensor.shape()[0]),
            2 => (tensor.shape()[0], tensor.shape()[1]),
            _ => {
                return Err(BrainError::invalid_value(format!(
                    "AudioBuffer requires 1D or 2D tensor, got {}D",
                    ndim
                )))
            }
        };
        Ok(AudioBuffer {
            data: tensor.data().to_vec(),
            channels,
            num_samples,
            sample_rate,
        })
    }

    /// Converts this AudioBuffer into a 2D `brain_core::Tensor` `[channels, num_samples]`.
    pub fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(&self.data, vec![self.channels, self.num_samples])
    }

    /// Returns the number of audio channels.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the number of samples per channel.
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Returns the audio sample rate.
    #[inline]
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Sets the sample rate metadata without resampling audio.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        self.sample_rate = sample_rate;
    }

    /// Returns duration in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.sample_rate.duration_seconds(self.num_samples)
    }

    /// Returns total number of elements across all channels.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice to all planar data.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable slice to all planar data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns a slice for a specific channel.
    pub fn channel(&self, ch: usize) -> BrainResult<&[f64]> {
        if ch >= self.channels {
            return Err(BrainError::index_out_of_bounds(
                ch as isize,
                self.channels,
                Some(0),
                "AudioBuffer::channel",
            ));
        }
        let start = ch * self.num_samples;
        let end = start + self.num_samples;
        Ok(&self.data[start..end])
    }

    /// Returns a mutable slice for a specific channel.
    pub fn channel_mut(&mut self, ch: usize) -> BrainResult<&mut [f64]> {
        if ch >= self.channels {
            return Err(BrainError::index_out_of_bounds(
                ch as isize,
                self.channels,
                Some(0),
                "AudioBuffer::channel_mut",
            ));
        }
        let start = ch * self.num_samples;
        let end = start + self.num_samples;
        Ok(&mut self.data[start..end])
    }

    /// Gets sample at `[channel, sample_idx]`.
    #[inline]
    pub fn get_sample(&self, ch: usize, idx: usize) -> Option<f64> {
        if ch < self.channels && idx < self.num_samples {
            Some(self.data[ch * self.num_samples + idx])
        } else {
            None
        }
    }

    /// Sets sample at `[channel, sample_idx]`.
    #[inline]
    pub fn set_sample(&mut self, ch: usize, idx: usize, val: f64) -> BrainResult<()> {
        if ch >= self.channels {
            return Err(BrainError::index_out_of_bounds(
                ch as isize,
                self.channels,
                Some(0),
                "AudioBuffer::set_sample channel",
            ));
        }
        if idx >= self.num_samples {
            return Err(BrainError::index_out_of_bounds(
                idx as isize,
                self.num_samples,
                Some(1),
                "AudioBuffer::set_sample sample_idx",
            ));
        }
        self.data[ch * self.num_samples + idx] = val;
        Ok(())
    }

    /// Slices the audio buffer temporally across all channels `[start_sample..end_sample]`.
    pub fn slice(&self, start: usize, end: usize) -> BrainResult<Self> {
        if start > end || end > self.num_samples {
            return Err(BrainError::invalid_value(format!(
                "invalid slice range [{}..{}] for length {}",
                start, end, self.num_samples
            )));
        }
        let sliced_samples = end - start;
        let mut sliced_data = Vec::with_capacity(self.channels * sliced_samples);
        for ch in 0..self.channels {
            let ch_start = ch * self.num_samples + start;
            let ch_end = ch * self.num_samples + end;
            sliced_data.extend_from_slice(&self.data[ch_start..ch_end]);
        }
        Ok(AudioBuffer {
            data: sliced_data,
            channels: self.channels,
            num_samples: sliced_samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Downmixes multi-channel audio to mono by averaging channels.
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }
        let inv_ch = 1.0 / self.channels as f64;
        let mut mono = vec![0.0; self.num_samples];
        for ch in 0..self.channels {
            let ch_data = &self.data[ch * self.num_samples..(ch + 1) * self.num_samples];
            for i in 0..self.num_samples {
                mono[i] += ch_data[i] * inv_ch;
            }
        }
        AudioBuffer {
            data: mono,
            channels: 1,
            num_samples: self.num_samples,
            sample_rate: self.sample_rate,
        }
    }

    /// Converts mono audio to multi-channel by duplicating the mono channel.
    pub fn to_stereo(&self) -> BrainResult<Self> {
        if self.channels == 2 {
            return Ok(self.clone());
        }
        if self.channels != 1 {
            return Err(BrainError::invalid_value("to_stereo requires mono input"));
        }
        let mut stereo_data = Vec::with_capacity(2 * self.num_samples);
        stereo_data.extend_from_slice(&self.data);
        stereo_data.extend_from_slice(&self.data);
        Ok(AudioBuffer {
            data: stereo_data,
            channels: 2,
            num_samples: self.num_samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Concatenates another AudioBuffer of the same channel count and sample rate.
    pub fn concat(&self, other: &AudioBuffer) -> BrainResult<Self> {
        if self.channels != other.channels {
            return Err(BrainError::shape_mismatch(
                format!("channels {}", self.channels),
                format!("channels {}", other.channels),
                "AudioBuffer::concat",
            ));
        }
        if self.sample_rate != other.sample_rate {
            return Err(BrainError::invalid_value(
                "sample rates must match to concatenate",
            ));
        }
        let out_samples = self.num_samples + other.num_samples;
        let mut out_data = Vec::with_capacity(self.channels * out_samples);
        for ch in 0..self.channels {
            let self_ch = self.channel(ch)?;
            let other_ch = other.channel(ch)?;
            out_data.extend_from_slice(self_ch);
            out_data.extend_from_slice(other_ch);
        }
        Ok(AudioBuffer {
            data: out_data,
            channels: self.channels,
            num_samples: out_samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Mixes (adds) another audio buffer with a given gain weight.
    pub fn mix(&mut self, other: &AudioBuffer, weight: f64) -> BrainResult<()> {
        if self.channels != other.channels || self.num_samples != other.num_samples {
            return Err(BrainError::shape_mismatch(
                format!("[{}, {}]", self.channels, self.num_samples),
                format!("[{}, {}]", other.channels, other.num_samples),
                "AudioBuffer::mix",
            ));
        }
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b * weight;
        }
        Ok(())
    }

    /// Scales all audio samples by a scalar multiplier.
    pub fn scale(&mut self, factor: f64) {
        for s in &mut self.data {
            *s *= factor;
        }
    }

    /// Normalizes audio to have maximum peak absolute amplitude of target_peak (default 1.0).
    pub fn normalize_peak(&mut self, target_peak: f64) {
        let max_val = self.peak_amplitude();
        if max_val > 1e-12 {
            let factor = target_peak / max_val;
            self.scale(factor);
        }
    }

    /// Computes the peak absolute amplitude across all channels.
    pub fn peak_amplitude(&self) -> f64 {
        self.data.iter().fold(0.0f64, |acc, &s| acc.max(s.abs()))
    }

    /// Computes the Root Mean Square (RMS) energy across all channels.
    pub fn rms_energy(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.data.iter().map(|&s| s * s).sum();
        (sum_sq / self.data.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
