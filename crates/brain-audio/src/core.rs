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
            return Err(BrainError::invalid_value("channel count must be at least 1"));
        }
        if count > 256 {
            return Err(BrainError::invalid_value("channel count exceeds maximum 256"));
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
    pub fn zeros(channels: usize, num_samples: usize, sample_rate: SampleRate) -> BrainResult<Self> {
        if channels == 0 || num_samples == 0 {
            return Err(BrainError::invalid_value("channels and num_samples must be non-zero"));
        }
        let total = channels.checked_mul(num_samples)
            .ok_or_else(|| BrainError::overflow("channels * samples", "usize", "AudioBuffer::zeros"))?;
        Ok(AudioBuffer {
            data: vec![0.0; total],
            channels,
            num_samples,
            sample_rate,
        })
    }

    /// Creates an audio buffer from planar slice data.
    pub fn from_slice(data: &[f64], channels: usize, num_samples: usize, sample_rate: SampleRate) -> BrainResult<Self> {
        if channels == 0 || num_samples == 0 {
            return Err(BrainError::invalid_value("channels and num_samples must be non-zero"));
        }
        if data.len() != channels * num_samples {
            return Err(BrainError::shape_mismatch(
                format!("channels({}) * samples({}) = {}", channels, num_samples, channels * num_samples),
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
            _ => return Err(BrainError::invalid_value(format!("AudioBuffer requires 1D or 2D tensor, got {}D", ndim))),
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
            return Err(BrainError::index_out_of_bounds(ch as isize, self.channels, Some(0), "AudioBuffer::channel"));
        }
        let start = ch * self.num_samples;
        let end = start + self.num_samples;
        Ok(&self.data[start..end])
    }

    /// Returns a mutable slice for a specific channel.
    pub fn channel_mut(&mut self, ch: usize) -> BrainResult<&mut [f64]> {
        if ch >= self.channels {
            return Err(BrainError::index_out_of_bounds(ch as isize, self.channels, Some(0), "AudioBuffer::channel_mut"));
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
            return Err(BrainError::index_out_of_bounds(ch as isize, self.channels, Some(0), "AudioBuffer::set_sample channel"));
        }
        if idx >= self.num_samples {
            return Err(BrainError::index_out_of_bounds(idx as isize, self.num_samples, Some(1), "AudioBuffer::set_sample sample_idx"));
        }
        self.data[ch * self.num_samples + idx] = val;
        Ok(())
    }

    /// Slices the audio buffer temporally across all channels `[start_sample..end_sample]`.
    pub fn slice(&self, start: usize, end: usize) -> BrainResult<Self> {
        if start > end || end > self.num_samples {
            return Err(BrainError::invalid_value(format!("invalid slice range [{}..{}] for length {}", start, end, self.num_samples)));
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
            return Err(BrainError::invalid_value("sample rates must match to concatenate"));
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

    #[test]
    fn test_audio_core_stress_001() {
        let sr = SampleRate::new(16100).unwrap();
        let ch = ((1 % 4) + 1) as usize;
        let num_samples = 128 + (1 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 1) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_002() {
        let sr = SampleRate::new(16200).unwrap();
        let ch = ((2 % 4) + 1) as usize;
        let num_samples = 128 + (2 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 2) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_003() {
        let sr = SampleRate::new(16300).unwrap();
        let ch = ((3 % 4) + 1) as usize;
        let num_samples = 128 + (3 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 3) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_004() {
        let sr = SampleRate::new(16400).unwrap();
        let ch = ((4 % 4) + 1) as usize;
        let num_samples = 128 + (4 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 4) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_005() {
        let sr = SampleRate::new(16500).unwrap();
        let ch = ((5 % 4) + 1) as usize;
        let num_samples = 128 + (5 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 5) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_006() {
        let sr = SampleRate::new(16600).unwrap();
        let ch = ((6 % 4) + 1) as usize;
        let num_samples = 128 + (6 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 6) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_007() {
        let sr = SampleRate::new(16700).unwrap();
        let ch = ((7 % 4) + 1) as usize;
        let num_samples = 128 + (7 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 7) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_008() {
        let sr = SampleRate::new(16800).unwrap();
        let ch = ((8 % 4) + 1) as usize;
        let num_samples = 128 + (8 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 8) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_009() {
        let sr = SampleRate::new(16900).unwrap();
        let ch = ((9 % 4) + 1) as usize;
        let num_samples = 128 + (9 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 9) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_010() {
        let sr = SampleRate::new(17000).unwrap();
        let ch = ((10 % 4) + 1) as usize;
        let num_samples = 128 + (10 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 10) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_011() {
        let sr = SampleRate::new(17100).unwrap();
        let ch = ((11 % 4) + 1) as usize;
        let num_samples = 128 + (11 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 11) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_012() {
        let sr = SampleRate::new(17200).unwrap();
        let ch = ((12 % 4) + 1) as usize;
        let num_samples = 128 + (12 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 12) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_013() {
        let sr = SampleRate::new(17300).unwrap();
        let ch = ((13 % 4) + 1) as usize;
        let num_samples = 128 + (13 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 13) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_014() {
        let sr = SampleRate::new(17400).unwrap();
        let ch = ((14 % 4) + 1) as usize;
        let num_samples = 128 + (14 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 14) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_015() {
        let sr = SampleRate::new(17500).unwrap();
        let ch = ((15 % 4) + 1) as usize;
        let num_samples = 128 + (15 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 15) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_016() {
        let sr = SampleRate::new(17600).unwrap();
        let ch = ((16 % 4) + 1) as usize;
        let num_samples = 128 + (16 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 16) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_017() {
        let sr = SampleRate::new(17700).unwrap();
        let ch = ((17 % 4) + 1) as usize;
        let num_samples = 128 + (17 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 17) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_018() {
        let sr = SampleRate::new(17800).unwrap();
        let ch = ((18 % 4) + 1) as usize;
        let num_samples = 128 + (18 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 18) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_019() {
        let sr = SampleRate::new(17900).unwrap();
        let ch = ((19 % 4) + 1) as usize;
        let num_samples = 128 + (19 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 19) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_020() {
        let sr = SampleRate::new(18000).unwrap();
        let ch = ((20 % 4) + 1) as usize;
        let num_samples = 128 + (20 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 20) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_021() {
        let sr = SampleRate::new(18100).unwrap();
        let ch = ((21 % 4) + 1) as usize;
        let num_samples = 128 + (21 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 21) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_022() {
        let sr = SampleRate::new(18200).unwrap();
        let ch = ((22 % 4) + 1) as usize;
        let num_samples = 128 + (22 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 22) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_023() {
        let sr = SampleRate::new(18300).unwrap();
        let ch = ((23 % 4) + 1) as usize;
        let num_samples = 128 + (23 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 23) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_024() {
        let sr = SampleRate::new(18400).unwrap();
        let ch = ((24 % 4) + 1) as usize;
        let num_samples = 128 + (24 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 24) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_025() {
        let sr = SampleRate::new(18500).unwrap();
        let ch = ((25 % 4) + 1) as usize;
        let num_samples = 128 + (25 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 25) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_026() {
        let sr = SampleRate::new(18600).unwrap();
        let ch = ((26 % 4) + 1) as usize;
        let num_samples = 128 + (26 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 26) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_027() {
        let sr = SampleRate::new(18700).unwrap();
        let ch = ((27 % 4) + 1) as usize;
        let num_samples = 128 + (27 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 27) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_028() {
        let sr = SampleRate::new(18800).unwrap();
        let ch = ((28 % 4) + 1) as usize;
        let num_samples = 128 + (28 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 28) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_029() {
        let sr = SampleRate::new(18900).unwrap();
        let ch = ((29 % 4) + 1) as usize;
        let num_samples = 128 + (29 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 29) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_030() {
        let sr = SampleRate::new(19000).unwrap();
        let ch = ((30 % 4) + 1) as usize;
        let num_samples = 128 + (30 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 30) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_031() {
        let sr = SampleRate::new(19100).unwrap();
        let ch = ((31 % 4) + 1) as usize;
        let num_samples = 128 + (31 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 31) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_032() {
        let sr = SampleRate::new(19200).unwrap();
        let ch = ((32 % 4) + 1) as usize;
        let num_samples = 128 + (32 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 32) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_033() {
        let sr = SampleRate::new(19300).unwrap();
        let ch = ((33 % 4) + 1) as usize;
        let num_samples = 128 + (33 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 33) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_034() {
        let sr = SampleRate::new(19400).unwrap();
        let ch = ((34 % 4) + 1) as usize;
        let num_samples = 128 + (34 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 34) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_035() {
        let sr = SampleRate::new(19500).unwrap();
        let ch = ((35 % 4) + 1) as usize;
        let num_samples = 128 + (35 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 35) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_036() {
        let sr = SampleRate::new(19600).unwrap();
        let ch = ((36 % 4) + 1) as usize;
        let num_samples = 128 + (36 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 36) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_037() {
        let sr = SampleRate::new(19700).unwrap();
        let ch = ((37 % 4) + 1) as usize;
        let num_samples = 128 + (37 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 37) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_038() {
        let sr = SampleRate::new(19800).unwrap();
        let ch = ((38 % 4) + 1) as usize;
        let num_samples = 128 + (38 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 38) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_039() {
        let sr = SampleRate::new(19900).unwrap();
        let ch = ((39 % 4) + 1) as usize;
        let num_samples = 128 + (39 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 39) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_040() {
        let sr = SampleRate::new(20000).unwrap();
        let ch = ((40 % 4) + 1) as usize;
        let num_samples = 128 + (40 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 40) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_041() {
        let sr = SampleRate::new(20100).unwrap();
        let ch = ((41 % 4) + 1) as usize;
        let num_samples = 128 + (41 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 41) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_042() {
        let sr = SampleRate::new(20200).unwrap();
        let ch = ((42 % 4) + 1) as usize;
        let num_samples = 128 + (42 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 42) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_043() {
        let sr = SampleRate::new(20300).unwrap();
        let ch = ((43 % 4) + 1) as usize;
        let num_samples = 128 + (43 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 43) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_044() {
        let sr = SampleRate::new(20400).unwrap();
        let ch = ((44 % 4) + 1) as usize;
        let num_samples = 128 + (44 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 44) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_045() {
        let sr = SampleRate::new(20500).unwrap();
        let ch = ((45 % 4) + 1) as usize;
        let num_samples = 128 + (45 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 45) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_046() {
        let sr = SampleRate::new(20600).unwrap();
        let ch = ((46 % 4) + 1) as usize;
        let num_samples = 128 + (46 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 46) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_047() {
        let sr = SampleRate::new(20700).unwrap();
        let ch = ((47 % 4) + 1) as usize;
        let num_samples = 128 + (47 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 47) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_048() {
        let sr = SampleRate::new(20800).unwrap();
        let ch = ((48 % 4) + 1) as usize;
        let num_samples = 128 + (48 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 48) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_049() {
        let sr = SampleRate::new(20900).unwrap();
        let ch = ((49 % 4) + 1) as usize;
        let num_samples = 128 + (49 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 49) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_050() {
        let sr = SampleRate::new(21000).unwrap();
        let ch = ((50 % 4) + 1) as usize;
        let num_samples = 128 + (50 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 50) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_051() {
        let sr = SampleRate::new(21100).unwrap();
        let ch = ((51 % 4) + 1) as usize;
        let num_samples = 128 + (51 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 51) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_052() {
        let sr = SampleRate::new(21200).unwrap();
        let ch = ((52 % 4) + 1) as usize;
        let num_samples = 128 + (52 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 52) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_053() {
        let sr = SampleRate::new(21300).unwrap();
        let ch = ((53 % 4) + 1) as usize;
        let num_samples = 128 + (53 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 53) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_054() {
        let sr = SampleRate::new(21400).unwrap();
        let ch = ((54 % 4) + 1) as usize;
        let num_samples = 128 + (54 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 54) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_055() {
        let sr = SampleRate::new(21500).unwrap();
        let ch = ((55 % 4) + 1) as usize;
        let num_samples = 128 + (55 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 55) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_056() {
        let sr = SampleRate::new(21600).unwrap();
        let ch = ((56 % 4) + 1) as usize;
        let num_samples = 128 + (56 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 56) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_057() {
        let sr = SampleRate::new(21700).unwrap();
        let ch = ((57 % 4) + 1) as usize;
        let num_samples = 128 + (57 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 57) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_058() {
        let sr = SampleRate::new(21800).unwrap();
        let ch = ((58 % 4) + 1) as usize;
        let num_samples = 128 + (58 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 58) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_059() {
        let sr = SampleRate::new(21900).unwrap();
        let ch = ((59 % 4) + 1) as usize;
        let num_samples = 128 + (59 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 59) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_060() {
        let sr = SampleRate::new(22000).unwrap();
        let ch = ((60 % 4) + 1) as usize;
        let num_samples = 128 + (60 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 60) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_061() {
        let sr = SampleRate::new(22100).unwrap();
        let ch = ((61 % 4) + 1) as usize;
        let num_samples = 128 + (61 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 61) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_062() {
        let sr = SampleRate::new(22200).unwrap();
        let ch = ((62 % 4) + 1) as usize;
        let num_samples = 128 + (62 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 62) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_063() {
        let sr = SampleRate::new(22300).unwrap();
        let ch = ((63 % 4) + 1) as usize;
        let num_samples = 128 + (63 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 63) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_064() {
        let sr = SampleRate::new(22400).unwrap();
        let ch = ((64 % 4) + 1) as usize;
        let num_samples = 128 + (64 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 64) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_065() {
        let sr = SampleRate::new(22500).unwrap();
        let ch = ((65 % 4) + 1) as usize;
        let num_samples = 128 + (65 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 65) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_066() {
        let sr = SampleRate::new(22600).unwrap();
        let ch = ((66 % 4) + 1) as usize;
        let num_samples = 128 + (66 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 66) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_067() {
        let sr = SampleRate::new(22700).unwrap();
        let ch = ((67 % 4) + 1) as usize;
        let num_samples = 128 + (67 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 67) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_068() {
        let sr = SampleRate::new(22800).unwrap();
        let ch = ((68 % 4) + 1) as usize;
        let num_samples = 128 + (68 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 68) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_069() {
        let sr = SampleRate::new(22900).unwrap();
        let ch = ((69 % 4) + 1) as usize;
        let num_samples = 128 + (69 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 69) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_070() {
        let sr = SampleRate::new(23000).unwrap();
        let ch = ((70 % 4) + 1) as usize;
        let num_samples = 128 + (70 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 70) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_071() {
        let sr = SampleRate::new(23100).unwrap();
        let ch = ((71 % 4) + 1) as usize;
        let num_samples = 128 + (71 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 71) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_072() {
        let sr = SampleRate::new(23200).unwrap();
        let ch = ((72 % 4) + 1) as usize;
        let num_samples = 128 + (72 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 72) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_073() {
        let sr = SampleRate::new(23300).unwrap();
        let ch = ((73 % 4) + 1) as usize;
        let num_samples = 128 + (73 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 73) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_074() {
        let sr = SampleRate::new(23400).unwrap();
        let ch = ((74 % 4) + 1) as usize;
        let num_samples = 128 + (74 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 74) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_075() {
        let sr = SampleRate::new(23500).unwrap();
        let ch = ((75 % 4) + 1) as usize;
        let num_samples = 128 + (75 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 75) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_076() {
        let sr = SampleRate::new(23600).unwrap();
        let ch = ((76 % 4) + 1) as usize;
        let num_samples = 128 + (76 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 76) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_077() {
        let sr = SampleRate::new(23700).unwrap();
        let ch = ((77 % 4) + 1) as usize;
        let num_samples = 128 + (77 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 77) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_078() {
        let sr = SampleRate::new(23800).unwrap();
        let ch = ((78 % 4) + 1) as usize;
        let num_samples = 128 + (78 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 78) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_079() {
        let sr = SampleRate::new(23900).unwrap();
        let ch = ((79 % 4) + 1) as usize;
        let num_samples = 128 + (79 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 79) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_080() {
        let sr = SampleRate::new(24000).unwrap();
        let ch = ((80 % 4) + 1) as usize;
        let num_samples = 128 + (80 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 80) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_081() {
        let sr = SampleRate::new(24100).unwrap();
        let ch = ((81 % 4) + 1) as usize;
        let num_samples = 128 + (81 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 81) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_082() {
        let sr = SampleRate::new(24200).unwrap();
        let ch = ((82 % 4) + 1) as usize;
        let num_samples = 128 + (82 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 82) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_083() {
        let sr = SampleRate::new(24300).unwrap();
        let ch = ((83 % 4) + 1) as usize;
        let num_samples = 128 + (83 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 83) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_084() {
        let sr = SampleRate::new(24400).unwrap();
        let ch = ((84 % 4) + 1) as usize;
        let num_samples = 128 + (84 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 84) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_085() {
        let sr = SampleRate::new(24500).unwrap();
        let ch = ((85 % 4) + 1) as usize;
        let num_samples = 128 + (85 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 85) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_086() {
        let sr = SampleRate::new(24600).unwrap();
        let ch = ((86 % 4) + 1) as usize;
        let num_samples = 128 + (86 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 86) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_087() {
        let sr = SampleRate::new(24700).unwrap();
        let ch = ((87 % 4) + 1) as usize;
        let num_samples = 128 + (87 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 87) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_088() {
        let sr = SampleRate::new(24800).unwrap();
        let ch = ((88 % 4) + 1) as usize;
        let num_samples = 128 + (88 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 88) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_audio_core_stress_089() {
        let sr = SampleRate::new(24900).unwrap();
        let ch = ((89 % 4) + 1) as usize;
        let num_samples = 128 + (89 * 7) % 256;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        assert_eq!(buf.channels(), ch);
        assert_eq!(buf.num_samples(), num_samples);
        assert_eq!(buf.duration(), sr.duration_seconds(num_samples));
        
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 100 + s + 89) as f64 * 0.01).sin();
                buf.set_sample(c, s, val).unwrap();
                assert_eq!(buf.get_sample(c, s), Some(val));
            }
        }
        
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_samples(), num_samples);
        
        let sliced = buf.slice(10, 50).unwrap();
        assert_eq!(sliced.num_samples(), 40);
        assert_eq!(sliced.channels(), ch);
        
        let t = buf.to_tensor();
        assert_eq!(t.shape(), &[ch, num_samples]);
        let buf2 = AudioBuffer::from_tensor(&t, sr).unwrap();
        assert_eq!(buf, buf2);
    }
}
