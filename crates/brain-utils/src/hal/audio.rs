//! # Audio Capture Abstraction (ALSA / PCM / Mock)
//!
//! Provides raw audio stream buffers and spectral representations.

use brain_core::Tensor;
use std::sync::{Arc, Mutex};

/// Audio Chunk representation.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl AudioChunk {
    pub fn to_tensor(&self) -> Tensor {
        let data: Vec<f64> = self.samples.iter().map(|&s| s as f64).collect();
        let len = data.len();
        Tensor::from_vec(data, vec![1, len])
    }
}

/// Abstract Audio Source Trait.
pub trait AudioSource: Send + Sync {
    /// Captures the latest audio chunk.
    fn capture_chunk(&self) -> Result<AudioChunk, String>;
}

/// Mock Audio Source generating synthetic sinusoidal tones.
#[derive(Debug, Clone)]
pub struct MockAudioSource {
    pub sample_rate: u32,
    pub chunk_size: usize,
    pub phase: Arc<Mutex<f32>>,
}

impl MockAudioSource {
    pub fn new(sample_rate: u32, chunk_size: usize) -> Self {
        Self {
            sample_rate,
            chunk_size,
            phase: Arc::new(Mutex::new(0.0)),
        }
    }
}

impl AudioSource for MockAudioSource {
    fn capture_chunk(&self) -> Result<AudioChunk, String> {
        let mut p = self.phase.lock().unwrap();
        let mut samples = Vec::with_capacity(self.chunk_size);
        let freq = 440.0f32; // A4
        let step = (2.0 * std::f32::consts::PI * freq) / (self.sample_rate as f32);

        for _ in 0..self.chunk_size {
            samples.push(p.sin());
            *p += step;
        }

        Ok(AudioChunk {
            sample_rate: self.sample_rate,
            channels: 1,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_audio_to_tensor() {
        let source = MockAudioSource::new(16000, 1600);
        let chunk = source.capture_chunk().unwrap();
        assert_eq!(chunk.samples.len(), 1600);

        let t = chunk.to_tensor();
        assert_eq!(t.shape(), &[1, 1600]);
    }
}
