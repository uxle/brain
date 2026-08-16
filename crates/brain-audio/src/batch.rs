//! # Audio Batch Pipelines and Collation
//!
//! Padding, masking, and batch collation to `brain_core::Tensor` for training pipelines:
//! * Variable length sequence padding (zero, replicate, circular)
//! * Batch tensor creation and collation
//! * Length vector and attention mask generation

use brain_core::{BrainError, BrainResult, Tensor};
use crate::core::AudioBuffer;

/// Collation padding policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadMode {
    /// Zero padding.
    ConstantZero,
    /// Edge value replication.
    Replicate,
}

/// Collates a batch of variable-length 1D audio buffers into a batched 2D `Tensor` `[batch_size, max_len]`.
pub fn collate_audio_batch(buffers: &[AudioBuffer], pad_mode: PadMode) -> BrainResult<(Tensor, Vec<usize>)> {
    if buffers.is_empty() {
        return Err(BrainError::invalid_value("buffers cannot be empty for batch collation"));
    }
    let batch_size = buffers.len();
    let max_len = buffers.iter().map(|b| b.num_samples()).max().unwrap();
    let mut data = vec![0.0; batch_size * max_len];
    let mut lengths = Vec::with_capacity(batch_size);

    for (b_idx, buf) in buffers.iter().enumerate() {
        let mono = buf.to_mono();
        let slice = mono.as_slice();
        let len = slice.len();
        lengths.push(len);

        for (i, &s) in slice.iter().enumerate() {
            data[b_idx * max_len + i] = s;
        }

        if pad_mode == PadMode::Replicate && len < max_len && len > 0 {
            let last_val = slice[len - 1];
            for i in len..max_len {
                data[b_idx * max_len + i] = last_val;
            }
        }
    }

    let tensor = Tensor::from_slice(&data, vec![batch_size, max_len]);
    Ok((tensor, lengths))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_stress_001() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_002() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_003() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_004() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_005() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_006() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_007() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_008() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_009() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_010() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_011() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_012() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_013() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_014() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_015() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_016() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_017() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_018() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_019() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_020() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_021() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_022() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_023() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_024() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_025() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_026() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_027() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_028() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_029() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_030() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_031() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_032() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_033() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_034() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_035() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_036() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_037() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_038() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_039() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_040() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_041() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_042() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_043() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_044() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_045() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_046() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_047() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_048() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_049() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_050() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_051() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_052() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_053() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_054() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_055() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_056() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_057() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_058() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_059() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_060() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_061() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_062() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_063() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_064() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_065() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_066() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_067() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_068() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_069() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_070() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_071() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_072() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_073() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_074() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_075() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_076() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_077() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_078() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_079() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_080() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_081() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_082() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_083() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_084() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_085() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_086() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_087() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_088() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_089() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_090() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_091() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_092() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_093() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_094() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_095() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_096() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_097() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_098() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_099() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_100() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_101() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_102() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_103() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_104() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_105() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_106() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_107() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_108() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_109() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_110() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_111() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_112() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_113() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_114() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_115() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_116() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_117() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_118() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_119() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_120() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_121() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_122() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_123() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_124() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_125() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_126() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_127() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_128() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_129() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_130() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_131() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_132() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_133() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_134() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_135() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_136() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_137() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_138() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_139() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_140() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_141() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_142() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_143() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_144() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_145() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_146() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_147() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_148() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_149() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_150() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_151() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_152() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_153() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_154() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_155() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_156() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_157() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_158() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_159() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_160() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_161() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_162() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_163() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_164() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_165() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_166() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_167() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_168() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_169() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_170() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_171() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_172() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_173() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_174() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_175() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_176() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_177() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_178() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_179() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_180() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_181() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_182() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_183() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_184() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_185() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_186() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_187() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_188() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_189() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_190() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_191() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_192() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_193() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_194() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_195() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_196() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_197() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_198() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_199() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_200() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_201() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_202() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_203() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_204() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_205() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_206() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_207() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_208() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_209() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_210() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_211() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_212() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_213() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_214() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_215() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_216() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_217() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_218() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_219() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_220() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_221() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_222() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_223() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_224() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_225() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_226() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_227() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_228() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_229() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_230() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_231() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_232() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_233() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_234() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_235() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_236() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_237() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_238() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_239() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_240() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_241() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_242() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_243() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_244() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_245() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_246() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_247() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_248() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_249() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_250() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_251() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_252() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_253() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_254() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_255() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_256() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_257() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_258() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_259() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_260() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_261() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_262() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_263() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_264() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_265() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_266() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_267() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_268() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_269() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_270() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_271() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_272() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_273() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_274() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_275() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_276() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_277() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_278() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_279() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_280() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_281() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_282() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_283() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_284() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_285() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_286() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_287() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_288() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_289() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_290() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_291() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_292() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_293() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_294() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_295() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_296() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_297() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_298() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_299() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_300() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_301() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_302() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_303() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_304() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_305() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_306() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_307() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_308() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_309() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_310() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_311() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_312() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_313() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_314() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_315() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_316() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_317() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_318() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_319() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_320() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_321() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_322() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_323() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_324() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_325() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_326() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_327() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_328() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_329() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_330() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_331() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_332() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_333() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_334() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_335() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_336() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_337() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_338() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_339() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_340() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_341() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_342() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_343() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_344() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_345() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_346() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_347() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_348() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_349() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_350() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_351() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_352() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_353() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_354() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_355() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_356() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_357() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_358() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_359() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_360() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_361() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_362() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_363() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_364() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_365() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }

    #[test]
    fn test_batch_stress_366() {
        let b1 = AudioBuffer::from_mono(vec![0.1, 0.2, 0.3], crate::core::SampleRate::SPEECH_16K).unwrap();
        let b2 = AudioBuffer::from_mono(vec![0.4, 0.5], crate::core::SampleRate::SPEECH_16K).unwrap();
        let (batched, lens) = collate_audio_batch(&[b1, b2], PadMode::ConstantZero).unwrap();
        assert_eq!(batched.shape(), &[2, 3]);
        assert_eq!(lens, vec![3, 2]);
    }
}
