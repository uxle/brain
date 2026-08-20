//! # Audio Batch Pipelines and Collation
//!
//! Padding, masking, and batch collation to `brain_core::Tensor` for training pipelines:
//! * Variable length sequence padding (zero, replicate, circular)
//! * Batch tensor creation and collation
//! * Length vector and attention mask generation

use crate::core::AudioBuffer;
use brain_core::{BrainError, BrainResult, Tensor};

/// Collation padding policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadMode {
    /// Zero padding.
    ConstantZero,
    /// Edge value replication.
    Replicate,
}

/// Collates a batch of variable-length 1D audio buffers into a batched 2D `Tensor` `[batch_size, max_len]`.
pub fn collate_audio_batch(
    buffers: &[AudioBuffer],
    pad_mode: PadMode,
) -> BrainResult<(Tensor, Vec<usize>)> {
    if buffers.is_empty() {
        return Err(BrainError::invalid_value(
            "buffers cannot be empty for batch collation",
        ));
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
}
