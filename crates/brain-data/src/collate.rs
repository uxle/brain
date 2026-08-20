//! # Sample Collation & Stacking
//!
//! Provides [`default_collate`] (tensor stacking), [`pad_collate`] (variable length sequences), and sequence masking.

use crate::core::{Sample, SampleBatch};
use brain_core::Tensor;

/// Collation function trait.
pub trait CollateFn: Send + Sync {
    fn collate(&self, samples: &[Sample]) -> SampleBatch;
}

/// Default collation function creating a batch from samples.
pub fn default_collate(samples: &[Sample]) -> SampleBatch {
    SampleBatch::new(samples.to_vec())
}

/// Stacks sample data tensors along dimension 0 into a single batched `Tensor` `[B, D1, D2, ...]`.
/// Also stacks labels if all samples contain labels.
pub fn stack_samples_to_tensor(samples: &[Sample]) -> (Tensor, Option<Tensor>) {
    assert!(!samples.is_empty(), "Cannot collate empty sample slice");

    let b = samples.len();
    let sample_shape = samples[0].data.shape();
    let numel_per_sample = samples[0].data.numel();

    let mut batched_data = Vec::with_capacity(b * numel_per_sample);
    for s in samples {
        assert_eq!(
            s.data.shape(),
            sample_shape,
            "Sample data shapes must match in default stacking collation"
        );
        batched_data.extend_from_slice(s.data.data());
    }

    let mut batched_shape = vec![b];
    batched_shape.extend_from_slice(sample_shape);
    let data_tensor = Tensor::from_slice(&batched_data, batched_shape);

    let labels_present = samples.iter().all(|s| s.label.is_some());
    let label_tensor = if labels_present {
        let label_shape = samples[0].label.as_ref().unwrap().shape();
        let label_numel = samples[0].label.as_ref().unwrap().numel();
        let mut batched_labels = Vec::with_capacity(b * label_numel);
        for s in samples {
            batched_labels.extend_from_slice(s.label.as_ref().unwrap().data());
        }
        let mut b_label_shape = vec![b];
        b_label_shape.extend_from_slice(label_shape);
        Some(Tensor::from_slice(&batched_labels, b_label_shape))
    } else {
        None
    };

    (data_tensor, label_tensor)
}

/// Collation function padding 1D variable-length sequences to `max_len` in batch.
/// Returns `(padded_data_tensor [B, max_len], mask_tensor [B, max_len])` where mask is 1.0 for valid tokens and 0.0 for padding.
pub fn pad_and_stack_sequences(samples: &[Sample], pad_value: f64) -> (Tensor, Tensor) {
    assert!(!samples.is_empty(), "Cannot pad empty sample slice");

    let b = samples.len();
    let max_len = samples.iter().map(|s| s.data.numel()).max().unwrap_or(0);

    let mut padded_data = vec![pad_value; b * max_len];
    let mut mask_data = vec![0.0f64; b * max_len];

    for (i, s) in samples.iter().enumerate() {
        let seq = s.data.data();
        let seq_len = seq.len();
        for t in 0..seq_len {
            padded_data[i * max_len + t] = seq[t];
            mask_data[i * max_len + t] = 1.0;
        }
    }

    let data_tensor = Tensor::from_slice(&padded_data, vec![b, max_len]);
    let mask_tensor = Tensor::from_slice(&mask_data, vec![b, max_len]);
    (data_tensor, mask_tensor)
}

/// Collation function padding variable-length tensors.
pub fn pad_collate(samples: &[Sample], pad_value: f64) -> SampleBatch {
    let (padded_data, _mask) = pad_and_stack_sequences(samples, pad_value);
    let mut new_samples = Vec::with_capacity(samples.len());
    let max_len = padded_data.shape()[1];
    for (i, s) in samples.iter().enumerate() {
        let row = &padded_data.data()[i * max_len..(i + 1) * max_len];
        let mut new_s = Sample::new(s.id, Tensor::from_slice(row, vec![max_len]));
        if let Some(ref l) = s.label {
            new_s = new_s.with_label(l.clone());
        }
        new_samples.push(new_s);
    }
    SampleBatch::new(new_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_samples_to_tensor() {
        let s1 = Sample::new(0, Tensor::from_slice(&[1.0, 2.0], vec![2]))
            .with_label(Tensor::scalar(0.0));
        let s2 = Sample::new(1, Tensor::from_slice(&[3.0, 4.0], vec![2]))
            .with_label(Tensor::scalar(1.0));

        let (data, labels) = stack_samples_to_tensor(&[s1, s2]);
        assert_eq!(data.shape(), &[2, 2]);
        assert_eq!(data.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(labels.is_some());
        assert_eq!(labels.unwrap().data(), &[0.0, 1.0]);
    }

    #[test]
    fn test_pad_and_stack_sequences() {
        let s1 = Sample::new(0, Tensor::from_slice(&[10.0, 20.0], vec![2]));
        let s2 = Sample::new(1, Tensor::from_slice(&[30.0, 40.0, 50.0], vec![3]));

        let (padded, mask) = pad_and_stack_sequences(&[s1, s2], 0.0);
        assert_eq!(padded.shape(), &[2, 3]);
        assert_eq!(mask.shape(), &[2, 3]);

        assert_eq!(padded.data(), &[10.0, 20.0, 0.0, 30.0, 40.0, 50.0]);
        assert_eq!(mask.data(), &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }
}
