//! # Comprehensive Dataset & DataLoader Pipeline Integration Tests

use brain_core::Tensor;
use brain_data::core::{DataSource, Sample, SampleBatch};
use brain_data::samplers::{Sampler, SequentialSampler, DistributedSampler};
use brain_data::batch::BatchIter;
use brain_data::collate::default_collate;
use brain_data::loading::MemoryLoader;

#[test]
fn test_memory_loader_datasource() {
    let tensors = vec![
        Tensor::from_slice(&[1.0, 2.0], vec![2]),
        Tensor::from_slice(&[3.0, 4.0], vec![2]),
        Tensor::from_slice(&[5.0, 6.0], vec![2]),
    ];
    let loader = MemoryLoader::from_tensors(tensors);

    assert_eq!(loader.len(), 3);
    assert!(!loader.is_empty());

    let s0 = loader.get(0).expect("Sample 0");
    assert_eq!(s0.id, 0);
    assert_eq!(s0.data.to_vec(), vec![1.0, 2.0]);

    let s2 = loader.get(2).expect("Sample 2");
    assert_eq!(s2.id, 2);
    assert_eq!(s2.data.to_vec(), vec![5.0, 6.0]);

    assert!(loader.get(3).is_none());
}

#[test]
fn test_sequential_and_distributed_samplers() {
    // Sequential sampler
    let seq = SequentialSampler::new(10);
    assert_eq!(seq.len(), 10);
    assert_eq!(seq.sample_indices(), (0..10).collect::<Vec<_>>());

    // Distributed sampler with 3 replicas
    let dist_rank0 = DistributedSampler::new(10, 3, 0);
    let dist_rank1 = DistributedSampler::new(10, 3, 1);
    let dist_rank2 = DistributedSampler::new(10, 3, 2);

    let idx0 = dist_rank0.sample_indices(); // 0, 3, 6, 9
    let idx1 = dist_rank1.sample_indices(); // 1, 4, 7
    let idx2 = dist_rank2.sample_indices(); // 2, 5, 8

    assert_eq!(idx0, vec![0, 3, 6, 9]);
    assert_eq!(idx1, vec![1, 4, 7]);
    assert_eq!(idx2, vec![2, 5, 8]);

    // All indices disjoint and cover 0..10
    let mut union = Vec::new();
    union.extend(idx0);
    union.extend(idx1);
    union.extend(idx2);
    union.sort();
    assert_eq!(union, (0..10).collect::<Vec<_>>());
}

#[test]
fn test_batch_iterator_drop_last() {
    let samples: Vec<Sample> = (0..5)
        .map(|i| Sample::new(i, Tensor::from_slice(&[i as f64], vec![1])))
        .collect();

    // Batch size 2 without drop_last -> yields [2, 2, 1]
    let iter1 = BatchIter::new(samples.clone().into_iter(), 2, false);
    let batches1: Vec<SampleBatch> = iter1.collect();
    assert_eq!(batches1.len(), 3);
    assert_eq!(batches1[0].len(), 2);
    assert_eq!(batches1[1].len(), 2);
    assert_eq!(batches1[2].len(), 1);

    // Batch size 2 with drop_last -> yields [2, 2]
    let iter2 = BatchIter::new(samples.into_iter(), 2, true);
    let batches2: Vec<SampleBatch> = iter2.collect();
    assert_eq!(batches2.len(), 2);
    assert_eq!(batches2[0].len(), 2);
    assert_eq!(batches2[1].len(), 2);
}

#[test]
fn test_default_collation() {
    let s1 = Sample::new(0, Tensor::from_slice(&[1.0, 2.0], vec![2])).with_label(Tensor::scalar(0.0));
    let s2 = Sample::new(1, Tensor::from_slice(&[3.0, 4.0], vec![2])).with_label(Tensor::scalar(1.0));

    let batch = default_collate(&[s1, s2]);
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.samples[0].id, 0);
    assert_eq!(batch.samples[1].id, 1);
}
