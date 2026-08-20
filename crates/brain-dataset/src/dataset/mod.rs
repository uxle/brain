//! # Abstract Dataset Interface & Combinators
//!
//! Provides the primary [`Dataset`] trait, [`TensorDataset`], [`ConcatDataset`], and [`Subset`].

pub mod audio;
pub mod tabular;
pub mod text;
pub mod vision;
pub mod vision_v2;

use crate::core::Item;
use brain_core::Tensor;

/// Abstract random-access dataset.
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, idx: usize) -> Option<Item>;
}

/// Slices a subset of an underlying dataset using an index map.
pub struct Subset<'a, D: Dataset> {
    pub dataset: &'a D,
    pub indices: Vec<usize>,
}

impl<'a, D: Dataset> Subset<'a, D> {
    /// Creates a new `Subset`.
    pub fn new(dataset: &'a D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<'a, D: Dataset> Dataset for Subset<'a, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, idx: usize) -> Option<Item> {
        self.indices
            .get(idx)
            .and_then(|&real_idx| self.dataset.get(real_idx))
    }
}

/// In-memory dataset wrapping feature and target tensors (PyTorch `TensorDataset` equivalent).
pub struct TensorDataset {
    data: Tensor,
    targets: Option<Tensor>,
}

impl TensorDataset {
    /// Creates a new `TensorDataset` from a features tensor and optional targets.
    pub fn new(data: Tensor, targets: Option<Tensor>) -> Self {
        if let Some(ref t) = targets {
            assert_eq!(
                data.shape()[0],
                t.shape()[0],
                "Data and targets must have matching batch dimension"
            );
        }
        Self { data, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        if self.data.shape().is_empty() {
            0
        } else {
            self.data.shape()[0]
        }
    }

    fn get(&self, idx: usize) -> Option<Item> {
        if idx >= self.len() {
            return None;
        }

        let slice_shape = self.data.shape()[1..].to_vec();
        let numel_per_item: usize = if slice_shape.is_empty() {
            1
        } else {
            slice_shape.iter().product()
        };

        let start = idx * numel_per_item;
        let end = start + numel_per_item;
        let item_data = Tensor::from_slice(&self.data.data()[start..end], slice_shape);

        let target_data = self.targets.as_ref().map(|t| {
            let t_slice_shape = t.shape()[1..].to_vec();
            let t_numel: usize = if t_slice_shape.is_empty() {
                1
            } else {
                t_slice_shape.iter().product()
            };
            let t_start = idx * t_numel;
            let t_end = t_start + t_numel;
            Tensor::from_slice(&t.data()[t_start..t_end], t_slice_shape)
        });

        let mut item = Item::new(idx, item_data);
        if let Some(target) = target_data {
            item = item.with_target(target);
        }
        Some(item)
    }
}

/// Concatenates multiple datasets into a single unified dataset.
pub struct ConcatDataset<'a, D: Dataset> {
    datasets: Vec<&'a D>,
    cumulative_sizes: Vec<usize>,
}

impl<'a, D: Dataset> ConcatDataset<'a, D> {
    /// Creates a new `ConcatDataset`.
    pub fn new(datasets: Vec<&'a D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut cum = 0;
        for d in &datasets {
            cum += d.len();
            cumulative_sizes.push(cum);
        }
        Self {
            datasets,
            cumulative_sizes,
        }
    }
}

impl<'a, D: Dataset> Dataset for ConcatDataset<'a, D> {
    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get(&self, idx: usize) -> Option<Item> {
        if idx >= self.len() {
            return None;
        }

        let dataset_idx = match self.cumulative_sizes.binary_search(&idx) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        let sample_idx = if dataset_idx == 0 {
            idx
        } else {
            idx - self.cumulative_sizes[dataset_idx - 1]
        };

        self.datasets
            .get(dataset_idx)
            .and_then(|d| d.get(sample_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dataset_and_subset() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let targets = Tensor::from_slice(&[0.0, 1.0, 2.0], vec![3, 1]);
        let ds = TensorDataset::new(data, Some(targets));

        assert_eq!(ds.len(), 3);
        let item1 = ds.get(1).unwrap();
        assert_eq!(item1.id, 1);
        assert_eq!(item1.data.data(), &[3.0, 4.0]);
        assert_eq!(item1.target.unwrap().data(), &[1.0]);

        let subset = Subset::new(&ds, vec![2, 0]);
        assert_eq!(subset.len(), 2);
        let s0 = subset.get(0).unwrap();
        assert_eq!(s0.data.data(), &[5.0, 6.0]);
    }
}
