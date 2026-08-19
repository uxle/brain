//! # Clustering & Unsupervised Metrics
//!
//! Cluster purity, Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI).
#![allow(missing_docs)]

use std::collections::HashMap;

/// Configuration for clustering metrics.
#[derive(Debug, Clone, Default)]
pub struct ClusterConfig {
    pub num_clusters: usize,
}

/// Cluster Purity: fraction of data points assigned to the majority class in each cluster.
pub fn cluster_purity(cluster_assignments: &[usize], true_labels: &[usize]) -> f64 {
    let n = cluster_assignments.len().min(true_labels.len());
    if n == 0 { return 0.0; }

    let mut cluster_class_counts: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for i in 0..n {
        let c = cluster_assignments[i];
        let y = true_labels[i];
        *cluster_class_counts.entry(c).or_default().entry(y).or_insert(0) += 1;
    }

    let mut correct_majority = 0usize;
    for class_counts in cluster_class_counts.values() {
        let max_in_cluster = class_counts.values().copied().max().unwrap_or(0);
        correct_majority += max_in_cluster;
    }

    correct_majority as f64 / n as f64
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
