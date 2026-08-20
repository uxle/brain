//! # Graph Utilities
//!
//! ID generators, name sanitizers, attribute hashers, and ASCII graph formatters.
#![allow(missing_docs)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Thread-safe ID counter generator.
#[derive(Debug, Default)]
pub struct IdGenerator {
    next_id: usize,
}

impl IdGenerator {
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn reset(&mut self) {
        self.next_id = 0;
    }
}

/// Sanitizes a string for use in DOT / JSON identifiers (alphanumeric and underscores only).
pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Computes a 64-bit hash of arbitrary key-value attribute pairs.
pub fn hash_attributes(attrs: &[(&str, &str)]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (k, v) in attrs {
        k.hash(&mut hasher);
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// Formats a simple tabular node/edge summary string.
pub fn format_graph_summary(num_nodes: usize, num_edges: usize, name: &str) -> String {
    format!("Graph '{}': {} nodes, {} edges", name, num_nodes, num_edges)
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
