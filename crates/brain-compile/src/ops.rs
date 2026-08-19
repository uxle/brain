//! # Compiler Operation Metadata & Cost Classification
//!
//! Categorizes operations and provides operational intensity and FLOP metrics for cost modeling.

/// High-level categorization of IR operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    UnaryElementwise,
    BinaryElementwise,
    Reduction,
    MatrixMultiplication,
    Convolution,
    MemoryMovement,
    FusedKernel,
}

/// Operational and memory cost metadata for an operation.
#[derive(Debug, Clone, Copy)]
pub struct OpCostInfo {
    pub flops_per_element: usize,
    pub bytes_read_per_element: usize,
    pub bytes_written_per_element: usize,
    pub is_fusable: bool,
}

impl OpCostInfo {
    /// Constructs a new `OpCostInfo`.
    pub const fn new(flops: usize, r_bytes: usize, w_bytes: usize, fusable: bool) -> Self {
        Self {
            flops_per_element: flops,
            bytes_read_per_element: r_bytes,
            bytes_written_per_element: w_bytes,
            is_fusable: fusable,
        }
    }

    /// Computes theoretical arithmetic intensity (FLOPs / Byte transferred).
    pub fn arithmetic_intensity(&self) -> f64 {
        let total_bytes = self.bytes_read_per_element + self.bytes_written_per_element;
        if total_bytes == 0 {
            0.0
        } else {
            self.flops_per_element as f64 / total_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
