//! # Operator Benchmark Registry & Grid Matrices
//!
//! Parameter sweeps and dimension grids for profiling tensor operations across various shapes.

use crate::core::BenchResult;
use crate::kernels::KernelSuite;
use brain_core::BrainResult;

/// Parameter grid runner for tensor operations.
#[derive(Debug, Clone, Default)]
pub struct OpsBenchMatrix {
    shapes: Vec<Vec<usize>>,
}

impl OpsBenchMatrix {
    /// Creates a standard 2D shape sweep matrix.
    pub fn standard_2d() -> Self {
        Self {
            shapes: vec![vec![32, 32], vec![64, 64], vec![128, 128], vec![256, 256]],
        }
    }

    /// Runs matrix multiplication across all configured shapes.
    pub fn run_matmul_sweep(&self) -> BrainResult<Vec<BenchResult>> {
        let mut results = Vec::new();
        for shape in &self.shapes {
            let (m, k, n) = (shape[0], shape[1], shape[1]);
            results.push(KernelSuite::bench_matmul(m, k, n)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
