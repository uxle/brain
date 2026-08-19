//! # CUDA C Kernel Source Generator
//!
//! Emits `.cu` CUDA C kernel sources with thread block indexing and grid mapping.

use crate::ir::IrGraph;

/// Generates a CUDA C elementwise kernel string.
pub fn generate_cuda_kernel(_graph: &IrGraph, kernel_name: &str) -> String {
    let mut cu = format!("extern \"C\" __global__ void {}(const double* in, double* out, int n) {{\n", kernel_name);
    cu.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    cu.push_str("    if (idx < n) {\n");
    cu.push_str("        out[idx] = in[idx];\n");
    cu.push_str("    }\n");
    cu.push_str("}\n");
    cu
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
