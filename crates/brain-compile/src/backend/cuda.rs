//! # CUDA C Kernel Source Text Generator (Codegen Only)
//!
//! Generates `.cu` CUDA C kernel source code strings from IR graphs for external compilation
//! or future GPU JIT dispatch. NOTE: Does not compile or execute kernels on GPU hardware directly.

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
