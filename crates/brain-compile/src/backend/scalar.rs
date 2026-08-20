//! # Scalar JIT Kernel Generator
//!
//! Generates pure Rust / C scalar loop code for compiled kernels.

use crate::ir::IrGraph;

/// Generates a Rust function source string for an IR graph.
pub fn generate_rust_kernel(_graph: &IrGraph, kernel_name: &str) -> String {
    let mut code = format!(
        "pub fn {}(_inputs: &[&[f64]], output: &mut [f64]) {{\n",
        kernel_name
    );
    code.push_str("    let n = output.len();\n");
    code.push_str("    for i in 0..n {\n");
    code.push_str("        output[i] = 0.0;\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    code
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
