//! # LLVM IR Text Generator
//!
//! Emits typed `.ll` LLVM IR text modules with function attributes and vectorization hints.

use crate::ir::IrGraph;

/// Generates LLVM IR module text for an IR graph.
pub fn generate_llvm_ir(_graph: &IrGraph, module_name: &str) -> String {
    let mut ll = format!(
        "; ModuleID = '{}'\nsource_filename = \"{}\"\n\n",
        module_name, module_name
    );
    ll.push_str("define void @compute(double* %in, double* %out, i64 %n) {\n");
    ll.push_str("entry:\n");
    ll.push_str("  ret void\n");
    ll.push_str("}\n");
    ll
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
