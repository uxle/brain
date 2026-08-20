//! # Tensor and Module Tree Pretty-Printing
//!
//! Formats multidimensional tensors with shape/strides and tree visualizers for neural network modules.

use brain_core::Tensor;

/// Formats a tensor with multidimensional truncation.
pub fn format_tensor_summary(t: &Tensor, max_elements: usize) -> String {
    let shape = t.shape();
    let numel = t.numel();
    let mut out = format!("Tensor(shape={:?}, numel={}, data=[", shape, numel);

    let display_count = numel.min(max_elements);
    for i in 0..display_count {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{:.4}", t.get(i)));
    }
    if numel > max_elements {
        out.push_str(", ...");
    }
    out.push(']');
    out.push(')');
    out
}

/// Formats a hierarchical module tree.
pub fn format_module_tree(name: &str, children: &[(&str, &str)]) -> String {
    let mut out = format!("Module: {}\n", name);
    for (i, (child_name, child_type)) in children.iter().enumerate() {
        let prefix = if i == children.len() - 1 {
            "└── "
        } else {
            "├── "
        };
        out.push_str(&format!("  {}{}: {}\n", prefix, child_name, child_type));
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
