//! # CLI Tensor Table Formatting & Operations
//!
//! Formats tensors into tabular text representations and visualizes training metrics.

use brain_core::Tensor;

/// Formats a 2D tensor into an aligned text table.
pub fn format_tensor_table(t: &Tensor) -> String {
    if t.ndim() != 2 {
        return format!("{:?}", t.shape());
    }

    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let mut out = String::new();

    for r in 0..rows.min(10) {
        out.push_str("[ ");
        for c in 0..cols.min(10) {
            let idx = r * cols + c;
            out.push_str(&format!("{:>8.4} ", t.get(idx)));
        }
        if cols > 10 {
            out.push_str("... ");
        }
        out.push_str("]\n");
    }
    if rows > 10 {
        out.push_str("...\n");
    }

    out
}

/// Formats a list of named metrics.
pub fn format_metrics(metrics: &[(&str, f64)]) -> String {
    let mut out = Vec::new();
    for (name, val) in metrics {
        out.push(format!("{}: {:.4}", name, val));
    }
    out.join(" | ")
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
