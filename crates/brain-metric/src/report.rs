//! # Evaluation Report Formatter
//!
//! Formats evaluation metrics into ASCII tables, Markdown tables, CSV, and JSON outputs.
#![allow(missing_docs)]

use std::collections::BTreeMap;

/// Target output format for evaluation summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportFormat {
    #[default]
    Markdown,
    TextTable,
    Csv,
    Json,
}

/// Formats a map of metric names to values into a clean Markdown table string.
pub fn format_markdown_report(metrics: &BTreeMap<String, f64>) -> String {
    let mut md = String::new();
    md.push_str("| Metric | Value |
|---|---|
");
    for (name, val) in metrics {
        md.push_str(&format!("| `{}` | {:.4} |
", name, val));
    }
    md
}

/// Formats metrics into a CSV string.
pub fn format_csv_report(metrics: &BTreeMap<String, f64>) -> String {
    let mut csv = String::from("metric,value
");
    for (name, val) in metrics {
        csv.push_str(&format!("{},{}
", name, val));
    }
    csv
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
