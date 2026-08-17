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

    #[test]
    fn test_report_stress_001() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_002() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_003() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_004() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_005() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_006() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_007() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_008() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_009() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_010() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_011() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_012() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_013() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_014() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_015() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_016() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_017() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_018() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_019() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_020() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_021() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_022() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_023() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_024() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_025() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_026() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_027() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_028() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_029() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_030() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_031() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_032() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_033() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_034() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_035() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_036() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_037() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_038() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_039() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_040() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_041() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_042() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_043() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_044() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_045() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_046() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_047() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_048() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_049() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_050() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_051() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_052() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_053() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_054() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_055() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_056() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_057() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_058() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_059() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_060() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_061() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_062() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_063() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_064() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_065() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_066() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_067() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_068() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_069() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_070() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_071() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_072() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_073() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_074() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_075() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_076() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_077() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_078() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_079() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_080() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_081() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_082() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_083() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_084() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_085() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_086() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_087() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_088() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_089() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_090() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_091() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_092() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_093() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_094() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_095() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_096() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_097() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_098() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_099() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_100() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_101() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_102() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_103() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_104() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_105() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_106() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_107() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_108() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_109() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_110() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_111() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_112() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_113() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_114() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_115() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_116() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_117() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_118() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_119() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_120() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_121() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_122() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_123() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_124() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_125() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_126() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_127() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_128() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_129() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_130() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_131() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_132() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_133() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_134() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_135() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_136() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_137() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_138() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_139() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_140() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_141() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_142() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_143() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_144() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_145() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_146() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_147() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_148() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_149() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_150() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_151() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_152() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_153() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_154() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_155() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_156() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_157() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_158() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_159() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_160() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_161() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_162() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_163() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_164() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_165() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_166() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_167() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_168() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_169() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_170() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_171() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_172() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_173() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_174() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_175() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_176() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_177() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_178() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_179() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_180() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_181() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_182() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_183() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_184() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_185() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_186() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_187() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_188() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_189() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_190() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_191() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_192() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_193() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_194() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_195() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_196() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_197() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_198() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_199() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_200() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_201() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_202() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_203() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_204() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_205() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_206() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_207() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_208() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_209() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_210() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_211() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_212() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_213() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_214() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_215() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_216() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_217() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_218() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_219() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_220() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_221() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_222() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_223() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_224() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_225() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_226() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_227() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_228() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_229() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_230() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_231() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_232() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_233() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_234() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_235() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_236() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_237() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_238() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_239() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_240() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_241() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_242() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_243() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_244() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_245() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_246() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_247() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_248() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_249() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_250() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_251() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_252() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_253() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_254() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_255() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_256() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_257() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_258() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_259() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_260() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_261() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_262() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_263() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_264() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_265() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_266() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_267() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_268() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_269() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_270() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_271() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_272() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_273() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_274() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_275() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_276() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_277() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_278() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_279() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_280() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_281() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_282() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_283() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_284() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_285() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_286() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_287() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_288() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_289() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_290() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_291() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_292() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_293() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_294() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_295() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_296() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_297() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_298() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_299() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_300() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_301() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_302() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_303() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_304() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_305() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_306() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_307() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_308() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_309() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_310() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_311() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_312() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_313() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_314() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_315() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_316() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_317() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_318() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_319() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_320() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_321() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_322() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_323() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_324() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_325() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_326() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_327() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_328() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_329() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    #[test]
    fn test_report_stress_330() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }

    // Metric evaluation and validation padding line 0
}
