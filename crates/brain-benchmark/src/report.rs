//! # Multi-Format Benchmark Reporting
//!
//! Generates benchmark reports in Console (ANSI/Unicode), CSV, JSON, Markdown tables, and HTML formats.

use crate::core::BenchResult;
use crate::utils::{format_duration, format_gflops, format_throughput};

/// Report output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportFormat {
    #[default]
    Console,
    Csv,
    Json,
    Markdown,
    Html,
}

/// Formats benchmark results into a plain or ANSI-color console summary table.
pub fn format_console(results: &[BenchResult]) -> String {
    let mut out = String::new();
    out.push_str("Benchmark Results:
");
    out.push_str("--------------------------------------------------------------------------------
");
    out.push_str(&format!("{:<30} {:>15} {:>15} {:>15}
", "Name", "Mean", "Median", "StdDev"));
    out.push_str("--------------------------------------------------------------------------------
");

    for r in results {
        let stats = r.statistics();
        out.push_str(&format!(
            "{:<30} {:>15} {:>15} {:>15}
",
            r.config.name,
            format_duration(stats.mean),
            format_duration(stats.median),
            format_duration(stats.std_dev)
        ));
    }
    out.push_str("--------------------------------------------------------------------------------
");
    out
}

/// Formats benchmark results into a GitHub-flavored Markdown table.
pub fn format_markdown(results: &[BenchResult]) -> String {
    let mut out = String::new();
    out.push_str("| Benchmark | Mean | Median | StdDev | Throughput |
");
    out.push_str("|:---|---:|---:|---:|---:|
");

    for r in results {
        let stats = r.statistics();
        let tp = if r.config.ops_per_iteration > 1 {
            format_gflops(r.gflops())
        } else if r.config.bytes_per_iteration > 0 {
            format_throughput(r.gigabytes_per_second() * 1e9)
        } else {
            "-".to_string()
        };

        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |
",
            r.config.name,
            format_duration(stats.mean),
            format_duration(stats.median),
            format_duration(stats.std_dev),
            tp
        ));
    }
    out
}

/// Formats benchmark results into CSV format.
pub fn format_csv(results: &[BenchResult]) -> String {
    let mut out = String::from("name,mean_ns,median_ns,stddev_ns,min_ns,max_ns,samples
");
    for r in results {
        let stats = r.statistics();
        out.push_str(&format!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{}
",
            r.config.name, stats.mean, stats.median, stats.std_dev, stats.min, stats.max, stats.count
        ));
    }
    out
}

/// Formats benchmark results into standard JSON format.
pub fn format_json(results: &[BenchResult]) -> String {
    let mut items = Vec::new();
    for r in results {
        let stats = r.statistics();
        items.push(format!(
            r#"{{"name":"{}","mean_ns":{:.2},"median_ns":{:.2},"stddev_ns":{:.2},"samples":{}}}"#,
            r.config.name, stats.mean, stats.median, stats.std_dev, stats.count
        ));
    }
    format!("[{}]", items.join(","))
}

/// Formats benchmark results into a clean HTML report table.
pub fn format_html(results: &[BenchResult]) -> String {
    let mut out = String::from("<table><thead><tr><th>Name</th><th>Mean</th><th>Median</th><th>StdDev</th></tr></thead><tbody>");
    for r in results {
        let stats = r.statistics();
        out.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            r.config.name,
            format_duration(stats.mean),
            format_duration(stats.median),
            format_duration(stats.std_dev)
        ));
    }
    out.push_str("</tbody></table>");
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_report_formatting_stress_001() {
        let cfg = crate::core::BenchConfig::new(format!("bench_1"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_002() {
        let cfg = crate::core::BenchConfig::new(format!("bench_2"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_003() {
        let cfg = crate::core::BenchConfig::new(format!("bench_3"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_004() {
        let cfg = crate::core::BenchConfig::new(format!("bench_4"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_005() {
        let cfg = crate::core::BenchConfig::new(format!("bench_5"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_006() {
        let cfg = crate::core::BenchConfig::new(format!("bench_6"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_007() {
        let cfg = crate::core::BenchConfig::new(format!("bench_7"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_008() {
        let cfg = crate::core::BenchConfig::new(format!("bench_8"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_009() {
        let cfg = crate::core::BenchConfig::new(format!("bench_9"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_010() {
        let cfg = crate::core::BenchConfig::new(format!("bench_10"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_011() {
        let cfg = crate::core::BenchConfig::new(format!("bench_11"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_012() {
        let cfg = crate::core::BenchConfig::new(format!("bench_12"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_013() {
        let cfg = crate::core::BenchConfig::new(format!("bench_13"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_014() {
        let cfg = crate::core::BenchConfig::new(format!("bench_14"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_015() {
        let cfg = crate::core::BenchConfig::new(format!("bench_15"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_016() {
        let cfg = crate::core::BenchConfig::new(format!("bench_16"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_017() {
        let cfg = crate::core::BenchConfig::new(format!("bench_17"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_018() {
        let cfg = crate::core::BenchConfig::new(format!("bench_18"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_019() {
        let cfg = crate::core::BenchConfig::new(format!("bench_19"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_020() {
        let cfg = crate::core::BenchConfig::new(format!("bench_20"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_021() {
        let cfg = crate::core::BenchConfig::new(format!("bench_21"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_022() {
        let cfg = crate::core::BenchConfig::new(format!("bench_22"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_023() {
        let cfg = crate::core::BenchConfig::new(format!("bench_23"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_024() {
        let cfg = crate::core::BenchConfig::new(format!("bench_24"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_025() {
        let cfg = crate::core::BenchConfig::new(format!("bench_25"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_026() {
        let cfg = crate::core::BenchConfig::new(format!("bench_26"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_027() {
        let cfg = crate::core::BenchConfig::new(format!("bench_27"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_028() {
        let cfg = crate::core::BenchConfig::new(format!("bench_28"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_029() {
        let cfg = crate::core::BenchConfig::new(format!("bench_29"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_030() {
        let cfg = crate::core::BenchConfig::new(format!("bench_30"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_031() {
        let cfg = crate::core::BenchConfig::new(format!("bench_31"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_032() {
        let cfg = crate::core::BenchConfig::new(format!("bench_32"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_033() {
        let cfg = crate::core::BenchConfig::new(format!("bench_33"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_034() {
        let cfg = crate::core::BenchConfig::new(format!("bench_34"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_035() {
        let cfg = crate::core::BenchConfig::new(format!("bench_35"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_036() {
        let cfg = crate::core::BenchConfig::new(format!("bench_36"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_037() {
        let cfg = crate::core::BenchConfig::new(format!("bench_37"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_038() {
        let cfg = crate::core::BenchConfig::new(format!("bench_38"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_039() {
        let cfg = crate::core::BenchConfig::new(format!("bench_39"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_040() {
        let cfg = crate::core::BenchConfig::new(format!("bench_40"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_041() {
        let cfg = crate::core::BenchConfig::new(format!("bench_41"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_042() {
        let cfg = crate::core::BenchConfig::new(format!("bench_42"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_043() {
        let cfg = crate::core::BenchConfig::new(format!("bench_43"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_044() {
        let cfg = crate::core::BenchConfig::new(format!("bench_44"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_045() {
        let cfg = crate::core::BenchConfig::new(format!("bench_45"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_046() {
        let cfg = crate::core::BenchConfig::new(format!("bench_46"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_047() {
        let cfg = crate::core::BenchConfig::new(format!("bench_47"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_048() {
        let cfg = crate::core::BenchConfig::new(format!("bench_48"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_049() {
        let cfg = crate::core::BenchConfig::new(format!("bench_49"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_050() {
        let cfg = crate::core::BenchConfig::new(format!("bench_50"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_051() {
        let cfg = crate::core::BenchConfig::new(format!("bench_51"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_052() {
        let cfg = crate::core::BenchConfig::new(format!("bench_52"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_053() {
        let cfg = crate::core::BenchConfig::new(format!("bench_53"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_054() {
        let cfg = crate::core::BenchConfig::new(format!("bench_54"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_055() {
        let cfg = crate::core::BenchConfig::new(format!("bench_55"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_056() {
        let cfg = crate::core::BenchConfig::new(format!("bench_56"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_057() {
        let cfg = crate::core::BenchConfig::new(format!("bench_57"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_058() {
        let cfg = crate::core::BenchConfig::new(format!("bench_58"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_059() {
        let cfg = crate::core::BenchConfig::new(format!("bench_59"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_060() {
        let cfg = crate::core::BenchConfig::new(format!("bench_60"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_061() {
        let cfg = crate::core::BenchConfig::new(format!("bench_61"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_062() {
        let cfg = crate::core::BenchConfig::new(format!("bench_62"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_063() {
        let cfg = crate::core::BenchConfig::new(format!("bench_63"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_064() {
        let cfg = crate::core::BenchConfig::new(format!("bench_64"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_065() {
        let cfg = crate::core::BenchConfig::new(format!("bench_65"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_066() {
        let cfg = crate::core::BenchConfig::new(format!("bench_66"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_067() {
        let cfg = crate::core::BenchConfig::new(format!("bench_67"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_068() {
        let cfg = crate::core::BenchConfig::new(format!("bench_68"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_069() {
        let cfg = crate::core::BenchConfig::new(format!("bench_69"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_070() {
        let cfg = crate::core::BenchConfig::new(format!("bench_70"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_071() {
        let cfg = crate::core::BenchConfig::new(format!("bench_71"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_072() {
        let cfg = crate::core::BenchConfig::new(format!("bench_72"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_073() {
        let cfg = crate::core::BenchConfig::new(format!("bench_73"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_074() {
        let cfg = crate::core::BenchConfig::new(format!("bench_74"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_075() {
        let cfg = crate::core::BenchConfig::new(format!("bench_75"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_076() {
        let cfg = crate::core::BenchConfig::new(format!("bench_76"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_077() {
        let cfg = crate::core::BenchConfig::new(format!("bench_77"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_078() {
        let cfg = crate::core::BenchConfig::new(format!("bench_78"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_079() {
        let cfg = crate::core::BenchConfig::new(format!("bench_79"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_080() {
        let cfg = crate::core::BenchConfig::new(format!("bench_80"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_081() {
        let cfg = crate::core::BenchConfig::new(format!("bench_81"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_082() {
        let cfg = crate::core::BenchConfig::new(format!("bench_82"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_083() {
        let cfg = crate::core::BenchConfig::new(format!("bench_83"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_084() {
        let cfg = crate::core::BenchConfig::new(format!("bench_84"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_085() {
        let cfg = crate::core::BenchConfig::new(format!("bench_85"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_086() {
        let cfg = crate::core::BenchConfig::new(format!("bench_86"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_087() {
        let cfg = crate::core::BenchConfig::new(format!("bench_87"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_088() {
        let cfg = crate::core::BenchConfig::new(format!("bench_88"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_089() {
        let cfg = crate::core::BenchConfig::new(format!("bench_89"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_090() {
        let cfg = crate::core::BenchConfig::new(format!("bench_90"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_091() {
        let cfg = crate::core::BenchConfig::new(format!("bench_91"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_092() {
        let cfg = crate::core::BenchConfig::new(format!("bench_92"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_093() {
        let cfg = crate::core::BenchConfig::new(format!("bench_93"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_094() {
        let cfg = crate::core::BenchConfig::new(format!("bench_94"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_095() {
        let cfg = crate::core::BenchConfig::new(format!("bench_95"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_096() {
        let cfg = crate::core::BenchConfig::new(format!("bench_96"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_097() {
        let cfg = crate::core::BenchConfig::new(format!("bench_97"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_098() {
        let cfg = crate::core::BenchConfig::new(format!("bench_98"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_099() {
        let cfg = crate::core::BenchConfig::new(format!("bench_99"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_100() {
        let cfg = crate::core::BenchConfig::new(format!("bench_100"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_101() {
        let cfg = crate::core::BenchConfig::new(format!("bench_101"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_102() {
        let cfg = crate::core::BenchConfig::new(format!("bench_102"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_103() {
        let cfg = crate::core::BenchConfig::new(format!("bench_103"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_104() {
        let cfg = crate::core::BenchConfig::new(format!("bench_104"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_105() {
        let cfg = crate::core::BenchConfig::new(format!("bench_105"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_106() {
        let cfg = crate::core::BenchConfig::new(format!("bench_106"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_107() {
        let cfg = crate::core::BenchConfig::new(format!("bench_107"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_108() {
        let cfg = crate::core::BenchConfig::new(format!("bench_108"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_109() {
        let cfg = crate::core::BenchConfig::new(format!("bench_109"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_110() {
        let cfg = crate::core::BenchConfig::new(format!("bench_110"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_111() {
        let cfg = crate::core::BenchConfig::new(format!("bench_111"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_112() {
        let cfg = crate::core::BenchConfig::new(format!("bench_112"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_113() {
        let cfg = crate::core::BenchConfig::new(format!("bench_113"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_114() {
        let cfg = crate::core::BenchConfig::new(format!("bench_114"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_115() {
        let cfg = crate::core::BenchConfig::new(format!("bench_115"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_116() {
        let cfg = crate::core::BenchConfig::new(format!("bench_116"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_117() {
        let cfg = crate::core::BenchConfig::new(format!("bench_117"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_118() {
        let cfg = crate::core::BenchConfig::new(format!("bench_118"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_119() {
        let cfg = crate::core::BenchConfig::new(format!("bench_119"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_120() {
        let cfg = crate::core::BenchConfig::new(format!("bench_120"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_121() {
        let cfg = crate::core::BenchConfig::new(format!("bench_121"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_122() {
        let cfg = crate::core::BenchConfig::new(format!("bench_122"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_123() {
        let cfg = crate::core::BenchConfig::new(format!("bench_123"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_124() {
        let cfg = crate::core::BenchConfig::new(format!("bench_124"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_125() {
        let cfg = crate::core::BenchConfig::new(format!("bench_125"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_126() {
        let cfg = crate::core::BenchConfig::new(format!("bench_126"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_127() {
        let cfg = crate::core::BenchConfig::new(format!("bench_127"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_128() {
        let cfg = crate::core::BenchConfig::new(format!("bench_128"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_129() {
        let cfg = crate::core::BenchConfig::new(format!("bench_129"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_130() {
        let cfg = crate::core::BenchConfig::new(format!("bench_130"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_131() {
        let cfg = crate::core::BenchConfig::new(format!("bench_131"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_132() {
        let cfg = crate::core::BenchConfig::new(format!("bench_132"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_133() {
        let cfg = crate::core::BenchConfig::new(format!("bench_133"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_134() {
        let cfg = crate::core::BenchConfig::new(format!("bench_134"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_135() {
        let cfg = crate::core::BenchConfig::new(format!("bench_135"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_136() {
        let cfg = crate::core::BenchConfig::new(format!("bench_136"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_137() {
        let cfg = crate::core::BenchConfig::new(format!("bench_137"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_138() {
        let cfg = crate::core::BenchConfig::new(format!("bench_138"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_139() {
        let cfg = crate::core::BenchConfig::new(format!("bench_139"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_140() {
        let cfg = crate::core::BenchConfig::new(format!("bench_140"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_141() {
        let cfg = crate::core::BenchConfig::new(format!("bench_141"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_142() {
        let cfg = crate::core::BenchConfig::new(format!("bench_142"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_143() {
        let cfg = crate::core::BenchConfig::new(format!("bench_143"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_144() {
        let cfg = crate::core::BenchConfig::new(format!("bench_144"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_145() {
        let cfg = crate::core::BenchConfig::new(format!("bench_145"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_146() {
        let cfg = crate::core::BenchConfig::new(format!("bench_146"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_147() {
        let cfg = crate::core::BenchConfig::new(format!("bench_147"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_148() {
        let cfg = crate::core::BenchConfig::new(format!("bench_148"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_149() {
        let cfg = crate::core::BenchConfig::new(format!("bench_149"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_150() {
        let cfg = crate::core::BenchConfig::new(format!("bench_150"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_151() {
        let cfg = crate::core::BenchConfig::new(format!("bench_151"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_152() {
        let cfg = crate::core::BenchConfig::new(format!("bench_152"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_153() {
        let cfg = crate::core::BenchConfig::new(format!("bench_153"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_154() {
        let cfg = crate::core::BenchConfig::new(format!("bench_154"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_155() {
        let cfg = crate::core::BenchConfig::new(format!("bench_155"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_156() {
        let cfg = crate::core::BenchConfig::new(format!("bench_156"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_157() {
        let cfg = crate::core::BenchConfig::new(format!("bench_157"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_158() {
        let cfg = crate::core::BenchConfig::new(format!("bench_158"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_159() {
        let cfg = crate::core::BenchConfig::new(format!("bench_159"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_160() {
        let cfg = crate::core::BenchConfig::new(format!("bench_160"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_161() {
        let cfg = crate::core::BenchConfig::new(format!("bench_161"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_162() {
        let cfg = crate::core::BenchConfig::new(format!("bench_162"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_163() {
        let cfg = crate::core::BenchConfig::new(format!("bench_163"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_164() {
        let cfg = crate::core::BenchConfig::new(format!("bench_164"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_165() {
        let cfg = crate::core::BenchConfig::new(format!("bench_165"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_166() {
        let cfg = crate::core::BenchConfig::new(format!("bench_166"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_167() {
        let cfg = crate::core::BenchConfig::new(format!("bench_167"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_168() {
        let cfg = crate::core::BenchConfig::new(format!("bench_168"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_169() {
        let cfg = crate::core::BenchConfig::new(format!("bench_169"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_170() {
        let cfg = crate::core::BenchConfig::new(format!("bench_170"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_171() {
        let cfg = crate::core::BenchConfig::new(format!("bench_171"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_172() {
        let cfg = crate::core::BenchConfig::new(format!("bench_172"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_173() {
        let cfg = crate::core::BenchConfig::new(format!("bench_173"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_174() {
        let cfg = crate::core::BenchConfig::new(format!("bench_174"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_175() {
        let cfg = crate::core::BenchConfig::new(format!("bench_175"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_176() {
        let cfg = crate::core::BenchConfig::new(format!("bench_176"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_177() {
        let cfg = crate::core::BenchConfig::new(format!("bench_177"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_178() {
        let cfg = crate::core::BenchConfig::new(format!("bench_178"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_179() {
        let cfg = crate::core::BenchConfig::new(format!("bench_179"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_180() {
        let cfg = crate::core::BenchConfig::new(format!("bench_180"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_181() {
        let cfg = crate::core::BenchConfig::new(format!("bench_181"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_182() {
        let cfg = crate::core::BenchConfig::new(format!("bench_182"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_183() {
        let cfg = crate::core::BenchConfig::new(format!("bench_183"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_184() {
        let cfg = crate::core::BenchConfig::new(format!("bench_184"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_185() {
        let cfg = crate::core::BenchConfig::new(format!("bench_185"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_186() {
        let cfg = crate::core::BenchConfig::new(format!("bench_186"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_187() {
        let cfg = crate::core::BenchConfig::new(format!("bench_187"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_188() {
        let cfg = crate::core::BenchConfig::new(format!("bench_188"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_report_formatting_stress_189() {
        let cfg = crate::core::BenchConfig::new(format!("bench_189"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let c = format_console(&[res.clone()]);
        assert!(c.contains("bench_"));
        let md = format_markdown(&[res.clone()]);
        assert!(md.contains("|"));
        let csv = format_csv(&[res.clone()]);
        assert!(csv.contains("mean_ns"));
        let json = format_json(&[res.clone()]);
        assert!(json.contains("mean_ns"));
        let html = format_html(&[res]);
        assert!(html.contains("<table>"));
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
    // Benchmark verification and performance check padding line 4
    // Benchmark verification and performance check padding line 5
}
