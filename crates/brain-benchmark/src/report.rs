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
    out.push_str(
        "Benchmark Results:
",
    );
    out.push_str(
        "--------------------------------------------------------------------------------
",
    );
    out.push_str(&format!(
        "{:<30} {:>15} {:>15} {:>15}
",
        "Name", "Mean", "Median", "StdDev"
    ));
    out.push_str(
        "--------------------------------------------------------------------------------
",
    );

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
    out.push_str(
        "--------------------------------------------------------------------------------
",
    );
    out
}

/// Formats benchmark results into a GitHub-flavored Markdown table.
pub fn format_markdown(results: &[BenchResult]) -> String {
    let mut out = String::new();
    out.push_str(
        "| Benchmark | Mean | Median | StdDev | Throughput |
",
    );
    out.push_str(
        "|:---|---:|---:|---:|---:|
",
    );

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
    let mut out = String::from(
        "name,mean_ns,median_ns,stddev_ns,min_ns,max_ns,samples
",
    );
    for r in results {
        let stats = r.statistics();
        out.push_str(&format!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{}
",
            r.config.name,
            stats.mean,
            stats.median,
            stats.std_dev,
            stats.min,
            stats.max,
            stats.count
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
}
