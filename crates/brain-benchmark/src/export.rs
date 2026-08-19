//! # Metric Exporters (Prometheus / OpenTelemetry)
//!
//! Exports benchmark statistics and performance metrics to observability backends.

use crate::core::BenchResult;

/// Exporter for Prometheus text exposition format.
pub struct PrometheusExporter;

impl PrometheusExporter {
    /// Formats results into Prometheus gauge and summary metrics.
    pub fn export(results: &[BenchResult]) -> String {
        let mut out = String::new();
        out.push_str("# HELP brain_benchmark_duration_seconds Average execution duration per iteration.\n");
        out.push_str("# TYPE brain_benchmark_duration_seconds gauge\n");

        for r in results {
            let mean_secs = r.mean_nanos() / 1e9;
            out.push_str(&format!(
                "brain_benchmark_duration_seconds{{benchmark=\"{}\"}} {:.9}\n",
                r.config.name, mean_secs
            ));
        }

        out
    }
}

/// Exporter for OpenTelemetry span JSON payloads.
pub struct OpenTelemetryExporter;

impl OpenTelemetryExporter {
    /// Formats results into OpenTelemetry-compatible span records.
    pub fn export(results: &[BenchResult]) -> String {
        let mut spans = Vec::new();
        for r in results {
            spans.push(format!(
                r#"{{"name":"{}","duration_ms":{:.3},"samples":{}}}"#,
                r.config.name,
                r.mean_nanos() / 1e6,
                r.samples.len()
            ));
        }
        format!(r#"{{"spans":[{}]}}"#, spans.join(","))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
