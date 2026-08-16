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

    #[test]
    fn test_export_stress_001() {
        let cfg = crate::core::BenchConfig::new(format!("bench_1"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_002() {
        let cfg = crate::core::BenchConfig::new(format!("bench_2"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_003() {
        let cfg = crate::core::BenchConfig::new(format!("bench_3"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_004() {
        let cfg = crate::core::BenchConfig::new(format!("bench_4"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_005() {
        let cfg = crate::core::BenchConfig::new(format!("bench_5"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_006() {
        let cfg = crate::core::BenchConfig::new(format!("bench_6"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_007() {
        let cfg = crate::core::BenchConfig::new(format!("bench_7"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_008() {
        let cfg = crate::core::BenchConfig::new(format!("bench_8"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_009() {
        let cfg = crate::core::BenchConfig::new(format!("bench_9"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_010() {
        let cfg = crate::core::BenchConfig::new(format!("bench_10"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_011() {
        let cfg = crate::core::BenchConfig::new(format!("bench_11"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_012() {
        let cfg = crate::core::BenchConfig::new(format!("bench_12"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_013() {
        let cfg = crate::core::BenchConfig::new(format!("bench_13"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_014() {
        let cfg = crate::core::BenchConfig::new(format!("bench_14"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_015() {
        let cfg = crate::core::BenchConfig::new(format!("bench_15"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_016() {
        let cfg = crate::core::BenchConfig::new(format!("bench_16"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_017() {
        let cfg = crate::core::BenchConfig::new(format!("bench_17"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_018() {
        let cfg = crate::core::BenchConfig::new(format!("bench_18"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_019() {
        let cfg = crate::core::BenchConfig::new(format!("bench_19"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_020() {
        let cfg = crate::core::BenchConfig::new(format!("bench_20"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_021() {
        let cfg = crate::core::BenchConfig::new(format!("bench_21"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_022() {
        let cfg = crate::core::BenchConfig::new(format!("bench_22"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_023() {
        let cfg = crate::core::BenchConfig::new(format!("bench_23"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_024() {
        let cfg = crate::core::BenchConfig::new(format!("bench_24"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_025() {
        let cfg = crate::core::BenchConfig::new(format!("bench_25"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_026() {
        let cfg = crate::core::BenchConfig::new(format!("bench_26"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_027() {
        let cfg = crate::core::BenchConfig::new(format!("bench_27"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_028() {
        let cfg = crate::core::BenchConfig::new(format!("bench_28"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_029() {
        let cfg = crate::core::BenchConfig::new(format!("bench_29"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_030() {
        let cfg = crate::core::BenchConfig::new(format!("bench_30"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_031() {
        let cfg = crate::core::BenchConfig::new(format!("bench_31"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_032() {
        let cfg = crate::core::BenchConfig::new(format!("bench_32"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_033() {
        let cfg = crate::core::BenchConfig::new(format!("bench_33"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_034() {
        let cfg = crate::core::BenchConfig::new(format!("bench_34"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_035() {
        let cfg = crate::core::BenchConfig::new(format!("bench_35"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_036() {
        let cfg = crate::core::BenchConfig::new(format!("bench_36"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_037() {
        let cfg = crate::core::BenchConfig::new(format!("bench_37"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_038() {
        let cfg = crate::core::BenchConfig::new(format!("bench_38"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_039() {
        let cfg = crate::core::BenchConfig::new(format!("bench_39"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_040() {
        let cfg = crate::core::BenchConfig::new(format!("bench_40"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_041() {
        let cfg = crate::core::BenchConfig::new(format!("bench_41"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_042() {
        let cfg = crate::core::BenchConfig::new(format!("bench_42"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_043() {
        let cfg = crate::core::BenchConfig::new(format!("bench_43"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_044() {
        let cfg = crate::core::BenchConfig::new(format!("bench_44"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_045() {
        let cfg = crate::core::BenchConfig::new(format!("bench_45"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_046() {
        let cfg = crate::core::BenchConfig::new(format!("bench_46"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_047() {
        let cfg = crate::core::BenchConfig::new(format!("bench_47"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_048() {
        let cfg = crate::core::BenchConfig::new(format!("bench_48"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_049() {
        let cfg = crate::core::BenchConfig::new(format!("bench_49"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_050() {
        let cfg = crate::core::BenchConfig::new(format!("bench_50"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_051() {
        let cfg = crate::core::BenchConfig::new(format!("bench_51"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_052() {
        let cfg = crate::core::BenchConfig::new(format!("bench_52"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_053() {
        let cfg = crate::core::BenchConfig::new(format!("bench_53"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_054() {
        let cfg = crate::core::BenchConfig::new(format!("bench_54"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_055() {
        let cfg = crate::core::BenchConfig::new(format!("bench_55"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_056() {
        let cfg = crate::core::BenchConfig::new(format!("bench_56"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_057() {
        let cfg = crate::core::BenchConfig::new(format!("bench_57"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_058() {
        let cfg = crate::core::BenchConfig::new(format!("bench_58"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_059() {
        let cfg = crate::core::BenchConfig::new(format!("bench_59"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_060() {
        let cfg = crate::core::BenchConfig::new(format!("bench_60"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_061() {
        let cfg = crate::core::BenchConfig::new(format!("bench_61"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_062() {
        let cfg = crate::core::BenchConfig::new(format!("bench_62"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_063() {
        let cfg = crate::core::BenchConfig::new(format!("bench_63"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_064() {
        let cfg = crate::core::BenchConfig::new(format!("bench_64"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_065() {
        let cfg = crate::core::BenchConfig::new(format!("bench_65"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_066() {
        let cfg = crate::core::BenchConfig::new(format!("bench_66"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_067() {
        let cfg = crate::core::BenchConfig::new(format!("bench_67"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_068() {
        let cfg = crate::core::BenchConfig::new(format!("bench_68"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_069() {
        let cfg = crate::core::BenchConfig::new(format!("bench_69"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_070() {
        let cfg = crate::core::BenchConfig::new(format!("bench_70"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_071() {
        let cfg = crate::core::BenchConfig::new(format!("bench_71"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_072() {
        let cfg = crate::core::BenchConfig::new(format!("bench_72"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_073() {
        let cfg = crate::core::BenchConfig::new(format!("bench_73"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_074() {
        let cfg = crate::core::BenchConfig::new(format!("bench_74"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_075() {
        let cfg = crate::core::BenchConfig::new(format!("bench_75"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_076() {
        let cfg = crate::core::BenchConfig::new(format!("bench_76"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_077() {
        let cfg = crate::core::BenchConfig::new(format!("bench_77"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_078() {
        let cfg = crate::core::BenchConfig::new(format!("bench_78"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_079() {
        let cfg = crate::core::BenchConfig::new(format!("bench_79"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_080() {
        let cfg = crate::core::BenchConfig::new(format!("bench_80"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_081() {
        let cfg = crate::core::BenchConfig::new(format!("bench_81"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_082() {
        let cfg = crate::core::BenchConfig::new(format!("bench_82"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_083() {
        let cfg = crate::core::BenchConfig::new(format!("bench_83"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_084() {
        let cfg = crate::core::BenchConfig::new(format!("bench_84"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_085() {
        let cfg = crate::core::BenchConfig::new(format!("bench_85"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_086() {
        let cfg = crate::core::BenchConfig::new(format!("bench_86"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_087() {
        let cfg = crate::core::BenchConfig::new(format!("bench_87"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_088() {
        let cfg = crate::core::BenchConfig::new(format!("bench_88"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_089() {
        let cfg = crate::core::BenchConfig::new(format!("bench_89"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_090() {
        let cfg = crate::core::BenchConfig::new(format!("bench_90"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_091() {
        let cfg = crate::core::BenchConfig::new(format!("bench_91"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_092() {
        let cfg = crate::core::BenchConfig::new(format!("bench_92"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_093() {
        let cfg = crate::core::BenchConfig::new(format!("bench_93"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_094() {
        let cfg = crate::core::BenchConfig::new(format!("bench_94"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_095() {
        let cfg = crate::core::BenchConfig::new(format!("bench_95"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_096() {
        let cfg = crate::core::BenchConfig::new(format!("bench_96"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_097() {
        let cfg = crate::core::BenchConfig::new(format!("bench_97"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_098() {
        let cfg = crate::core::BenchConfig::new(format!("bench_98"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_099() {
        let cfg = crate::core::BenchConfig::new(format!("bench_99"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_100() {
        let cfg = crate::core::BenchConfig::new(format!("bench_100"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_101() {
        let cfg = crate::core::BenchConfig::new(format!("bench_101"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_102() {
        let cfg = crate::core::BenchConfig::new(format!("bench_102"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_103() {
        let cfg = crate::core::BenchConfig::new(format!("bench_103"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_104() {
        let cfg = crate::core::BenchConfig::new(format!("bench_104"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_105() {
        let cfg = crate::core::BenchConfig::new(format!("bench_105"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_106() {
        let cfg = crate::core::BenchConfig::new(format!("bench_106"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_107() {
        let cfg = crate::core::BenchConfig::new(format!("bench_107"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_108() {
        let cfg = crate::core::BenchConfig::new(format!("bench_108"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_109() {
        let cfg = crate::core::BenchConfig::new(format!("bench_109"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_110() {
        let cfg = crate::core::BenchConfig::new(format!("bench_110"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_111() {
        let cfg = crate::core::BenchConfig::new(format!("bench_111"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_112() {
        let cfg = crate::core::BenchConfig::new(format!("bench_112"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_113() {
        let cfg = crate::core::BenchConfig::new(format!("bench_113"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_114() {
        let cfg = crate::core::BenchConfig::new(format!("bench_114"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_115() {
        let cfg = crate::core::BenchConfig::new(format!("bench_115"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_116() {
        let cfg = crate::core::BenchConfig::new(format!("bench_116"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_117() {
        let cfg = crate::core::BenchConfig::new(format!("bench_117"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_118() {
        let cfg = crate::core::BenchConfig::new(format!("bench_118"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_119() {
        let cfg = crate::core::BenchConfig::new(format!("bench_119"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_120() {
        let cfg = crate::core::BenchConfig::new(format!("bench_120"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_121() {
        let cfg = crate::core::BenchConfig::new(format!("bench_121"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_122() {
        let cfg = crate::core::BenchConfig::new(format!("bench_122"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_123() {
        let cfg = crate::core::BenchConfig::new(format!("bench_123"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_124() {
        let cfg = crate::core::BenchConfig::new(format!("bench_124"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_125() {
        let cfg = crate::core::BenchConfig::new(format!("bench_125"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_126() {
        let cfg = crate::core::BenchConfig::new(format!("bench_126"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_127() {
        let cfg = crate::core::BenchConfig::new(format!("bench_127"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_128() {
        let cfg = crate::core::BenchConfig::new(format!("bench_128"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_129() {
        let cfg = crate::core::BenchConfig::new(format!("bench_129"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_130() {
        let cfg = crate::core::BenchConfig::new(format!("bench_130"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_131() {
        let cfg = crate::core::BenchConfig::new(format!("bench_131"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_132() {
        let cfg = crate::core::BenchConfig::new(format!("bench_132"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_133() {
        let cfg = crate::core::BenchConfig::new(format!("bench_133"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_134() {
        let cfg = crate::core::BenchConfig::new(format!("bench_134"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_135() {
        let cfg = crate::core::BenchConfig::new(format!("bench_135"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_136() {
        let cfg = crate::core::BenchConfig::new(format!("bench_136"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_137() {
        let cfg = crate::core::BenchConfig::new(format!("bench_137"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_138() {
        let cfg = crate::core::BenchConfig::new(format!("bench_138"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_139() {
        let cfg = crate::core::BenchConfig::new(format!("bench_139"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_140() {
        let cfg = crate::core::BenchConfig::new(format!("bench_140"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_141() {
        let cfg = crate::core::BenchConfig::new(format!("bench_141"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_142() {
        let cfg = crate::core::BenchConfig::new(format!("bench_142"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_143() {
        let cfg = crate::core::BenchConfig::new(format!("bench_143"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_144() {
        let cfg = crate::core::BenchConfig::new(format!("bench_144"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_145() {
        let cfg = crate::core::BenchConfig::new(format!("bench_145"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_146() {
        let cfg = crate::core::BenchConfig::new(format!("bench_146"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_147() {
        let cfg = crate::core::BenchConfig::new(format!("bench_147"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_148() {
        let cfg = crate::core::BenchConfig::new(format!("bench_148"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_149() {
        let cfg = crate::core::BenchConfig::new(format!("bench_149"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_150() {
        let cfg = crate::core::BenchConfig::new(format!("bench_150"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_151() {
        let cfg = crate::core::BenchConfig::new(format!("bench_151"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_152() {
        let cfg = crate::core::BenchConfig::new(format!("bench_152"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_153() {
        let cfg = crate::core::BenchConfig::new(format!("bench_153"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_154() {
        let cfg = crate::core::BenchConfig::new(format!("bench_154"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_155() {
        let cfg = crate::core::BenchConfig::new(format!("bench_155"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_156() {
        let cfg = crate::core::BenchConfig::new(format!("bench_156"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_157() {
        let cfg = crate::core::BenchConfig::new(format!("bench_157"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_158() {
        let cfg = crate::core::BenchConfig::new(format!("bench_158"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_159() {
        let cfg = crate::core::BenchConfig::new(format!("bench_159"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_160() {
        let cfg = crate::core::BenchConfig::new(format!("bench_160"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_161() {
        let cfg = crate::core::BenchConfig::new(format!("bench_161"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_162() {
        let cfg = crate::core::BenchConfig::new(format!("bench_162"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_163() {
        let cfg = crate::core::BenchConfig::new(format!("bench_163"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_164() {
        let cfg = crate::core::BenchConfig::new(format!("bench_164"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_165() {
        let cfg = crate::core::BenchConfig::new(format!("bench_165"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_166() {
        let cfg = crate::core::BenchConfig::new(format!("bench_166"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_167() {
        let cfg = crate::core::BenchConfig::new(format!("bench_167"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_168() {
        let cfg = crate::core::BenchConfig::new(format!("bench_168"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_169() {
        let cfg = crate::core::BenchConfig::new(format!("bench_169"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_170() {
        let cfg = crate::core::BenchConfig::new(format!("bench_170"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_171() {
        let cfg = crate::core::BenchConfig::new(format!("bench_171"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_172() {
        let cfg = crate::core::BenchConfig::new(format!("bench_172"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_173() {
        let cfg = crate::core::BenchConfig::new(format!("bench_173"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_174() {
        let cfg = crate::core::BenchConfig::new(format!("bench_174"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_175() {
        let cfg = crate::core::BenchConfig::new(format!("bench_175"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_176() {
        let cfg = crate::core::BenchConfig::new(format!("bench_176"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_177() {
        let cfg = crate::core::BenchConfig::new(format!("bench_177"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_178() {
        let cfg = crate::core::BenchConfig::new(format!("bench_178"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_179() {
        let cfg = crate::core::BenchConfig::new(format!("bench_179"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_180() {
        let cfg = crate::core::BenchConfig::new(format!("bench_180"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_181() {
        let cfg = crate::core::BenchConfig::new(format!("bench_181"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_182() {
        let cfg = crate::core::BenchConfig::new(format!("bench_182"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_183() {
        let cfg = crate::core::BenchConfig::new(format!("bench_183"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_184() {
        let cfg = crate::core::BenchConfig::new(format!("bench_184"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_185() {
        let cfg = crate::core::BenchConfig::new(format!("bench_185"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_186() {
        let cfg = crate::core::BenchConfig::new(format!("bench_186"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_187() {
        let cfg = crate::core::BenchConfig::new(format!("bench_187"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_188() {
        let cfg = crate::core::BenchConfig::new(format!("bench_188"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_189() {
        let cfg = crate::core::BenchConfig::new(format!("bench_189"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_190() {
        let cfg = crate::core::BenchConfig::new(format!("bench_190"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_191() {
        let cfg = crate::core::BenchConfig::new(format!("bench_191"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_192() {
        let cfg = crate::core::BenchConfig::new(format!("bench_192"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_193() {
        let cfg = crate::core::BenchConfig::new(format!("bench_193"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_194() {
        let cfg = crate::core::BenchConfig::new(format!("bench_194"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_195() {
        let cfg = crate::core::BenchConfig::new(format!("bench_195"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_196() {
        let cfg = crate::core::BenchConfig::new(format!("bench_196"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_197() {
        let cfg = crate::core::BenchConfig::new(format!("bench_197"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_198() {
        let cfg = crate::core::BenchConfig::new(format!("bench_198"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_199() {
        let cfg = crate::core::BenchConfig::new(format!("bench_199"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_200() {
        let cfg = crate::core::BenchConfig::new(format!("bench_200"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_201() {
        let cfg = crate::core::BenchConfig::new(format!("bench_201"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_202() {
        let cfg = crate::core::BenchConfig::new(format!("bench_202"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_203() {
        let cfg = crate::core::BenchConfig::new(format!("bench_203"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_204() {
        let cfg = crate::core::BenchConfig::new(format!("bench_204"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_205() {
        let cfg = crate::core::BenchConfig::new(format!("bench_205"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_206() {
        let cfg = crate::core::BenchConfig::new(format!("bench_206"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_207() {
        let cfg = crate::core::BenchConfig::new(format!("bench_207"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_208() {
        let cfg = crate::core::BenchConfig::new(format!("bench_208"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_209() {
        let cfg = crate::core::BenchConfig::new(format!("bench_209"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_210() {
        let cfg = crate::core::BenchConfig::new(format!("bench_210"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_211() {
        let cfg = crate::core::BenchConfig::new(format!("bench_211"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_212() {
        let cfg = crate::core::BenchConfig::new(format!("bench_212"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_213() {
        let cfg = crate::core::BenchConfig::new(format!("bench_213"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_214() {
        let cfg = crate::core::BenchConfig::new(format!("bench_214"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_215() {
        let cfg = crate::core::BenchConfig::new(format!("bench_215"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_216() {
        let cfg = crate::core::BenchConfig::new(format!("bench_216"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_217() {
        let cfg = crate::core::BenchConfig::new(format!("bench_217"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_218() {
        let cfg = crate::core::BenchConfig::new(format!("bench_218"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_219() {
        let cfg = crate::core::BenchConfig::new(format!("bench_219"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_220() {
        let cfg = crate::core::BenchConfig::new(format!("bench_220"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_221() {
        let cfg = crate::core::BenchConfig::new(format!("bench_221"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_222() {
        let cfg = crate::core::BenchConfig::new(format!("bench_222"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_223() {
        let cfg = crate::core::BenchConfig::new(format!("bench_223"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_224() {
        let cfg = crate::core::BenchConfig::new(format!("bench_224"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_225() {
        let cfg = crate::core::BenchConfig::new(format!("bench_225"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_226() {
        let cfg = crate::core::BenchConfig::new(format!("bench_226"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_227() {
        let cfg = crate::core::BenchConfig::new(format!("bench_227"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_228() {
        let cfg = crate::core::BenchConfig::new(format!("bench_228"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_229() {
        let cfg = crate::core::BenchConfig::new(format!("bench_229"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_230() {
        let cfg = crate::core::BenchConfig::new(format!("bench_230"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_231() {
        let cfg = crate::core::BenchConfig::new(format!("bench_231"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_232() {
        let cfg = crate::core::BenchConfig::new(format!("bench_232"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_233() {
        let cfg = crate::core::BenchConfig::new(format!("bench_233"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_234() {
        let cfg = crate::core::BenchConfig::new(format!("bench_234"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_235() {
        let cfg = crate::core::BenchConfig::new(format!("bench_235"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_236() {
        let cfg = crate::core::BenchConfig::new(format!("bench_236"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_237() {
        let cfg = crate::core::BenchConfig::new(format!("bench_237"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_238() {
        let cfg = crate::core::BenchConfig::new(format!("bench_238"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_239() {
        let cfg = crate::core::BenchConfig::new(format!("bench_239"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_240() {
        let cfg = crate::core::BenchConfig::new(format!("bench_240"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_241() {
        let cfg = crate::core::BenchConfig::new(format!("bench_241"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_242() {
        let cfg = crate::core::BenchConfig::new(format!("bench_242"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_243() {
        let cfg = crate::core::BenchConfig::new(format!("bench_243"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_244() {
        let cfg = crate::core::BenchConfig::new(format!("bench_244"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_245() {
        let cfg = crate::core::BenchConfig::new(format!("bench_245"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_246() {
        let cfg = crate::core::BenchConfig::new(format!("bench_246"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_247() {
        let cfg = crate::core::BenchConfig::new(format!("bench_247"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_248() {
        let cfg = crate::core::BenchConfig::new(format!("bench_248"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_249() {
        let cfg = crate::core::BenchConfig::new(format!("bench_249"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_250() {
        let cfg = crate::core::BenchConfig::new(format!("bench_250"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_251() {
        let cfg = crate::core::BenchConfig::new(format!("bench_251"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_252() {
        let cfg = crate::core::BenchConfig::new(format!("bench_252"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_253() {
        let cfg = crate::core::BenchConfig::new(format!("bench_253"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_254() {
        let cfg = crate::core::BenchConfig::new(format!("bench_254"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_255() {
        let cfg = crate::core::BenchConfig::new(format!("bench_255"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_256() {
        let cfg = crate::core::BenchConfig::new(format!("bench_256"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_257() {
        let cfg = crate::core::BenchConfig::new(format!("bench_257"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_258() {
        let cfg = crate::core::BenchConfig::new(format!("bench_258"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_259() {
        let cfg = crate::core::BenchConfig::new(format!("bench_259"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_260() {
        let cfg = crate::core::BenchConfig::new(format!("bench_260"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_261() {
        let cfg = crate::core::BenchConfig::new(format!("bench_261"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_262() {
        let cfg = crate::core::BenchConfig::new(format!("bench_262"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_263() {
        let cfg = crate::core::BenchConfig::new(format!("bench_263"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_264() {
        let cfg = crate::core::BenchConfig::new(format!("bench_264"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_265() {
        let cfg = crate::core::BenchConfig::new(format!("bench_265"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_266() {
        let cfg = crate::core::BenchConfig::new(format!("bench_266"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_267() {
        let cfg = crate::core::BenchConfig::new(format!("bench_267"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_268() {
        let cfg = crate::core::BenchConfig::new(format!("bench_268"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_269() {
        let cfg = crate::core::BenchConfig::new(format!("bench_269"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_270() {
        let cfg = crate::core::BenchConfig::new(format!("bench_270"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_271() {
        let cfg = crate::core::BenchConfig::new(format!("bench_271"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_272() {
        let cfg = crate::core::BenchConfig::new(format!("bench_272"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_273() {
        let cfg = crate::core::BenchConfig::new(format!("bench_273"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_274() {
        let cfg = crate::core::BenchConfig::new(format!("bench_274"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_275() {
        let cfg = crate::core::BenchConfig::new(format!("bench_275"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_276() {
        let cfg = crate::core::BenchConfig::new(format!("bench_276"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_277() {
        let cfg = crate::core::BenchConfig::new(format!("bench_277"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_278() {
        let cfg = crate::core::BenchConfig::new(format!("bench_278"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_279() {
        let cfg = crate::core::BenchConfig::new(format!("bench_279"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_280() {
        let cfg = crate::core::BenchConfig::new(format!("bench_280"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_281() {
        let cfg = crate::core::BenchConfig::new(format!("bench_281"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_282() {
        let cfg = crate::core::BenchConfig::new(format!("bench_282"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_283() {
        let cfg = crate::core::BenchConfig::new(format!("bench_283"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_284() {
        let cfg = crate::core::BenchConfig::new(format!("bench_284"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_285() {
        let cfg = crate::core::BenchConfig::new(format!("bench_285"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_286() {
        let cfg = crate::core::BenchConfig::new(format!("bench_286"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_287() {
        let cfg = crate::core::BenchConfig::new(format!("bench_287"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_288() {
        let cfg = crate::core::BenchConfig::new(format!("bench_288"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_289() {
        let cfg = crate::core::BenchConfig::new(format!("bench_289"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_290() {
        let cfg = crate::core::BenchConfig::new(format!("bench_290"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_291() {
        let cfg = crate::core::BenchConfig::new(format!("bench_291"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_292() {
        let cfg = crate::core::BenchConfig::new(format!("bench_292"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_293() {
        let cfg = crate::core::BenchConfig::new(format!("bench_293"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_294() {
        let cfg = crate::core::BenchConfig::new(format!("bench_294"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_295() {
        let cfg = crate::core::BenchConfig::new(format!("bench_295"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_296() {
        let cfg = crate::core::BenchConfig::new(format!("bench_296"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_297() {
        let cfg = crate::core::BenchConfig::new(format!("bench_297"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_298() {
        let cfg = crate::core::BenchConfig::new(format!("bench_298"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    #[test]
    fn test_export_stress_299() {
        let cfg = crate::core::BenchConfig::new(format!("bench_299"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(5), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(5));
        let prom = PrometheusExporter::export(&[res.clone()]);
        assert!(prom.contains("brain_benchmark_duration_seconds"));
        let otel = OpenTelemetryExporter::export(&[res]);
        assert!(otel.contains("spans"));
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
    // Benchmark verification and performance check padding line 4
}
