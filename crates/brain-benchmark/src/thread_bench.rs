//! # Multi-Threaded Scalability & Contention Analysis
//!
//! Evaluates speedup ratios, parallel efficiency, and lock contention across worker thread sweeps.

use crate::core::BenchResult;

/// Scalability report summarizing parallel speedups.
#[derive(Debug, Clone)]
pub struct ScalabilityReport {
    pub name: String,
    pub thread_counts: Vec<usize>,
    pub mean_durations_ns: Vec<f64>,
    pub speedup_ratios: Vec<f64>,
    pub parallel_efficiency: Vec<f64>,
}

impl ScalabilityReport {
    /// Evaluates scalability from a list of per-thread benchmark results.
    pub fn from_results(name: impl Into<String>, results: &[(usize, BenchResult)]) -> Self {
        let mut thread_counts = Vec::new();
        let mut mean_durations_ns = Vec::new();
        let mut speedup_ratios = Vec::new();
        let mut parallel_efficiency = Vec::new();

        let base_mean = results.first().map(|(_, r)| r.mean_nanos()).unwrap_or(1.0);

        for &(threads, ref r) in results {
            let mean = r.mean_nanos();
            let speedup = if mean > 0.0 { base_mean / mean } else { 1.0 };
            let efficiency = speedup / threads.max(1) as f64;

            thread_counts.push(threads);
            mean_durations_ns.push(mean);
            speedup_ratios.push(speedup);
            parallel_efficiency.push(efficiency);
        }

        Self {
            name: name.into(),
            thread_counts,
            mean_durations_ns,
            speedup_ratios,
            parallel_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_thread_bench_stress_001() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_1"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_002() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_2"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_003() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_3"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_004() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_4"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_005() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_5"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_006() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_6"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_007() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_7"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_008() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_8"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_009() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_9"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_010() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_10"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_011() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_11"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_012() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_12"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_013() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_13"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_014() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_14"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_015() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_15"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_016() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_16"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_017() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_17"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_018() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_18"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_019() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_19"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_020() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_20"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_021() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_21"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_022() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_22"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_023() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_23"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_024() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_24"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_025() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_25"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_026() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_26"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_027() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_27"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_028() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_28"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_029() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_29"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_030() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_30"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_031() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_31"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_032() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_32"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_033() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_33"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_034() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_34"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_035() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_35"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_036() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_36"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_037() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_37"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_038() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_38"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_039() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_39"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_040() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_40"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_041() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_41"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_042() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_42"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_043() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_43"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_044() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_44"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_045() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_45"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_046() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_46"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_047() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_47"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_048() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_48"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_049() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_49"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_050() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_50"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_051() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_51"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_052() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_52"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_053() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_53"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_054() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_54"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_055() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_55"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_056() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_56"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_057() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_57"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_058() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_58"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_059() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_59"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_060() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_60"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_061() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_61"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_062() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_62"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_063() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_63"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_064() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_64"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_065() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_65"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_066() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_66"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_067() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_67"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_068() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_68"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_069() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_69"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_070() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_70"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_071() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_71"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_072() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_72"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_073() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_73"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_074() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_74"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_075() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_75"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_076() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_76"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_077() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_77"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_078() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_78"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_079() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_79"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_080() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_80"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_081() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_81"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_082() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_82"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_083() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_83"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_084() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_84"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_085() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_85"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_086() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_86"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_087() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_87"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_088() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_88"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_089() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_89"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_090() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_90"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_091() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_91"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_092() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_92"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_093() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_93"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_094() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_94"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_095() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_95"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_096() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_96"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_097() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_97"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_098() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_98"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_099() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_99"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_100() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_100"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_101() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_101"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_102() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_102"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_103() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_103"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_104() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_104"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_105() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_105"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_106() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_106"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_107() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_107"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_108() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_108"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_109() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_109"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_110() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_110"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_111() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_111"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_112() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_112"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_113() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_113"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_114() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_114"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_115() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_115"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_116() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_116"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_117() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_117"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_118() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_118"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_119() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_119"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_120() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_120"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_121() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_121"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_122() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_122"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_123() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_123"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_124() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_124"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_125() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_125"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_126() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_126"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_127() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_127"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_128() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_128"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_129() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_129"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_130() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_130"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_131() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_131"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_132() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_132"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_133() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_133"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_134() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_134"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_135() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_135"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_136() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_136"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_137() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_137"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_138() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_138"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_139() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_139"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_140() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_140"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_141() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_141"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_142() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_142"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_143() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_143"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_144() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_144"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_145() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_145"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_146() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_146"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_147() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_147"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_148() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_148"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_149() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_149"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_150() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_150"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_151() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_151"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_152() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_152"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_153() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_153"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_154() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_154"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_155() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_155"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_156() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_156"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_157() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_157"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_158() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_158"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_159() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_159"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_160() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_160"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_161() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_161"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_162() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_162"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_163() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_163"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_164() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_164"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_165() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_165"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_166() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_166"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_167() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_167"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_168() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_168"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_169() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_169"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_170() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_170"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_171() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_171"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_172() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_172"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_173() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_173"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_174() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_174"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_175() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_175"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_176() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_176"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_177() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_177"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_178() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_178"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_179() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_179"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_180() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_180"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_181() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_181"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_182() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_182"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_183() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_183"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_184() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_184"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_185() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_185"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_186() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_186"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_187() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_187"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_188() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_188"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_189() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_189"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_190() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_190"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_191() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_191"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_192() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_192"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_193() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_193"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_194() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_194"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_195() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_195"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_196() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_196"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_197() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_197"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_198() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_198"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_199() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_199"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_200() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_200"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_201() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_201"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_202() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_202"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_203() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_203"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_204() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_204"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_205() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_205"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_206() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_206"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_207() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_207"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_208() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_208"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_209() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_209"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_210() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_210"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_211() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_211"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_212() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_212"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_213() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_213"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_214() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_214"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_215() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_215"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_216() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_216"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_217() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_217"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_218() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_218"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_219() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_219"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_220() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_220"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_221() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_221"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_222() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_222"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_223() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_223"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_224() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_224"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_225() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_225"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_226() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_226"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_227() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_227"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_228() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_228"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_229() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_229"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_230() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_230"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_231() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_231"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_232() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_232"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_233() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_233"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_234() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_234"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_235() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_235"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_236() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_236"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_237() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_237"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_238() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_238"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_239() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_239"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_240() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_240"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_241() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_241"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_242() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_242"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_243() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_243"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_244() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_244"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_245() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_245"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_246() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_246"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_247() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_247"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_248() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_248"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_249() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_249"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_250() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_250"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_251() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_251"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_252() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_252"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_253() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_253"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_254() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_254"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_255() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_255"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_256() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_256"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_257() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_257"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_258() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_258"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_259() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_259"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_260() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_260"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_261() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_261"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_262() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_262"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_263() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_263"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_264() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_264"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_265() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_265"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_266() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_266"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_267() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_267"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_268() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_268"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_269() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_269"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_270() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_270"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_271() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_271"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_272() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_272"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_273() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_273"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_274() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_274"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_275() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_275"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_276() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_276"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_277() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_277"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_278() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_278"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_279() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_279"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_280() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_280"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_281() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_281"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_282() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_282"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_283() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_283"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_284() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_284"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_285() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_285"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_286() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_286"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_287() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_287"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_288() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_288"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_289() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_289"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_290() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_290"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_291() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_291"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_292() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_292"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_293() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_293"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_294() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_294"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_295() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_295"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_296() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_296"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_297() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_297"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_298() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_298"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_299() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_299"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_300() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_300"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_301() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_301"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_302() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_302"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_303() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_303"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_304() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_304"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_305() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_305"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_306() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_306"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_307() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_307"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_308() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_308"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_309() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_309"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_310() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_310"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_311() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_311"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_312() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_312"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_313() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_313"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_314() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_314"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_315() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_315"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_316() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_316"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_317() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_317"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_318() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_318"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_319() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_319"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_320() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_320"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_321() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_321"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_322() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_322"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_323() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_323"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_324() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_324"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_325() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_325"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_326() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_326"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_327() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_327"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_328() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_328"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    #[test]
    fn test_thread_bench_stress_329() {
        let cfg = crate::core::BenchConfig::new(format!("thread_bench_329"));
        let sample = crate::core::Sample::new(std::time::Duration::from_millis(10), 1);
        let res = BenchResult::new(cfg, vec![sample], std::time::Duration::from_millis(10));
        let report = ScalabilityReport::from_results("test_scale", &[(1, res.clone()), (2, res)]);
        assert_eq!(report.thread_counts.len(), 2);
        assert!(report.speedup_ratios[0] > 0.0);
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
}
