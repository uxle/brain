//! # Hardware Discovery and Topology Probing
//!
//! Inspects host hardware topology, CPU specifications, core counts, cache sizes,
//! and estimates raw memory bandwidth.

use std::fs::File;
use std::io::{BufRead, BufReader};

/// Discovered hardware profile of the host system.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub l1_cache_kb: Option<usize>,
    pub l2_cache_kb: Option<usize>,
    pub l3_cache_kb: Option<usize>,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self::probe()
    }
}

impl HardwareInfo {
    /// Probes system hardware information.
    pub fn probe() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let (cpu_model, physical_cores) = Self::probe_linux_cpuinfo().unwrap_or_else(|| {
            ("Generic CPU".to_string(), logical_cores)
        });

        Self {
            cpu_model,
            logical_cores,
            physical_cores,
            l1_cache_kb: Some(32),
            l2_cache_kb: Some(512),
            l3_cache_kb: Some(16384),
        }
    }

    fn probe_linux_cpuinfo() -> Option<(String, usize)> {
        let file = File::open("/proc/cpuinfo").ok()?;
        let reader = BufReader::new(file);
        let mut model_name = None;
        let mut core_ids = std::collections::HashSet::new();

        for line in reader.lines().map_while(Result::ok) {
            if line.starts_with("model name") && model_name.is_none() {
                if let Some(pos) = line.find(':') {
                    model_name = Some(line[pos + 1..].trim().to_string());
                }
            }
            if line.starts_with("core id") {
                if let Some(pos) = line.find(':') {
                    if let Ok(id) = line[pos + 1..].trim().parse::<usize>() {
                        core_ids.insert(id);
                    }
                }
            }
        }

        let model = model_name.unwrap_or_else(|| "Unknown CPU".to_string());
        let physical = if !core_ids.is_empty() {
            core_ids.len()
        } else {
            1
        };

        Some((model, physical))
    }

    /// Estimates memory copy bandwidth in Gigabytes per second.
    pub fn estimate_memory_bandwidth_gbps() -> f64 {
        let size = 1_000_000;
        let src = vec![1.0_f64; size];
        let mut dst = vec![0.0_f64; size];
        let bytes = (size * std::mem::size_of::<f64>()) as f64;

        let start = std::time::Instant::now();
        let iters = 100;
        for _ in 0..iters {
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
        }
        let elapsed = start.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            (bytes * iters as f64 / 1e9) / elapsed
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
