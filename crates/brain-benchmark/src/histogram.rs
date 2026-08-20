//! # High-Dynamic-Range (HDR) Latency Histogram
//!
//! Provides a memory-efficient logarithmic bucket histogram for nanosecond-to-minute latency distributions.

/// High dynamic range logarithmic latency histogram.
#[derive(Debug, Clone)]
pub struct HdrHistogram {
    buckets: Vec<u64>,
    min_value: u64,
    max_value: u64,
    total_count: u64,
}

impl Default for HdrHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl HdrHistogram {
    /// Creates a new `HdrHistogram` with 1024 logarithmic buckets.
    pub fn new() -> Self {
        Self {
            buckets: vec![0; 1024],
            min_value: u64::MAX,
            max_value: 0,
            total_count: 0,
        }
    }

    /// Records a latency value in nanoseconds.
    pub fn record(&mut self, value_nanos: u64) {
        self.min_value = self.min_value.min(value_nanos);
        self.max_value = self.max_value.max(value_nanos);
        self.total_count += 1;

        let bucket_idx = self.value_to_bucket(value_nanos);
        self.buckets[bucket_idx] += 1;
    }

    fn value_to_bucket(&self, val: u64) -> usize {
        if val == 0 {
            0
        } else {
            let leading_zeros = val.leading_zeros() as usize;
            let log_bucket = (64 - leading_zeros).min(63);
            let sub_bucket = ((val >> log_bucket.saturating_sub(4)) & 0x0F) as usize;
            (log_bucket * 16 + sub_bucket).min(self.buckets.len() - 1)
        }
    }

    /// Returns the total number of recorded observations.
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Returns the minimum recorded value in nanoseconds.
    pub fn min(&self) -> u64 {
        if self.total_count == 0 {
            0
        } else {
            self.min_value
        }
    }

    /// Returns the maximum recorded value in nanoseconds.
    pub fn max(&self) -> u64 {
        self.max_value
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
