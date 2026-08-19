//! # JIT Compiler & Cache Configuration
//!
//! Controls cache capacities, parallel compilation thresholds, and target hardware presets.

/// JIT caching policy and capacity limits.
#[derive(Debug, Clone)]
pub struct JitCacheConfig {
    pub max_entries: usize,
    pub ttl_seconds: Option<u64>,
    pub enable_persistence: bool,
}

impl Default for JitCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            ttl_seconds: None,
            enable_persistence: false,
        }
    }
}

/// Global compiler settings.
#[derive(Debug, Clone, Default)]
pub struct CompilerConfig {
    pub cache: JitCacheConfig,
    pub num_worker_threads: usize,
    pub debug_dump_ir: bool,
}

impl CompilerConfig {
    /// Creates a new `CompilerConfig`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of cached JIT graph kernels.
    pub fn with_cache_capacity(mut self, cap: usize) -> Self {
        self.cache.max_entries = cap;
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
