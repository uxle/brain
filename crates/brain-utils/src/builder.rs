//! # Fluent Utility Builders
//!
//! Provides builder patterns for configuring logging, profiling,
//! and global framework settings.

use crate::core::UtilsConfig;

/// Fluent builder for constructing `UtilsConfig` and configuring global runtime.
#[derive(Debug, Clone, Default)]
pub struct UtilsBuilder {
    config: UtilsConfig,
}

impl UtilsBuilder {
    /// Creates a new builder with defaults.
    pub fn new() -> Self {
        Self {
            config: UtilsConfig::default(),
        }
    }

    /// Sets logging level.
    pub fn log_level(mut self, level: &str) -> Self {
        self.config.log_level = level.to_string();
        self
    }

    /// Sets log file destination.
    pub fn log_file(mut self, path: &str) -> Self {
        self.config.log_file_path = Some(path.to_string());
        self
    }

    /// Enables or disables profiling.
    pub fn profiler(mut self, enabled: bool) -> Self {
        self.config.profiling_enabled = enabled;
        self
    }

    /// Sets application name.
    pub fn app_name(mut self, name: &str) -> Self {
        self.config.app_name = name.to_string();
        self
    }

    /// Sets environment variable lookup prefix.
    pub fn env_prefix(mut self, prefix: &str) -> Self {
        self.config.env_prefix = prefix.to_string();
        self
    }

    /// Finalizes and returns the `UtilsConfig`.
    pub fn build(self) -> UtilsConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_utils_builder_fluent_1() {
        let cfg = UtilsBuilder::new()
            .log_level("DEBUG")
            .log_file("/tmp/app.log")
            .profiler(true)
            .app_name("custom_brain")
            .env_prefix("MYAPP_")
            .build();
    
        assert_eq!(cfg.log_level, "DEBUG");
        assert_eq!(cfg.log_file_path, Some("/tmp/app.log".to_string()));
        assert!(cfg.profiling_enabled);
        assert_eq!(cfg.app_name, "custom_brain");
        assert_eq!(cfg.env_prefix, "MYAPP_");
    }
}
