//! # Core Utilities for Brain
//!
//! Provides fundamental configuration structures, error handling types,
//! global state containers, and execution context primitives.

use std::fmt;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Result type alias for brain-utils operations.
pub type UtilsResult<T> = Result<T, UtilsError>;

/// Master error enumeration for the brain-utils crate.
#[derive(Debug, Clone, PartialEq)]
pub enum UtilsError {
    /// Generic I/O error with descriptive message.
    IoError(String),
    /// File not found at the specified path.
    FileNotFound(String),
    /// Permission denied for requested operation.
    PermissionDenied(String),
    /// Configuration key missing or invalid.
    ConfigError(String),
    /// Schema validation failed.
    ValidationError(String),
    /// JSON parsing or serialization failure.
    JsonError(String),
    /// CSV parsing or formatting failure.
    CsvError(String),
    /// INI format syntax or lookup error.
    IniError(String),
    /// Profiling session or timer error.
    ProfileError(String),
    /// Logging sink or formatting failure.
    LogError(String),
    /// Conversion or parsing error.
    ParseError(String),
    /// Operation timed out.
    Timeout(String),
    /// Feature or operation unsupported.
    Unsupported(String),
}

impl fmt::Display for UtilsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "I/O Error: {}", msg),
            Self::FileNotFound(path) => write!(f, "File not found: {}", path),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            Self::ConfigError(msg) => write!(f, "Config error: {}", msg),
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            Self::JsonError(msg) => write!(f, "JSON error: {}", msg),
            Self::CsvError(msg) => write!(f, "CSV error: {}", msg),
            Self::IniError(msg) => write!(f, "INI error: {}", msg),
            Self::ProfileError(msg) => write!(f, "Profile error: {}", msg),
            Self::LogError(msg) => write!(f, "Log error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Self::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for UtilsError {}

/// Unified configuration umbrella for brain framework utilities.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilsConfig {
    /// Logging level threshold (e.g., "INFO", "DEBUG", "TRACE").
    pub log_level: String,
    /// Path to output log file (if file logging is enabled).
    pub log_file_path: Option<String>,
    /// Whether profiling is globally enabled.
    pub profiling_enabled: bool,
    /// Maximum ring buffer entries for in-memory logs.
    pub max_log_buffer_size: usize,
    /// Application name tag.
    pub app_name: String,
    /// Environment prefix for variable lookups.
    pub env_prefix: String,
    /// Metrics collection interval in seconds.
    pub metrics_interval_sec: u64,
}

impl Default for UtilsConfig {
    fn default() -> Self {
        Self {
            log_level: "INFO".to_string(),
            log_file_path: None,
            profiling_enabled: true,
            max_log_buffer_size: 4096,
            app_name: "brain-engine".to_string(),
            env_prefix: "BRAIN_".to_string(),
            metrics_interval_sec: 10,
        }
    }
}

impl UtilsConfig {
    /// Creates a new UtilsConfig with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the logging level.
    pub fn with_log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = level.into();
        self
    }

    /// Sets the log file path.
    pub fn with_log_file(mut self, path: impl Into<String>) -> Self {
        self.log_file_path = Some(path.into());
        self
    }

    /// Toggles profiling.
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling_enabled = enabled;
        self
    }

    /// Sets app name.
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
        self
    }
}

/// Global system runtime information.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemInfo {
    /// Operating system name.
    pub os: String,
    /// Process ID.
    pub pid: u32,
    /// Framework version.
    pub version: String,
    /// Timestamp when process initialized.
    pub start_time_ms: u64,
}

impl SystemInfo {
    /// Retrieves current system runtime info.
    pub fn current() -> Self {
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            os: std::env::consts::OS.to_string(),
            pid: std::process::id(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            start_time_ms: start,
        }
    }
}

/// Global state holder for logging, profiling, and registry coordination.
#[derive(Debug)]
pub struct GlobalState {
    config: RwLock<UtilsConfig>,
    initialized: Mutex<bool>,
    start_instant: Instant,
}

impl GlobalState {
    /// Constructs a new global state container.
    pub fn new(config: UtilsConfig) -> Self {
        Self {
            config: RwLock::new(config),
            initialized: Mutex::new(true),
            start_instant: Instant::now(),
        }
    }

    /// Gets a clone of current config.
    pub fn get_config(&self) -> UtilsConfig {
        self.config.read().map(|c| c.clone()).unwrap_or_default()
    }

    /// Updates current config.
    pub fn set_config(&self, config: UtilsConfig) -> UtilsResult<()> {
        let mut w = self
            .config
            .write()
            .map_err(|_| UtilsError::ConfigError("Lock poison".to_string()))?;
        *w = config;
        Ok(())
    }

    /// Checks if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.lock().map(|i| *i).unwrap_or(false)
    }

    /// Elapsed process uptime.
    pub fn uptime(&self) -> Duration {
        self.start_instant.elapsed()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_core_utils_lifecycle_1() {
        let mut cfg = UtilsConfig::default()
            .with_log_level("DEBUG")
            .with_app_name("test_suite_1")
            .with_profiling(true);
        assert_eq!(cfg.log_level, "DEBUG");
        assert_eq!(cfg.app_name, "test_suite_1");
        assert!(cfg.profiling_enabled);

        let state = GlobalState::new(cfg.clone());
        assert!(state.is_initialized());
        let _ = state.uptime();

        cfg.metrics_interval_sec = 2;
        assert!(state.set_config(cfg.clone()).is_ok());
        let retrieved = state.get_config();
        assert_eq!(retrieved.metrics_interval_sec, 2);

        let err = UtilsError::ConfigError(format!("err_{}", 1));
        assert_eq!(err.to_string(), format!("Config error: err_{}", 1));

        let sys = SystemInfo::current();
        assert!(!sys.os.is_empty());
        assert!(sys.pid > 0);
    }
}
