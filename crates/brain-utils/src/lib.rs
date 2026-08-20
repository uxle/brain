//! # brain-utils 🧠🛠️
//!
//! Production-grade logging, profiling, I/O, configuration, hashing, checksums,
//! and diagnostics platform for the **Brain** deep learning framework.

pub mod builder;
pub mod checksum;
pub mod config;
pub mod core;
pub mod env;
pub mod fs_stats;
pub mod hal;
pub mod hash;
pub mod r#impl;
pub mod io;
pub mod log;
pub mod metrics_utils;
pub mod ops;
pub mod panic;
pub mod prof;
pub mod serialize_helpers;
pub mod units;
pub mod utils;
pub mod version;

/// Unified prelude for brain-utils crate.
pub mod prelude {
    pub use crate::builder::UtilsBuilder;
    pub use crate::checksum::{Adler32, Crc32};
    pub use crate::config::schema::{Constraint, FieldDef, FieldType, SchemaValidator};
    pub use crate::config::{ConfigEntry, ConfigManager, ConfigSource};
    pub use crate::core::{GlobalState, SystemInfo, UtilsConfig, UtilsError, UtilsResult};
    pub use crate::env::{
        env_bool, env_f64, env_get, env_get_or, env_i64, env_list, env_prefix_map, EnvConfig,
    };
    pub use crate::fs_stats::{dir_size, file_count, scan_dir_stats, FsStats};
    pub use crate::hal::*;
    pub use crate::hash::{fnv1a_32, fnv1a_64, hash_combine_64, hash_str, murmur3_32};
    pub use crate::io::csv::{CsvConfig, CsvReader, CsvRecord, CsvWriter};
    pub use crate::io::ini::IniFile;
    pub use crate::io::json::{parse_json, JsonValue};
    pub use crate::io::paths::{
        ensure_dir, extension, join_safe, normalize_slashes, unique_temp_path,
    };
    pub use crate::io::{
        append_file, atomic_write_file, delete_file, read_file_bytes, read_file_str, write_file,
        IoConfig,
    };
    pub use crate::log::formatters::{
        JsonFormatter, KeyValueFormatter, LogFormatter, PlainFormatter, TimestampedFormatter,
    };
    pub use crate::log::sinks::{ConsoleSink, LogSink, MultiSink, RingBufferSink};
    pub use crate::log::{LogLevel, LogRecord, Logger, StandardLogger};
    pub use crate::metrics_utils::{Metric, MetricType, MetricsRegistry};
    pub use crate::ops::{chunk_slice, measure_block, retry_with_backoff, TokenBucketRateLimiter};
    pub use crate::panic::{last_panic_message, set_panic_hook};
    pub use crate::prof::counters::{AtomicCounter, AtomicGauge, CounterSet};
    pub use crate::prof::scope::{ScopeGuard, ScopeReport};
    pub use crate::prof::timer::{Timer, TimingStats, TimingTree};
    pub use crate::prof::{ProfConfig, Profiler};
    pub use crate::r#impl::{get_global_state, init_utils, shutdown_utils, utils_summary};
    pub use crate::serialize_helpers::{base64_decode, base64_encode, bytes_to_hex, hex_to_bytes};
    pub use crate::units::{
        format_bytes, format_bytes_binary, format_duration, format_percent, parse_size,
    };
    pub use crate::utils::{
        now_ms, now_ns, now_us, pad_left, pad_right, random_uuid_lite, sanitize_filename,
        shell_quote, truncate_str,
    };
    pub use crate::version::Version;
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_prelude_integration_1() {
        use crate::prelude::*;
        let cfg = UtilsBuilder::new().app_name("lib_test_1").build();
        assert_eq!(cfg.app_name, "lib_test_1");

        let h = fnv1a_64(b"test_1");
        assert_ne!(h, 0);

        let v = Version::new(0, 2, 0);
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 0);
    }
}
