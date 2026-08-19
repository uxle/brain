# `brain-utils`

> Shared toolkit: hashing and checksums, logging, config, profiling, I/O parsers, environment access, and formatting utilities.

## Overview

`brain-utils` is the workspace utility crate: zero-dependency hashing (FNV-1a, Murmur3, CRC32, Adler32), a structured logging system with sinks and formatters, layered `ConfigManager` with source precedence, profiling timers and atomic counters, dependency-free JSON/CSV/INI parsers, environment variable helpers, and assorted string/format/time utilities.

## Features

- Hashing & checksums: `fnv1a_64`/`fnv1a_32`, `hash_combine_64`, `murmur3_32`, `hash_str`; `Crc32` and `Adler32` streaming checksums
- Logging: `LogLevel`, `LogRecord`, `StandardLogger` with pluggable `LogSink`/`LogFormatter` sinks, and `log_trace!`/`log_debug!`/`log_info!`/`log_warn!`/`log_error!` macros
- Config: `ConfigManager` with `set`/`force_set`/`get` by `ConfigSource`, `EnvConfig::from_prefix` typed env access, `UtilsConfig` global state
- Profiling: `TimingStats` (mean, p50/p95/p99), `TimingTree`, `AtomicCounter`, scope-timed regions
- I/O: zero-dependency `parse_json` + `JsonValue`, `CsvReader`/`CsvRecord`/`CsvConfig`, `IniFile`, path helpers
- Utilities: `FastRng` seeded RNG, `now_ms`/`now_us`/`now_ns`, `random_uuid_lite`, `shell_quote`, `sanitize_filename`, string pad/truncate, `measure_block`, `retry_with_backoff`, `RateLimiter`, `chunk_slice`
- `SystemInfo::current`, `Version::parse` with comparison, `FsStats::scan_dir_stats` / `dir_size`, Prometheus-style `Metric` registry helpers, panic helpers

## Modules

| Module | Description |
|---|---|
| `hash` | FNV-1a, Murmur3, hash combine, string hashing |
| `checksum` | `Crc32`, `Adler32` |
| `log` | Levels, `StandardLogger`, sinks, formatters, macros |
| `config` | `ConfigManager`, `ConfigEntry`, schema |
| `prof` | `TimingStats`, `TimingTree`, `AtomicCounter`, scope/timer |
| `io` | JSON, CSV, INI parsers and path helpers |
| `env` | Typed environment access, `EnvConfig` |
| `utils` | `FastRng`, time, UUID-lite, string helpers |
| `ops` | `measure_block`, `retry_with_backoff`, `RateLimiter`, `chunk_slice` |
| `metrics_utils` | `Metric`/registry with labels |
| `fs_stats` | Directory size and stats scanning |
| `version` | `Version` parse/compare |
| `core` | `UtilsConfig`, `SystemInfo`, `GlobalState` |
| `units` / `panic` / `serialize_helpers` / `builder` | Formatting, panic, serialization, construction helpers |

## Quick Start

```rust
use brain_utils::hash::{fnv1a_64, murmur3_32};
use brain_utils::checksum::Crc32;
use brain_utils::utils::FastRng;

let h = fnv1a_64(b"brain");
let crc = Crc32::compute(b"brain");
let mut rng = FastRng::new(42);
let x: f64 = rng.next_f64();
println!("hash={h} crc={crc} rand={x}");
```

## Testing

```bash
cargo test -p brain-utils -j 2
```

## Workspace Role

Depends only on `brain-core`. `brain-utils` is the shared plumbing for the rest of the workspace: crates reuse its hashing (cache keys), logging, config precedence, and profiling timers so the framework stays dependency-free.