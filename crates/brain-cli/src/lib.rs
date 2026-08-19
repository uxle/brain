//! # Brain CLI — Developer-Grade Deep Learning Command-Line Suite
//!
//! Features an interactive REPL with mathematical expressions, subcommands for tensor/model/training/benchmarking,
//! project scaffolding, shell completions, and diagnostic system inspection.
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_cli::prelude::*;
//!
//! let sink = OutputSink::memory();
//! let exit_code = run_cli(&["--version".to_string()], &sink);
//! assert!(exit_code.is_success());
//! assert!(sink.captured().unwrap().contains("brain-cli v"));
//! ```
//!
//! ## Running Subcommands
//!
//! ```rust
//! use brain_cli::prelude::*;
//!
//! let sink = OutputSink::memory();
//! let exit_code = run_cli(&["doctor".to_string()], &sink);
//! assert!(exit_code.is_success());
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod cache;
pub mod commands;
pub mod completion;
pub mod config;
pub mod config_file;
pub mod core;
pub mod datafile;
pub mod diagnostics;
pub mod errors;
pub mod r#impl;
pub mod init;
pub mod interactive;
pub mod ops;
pub mod parser;
pub mod plugin;
pub mod pretty;
pub mod repl;
pub mod script;
pub mod term;
pub mod utils;

// Re-exports
pub use config::{CliConfig, ColorChoice, Verbosity};
pub use config_file::ConfigFile;
pub use core::{CommandSpec, ExitCode, OutputFormat, OutputSink};
pub use r#impl::run_cli;

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_cli::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_cli::version_string;
/// assert_eq!(version_string(), "brain-cli v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-cli v{}", VERSION)
}

/// Standard prelude imports for CLI utilities.
///
/// ```rust
/// use brain_cli::prelude::*;
/// let sink = OutputSink::memory();
/// sink.println("hello");
/// assert_eq!(sink.captured().unwrap().trim(), "hello");
/// ```
pub mod prelude {
    pub use crate::config::{CliConfig, ColorChoice, Verbosity};
    pub use crate::core::{CommandSpec, ExitCode, OutputFormat, OutputSink};
    pub use crate::r#impl::run_cli;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
