//! # CLI Runtime Configuration & Environment Options
//!
//! Controls output verbosity levels, ANSI color modes, timeout budgets, and compute devices.

/// Console logging verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
    Debug,
    Trace,
}

/// Terminal ANSI color preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    #[default]
    Auto,
    Always,
    Never,
}

/// Global CLI invocation options.
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub verbosity: Verbosity,
    pub color: ColorChoice,
    pub device: String,
    pub timeout_seconds: Option<u64>,
    pub num_threads: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            verbosity: Verbosity::Normal,
            color: ColorChoice::Auto,
            device: "cpu".to_string(),
            timeout_seconds: None,
            num_threads: 1,
        }
    }
}

impl CliConfig {
    /// Creates a new default `CliConfig`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets verbosity level.
    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Sets color mode.
    pub fn with_color(mut self, color: ColorChoice) -> Self {
        self.color = color;
        self
    }

    /// Sets compute device target string.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
