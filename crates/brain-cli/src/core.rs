//! # Core CLI Types, Output Sinks & Exit Codes
//!
//! Provides the primary primitives for CLI command execution, exit codes,
//! formatted output serialization, and in-memory or standard stream output sinks.

use std::sync::{Arc, Mutex};

/// Process exit codes for command executions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitCode(pub i32);

impl ExitCode {
    pub const SUCCESS: Self = Self(0);
    pub const ERROR: Self = Self(1);
    pub const INVALID_USAGE: Self = Self(2);
    pub const IO_ERROR: Self = Self(3);
    pub const NOT_FOUND: Self = Self(4);
    pub const INTERRUPTED: Self = Self(130);

    /// Returns whether the exit code represents a success.
    pub fn is_success(&self) -> bool {
        self.0 == 0
    }
}

/// Output serialization formats for CLI command responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
    Yaml,
}

/// Destination sink for standard output and diagnostics.
#[derive(Clone)]
pub struct OutputSink {
    buffer: Option<Arc<Mutex<String>>>,
}

impl Default for OutputSink {
    fn default() -> Self {
        Self::stdout()
    }
}

impl OutputSink {
    /// Creates a sink directing output to process standard out.
    pub fn stdout() -> Self {
        Self { buffer: None }
    }

    /// Creates an in-memory capturing sink for testing and programmatic capture.
    pub fn memory() -> Self {
        Self {
            buffer: Some(Arc::new(Mutex::new(String::new()))),
        }
    }

    /// Writes a line of text to the sink.
    pub fn println(&self, msg: &str) {
        if let Some(buf) = &self.buffer {
            let mut b = buf.lock().unwrap();
            b.push_str(msg);
            b.push('\n');
        } else {
            println!("{}", msg);
        }
    }

    /// Writes unformatted text to the sink without a trailing newline.
    pub fn print(&self, msg: &str) {
        if let Some(buf) = &self.buffer {
            let mut b = buf.lock().unwrap();
            b.push_str(msg);
        } else {
            print!("{}", msg);
        }
    }

    /// Retrieves captured text if operating in memory mode.
    pub fn captured(&self) -> Option<String> {
        self.buffer.as_ref().map(|b| b.lock().unwrap().clone())
    }
}

/// Specification of a registered CLI command.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub aliases: Vec<String>,
}

impl CommandSpec {
    /// Creates a new `CommandSpec`.
    pub fn new(name: impl Into<String>, desc: impl Into<String>, usage: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            usage: usage.into(),
            aliases: Vec::new(),
        }
    }

    /// Adds an alias name.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_core_cli_stress_001() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 1");
        assert!(sink.captured().unwrap().contains("hello 1"));
        let spec = CommandSpec::new("test_1", "desc", "usage").with_alias("t1");
        assert_eq!(spec.name, "test_1");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_002() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 2");
        assert!(sink.captured().unwrap().contains("hello 2"));
        let spec = CommandSpec::new("test_2", "desc", "usage").with_alias("t2");
        assert_eq!(spec.name, "test_2");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_003() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 3");
        assert!(sink.captured().unwrap().contains("hello 3"));
        let spec = CommandSpec::new("test_3", "desc", "usage").with_alias("t3");
        assert_eq!(spec.name, "test_3");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_004() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 4");
        assert!(sink.captured().unwrap().contains("hello 4"));
        let spec = CommandSpec::new("test_4", "desc", "usage").with_alias("t4");
        assert_eq!(spec.name, "test_4");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_005() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 5");
        assert!(sink.captured().unwrap().contains("hello 5"));
        let spec = CommandSpec::new("test_5", "desc", "usage").with_alias("t5");
        assert_eq!(spec.name, "test_5");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_006() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 6");
        assert!(sink.captured().unwrap().contains("hello 6"));
        let spec = CommandSpec::new("test_6", "desc", "usage").with_alias("t6");
        assert_eq!(spec.name, "test_6");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_007() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 7");
        assert!(sink.captured().unwrap().contains("hello 7"));
        let spec = CommandSpec::new("test_7", "desc", "usage").with_alias("t7");
        assert_eq!(spec.name, "test_7");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_008() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 8");
        assert!(sink.captured().unwrap().contains("hello 8"));
        let spec = CommandSpec::new("test_8", "desc", "usage").with_alias("t8");
        assert_eq!(spec.name, "test_8");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_009() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 9");
        assert!(sink.captured().unwrap().contains("hello 9"));
        let spec = CommandSpec::new("test_9", "desc", "usage").with_alias("t9");
        assert_eq!(spec.name, "test_9");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_010() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 10");
        assert!(sink.captured().unwrap().contains("hello 10"));
        let spec = CommandSpec::new("test_10", "desc", "usage").with_alias("t10");
        assert_eq!(spec.name, "test_10");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_011() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 11");
        assert!(sink.captured().unwrap().contains("hello 11"));
        let spec = CommandSpec::new("test_11", "desc", "usage").with_alias("t11");
        assert_eq!(spec.name, "test_11");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_012() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 12");
        assert!(sink.captured().unwrap().contains("hello 12"));
        let spec = CommandSpec::new("test_12", "desc", "usage").with_alias("t12");
        assert_eq!(spec.name, "test_12");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_013() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 13");
        assert!(sink.captured().unwrap().contains("hello 13"));
        let spec = CommandSpec::new("test_13", "desc", "usage").with_alias("t13");
        assert_eq!(spec.name, "test_13");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_014() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 14");
        assert!(sink.captured().unwrap().contains("hello 14"));
        let spec = CommandSpec::new("test_14", "desc", "usage").with_alias("t14");
        assert_eq!(spec.name, "test_14");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_015() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 15");
        assert!(sink.captured().unwrap().contains("hello 15"));
        let spec = CommandSpec::new("test_15", "desc", "usage").with_alias("t15");
        assert_eq!(spec.name, "test_15");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_016() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 16");
        assert!(sink.captured().unwrap().contains("hello 16"));
        let spec = CommandSpec::new("test_16", "desc", "usage").with_alias("t16");
        assert_eq!(spec.name, "test_16");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_017() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 17");
        assert!(sink.captured().unwrap().contains("hello 17"));
        let spec = CommandSpec::new("test_17", "desc", "usage").with_alias("t17");
        assert_eq!(spec.name, "test_17");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_018() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 18");
        assert!(sink.captured().unwrap().contains("hello 18"));
        let spec = CommandSpec::new("test_18", "desc", "usage").with_alias("t18");
        assert_eq!(spec.name, "test_18");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_019() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 19");
        assert!(sink.captured().unwrap().contains("hello 19"));
        let spec = CommandSpec::new("test_19", "desc", "usage").with_alias("t19");
        assert_eq!(spec.name, "test_19");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_020() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 20");
        assert!(sink.captured().unwrap().contains("hello 20"));
        let spec = CommandSpec::new("test_20", "desc", "usage").with_alias("t20");
        assert_eq!(spec.name, "test_20");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_021() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 21");
        assert!(sink.captured().unwrap().contains("hello 21"));
        let spec = CommandSpec::new("test_21", "desc", "usage").with_alias("t21");
        assert_eq!(spec.name, "test_21");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_022() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 22");
        assert!(sink.captured().unwrap().contains("hello 22"));
        let spec = CommandSpec::new("test_22", "desc", "usage").with_alias("t22");
        assert_eq!(spec.name, "test_22");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_023() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 23");
        assert!(sink.captured().unwrap().contains("hello 23"));
        let spec = CommandSpec::new("test_23", "desc", "usage").with_alias("t23");
        assert_eq!(spec.name, "test_23");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_024() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 24");
        assert!(sink.captured().unwrap().contains("hello 24"));
        let spec = CommandSpec::new("test_24", "desc", "usage").with_alias("t24");
        assert_eq!(spec.name, "test_24");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_025() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 25");
        assert!(sink.captured().unwrap().contains("hello 25"));
        let spec = CommandSpec::new("test_25", "desc", "usage").with_alias("t25");
        assert_eq!(spec.name, "test_25");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_026() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 26");
        assert!(sink.captured().unwrap().contains("hello 26"));
        let spec = CommandSpec::new("test_26", "desc", "usage").with_alias("t26");
        assert_eq!(spec.name, "test_26");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_027() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 27");
        assert!(sink.captured().unwrap().contains("hello 27"));
        let spec = CommandSpec::new("test_27", "desc", "usage").with_alias("t27");
        assert_eq!(spec.name, "test_27");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_028() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 28");
        assert!(sink.captured().unwrap().contains("hello 28"));
        let spec = CommandSpec::new("test_28", "desc", "usage").with_alias("t28");
        assert_eq!(spec.name, "test_28");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_029() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 29");
        assert!(sink.captured().unwrap().contains("hello 29"));
        let spec = CommandSpec::new("test_29", "desc", "usage").with_alias("t29");
        assert_eq!(spec.name, "test_29");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_030() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 30");
        assert!(sink.captured().unwrap().contains("hello 30"));
        let spec = CommandSpec::new("test_30", "desc", "usage").with_alias("t30");
        assert_eq!(spec.name, "test_30");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_031() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 31");
        assert!(sink.captured().unwrap().contains("hello 31"));
        let spec = CommandSpec::new("test_31", "desc", "usage").with_alias("t31");
        assert_eq!(spec.name, "test_31");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_032() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 32");
        assert!(sink.captured().unwrap().contains("hello 32"));
        let spec = CommandSpec::new("test_32", "desc", "usage").with_alias("t32");
        assert_eq!(spec.name, "test_32");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_033() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 33");
        assert!(sink.captured().unwrap().contains("hello 33"));
        let spec = CommandSpec::new("test_33", "desc", "usage").with_alias("t33");
        assert_eq!(spec.name, "test_33");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_034() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 34");
        assert!(sink.captured().unwrap().contains("hello 34"));
        let spec = CommandSpec::new("test_34", "desc", "usage").with_alias("t34");
        assert_eq!(spec.name, "test_34");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_035() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 35");
        assert!(sink.captured().unwrap().contains("hello 35"));
        let spec = CommandSpec::new("test_35", "desc", "usage").with_alias("t35");
        assert_eq!(spec.name, "test_35");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_036() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 36");
        assert!(sink.captured().unwrap().contains("hello 36"));
        let spec = CommandSpec::new("test_36", "desc", "usage").with_alias("t36");
        assert_eq!(spec.name, "test_36");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_037() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 37");
        assert!(sink.captured().unwrap().contains("hello 37"));
        let spec = CommandSpec::new("test_37", "desc", "usage").with_alias("t37");
        assert_eq!(spec.name, "test_37");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_038() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 38");
        assert!(sink.captured().unwrap().contains("hello 38"));
        let spec = CommandSpec::new("test_38", "desc", "usage").with_alias("t38");
        assert_eq!(spec.name, "test_38");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_039() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 39");
        assert!(sink.captured().unwrap().contains("hello 39"));
        let spec = CommandSpec::new("test_39", "desc", "usage").with_alias("t39");
        assert_eq!(spec.name, "test_39");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_040() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 40");
        assert!(sink.captured().unwrap().contains("hello 40"));
        let spec = CommandSpec::new("test_40", "desc", "usage").with_alias("t40");
        assert_eq!(spec.name, "test_40");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_041() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 41");
        assert!(sink.captured().unwrap().contains("hello 41"));
        let spec = CommandSpec::new("test_41", "desc", "usage").with_alias("t41");
        assert_eq!(spec.name, "test_41");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_042() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 42");
        assert!(sink.captured().unwrap().contains("hello 42"));
        let spec = CommandSpec::new("test_42", "desc", "usage").with_alias("t42");
        assert_eq!(spec.name, "test_42");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_043() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 43");
        assert!(sink.captured().unwrap().contains("hello 43"));
        let spec = CommandSpec::new("test_43", "desc", "usage").with_alias("t43");
        assert_eq!(spec.name, "test_43");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_044() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 44");
        assert!(sink.captured().unwrap().contains("hello 44"));
        let spec = CommandSpec::new("test_44", "desc", "usage").with_alias("t44");
        assert_eq!(spec.name, "test_44");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_045() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 45");
        assert!(sink.captured().unwrap().contains("hello 45"));
        let spec = CommandSpec::new("test_45", "desc", "usage").with_alias("t45");
        assert_eq!(spec.name, "test_45");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_046() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 46");
        assert!(sink.captured().unwrap().contains("hello 46"));
        let spec = CommandSpec::new("test_46", "desc", "usage").with_alias("t46");
        assert_eq!(spec.name, "test_46");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_047() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 47");
        assert!(sink.captured().unwrap().contains("hello 47"));
        let spec = CommandSpec::new("test_47", "desc", "usage").with_alias("t47");
        assert_eq!(spec.name, "test_47");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_048() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 48");
        assert!(sink.captured().unwrap().contains("hello 48"));
        let spec = CommandSpec::new("test_48", "desc", "usage").with_alias("t48");
        assert_eq!(spec.name, "test_48");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_049() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 49");
        assert!(sink.captured().unwrap().contains("hello 49"));
        let spec = CommandSpec::new("test_49", "desc", "usage").with_alias("t49");
        assert_eq!(spec.name, "test_49");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_050() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 50");
        assert!(sink.captured().unwrap().contains("hello 50"));
        let spec = CommandSpec::new("test_50", "desc", "usage").with_alias("t50");
        assert_eq!(spec.name, "test_50");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_051() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 51");
        assert!(sink.captured().unwrap().contains("hello 51"));
        let spec = CommandSpec::new("test_51", "desc", "usage").with_alias("t51");
        assert_eq!(spec.name, "test_51");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_052() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 52");
        assert!(sink.captured().unwrap().contains("hello 52"));
        let spec = CommandSpec::new("test_52", "desc", "usage").with_alias("t52");
        assert_eq!(spec.name, "test_52");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_053() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 53");
        assert!(sink.captured().unwrap().contains("hello 53"));
        let spec = CommandSpec::new("test_53", "desc", "usage").with_alias("t53");
        assert_eq!(spec.name, "test_53");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_054() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 54");
        assert!(sink.captured().unwrap().contains("hello 54"));
        let spec = CommandSpec::new("test_54", "desc", "usage").with_alias("t54");
        assert_eq!(spec.name, "test_54");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_055() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 55");
        assert!(sink.captured().unwrap().contains("hello 55"));
        let spec = CommandSpec::new("test_55", "desc", "usage").with_alias("t55");
        assert_eq!(spec.name, "test_55");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_056() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 56");
        assert!(sink.captured().unwrap().contains("hello 56"));
        let spec = CommandSpec::new("test_56", "desc", "usage").with_alias("t56");
        assert_eq!(spec.name, "test_56");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_057() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 57");
        assert!(sink.captured().unwrap().contains("hello 57"));
        let spec = CommandSpec::new("test_57", "desc", "usage").with_alias("t57");
        assert_eq!(spec.name, "test_57");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_058() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 58");
        assert!(sink.captured().unwrap().contains("hello 58"));
        let spec = CommandSpec::new("test_58", "desc", "usage").with_alias("t58");
        assert_eq!(spec.name, "test_58");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_059() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 59");
        assert!(sink.captured().unwrap().contains("hello 59"));
        let spec = CommandSpec::new("test_59", "desc", "usage").with_alias("t59");
        assert_eq!(spec.name, "test_59");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_060() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 60");
        assert!(sink.captured().unwrap().contains("hello 60"));
        let spec = CommandSpec::new("test_60", "desc", "usage").with_alias("t60");
        assert_eq!(spec.name, "test_60");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_061() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 61");
        assert!(sink.captured().unwrap().contains("hello 61"));
        let spec = CommandSpec::new("test_61", "desc", "usage").with_alias("t61");
        assert_eq!(spec.name, "test_61");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_062() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 62");
        assert!(sink.captured().unwrap().contains("hello 62"));
        let spec = CommandSpec::new("test_62", "desc", "usage").with_alias("t62");
        assert_eq!(spec.name, "test_62");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_063() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 63");
        assert!(sink.captured().unwrap().contains("hello 63"));
        let spec = CommandSpec::new("test_63", "desc", "usage").with_alias("t63");
        assert_eq!(spec.name, "test_63");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_064() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 64");
        assert!(sink.captured().unwrap().contains("hello 64"));
        let spec = CommandSpec::new("test_64", "desc", "usage").with_alias("t64");
        assert_eq!(spec.name, "test_64");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_065() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 65");
        assert!(sink.captured().unwrap().contains("hello 65"));
        let spec = CommandSpec::new("test_65", "desc", "usage").with_alias("t65");
        assert_eq!(spec.name, "test_65");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_066() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 66");
        assert!(sink.captured().unwrap().contains("hello 66"));
        let spec = CommandSpec::new("test_66", "desc", "usage").with_alias("t66");
        assert_eq!(spec.name, "test_66");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_067() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 67");
        assert!(sink.captured().unwrap().contains("hello 67"));
        let spec = CommandSpec::new("test_67", "desc", "usage").with_alias("t67");
        assert_eq!(spec.name, "test_67");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_068() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 68");
        assert!(sink.captured().unwrap().contains("hello 68"));
        let spec = CommandSpec::new("test_68", "desc", "usage").with_alias("t68");
        assert_eq!(spec.name, "test_68");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_069() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 69");
        assert!(sink.captured().unwrap().contains("hello 69"));
        let spec = CommandSpec::new("test_69", "desc", "usage").with_alias("t69");
        assert_eq!(spec.name, "test_69");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_070() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 70");
        assert!(sink.captured().unwrap().contains("hello 70"));
        let spec = CommandSpec::new("test_70", "desc", "usage").with_alias("t70");
        assert_eq!(spec.name, "test_70");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_071() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 71");
        assert!(sink.captured().unwrap().contains("hello 71"));
        let spec = CommandSpec::new("test_71", "desc", "usage").with_alias("t71");
        assert_eq!(spec.name, "test_71");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_072() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 72");
        assert!(sink.captured().unwrap().contains("hello 72"));
        let spec = CommandSpec::new("test_72", "desc", "usage").with_alias("t72");
        assert_eq!(spec.name, "test_72");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_073() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 73");
        assert!(sink.captured().unwrap().contains("hello 73"));
        let spec = CommandSpec::new("test_73", "desc", "usage").with_alias("t73");
        assert_eq!(spec.name, "test_73");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_074() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 74");
        assert!(sink.captured().unwrap().contains("hello 74"));
        let spec = CommandSpec::new("test_74", "desc", "usage").with_alias("t74");
        assert_eq!(spec.name, "test_74");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_075() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 75");
        assert!(sink.captured().unwrap().contains("hello 75"));
        let spec = CommandSpec::new("test_75", "desc", "usage").with_alias("t75");
        assert_eq!(spec.name, "test_75");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_076() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 76");
        assert!(sink.captured().unwrap().contains("hello 76"));
        let spec = CommandSpec::new("test_76", "desc", "usage").with_alias("t76");
        assert_eq!(spec.name, "test_76");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_077() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 77");
        assert!(sink.captured().unwrap().contains("hello 77"));
        let spec = CommandSpec::new("test_77", "desc", "usage").with_alias("t77");
        assert_eq!(spec.name, "test_77");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_078() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 78");
        assert!(sink.captured().unwrap().contains("hello 78"));
        let spec = CommandSpec::new("test_78", "desc", "usage").with_alias("t78");
        assert_eq!(spec.name, "test_78");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_079() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 79");
        assert!(sink.captured().unwrap().contains("hello 79"));
        let spec = CommandSpec::new("test_79", "desc", "usage").with_alias("t79");
        assert_eq!(spec.name, "test_79");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_080() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 80");
        assert!(sink.captured().unwrap().contains("hello 80"));
        let spec = CommandSpec::new("test_80", "desc", "usage").with_alias("t80");
        assert_eq!(spec.name, "test_80");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_081() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 81");
        assert!(sink.captured().unwrap().contains("hello 81"));
        let spec = CommandSpec::new("test_81", "desc", "usage").with_alias("t81");
        assert_eq!(spec.name, "test_81");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_082() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 82");
        assert!(sink.captured().unwrap().contains("hello 82"));
        let spec = CommandSpec::new("test_82", "desc", "usage").with_alias("t82");
        assert_eq!(spec.name, "test_82");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_083() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 83");
        assert!(sink.captured().unwrap().contains("hello 83"));
        let spec = CommandSpec::new("test_83", "desc", "usage").with_alias("t83");
        assert_eq!(spec.name, "test_83");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_084() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 84");
        assert!(sink.captured().unwrap().contains("hello 84"));
        let spec = CommandSpec::new("test_84", "desc", "usage").with_alias("t84");
        assert_eq!(spec.name, "test_84");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_085() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 85");
        assert!(sink.captured().unwrap().contains("hello 85"));
        let spec = CommandSpec::new("test_85", "desc", "usage").with_alias("t85");
        assert_eq!(spec.name, "test_85");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_086() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 86");
        assert!(sink.captured().unwrap().contains("hello 86"));
        let spec = CommandSpec::new("test_86", "desc", "usage").with_alias("t86");
        assert_eq!(spec.name, "test_86");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_087() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 87");
        assert!(sink.captured().unwrap().contains("hello 87"));
        let spec = CommandSpec::new("test_87", "desc", "usage").with_alias("t87");
        assert_eq!(spec.name, "test_87");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_088() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 88");
        assert!(sink.captured().unwrap().contains("hello 88"));
        let spec = CommandSpec::new("test_88", "desc", "usage").with_alias("t88");
        assert_eq!(spec.name, "test_88");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_089() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 89");
        assert!(sink.captured().unwrap().contains("hello 89"));
        let spec = CommandSpec::new("test_89", "desc", "usage").with_alias("t89");
        assert_eq!(spec.name, "test_89");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_090() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 90");
        assert!(sink.captured().unwrap().contains("hello 90"));
        let spec = CommandSpec::new("test_90", "desc", "usage").with_alias("t90");
        assert_eq!(spec.name, "test_90");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_091() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 91");
        assert!(sink.captured().unwrap().contains("hello 91"));
        let spec = CommandSpec::new("test_91", "desc", "usage").with_alias("t91");
        assert_eq!(spec.name, "test_91");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_092() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 92");
        assert!(sink.captured().unwrap().contains("hello 92"));
        let spec = CommandSpec::new("test_92", "desc", "usage").with_alias("t92");
        assert_eq!(spec.name, "test_92");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_093() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 93");
        assert!(sink.captured().unwrap().contains("hello 93"));
        let spec = CommandSpec::new("test_93", "desc", "usage").with_alias("t93");
        assert_eq!(spec.name, "test_93");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_094() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 94");
        assert!(sink.captured().unwrap().contains("hello 94"));
        let spec = CommandSpec::new("test_94", "desc", "usage").with_alias("t94");
        assert_eq!(spec.name, "test_94");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_095() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 95");
        assert!(sink.captured().unwrap().contains("hello 95"));
        let spec = CommandSpec::new("test_95", "desc", "usage").with_alias("t95");
        assert_eq!(spec.name, "test_95");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_096() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 96");
        assert!(sink.captured().unwrap().contains("hello 96"));
        let spec = CommandSpec::new("test_96", "desc", "usage").with_alias("t96");
        assert_eq!(spec.name, "test_96");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_097() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 97");
        assert!(sink.captured().unwrap().contains("hello 97"));
        let spec = CommandSpec::new("test_97", "desc", "usage").with_alias("t97");
        assert_eq!(spec.name, "test_97");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_098() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 98");
        assert!(sink.captured().unwrap().contains("hello 98"));
        let spec = CommandSpec::new("test_98", "desc", "usage").with_alias("t98");
        assert_eq!(spec.name, "test_98");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_099() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 99");
        assert!(sink.captured().unwrap().contains("hello 99"));
        let spec = CommandSpec::new("test_99", "desc", "usage").with_alias("t99");
        assert_eq!(spec.name, "test_99");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_100() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 100");
        assert!(sink.captured().unwrap().contains("hello 100"));
        let spec = CommandSpec::new("test_100", "desc", "usage").with_alias("t100");
        assert_eq!(spec.name, "test_100");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_101() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 101");
        assert!(sink.captured().unwrap().contains("hello 101"));
        let spec = CommandSpec::new("test_101", "desc", "usage").with_alias("t101");
        assert_eq!(spec.name, "test_101");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_102() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 102");
        assert!(sink.captured().unwrap().contains("hello 102"));
        let spec = CommandSpec::new("test_102", "desc", "usage").with_alias("t102");
        assert_eq!(spec.name, "test_102");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_103() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 103");
        assert!(sink.captured().unwrap().contains("hello 103"));
        let spec = CommandSpec::new("test_103", "desc", "usage").with_alias("t103");
        assert_eq!(spec.name, "test_103");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_104() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 104");
        assert!(sink.captured().unwrap().contains("hello 104"));
        let spec = CommandSpec::new("test_104", "desc", "usage").with_alias("t104");
        assert_eq!(spec.name, "test_104");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_105() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 105");
        assert!(sink.captured().unwrap().contains("hello 105"));
        let spec = CommandSpec::new("test_105", "desc", "usage").with_alias("t105");
        assert_eq!(spec.name, "test_105");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_106() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 106");
        assert!(sink.captured().unwrap().contains("hello 106"));
        let spec = CommandSpec::new("test_106", "desc", "usage").with_alias("t106");
        assert_eq!(spec.name, "test_106");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_107() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 107");
        assert!(sink.captured().unwrap().contains("hello 107"));
        let spec = CommandSpec::new("test_107", "desc", "usage").with_alias("t107");
        assert_eq!(spec.name, "test_107");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_108() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 108");
        assert!(sink.captured().unwrap().contains("hello 108"));
        let spec = CommandSpec::new("test_108", "desc", "usage").with_alias("t108");
        assert_eq!(spec.name, "test_108");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_109() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 109");
        assert!(sink.captured().unwrap().contains("hello 109"));
        let spec = CommandSpec::new("test_109", "desc", "usage").with_alias("t109");
        assert_eq!(spec.name, "test_109");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_110() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 110");
        assert!(sink.captured().unwrap().contains("hello 110"));
        let spec = CommandSpec::new("test_110", "desc", "usage").with_alias("t110");
        assert_eq!(spec.name, "test_110");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_111() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 111");
        assert!(sink.captured().unwrap().contains("hello 111"));
        let spec = CommandSpec::new("test_111", "desc", "usage").with_alias("t111");
        assert_eq!(spec.name, "test_111");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_112() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 112");
        assert!(sink.captured().unwrap().contains("hello 112"));
        let spec = CommandSpec::new("test_112", "desc", "usage").with_alias("t112");
        assert_eq!(spec.name, "test_112");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_113() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 113");
        assert!(sink.captured().unwrap().contains("hello 113"));
        let spec = CommandSpec::new("test_113", "desc", "usage").with_alias("t113");
        assert_eq!(spec.name, "test_113");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_114() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 114");
        assert!(sink.captured().unwrap().contains("hello 114"));
        let spec = CommandSpec::new("test_114", "desc", "usage").with_alias("t114");
        assert_eq!(spec.name, "test_114");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_115() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 115");
        assert!(sink.captured().unwrap().contains("hello 115"));
        let spec = CommandSpec::new("test_115", "desc", "usage").with_alias("t115");
        assert_eq!(spec.name, "test_115");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_116() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 116");
        assert!(sink.captured().unwrap().contains("hello 116"));
        let spec = CommandSpec::new("test_116", "desc", "usage").with_alias("t116");
        assert_eq!(spec.name, "test_116");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_117() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 117");
        assert!(sink.captured().unwrap().contains("hello 117"));
        let spec = CommandSpec::new("test_117", "desc", "usage").with_alias("t117");
        assert_eq!(spec.name, "test_117");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_118() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 118");
        assert!(sink.captured().unwrap().contains("hello 118"));
        let spec = CommandSpec::new("test_118", "desc", "usage").with_alias("t118");
        assert_eq!(spec.name, "test_118");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_119() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 119");
        assert!(sink.captured().unwrap().contains("hello 119"));
        let spec = CommandSpec::new("test_119", "desc", "usage").with_alias("t119");
        assert_eq!(spec.name, "test_119");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_120() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 120");
        assert!(sink.captured().unwrap().contains("hello 120"));
        let spec = CommandSpec::new("test_120", "desc", "usage").with_alias("t120");
        assert_eq!(spec.name, "test_120");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_121() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 121");
        assert!(sink.captured().unwrap().contains("hello 121"));
        let spec = CommandSpec::new("test_121", "desc", "usage").with_alias("t121");
        assert_eq!(spec.name, "test_121");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_122() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 122");
        assert!(sink.captured().unwrap().contains("hello 122"));
        let spec = CommandSpec::new("test_122", "desc", "usage").with_alias("t122");
        assert_eq!(spec.name, "test_122");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_123() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 123");
        assert!(sink.captured().unwrap().contains("hello 123"));
        let spec = CommandSpec::new("test_123", "desc", "usage").with_alias("t123");
        assert_eq!(spec.name, "test_123");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_124() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 124");
        assert!(sink.captured().unwrap().contains("hello 124"));
        let spec = CommandSpec::new("test_124", "desc", "usage").with_alias("t124");
        assert_eq!(spec.name, "test_124");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_125() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 125");
        assert!(sink.captured().unwrap().contains("hello 125"));
        let spec = CommandSpec::new("test_125", "desc", "usage").with_alias("t125");
        assert_eq!(spec.name, "test_125");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_126() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 126");
        assert!(sink.captured().unwrap().contains("hello 126"));
        let spec = CommandSpec::new("test_126", "desc", "usage").with_alias("t126");
        assert_eq!(spec.name, "test_126");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_127() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 127");
        assert!(sink.captured().unwrap().contains("hello 127"));
        let spec = CommandSpec::new("test_127", "desc", "usage").with_alias("t127");
        assert_eq!(spec.name, "test_127");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_128() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 128");
        assert!(sink.captured().unwrap().contains("hello 128"));
        let spec = CommandSpec::new("test_128", "desc", "usage").with_alias("t128");
        assert_eq!(spec.name, "test_128");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_129() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 129");
        assert!(sink.captured().unwrap().contains("hello 129"));
        let spec = CommandSpec::new("test_129", "desc", "usage").with_alias("t129");
        assert_eq!(spec.name, "test_129");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_130() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 130");
        assert!(sink.captured().unwrap().contains("hello 130"));
        let spec = CommandSpec::new("test_130", "desc", "usage").with_alias("t130");
        assert_eq!(spec.name, "test_130");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_131() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 131");
        assert!(sink.captured().unwrap().contains("hello 131"));
        let spec = CommandSpec::new("test_131", "desc", "usage").with_alias("t131");
        assert_eq!(spec.name, "test_131");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_132() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 132");
        assert!(sink.captured().unwrap().contains("hello 132"));
        let spec = CommandSpec::new("test_132", "desc", "usage").with_alias("t132");
        assert_eq!(spec.name, "test_132");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_133() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 133");
        assert!(sink.captured().unwrap().contains("hello 133"));
        let spec = CommandSpec::new("test_133", "desc", "usage").with_alias("t133");
        assert_eq!(spec.name, "test_133");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_134() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 134");
        assert!(sink.captured().unwrap().contains("hello 134"));
        let spec = CommandSpec::new("test_134", "desc", "usage").with_alias("t134");
        assert_eq!(spec.name, "test_134");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_135() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 135");
        assert!(sink.captured().unwrap().contains("hello 135"));
        let spec = CommandSpec::new("test_135", "desc", "usage").with_alias("t135");
        assert_eq!(spec.name, "test_135");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_136() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 136");
        assert!(sink.captured().unwrap().contains("hello 136"));
        let spec = CommandSpec::new("test_136", "desc", "usage").with_alias("t136");
        assert_eq!(spec.name, "test_136");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_137() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 137");
        assert!(sink.captured().unwrap().contains("hello 137"));
        let spec = CommandSpec::new("test_137", "desc", "usage").with_alias("t137");
        assert_eq!(spec.name, "test_137");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_138() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 138");
        assert!(sink.captured().unwrap().contains("hello 138"));
        let spec = CommandSpec::new("test_138", "desc", "usage").with_alias("t138");
        assert_eq!(spec.name, "test_138");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_139() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 139");
        assert!(sink.captured().unwrap().contains("hello 139"));
        let spec = CommandSpec::new("test_139", "desc", "usage").with_alias("t139");
        assert_eq!(spec.name, "test_139");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_140() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 140");
        assert!(sink.captured().unwrap().contains("hello 140"));
        let spec = CommandSpec::new("test_140", "desc", "usage").with_alias("t140");
        assert_eq!(spec.name, "test_140");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_141() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 141");
        assert!(sink.captured().unwrap().contains("hello 141"));
        let spec = CommandSpec::new("test_141", "desc", "usage").with_alias("t141");
        assert_eq!(spec.name, "test_141");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_142() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 142");
        assert!(sink.captured().unwrap().contains("hello 142"));
        let spec = CommandSpec::new("test_142", "desc", "usage").with_alias("t142");
        assert_eq!(spec.name, "test_142");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_143() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 143");
        assert!(sink.captured().unwrap().contains("hello 143"));
        let spec = CommandSpec::new("test_143", "desc", "usage").with_alias("t143");
        assert_eq!(spec.name, "test_143");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_144() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 144");
        assert!(sink.captured().unwrap().contains("hello 144"));
        let spec = CommandSpec::new("test_144", "desc", "usage").with_alias("t144");
        assert_eq!(spec.name, "test_144");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_145() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 145");
        assert!(sink.captured().unwrap().contains("hello 145"));
        let spec = CommandSpec::new("test_145", "desc", "usage").with_alias("t145");
        assert_eq!(spec.name, "test_145");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_146() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 146");
        assert!(sink.captured().unwrap().contains("hello 146"));
        let spec = CommandSpec::new("test_146", "desc", "usage").with_alias("t146");
        assert_eq!(spec.name, "test_146");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_147() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 147");
        assert!(sink.captured().unwrap().contains("hello 147"));
        let spec = CommandSpec::new("test_147", "desc", "usage").with_alias("t147");
        assert_eq!(spec.name, "test_147");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_148() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 148");
        assert!(sink.captured().unwrap().contains("hello 148"));
        let spec = CommandSpec::new("test_148", "desc", "usage").with_alias("t148");
        assert_eq!(spec.name, "test_148");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_149() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 149");
        assert!(sink.captured().unwrap().contains("hello 149"));
        let spec = CommandSpec::new("test_149", "desc", "usage").with_alias("t149");
        assert_eq!(spec.name, "test_149");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_150() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 150");
        assert!(sink.captured().unwrap().contains("hello 150"));
        let spec = CommandSpec::new("test_150", "desc", "usage").with_alias("t150");
        assert_eq!(spec.name, "test_150");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_151() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 151");
        assert!(sink.captured().unwrap().contains("hello 151"));
        let spec = CommandSpec::new("test_151", "desc", "usage").with_alias("t151");
        assert_eq!(spec.name, "test_151");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_152() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 152");
        assert!(sink.captured().unwrap().contains("hello 152"));
        let spec = CommandSpec::new("test_152", "desc", "usage").with_alias("t152");
        assert_eq!(spec.name, "test_152");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_153() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 153");
        assert!(sink.captured().unwrap().contains("hello 153"));
        let spec = CommandSpec::new("test_153", "desc", "usage").with_alias("t153");
        assert_eq!(spec.name, "test_153");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_154() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 154");
        assert!(sink.captured().unwrap().contains("hello 154"));
        let spec = CommandSpec::new("test_154", "desc", "usage").with_alias("t154");
        assert_eq!(spec.name, "test_154");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_155() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 155");
        assert!(sink.captured().unwrap().contains("hello 155"));
        let spec = CommandSpec::new("test_155", "desc", "usage").with_alias("t155");
        assert_eq!(spec.name, "test_155");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_156() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 156");
        assert!(sink.captured().unwrap().contains("hello 156"));
        let spec = CommandSpec::new("test_156", "desc", "usage").with_alias("t156");
        assert_eq!(spec.name, "test_156");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_157() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 157");
        assert!(sink.captured().unwrap().contains("hello 157"));
        let spec = CommandSpec::new("test_157", "desc", "usage").with_alias("t157");
        assert_eq!(spec.name, "test_157");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_158() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 158");
        assert!(sink.captured().unwrap().contains("hello 158"));
        let spec = CommandSpec::new("test_158", "desc", "usage").with_alias("t158");
        assert_eq!(spec.name, "test_158");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_159() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 159");
        assert!(sink.captured().unwrap().contains("hello 159"));
        let spec = CommandSpec::new("test_159", "desc", "usage").with_alias("t159");
        assert_eq!(spec.name, "test_159");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_160() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 160");
        assert!(sink.captured().unwrap().contains("hello 160"));
        let spec = CommandSpec::new("test_160", "desc", "usage").with_alias("t160");
        assert_eq!(spec.name, "test_160");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_161() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 161");
        assert!(sink.captured().unwrap().contains("hello 161"));
        let spec = CommandSpec::new("test_161", "desc", "usage").with_alias("t161");
        assert_eq!(spec.name, "test_161");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_162() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 162");
        assert!(sink.captured().unwrap().contains("hello 162"));
        let spec = CommandSpec::new("test_162", "desc", "usage").with_alias("t162");
        assert_eq!(spec.name, "test_162");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_163() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 163");
        assert!(sink.captured().unwrap().contains("hello 163"));
        let spec = CommandSpec::new("test_163", "desc", "usage").with_alias("t163");
        assert_eq!(spec.name, "test_163");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_164() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 164");
        assert!(sink.captured().unwrap().contains("hello 164"));
        let spec = CommandSpec::new("test_164", "desc", "usage").with_alias("t164");
        assert_eq!(spec.name, "test_164");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_165() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 165");
        assert!(sink.captured().unwrap().contains("hello 165"));
        let spec = CommandSpec::new("test_165", "desc", "usage").with_alias("t165");
        assert_eq!(spec.name, "test_165");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_166() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 166");
        assert!(sink.captured().unwrap().contains("hello 166"));
        let spec = CommandSpec::new("test_166", "desc", "usage").with_alias("t166");
        assert_eq!(spec.name, "test_166");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_167() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 167");
        assert!(sink.captured().unwrap().contains("hello 167"));
        let spec = CommandSpec::new("test_167", "desc", "usage").with_alias("t167");
        assert_eq!(spec.name, "test_167");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_168() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 168");
        assert!(sink.captured().unwrap().contains("hello 168"));
        let spec = CommandSpec::new("test_168", "desc", "usage").with_alias("t168");
        assert_eq!(spec.name, "test_168");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_169() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 169");
        assert!(sink.captured().unwrap().contains("hello 169"));
        let spec = CommandSpec::new("test_169", "desc", "usage").with_alias("t169");
        assert_eq!(spec.name, "test_169");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_170() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 170");
        assert!(sink.captured().unwrap().contains("hello 170"));
        let spec = CommandSpec::new("test_170", "desc", "usage").with_alias("t170");
        assert_eq!(spec.name, "test_170");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_171() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 171");
        assert!(sink.captured().unwrap().contains("hello 171"));
        let spec = CommandSpec::new("test_171", "desc", "usage").with_alias("t171");
        assert_eq!(spec.name, "test_171");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_172() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 172");
        assert!(sink.captured().unwrap().contains("hello 172"));
        let spec = CommandSpec::new("test_172", "desc", "usage").with_alias("t172");
        assert_eq!(spec.name, "test_172");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_173() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 173");
        assert!(sink.captured().unwrap().contains("hello 173"));
        let spec = CommandSpec::new("test_173", "desc", "usage").with_alias("t173");
        assert_eq!(spec.name, "test_173");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_174() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 174");
        assert!(sink.captured().unwrap().contains("hello 174"));
        let spec = CommandSpec::new("test_174", "desc", "usage").with_alias("t174");
        assert_eq!(spec.name, "test_174");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_175() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 175");
        assert!(sink.captured().unwrap().contains("hello 175"));
        let spec = CommandSpec::new("test_175", "desc", "usage").with_alias("t175");
        assert_eq!(spec.name, "test_175");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_176() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 176");
        assert!(sink.captured().unwrap().contains("hello 176"));
        let spec = CommandSpec::new("test_176", "desc", "usage").with_alias("t176");
        assert_eq!(spec.name, "test_176");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_177() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 177");
        assert!(sink.captured().unwrap().contains("hello 177"));
        let spec = CommandSpec::new("test_177", "desc", "usage").with_alias("t177");
        assert_eq!(spec.name, "test_177");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_178() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 178");
        assert!(sink.captured().unwrap().contains("hello 178"));
        let spec = CommandSpec::new("test_178", "desc", "usage").with_alias("t178");
        assert_eq!(spec.name, "test_178");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_179() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 179");
        assert!(sink.captured().unwrap().contains("hello 179"));
        let spec = CommandSpec::new("test_179", "desc", "usage").with_alias("t179");
        assert_eq!(spec.name, "test_179");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_180() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 180");
        assert!(sink.captured().unwrap().contains("hello 180"));
        let spec = CommandSpec::new("test_180", "desc", "usage").with_alias("t180");
        assert_eq!(spec.name, "test_180");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_181() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 181");
        assert!(sink.captured().unwrap().contains("hello 181"));
        let spec = CommandSpec::new("test_181", "desc", "usage").with_alias("t181");
        assert_eq!(spec.name, "test_181");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_182() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 182");
        assert!(sink.captured().unwrap().contains("hello 182"));
        let spec = CommandSpec::new("test_182", "desc", "usage").with_alias("t182");
        assert_eq!(spec.name, "test_182");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_183() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 183");
        assert!(sink.captured().unwrap().contains("hello 183"));
        let spec = CommandSpec::new("test_183", "desc", "usage").with_alias("t183");
        assert_eq!(spec.name, "test_183");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_184() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 184");
        assert!(sink.captured().unwrap().contains("hello 184"));
        let spec = CommandSpec::new("test_184", "desc", "usage").with_alias("t184");
        assert_eq!(spec.name, "test_184");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_185() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 185");
        assert!(sink.captured().unwrap().contains("hello 185"));
        let spec = CommandSpec::new("test_185", "desc", "usage").with_alias("t185");
        assert_eq!(spec.name, "test_185");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_186() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 186");
        assert!(sink.captured().unwrap().contains("hello 186"));
        let spec = CommandSpec::new("test_186", "desc", "usage").with_alias("t186");
        assert_eq!(spec.name, "test_186");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_187() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 187");
        assert!(sink.captured().unwrap().contains("hello 187"));
        let spec = CommandSpec::new("test_187", "desc", "usage").with_alias("t187");
        assert_eq!(spec.name, "test_187");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_188() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 188");
        assert!(sink.captured().unwrap().contains("hello 188"));
        let spec = CommandSpec::new("test_188", "desc", "usage").with_alias("t188");
        assert_eq!(spec.name, "test_188");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_189() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 189");
        assert!(sink.captured().unwrap().contains("hello 189"));
        let spec = CommandSpec::new("test_189", "desc", "usage").with_alias("t189");
        assert_eq!(spec.name, "test_189");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_190() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 190");
        assert!(sink.captured().unwrap().contains("hello 190"));
        let spec = CommandSpec::new("test_190", "desc", "usage").with_alias("t190");
        assert_eq!(spec.name, "test_190");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_191() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 191");
        assert!(sink.captured().unwrap().contains("hello 191"));
        let spec = CommandSpec::new("test_191", "desc", "usage").with_alias("t191");
        assert_eq!(spec.name, "test_191");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_192() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 192");
        assert!(sink.captured().unwrap().contains("hello 192"));
        let spec = CommandSpec::new("test_192", "desc", "usage").with_alias("t192");
        assert_eq!(spec.name, "test_192");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_193() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 193");
        assert!(sink.captured().unwrap().contains("hello 193"));
        let spec = CommandSpec::new("test_193", "desc", "usage").with_alias("t193");
        assert_eq!(spec.name, "test_193");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_194() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 194");
        assert!(sink.captured().unwrap().contains("hello 194"));
        let spec = CommandSpec::new("test_194", "desc", "usage").with_alias("t194");
        assert_eq!(spec.name, "test_194");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_195() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 195");
        assert!(sink.captured().unwrap().contains("hello 195"));
        let spec = CommandSpec::new("test_195", "desc", "usage").with_alias("t195");
        assert_eq!(spec.name, "test_195");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_196() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 196");
        assert!(sink.captured().unwrap().contains("hello 196"));
        let spec = CommandSpec::new("test_196", "desc", "usage").with_alias("t196");
        assert_eq!(spec.name, "test_196");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_197() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 197");
        assert!(sink.captured().unwrap().contains("hello 197"));
        let spec = CommandSpec::new("test_197", "desc", "usage").with_alias("t197");
        assert_eq!(spec.name, "test_197");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_198() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 198");
        assert!(sink.captured().unwrap().contains("hello 198"));
        let spec = CommandSpec::new("test_198", "desc", "usage").with_alias("t198");
        assert_eq!(spec.name, "test_198");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_199() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 199");
        assert!(sink.captured().unwrap().contains("hello 199"));
        let spec = CommandSpec::new("test_199", "desc", "usage").with_alias("t199");
        assert_eq!(spec.name, "test_199");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_200() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 200");
        assert!(sink.captured().unwrap().contains("hello 200"));
        let spec = CommandSpec::new("test_200", "desc", "usage").with_alias("t200");
        assert_eq!(spec.name, "test_200");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_201() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 201");
        assert!(sink.captured().unwrap().contains("hello 201"));
        let spec = CommandSpec::new("test_201", "desc", "usage").with_alias("t201");
        assert_eq!(spec.name, "test_201");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_202() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 202");
        assert!(sink.captured().unwrap().contains("hello 202"));
        let spec = CommandSpec::new("test_202", "desc", "usage").with_alias("t202");
        assert_eq!(spec.name, "test_202");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_203() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 203");
        assert!(sink.captured().unwrap().contains("hello 203"));
        let spec = CommandSpec::new("test_203", "desc", "usage").with_alias("t203");
        assert_eq!(spec.name, "test_203");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_204() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 204");
        assert!(sink.captured().unwrap().contains("hello 204"));
        let spec = CommandSpec::new("test_204", "desc", "usage").with_alias("t204");
        assert_eq!(spec.name, "test_204");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_205() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 205");
        assert!(sink.captured().unwrap().contains("hello 205"));
        let spec = CommandSpec::new("test_205", "desc", "usage").with_alias("t205");
        assert_eq!(spec.name, "test_205");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_206() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 206");
        assert!(sink.captured().unwrap().contains("hello 206"));
        let spec = CommandSpec::new("test_206", "desc", "usage").with_alias("t206");
        assert_eq!(spec.name, "test_206");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_207() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 207");
        assert!(sink.captured().unwrap().contains("hello 207"));
        let spec = CommandSpec::new("test_207", "desc", "usage").with_alias("t207");
        assert_eq!(spec.name, "test_207");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_208() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 208");
        assert!(sink.captured().unwrap().contains("hello 208"));
        let spec = CommandSpec::new("test_208", "desc", "usage").with_alias("t208");
        assert_eq!(spec.name, "test_208");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_209() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 209");
        assert!(sink.captured().unwrap().contains("hello 209"));
        let spec = CommandSpec::new("test_209", "desc", "usage").with_alias("t209");
        assert_eq!(spec.name, "test_209");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_210() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 210");
        assert!(sink.captured().unwrap().contains("hello 210"));
        let spec = CommandSpec::new("test_210", "desc", "usage").with_alias("t210");
        assert_eq!(spec.name, "test_210");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_211() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 211");
        assert!(sink.captured().unwrap().contains("hello 211"));
        let spec = CommandSpec::new("test_211", "desc", "usage").with_alias("t211");
        assert_eq!(spec.name, "test_211");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_212() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 212");
        assert!(sink.captured().unwrap().contains("hello 212"));
        let spec = CommandSpec::new("test_212", "desc", "usage").with_alias("t212");
        assert_eq!(spec.name, "test_212");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_213() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 213");
        assert!(sink.captured().unwrap().contains("hello 213"));
        let spec = CommandSpec::new("test_213", "desc", "usage").with_alias("t213");
        assert_eq!(spec.name, "test_213");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_214() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 214");
        assert!(sink.captured().unwrap().contains("hello 214"));
        let spec = CommandSpec::new("test_214", "desc", "usage").with_alias("t214");
        assert_eq!(spec.name, "test_214");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_215() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 215");
        assert!(sink.captured().unwrap().contains("hello 215"));
        let spec = CommandSpec::new("test_215", "desc", "usage").with_alias("t215");
        assert_eq!(spec.name, "test_215");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_216() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 216");
        assert!(sink.captured().unwrap().contains("hello 216"));
        let spec = CommandSpec::new("test_216", "desc", "usage").with_alias("t216");
        assert_eq!(spec.name, "test_216");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_217() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 217");
        assert!(sink.captured().unwrap().contains("hello 217"));
        let spec = CommandSpec::new("test_217", "desc", "usage").with_alias("t217");
        assert_eq!(spec.name, "test_217");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_218() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 218");
        assert!(sink.captured().unwrap().contains("hello 218"));
        let spec = CommandSpec::new("test_218", "desc", "usage").with_alias("t218");
        assert_eq!(spec.name, "test_218");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_219() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 219");
        assert!(sink.captured().unwrap().contains("hello 219"));
        let spec = CommandSpec::new("test_219", "desc", "usage").with_alias("t219");
        assert_eq!(spec.name, "test_219");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_220() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 220");
        assert!(sink.captured().unwrap().contains("hello 220"));
        let spec = CommandSpec::new("test_220", "desc", "usage").with_alias("t220");
        assert_eq!(spec.name, "test_220");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_221() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 221");
        assert!(sink.captured().unwrap().contains("hello 221"));
        let spec = CommandSpec::new("test_221", "desc", "usage").with_alias("t221");
        assert_eq!(spec.name, "test_221");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_222() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 222");
        assert!(sink.captured().unwrap().contains("hello 222"));
        let spec = CommandSpec::new("test_222", "desc", "usage").with_alias("t222");
        assert_eq!(spec.name, "test_222");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_223() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 223");
        assert!(sink.captured().unwrap().contains("hello 223"));
        let spec = CommandSpec::new("test_223", "desc", "usage").with_alias("t223");
        assert_eq!(spec.name, "test_223");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_224() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 224");
        assert!(sink.captured().unwrap().contains("hello 224"));
        let spec = CommandSpec::new("test_224", "desc", "usage").with_alias("t224");
        assert_eq!(spec.name, "test_224");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_225() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 225");
        assert!(sink.captured().unwrap().contains("hello 225"));
        let spec = CommandSpec::new("test_225", "desc", "usage").with_alias("t225");
        assert_eq!(spec.name, "test_225");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_226() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 226");
        assert!(sink.captured().unwrap().contains("hello 226"));
        let spec = CommandSpec::new("test_226", "desc", "usage").with_alias("t226");
        assert_eq!(spec.name, "test_226");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_227() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 227");
        assert!(sink.captured().unwrap().contains("hello 227"));
        let spec = CommandSpec::new("test_227", "desc", "usage").with_alias("t227");
        assert_eq!(spec.name, "test_227");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_228() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 228");
        assert!(sink.captured().unwrap().contains("hello 228"));
        let spec = CommandSpec::new("test_228", "desc", "usage").with_alias("t228");
        assert_eq!(spec.name, "test_228");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_229() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 229");
        assert!(sink.captured().unwrap().contains("hello 229"));
        let spec = CommandSpec::new("test_229", "desc", "usage").with_alias("t229");
        assert_eq!(spec.name, "test_229");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_230() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 230");
        assert!(sink.captured().unwrap().contains("hello 230"));
        let spec = CommandSpec::new("test_230", "desc", "usage").with_alias("t230");
        assert_eq!(spec.name, "test_230");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_231() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 231");
        assert!(sink.captured().unwrap().contains("hello 231"));
        let spec = CommandSpec::new("test_231", "desc", "usage").with_alias("t231");
        assert_eq!(spec.name, "test_231");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_232() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 232");
        assert!(sink.captured().unwrap().contains("hello 232"));
        let spec = CommandSpec::new("test_232", "desc", "usage").with_alias("t232");
        assert_eq!(spec.name, "test_232");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_233() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 233");
        assert!(sink.captured().unwrap().contains("hello 233"));
        let spec = CommandSpec::new("test_233", "desc", "usage").with_alias("t233");
        assert_eq!(spec.name, "test_233");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_234() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 234");
        assert!(sink.captured().unwrap().contains("hello 234"));
        let spec = CommandSpec::new("test_234", "desc", "usage").with_alias("t234");
        assert_eq!(spec.name, "test_234");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_235() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 235");
        assert!(sink.captured().unwrap().contains("hello 235"));
        let spec = CommandSpec::new("test_235", "desc", "usage").with_alias("t235");
        assert_eq!(spec.name, "test_235");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_236() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 236");
        assert!(sink.captured().unwrap().contains("hello 236"));
        let spec = CommandSpec::new("test_236", "desc", "usage").with_alias("t236");
        assert_eq!(spec.name, "test_236");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_237() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 237");
        assert!(sink.captured().unwrap().contains("hello 237"));
        let spec = CommandSpec::new("test_237", "desc", "usage").with_alias("t237");
        assert_eq!(spec.name, "test_237");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_238() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 238");
        assert!(sink.captured().unwrap().contains("hello 238"));
        let spec = CommandSpec::new("test_238", "desc", "usage").with_alias("t238");
        assert_eq!(spec.name, "test_238");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_239() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 239");
        assert!(sink.captured().unwrap().contains("hello 239"));
        let spec = CommandSpec::new("test_239", "desc", "usage").with_alias("t239");
        assert_eq!(spec.name, "test_239");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_240() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 240");
        assert!(sink.captured().unwrap().contains("hello 240"));
        let spec = CommandSpec::new("test_240", "desc", "usage").with_alias("t240");
        assert_eq!(spec.name, "test_240");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_241() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 241");
        assert!(sink.captured().unwrap().contains("hello 241"));
        let spec = CommandSpec::new("test_241", "desc", "usage").with_alias("t241");
        assert_eq!(spec.name, "test_241");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_242() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 242");
        assert!(sink.captured().unwrap().contains("hello 242"));
        let spec = CommandSpec::new("test_242", "desc", "usage").with_alias("t242");
        assert_eq!(spec.name, "test_242");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_243() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 243");
        assert!(sink.captured().unwrap().contains("hello 243"));
        let spec = CommandSpec::new("test_243", "desc", "usage").with_alias("t243");
        assert_eq!(spec.name, "test_243");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_244() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 244");
        assert!(sink.captured().unwrap().contains("hello 244"));
        let spec = CommandSpec::new("test_244", "desc", "usage").with_alias("t244");
        assert_eq!(spec.name, "test_244");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_245() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 245");
        assert!(sink.captured().unwrap().contains("hello 245"));
        let spec = CommandSpec::new("test_245", "desc", "usage").with_alias("t245");
        assert_eq!(spec.name, "test_245");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_246() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 246");
        assert!(sink.captured().unwrap().contains("hello 246"));
        let spec = CommandSpec::new("test_246", "desc", "usage").with_alias("t246");
        assert_eq!(spec.name, "test_246");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_247() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 247");
        assert!(sink.captured().unwrap().contains("hello 247"));
        let spec = CommandSpec::new("test_247", "desc", "usage").with_alias("t247");
        assert_eq!(spec.name, "test_247");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_248() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 248");
        assert!(sink.captured().unwrap().contains("hello 248"));
        let spec = CommandSpec::new("test_248", "desc", "usage").with_alias("t248");
        assert_eq!(spec.name, "test_248");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_249() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 249");
        assert!(sink.captured().unwrap().contains("hello 249"));
        let spec = CommandSpec::new("test_249", "desc", "usage").with_alias("t249");
        assert_eq!(spec.name, "test_249");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_250() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 250");
        assert!(sink.captured().unwrap().contains("hello 250"));
        let spec = CommandSpec::new("test_250", "desc", "usage").with_alias("t250");
        assert_eq!(spec.name, "test_250");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_251() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 251");
        assert!(sink.captured().unwrap().contains("hello 251"));
        let spec = CommandSpec::new("test_251", "desc", "usage").with_alias("t251");
        assert_eq!(spec.name, "test_251");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_252() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 252");
        assert!(sink.captured().unwrap().contains("hello 252"));
        let spec = CommandSpec::new("test_252", "desc", "usage").with_alias("t252");
        assert_eq!(spec.name, "test_252");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_253() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 253");
        assert!(sink.captured().unwrap().contains("hello 253"));
        let spec = CommandSpec::new("test_253", "desc", "usage").with_alias("t253");
        assert_eq!(spec.name, "test_253");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_254() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 254");
        assert!(sink.captured().unwrap().contains("hello 254"));
        let spec = CommandSpec::new("test_254", "desc", "usage").with_alias("t254");
        assert_eq!(spec.name, "test_254");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_255() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 255");
        assert!(sink.captured().unwrap().contains("hello 255"));
        let spec = CommandSpec::new("test_255", "desc", "usage").with_alias("t255");
        assert_eq!(spec.name, "test_255");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_256() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 256");
        assert!(sink.captured().unwrap().contains("hello 256"));
        let spec = CommandSpec::new("test_256", "desc", "usage").with_alias("t256");
        assert_eq!(spec.name, "test_256");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_257() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 257");
        assert!(sink.captured().unwrap().contains("hello 257"));
        let spec = CommandSpec::new("test_257", "desc", "usage").with_alias("t257");
        assert_eq!(spec.name, "test_257");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_258() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 258");
        assert!(sink.captured().unwrap().contains("hello 258"));
        let spec = CommandSpec::new("test_258", "desc", "usage").with_alias("t258");
        assert_eq!(spec.name, "test_258");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_259() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 259");
        assert!(sink.captured().unwrap().contains("hello 259"));
        let spec = CommandSpec::new("test_259", "desc", "usage").with_alias("t259");
        assert_eq!(spec.name, "test_259");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_260() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 260");
        assert!(sink.captured().unwrap().contains("hello 260"));
        let spec = CommandSpec::new("test_260", "desc", "usage").with_alias("t260");
        assert_eq!(spec.name, "test_260");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_261() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 261");
        assert!(sink.captured().unwrap().contains("hello 261"));
        let spec = CommandSpec::new("test_261", "desc", "usage").with_alias("t261");
        assert_eq!(spec.name, "test_261");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_262() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 262");
        assert!(sink.captured().unwrap().contains("hello 262"));
        let spec = CommandSpec::new("test_262", "desc", "usage").with_alias("t262");
        assert_eq!(spec.name, "test_262");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_263() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 263");
        assert!(sink.captured().unwrap().contains("hello 263"));
        let spec = CommandSpec::new("test_263", "desc", "usage").with_alias("t263");
        assert_eq!(spec.name, "test_263");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_264() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 264");
        assert!(sink.captured().unwrap().contains("hello 264"));
        let spec = CommandSpec::new("test_264", "desc", "usage").with_alias("t264");
        assert_eq!(spec.name, "test_264");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_265() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 265");
        assert!(sink.captured().unwrap().contains("hello 265"));
        let spec = CommandSpec::new("test_265", "desc", "usage").with_alias("t265");
        assert_eq!(spec.name, "test_265");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_266() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 266");
        assert!(sink.captured().unwrap().contains("hello 266"));
        let spec = CommandSpec::new("test_266", "desc", "usage").with_alias("t266");
        assert_eq!(spec.name, "test_266");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_267() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 267");
        assert!(sink.captured().unwrap().contains("hello 267"));
        let spec = CommandSpec::new("test_267", "desc", "usage").with_alias("t267");
        assert_eq!(spec.name, "test_267");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_268() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 268");
        assert!(sink.captured().unwrap().contains("hello 268"));
        let spec = CommandSpec::new("test_268", "desc", "usage").with_alias("t268");
        assert_eq!(spec.name, "test_268");
        assert_eq!(spec.aliases.len(), 1);
    }

    #[test]
    fn test_core_cli_stress_269() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::ERROR.is_success());
        let sink = OutputSink::memory();
        sink.println("hello 269");
        assert!(sink.captured().unwrap().contains("hello 269"));
        let spec = CommandSpec::new("test_269", "desc", "usage").with_alias("t269");
        assert_eq!(spec.name, "test_269");
        assert_eq!(spec.aliases.len(), 1);
    }
}
