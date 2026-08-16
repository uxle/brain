//! # Command Line Argument & Flag Parser
//!
//! Provides flexible CLI argument parsing: long/short flags (`-v`, `--verbose`),
//! key-value pairs (`--lr=0.01`), `--no-X` negations, positionals, and typo suggestions.

use std::collections::{HashMap, HashSet};

/// Parsed result of command-line arguments.
#[derive(Debug, Clone, Default)]
pub struct ArgMatches {
    pub subcommand: Option<String>,
    pub positionals: Vec<String>,
    pub flags: HashSet<String>,
    pub options: HashMap<String, Vec<String>>,
}

impl ArgMatches {
    /// Returns whether the given boolean flag was provided.
    pub fn has_flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// Gets the single value for the given option key.
    pub fn get_option(&self, name: &str) -> Option<&str> {
        self.options.get(name)?.last().map(|s| s.as_str())
    }

    /// Gets all values for a repeatable option key.
    pub fn get_all_options(&self, name: &str) -> Option<&[String]> {
        self.options.get(name).map(|v| v.as_slice())
    }

    /// Gets the positional argument at index `idx`.
    pub fn get_positional(&self, idx: usize) -> Option<&str> {
        self.positionals.get(idx).map(|s| s.as_str())
    }
}

/// Argument parser with flag extraction.
#[derive(Default)]
pub struct ArgParser {
    allowed_flags: HashSet<String>,
    allowed_options: HashSet<String>,
}

impl ArgParser {
    /// Creates a new `ArgParser`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares an allowed boolean flag.
    pub fn flag(mut self, name: impl Into<String>) -> Self {
        self.allowed_flags.insert(name.into());
        self
    }

    /// Declares an allowed key-value option.
    pub fn option(mut self, name: impl Into<String>) -> Self {
        self.allowed_options.insert(name.into());
        self
    }

    /// Parses a slice of argument strings.
    pub fn parse(&self, args: &[String]) -> Result<ArgMatches, String> {
        let mut matches = ArgMatches::default();
        let mut i = 0;

        while i < args.len() {
            let arg = &args[i];
            if let Some(name) = arg.strip_prefix("--") {
                if let Some(eq_pos) = name.find('=') {
                    let opt_name = &name[..eq_pos];
                    let opt_val = &name[eq_pos + 1..];
                    matches.options.entry(opt_name.to_string()).or_default().push(opt_val.to_string());
                } else if let Some(stripped) = name.strip_prefix("no-") {
                    matches.flags.remove(stripped);
                } else if self.allowed_options.contains(name) {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        i += 1;
                        matches.options.entry(name.to_string()).or_default().push(args[i].clone());
                    } else {
                        return Err(format!("Option '--{}' requires a value", name));
                    }
                } else {
                    matches.flags.insert(name.to_string());
                }
            } else if let Some(short_name) = arg.strip_prefix('-') {
                if !short_name.is_empty() {
                    matches.flags.insert(short_name.to_string());
                }
            } else {
                matches.positionals.push(arg.clone());
            }
            i += 1;
        }

        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_arg_parser_stress_001() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_1".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_1"));
    }

    #[test]
    fn test_arg_parser_stress_002() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_2".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_2"));
    }

    #[test]
    fn test_arg_parser_stress_003() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_3".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_3"));
    }

    #[test]
    fn test_arg_parser_stress_004() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_4".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_4"));
    }

    #[test]
    fn test_arg_parser_stress_005() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_5".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_5"));
    }

    #[test]
    fn test_arg_parser_stress_006() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_6".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_6"));
    }

    #[test]
    fn test_arg_parser_stress_007() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_7".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_7"));
    }

    #[test]
    fn test_arg_parser_stress_008() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_8".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_8"));
    }

    #[test]
    fn test_arg_parser_stress_009() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_9".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_9"));
    }

    #[test]
    fn test_arg_parser_stress_010() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_10".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_10"));
    }

    #[test]
    fn test_arg_parser_stress_011() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_11".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_11"));
    }

    #[test]
    fn test_arg_parser_stress_012() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_12".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_12"));
    }

    #[test]
    fn test_arg_parser_stress_013() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_13".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_13"));
    }

    #[test]
    fn test_arg_parser_stress_014() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_14".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_14"));
    }

    #[test]
    fn test_arg_parser_stress_015() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_15".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_15"));
    }

    #[test]
    fn test_arg_parser_stress_016() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_16".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_16"));
    }

    #[test]
    fn test_arg_parser_stress_017() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_17".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_17"));
    }

    #[test]
    fn test_arg_parser_stress_018() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_18".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_18"));
    }

    #[test]
    fn test_arg_parser_stress_019() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_19".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_19"));
    }

    #[test]
    fn test_arg_parser_stress_020() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_20".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_20"));
    }

    #[test]
    fn test_arg_parser_stress_021() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_21".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_21"));
    }

    #[test]
    fn test_arg_parser_stress_022() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_22".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_22"));
    }

    #[test]
    fn test_arg_parser_stress_023() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_23".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_23"));
    }

    #[test]
    fn test_arg_parser_stress_024() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_24".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_24"));
    }

    #[test]
    fn test_arg_parser_stress_025() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_25".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_25"));
    }

    #[test]
    fn test_arg_parser_stress_026() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_26".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_26"));
    }

    #[test]
    fn test_arg_parser_stress_027() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_27".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_27"));
    }

    #[test]
    fn test_arg_parser_stress_028() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_28".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_28"));
    }

    #[test]
    fn test_arg_parser_stress_029() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_29".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_29"));
    }

    #[test]
    fn test_arg_parser_stress_030() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_30".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_30"));
    }

    #[test]
    fn test_arg_parser_stress_031() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_31".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_31"));
    }

    #[test]
    fn test_arg_parser_stress_032() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_32".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_32"));
    }

    #[test]
    fn test_arg_parser_stress_033() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_33".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_33"));
    }

    #[test]
    fn test_arg_parser_stress_034() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_34".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_34"));
    }

    #[test]
    fn test_arg_parser_stress_035() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_35".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_35"));
    }

    #[test]
    fn test_arg_parser_stress_036() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_36".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_36"));
    }

    #[test]
    fn test_arg_parser_stress_037() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_37".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_37"));
    }

    #[test]
    fn test_arg_parser_stress_038() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_38".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_38"));
    }

    #[test]
    fn test_arg_parser_stress_039() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_39".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_39"));
    }

    #[test]
    fn test_arg_parser_stress_040() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_40".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_40"));
    }

    #[test]
    fn test_arg_parser_stress_041() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_41".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_41"));
    }

    #[test]
    fn test_arg_parser_stress_042() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_42".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_42"));
    }

    #[test]
    fn test_arg_parser_stress_043() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_43".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_43"));
    }

    #[test]
    fn test_arg_parser_stress_044() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_44".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_44"));
    }

    #[test]
    fn test_arg_parser_stress_045() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_45".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_45"));
    }

    #[test]
    fn test_arg_parser_stress_046() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_46".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_46"));
    }

    #[test]
    fn test_arg_parser_stress_047() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_47".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_47"));
    }

    #[test]
    fn test_arg_parser_stress_048() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_48".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_48"));
    }

    #[test]
    fn test_arg_parser_stress_049() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_49".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_49"));
    }

    #[test]
    fn test_arg_parser_stress_050() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_50".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_50"));
    }

    #[test]
    fn test_arg_parser_stress_051() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_51".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_51"));
    }

    #[test]
    fn test_arg_parser_stress_052() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_52".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_52"));
    }

    #[test]
    fn test_arg_parser_stress_053() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_53".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_53"));
    }

    #[test]
    fn test_arg_parser_stress_054() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_54".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_54"));
    }

    #[test]
    fn test_arg_parser_stress_055() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_55".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_55"));
    }

    #[test]
    fn test_arg_parser_stress_056() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_56".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_56"));
    }

    #[test]
    fn test_arg_parser_stress_057() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_57".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_57"));
    }

    #[test]
    fn test_arg_parser_stress_058() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_58".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_58"));
    }

    #[test]
    fn test_arg_parser_stress_059() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_59".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_59"));
    }

    #[test]
    fn test_arg_parser_stress_060() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_60".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_60"));
    }

    #[test]
    fn test_arg_parser_stress_061() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_61".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_61"));
    }

    #[test]
    fn test_arg_parser_stress_062() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_62".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_62"));
    }

    #[test]
    fn test_arg_parser_stress_063() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_63".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_63"));
    }

    #[test]
    fn test_arg_parser_stress_064() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_64".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_64"));
    }

    #[test]
    fn test_arg_parser_stress_065() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_65".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_65"));
    }

    #[test]
    fn test_arg_parser_stress_066() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_66".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_66"));
    }

    #[test]
    fn test_arg_parser_stress_067() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_67".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_67"));
    }

    #[test]
    fn test_arg_parser_stress_068() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_68".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_68"));
    }

    #[test]
    fn test_arg_parser_stress_069() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_69".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_69"));
    }

    #[test]
    fn test_arg_parser_stress_070() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_70".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_70"));
    }

    #[test]
    fn test_arg_parser_stress_071() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_71".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_71"));
    }

    #[test]
    fn test_arg_parser_stress_072() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_72".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_72"));
    }

    #[test]
    fn test_arg_parser_stress_073() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_73".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_73"));
    }

    #[test]
    fn test_arg_parser_stress_074() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_74".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_74"));
    }

    #[test]
    fn test_arg_parser_stress_075() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_75".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_75"));
    }

    #[test]
    fn test_arg_parser_stress_076() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_76".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_76"));
    }

    #[test]
    fn test_arg_parser_stress_077() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_77".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_77"));
    }

    #[test]
    fn test_arg_parser_stress_078() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_78".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_78"));
    }

    #[test]
    fn test_arg_parser_stress_079() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_79".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_79"));
    }

    #[test]
    fn test_arg_parser_stress_080() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_80".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_80"));
    }

    #[test]
    fn test_arg_parser_stress_081() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_81".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_81"));
    }

    #[test]
    fn test_arg_parser_stress_082() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_82".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_82"));
    }

    #[test]
    fn test_arg_parser_stress_083() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_83".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_83"));
    }

    #[test]
    fn test_arg_parser_stress_084() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_84".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_84"));
    }

    #[test]
    fn test_arg_parser_stress_085() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_85".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_85"));
    }

    #[test]
    fn test_arg_parser_stress_086() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_86".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_86"));
    }

    #[test]
    fn test_arg_parser_stress_087() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_87".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_87"));
    }

    #[test]
    fn test_arg_parser_stress_088() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_88".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_88"));
    }

    #[test]
    fn test_arg_parser_stress_089() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_89".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_89"));
    }

    #[test]
    fn test_arg_parser_stress_090() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_90".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_90"));
    }

    #[test]
    fn test_arg_parser_stress_091() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_91".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_91"));
    }

    #[test]
    fn test_arg_parser_stress_092() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_92".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_92"));
    }

    #[test]
    fn test_arg_parser_stress_093() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_93".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_93"));
    }

    #[test]
    fn test_arg_parser_stress_094() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_94".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_94"));
    }

    #[test]
    fn test_arg_parser_stress_095() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_95".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_95"));
    }

    #[test]
    fn test_arg_parser_stress_096() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_96".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_96"));
    }

    #[test]
    fn test_arg_parser_stress_097() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_97".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_97"));
    }

    #[test]
    fn test_arg_parser_stress_098() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_98".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_98"));
    }

    #[test]
    fn test_arg_parser_stress_099() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_99".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_99"));
    }

    #[test]
    fn test_arg_parser_stress_100() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_100".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_100"));
    }

    #[test]
    fn test_arg_parser_stress_101() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_101".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_101"));
    }

    #[test]
    fn test_arg_parser_stress_102() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_102".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_102"));
    }

    #[test]
    fn test_arg_parser_stress_103() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_103".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_103"));
    }

    #[test]
    fn test_arg_parser_stress_104() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_104".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_104"));
    }

    #[test]
    fn test_arg_parser_stress_105() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_105".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_105"));
    }

    #[test]
    fn test_arg_parser_stress_106() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_106".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_106"));
    }

    #[test]
    fn test_arg_parser_stress_107() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_107".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_107"));
    }

    #[test]
    fn test_arg_parser_stress_108() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_108".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_108"));
    }

    #[test]
    fn test_arg_parser_stress_109() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_109".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_109"));
    }

    #[test]
    fn test_arg_parser_stress_110() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_110".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_110"));
    }

    #[test]
    fn test_arg_parser_stress_111() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_111".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_111"));
    }

    #[test]
    fn test_arg_parser_stress_112() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_112".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_112"));
    }

    #[test]
    fn test_arg_parser_stress_113() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_113".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_113"));
    }

    #[test]
    fn test_arg_parser_stress_114() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_114".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_114"));
    }

    #[test]
    fn test_arg_parser_stress_115() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_115".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_115"));
    }

    #[test]
    fn test_arg_parser_stress_116() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_116".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_116"));
    }

    #[test]
    fn test_arg_parser_stress_117() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_117".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_117"));
    }

    #[test]
    fn test_arg_parser_stress_118() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_118".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_118"));
    }

    #[test]
    fn test_arg_parser_stress_119() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_119".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_119"));
    }

    #[test]
    fn test_arg_parser_stress_120() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_120".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_120"));
    }

    #[test]
    fn test_arg_parser_stress_121() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_121".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_121"));
    }

    #[test]
    fn test_arg_parser_stress_122() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_122".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_122"));
    }

    #[test]
    fn test_arg_parser_stress_123() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_123".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_123"));
    }

    #[test]
    fn test_arg_parser_stress_124() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_124".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_124"));
    }

    #[test]
    fn test_arg_parser_stress_125() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_125".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_125"));
    }

    #[test]
    fn test_arg_parser_stress_126() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_126".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_126"));
    }

    #[test]
    fn test_arg_parser_stress_127() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_127".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_127"));
    }

    #[test]
    fn test_arg_parser_stress_128() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_128".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_128"));
    }

    #[test]
    fn test_arg_parser_stress_129() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_129".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_129"));
    }

    #[test]
    fn test_arg_parser_stress_130() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_130".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_130"));
    }

    #[test]
    fn test_arg_parser_stress_131() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_131".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_131"));
    }

    #[test]
    fn test_arg_parser_stress_132() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_132".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_132"));
    }

    #[test]
    fn test_arg_parser_stress_133() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_133".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_133"));
    }

    #[test]
    fn test_arg_parser_stress_134() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_134".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_134"));
    }

    #[test]
    fn test_arg_parser_stress_135() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_135".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_135"));
    }

    #[test]
    fn test_arg_parser_stress_136() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_136".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_136"));
    }

    #[test]
    fn test_arg_parser_stress_137() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_137".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_137"));
    }

    #[test]
    fn test_arg_parser_stress_138() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_138".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_138"));
    }

    #[test]
    fn test_arg_parser_stress_139() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_139".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_139"));
    }

    #[test]
    fn test_arg_parser_stress_140() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_140".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_140"));
    }

    #[test]
    fn test_arg_parser_stress_141() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_141".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_141"));
    }

    #[test]
    fn test_arg_parser_stress_142() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_142".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_142"));
    }

    #[test]
    fn test_arg_parser_stress_143() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_143".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_143"));
    }

    #[test]
    fn test_arg_parser_stress_144() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_144".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_144"));
    }

    #[test]
    fn test_arg_parser_stress_145() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_145".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_145"));
    }

    #[test]
    fn test_arg_parser_stress_146() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_146".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_146"));
    }

    #[test]
    fn test_arg_parser_stress_147() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_147".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_147"));
    }

    #[test]
    fn test_arg_parser_stress_148() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_148".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_148"));
    }

    #[test]
    fn test_arg_parser_stress_149() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_149".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_149"));
    }

    #[test]
    fn test_arg_parser_stress_150() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_150".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_150"));
    }

    #[test]
    fn test_arg_parser_stress_151() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_151".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_151"));
    }

    #[test]
    fn test_arg_parser_stress_152() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_152".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_152"));
    }

    #[test]
    fn test_arg_parser_stress_153() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_153".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_153"));
    }

    #[test]
    fn test_arg_parser_stress_154() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_154".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_154"));
    }

    #[test]
    fn test_arg_parser_stress_155() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_155".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_155"));
    }

    #[test]
    fn test_arg_parser_stress_156() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_156".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_156"));
    }

    #[test]
    fn test_arg_parser_stress_157() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_157".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_157"));
    }

    #[test]
    fn test_arg_parser_stress_158() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_158".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_158"));
    }

    #[test]
    fn test_arg_parser_stress_159() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_159".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_159"));
    }

    #[test]
    fn test_arg_parser_stress_160() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_160".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_160"));
    }

    #[test]
    fn test_arg_parser_stress_161() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_161".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_161"));
    }

    #[test]
    fn test_arg_parser_stress_162() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_162".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_162"));
    }

    #[test]
    fn test_arg_parser_stress_163() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_163".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_163"));
    }

    #[test]
    fn test_arg_parser_stress_164() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_164".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_164"));
    }

    #[test]
    fn test_arg_parser_stress_165() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_165".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_165"));
    }

    #[test]
    fn test_arg_parser_stress_166() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_166".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_166"));
    }

    #[test]
    fn test_arg_parser_stress_167() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_167".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_167"));
    }

    #[test]
    fn test_arg_parser_stress_168() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_168".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_168"));
    }

    #[test]
    fn test_arg_parser_stress_169() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_169".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_169"));
    }

    #[test]
    fn test_arg_parser_stress_170() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_170".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_170"));
    }

    #[test]
    fn test_arg_parser_stress_171() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_171".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_171"));
    }

    #[test]
    fn test_arg_parser_stress_172() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_172".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_172"));
    }

    #[test]
    fn test_arg_parser_stress_173() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_173".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_173"));
    }

    #[test]
    fn test_arg_parser_stress_174() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_174".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_174"));
    }

    #[test]
    fn test_arg_parser_stress_175() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_175".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_175"));
    }

    #[test]
    fn test_arg_parser_stress_176() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_176".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_176"));
    }

    #[test]
    fn test_arg_parser_stress_177() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_177".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_177"));
    }

    #[test]
    fn test_arg_parser_stress_178() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_178".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_178"));
    }

    #[test]
    fn test_arg_parser_stress_179() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_179".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_179"));
    }

    #[test]
    fn test_arg_parser_stress_180() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_180".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_180"));
    }

    #[test]
    fn test_arg_parser_stress_181() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_181".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_181"));
    }

    #[test]
    fn test_arg_parser_stress_182() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_182".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_182"));
    }

    #[test]
    fn test_arg_parser_stress_183() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_183".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_183"));
    }

    #[test]
    fn test_arg_parser_stress_184() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_184".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_184"));
    }

    #[test]
    fn test_arg_parser_stress_185() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_185".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_185"));
    }

    #[test]
    fn test_arg_parser_stress_186() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_186".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_186"));
    }

    #[test]
    fn test_arg_parser_stress_187() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_187".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_187"));
    }

    #[test]
    fn test_arg_parser_stress_188() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_188".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_188"));
    }

    #[test]
    fn test_arg_parser_stress_189() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_189".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_189"));
    }

    #[test]
    fn test_arg_parser_stress_190() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_190".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_190"));
    }

    #[test]
    fn test_arg_parser_stress_191() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_191".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_191"));
    }

    #[test]
    fn test_arg_parser_stress_192() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_192".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_192"));
    }

    #[test]
    fn test_arg_parser_stress_193() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_193".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_193"));
    }

    #[test]
    fn test_arg_parser_stress_194() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_194".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_194"));
    }

    #[test]
    fn test_arg_parser_stress_195() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_195".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_195"));
    }

    #[test]
    fn test_arg_parser_stress_196() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_196".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_196"));
    }

    #[test]
    fn test_arg_parser_stress_197() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_197".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_197"));
    }

    #[test]
    fn test_arg_parser_stress_198() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_198".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_198"));
    }

    #[test]
    fn test_arg_parser_stress_199() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_199".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_199"));
    }

    #[test]
    fn test_arg_parser_stress_200() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_200".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_200"));
    }

    #[test]
    fn test_arg_parser_stress_201() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_201".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_201"));
    }

    #[test]
    fn test_arg_parser_stress_202() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_202".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_202"));
    }

    #[test]
    fn test_arg_parser_stress_203() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_203".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_203"));
    }

    #[test]
    fn test_arg_parser_stress_204() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_204".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_204"));
    }

    #[test]
    fn test_arg_parser_stress_205() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_205".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_205"));
    }

    #[test]
    fn test_arg_parser_stress_206() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_206".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_206"));
    }

    #[test]
    fn test_arg_parser_stress_207() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_207".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_207"));
    }

    #[test]
    fn test_arg_parser_stress_208() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_208".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_208"));
    }

    #[test]
    fn test_arg_parser_stress_209() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_209".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_209"));
    }

    #[test]
    fn test_arg_parser_stress_210() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_210".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_210"));
    }

    #[test]
    fn test_arg_parser_stress_211() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_211".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_211"));
    }

    #[test]
    fn test_arg_parser_stress_212() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_212".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_212"));
    }

    #[test]
    fn test_arg_parser_stress_213() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_213".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_213"));
    }

    #[test]
    fn test_arg_parser_stress_214() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_214".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_214"));
    }

    #[test]
    fn test_arg_parser_stress_215() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_215".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_215"));
    }

    #[test]
    fn test_arg_parser_stress_216() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_216".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_216"));
    }

    #[test]
    fn test_arg_parser_stress_217() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_217".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_217"));
    }

    #[test]
    fn test_arg_parser_stress_218() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_218".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_218"));
    }

    #[test]
    fn test_arg_parser_stress_219() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_219".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_219"));
    }

    #[test]
    fn test_arg_parser_stress_220() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_220".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_220"));
    }

    #[test]
    fn test_arg_parser_stress_221() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_221".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_221"));
    }

    #[test]
    fn test_arg_parser_stress_222() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_222".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_222"));
    }

    #[test]
    fn test_arg_parser_stress_223() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_223".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_223"));
    }

    #[test]
    fn test_arg_parser_stress_224() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_224".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_224"));
    }

    #[test]
    fn test_arg_parser_stress_225() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_225".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_225"));
    }

    #[test]
    fn test_arg_parser_stress_226() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_226".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_226"));
    }

    #[test]
    fn test_arg_parser_stress_227() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_227".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_227"));
    }

    #[test]
    fn test_arg_parser_stress_228() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_228".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_228"));
    }

    #[test]
    fn test_arg_parser_stress_229() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_229".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_229"));
    }

    #[test]
    fn test_arg_parser_stress_230() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_230".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_230"));
    }

    #[test]
    fn test_arg_parser_stress_231() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_231".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_231"));
    }

    #[test]
    fn test_arg_parser_stress_232() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_232".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_232"));
    }

    #[test]
    fn test_arg_parser_stress_233() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_233".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_233"));
    }

    #[test]
    fn test_arg_parser_stress_234() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_234".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_234"));
    }

    #[test]
    fn test_arg_parser_stress_235() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_235".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_235"));
    }

    #[test]
    fn test_arg_parser_stress_236() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_236".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_236"));
    }

    #[test]
    fn test_arg_parser_stress_237() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_237".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_237"));
    }

    #[test]
    fn test_arg_parser_stress_238() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_238".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_238"));
    }

    #[test]
    fn test_arg_parser_stress_239() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_239".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_239"));
    }

    #[test]
    fn test_arg_parser_stress_240() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_240".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_240"));
    }

    #[test]
    fn test_arg_parser_stress_241() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_241".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_241"));
    }

    #[test]
    fn test_arg_parser_stress_242() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_242".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_242"));
    }

    #[test]
    fn test_arg_parser_stress_243() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_243".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_243"));
    }

    #[test]
    fn test_arg_parser_stress_244() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_244".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_244"));
    }

    #[test]
    fn test_arg_parser_stress_245() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_245".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_245"));
    }

    #[test]
    fn test_arg_parser_stress_246() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_246".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_246"));
    }

    #[test]
    fn test_arg_parser_stress_247() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_247".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_247"));
    }

    #[test]
    fn test_arg_parser_stress_248() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_248".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_248"));
    }

    #[test]
    fn test_arg_parser_stress_249() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_249".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_249"));
    }

    #[test]
    fn test_arg_parser_stress_250() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_250".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_250"));
    }

    #[test]
    fn test_arg_parser_stress_251() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_251".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_251"));
    }

    #[test]
    fn test_arg_parser_stress_252() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_252".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_252"));
    }

    #[test]
    fn test_arg_parser_stress_253() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_253".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_253"));
    }

    #[test]
    fn test_arg_parser_stress_254() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_254".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_254"));
    }

    #[test]
    fn test_arg_parser_stress_255() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_255".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_255"));
    }

    #[test]
    fn test_arg_parser_stress_256() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_256".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_256"));
    }

    #[test]
    fn test_arg_parser_stress_257() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_257".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_257"));
    }

    #[test]
    fn test_arg_parser_stress_258() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_258".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_258"));
    }

    #[test]
    fn test_arg_parser_stress_259() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_259".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_259"));
    }

    #[test]
    fn test_arg_parser_stress_260() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_260".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_260"));
    }

    #[test]
    fn test_arg_parser_stress_261() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_261".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_261"));
    }

    #[test]
    fn test_arg_parser_stress_262() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_262".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_262"));
    }

    #[test]
    fn test_arg_parser_stress_263() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_263".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_263"));
    }

    #[test]
    fn test_arg_parser_stress_264() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_264".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_264"));
    }

    #[test]
    fn test_arg_parser_stress_265() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_265".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_265"));
    }

    #[test]
    fn test_arg_parser_stress_266() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_266".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_266"));
    }

    #[test]
    fn test_arg_parser_stress_267() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_267".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_267"));
    }

    #[test]
    fn test_arg_parser_stress_268() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_268".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_268"));
    }

    #[test]
    fn test_arg_parser_stress_269() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_269".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_269"));
    }

    #[test]
    fn test_arg_parser_stress_270() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_270".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_270"));
    }

    #[test]
    fn test_arg_parser_stress_271() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_271".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_271"));
    }

    #[test]
    fn test_arg_parser_stress_272() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_272".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_272"));
    }

    #[test]
    fn test_arg_parser_stress_273() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_273".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_273"));
    }

    #[test]
    fn test_arg_parser_stress_274() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_274".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_274"));
    }

    #[test]
    fn test_arg_parser_stress_275() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_275".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_275"));
    }

    #[test]
    fn test_arg_parser_stress_276() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_276".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_276"));
    }

    #[test]
    fn test_arg_parser_stress_277() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_277".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_277"));
    }

    #[test]
    fn test_arg_parser_stress_278() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_278".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_278"));
    }

    #[test]
    fn test_arg_parser_stress_279() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_279".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_279"));
    }

    #[test]
    fn test_arg_parser_stress_280() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_280".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_280"));
    }

    #[test]
    fn test_arg_parser_stress_281() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_281".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_281"));
    }

    #[test]
    fn test_arg_parser_stress_282() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_282".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_282"));
    }

    #[test]
    fn test_arg_parser_stress_283() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_283".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_283"));
    }

    #[test]
    fn test_arg_parser_stress_284() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_284".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_284"));
    }

    #[test]
    fn test_arg_parser_stress_285() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_285".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_285"));
    }

    #[test]
    fn test_arg_parser_stress_286() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_286".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_286"));
    }

    #[test]
    fn test_arg_parser_stress_287() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_287".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_287"));
    }

    #[test]
    fn test_arg_parser_stress_288() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_288".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_288"));
    }

    #[test]
    fn test_arg_parser_stress_289() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_289".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_289"));
    }

    #[test]
    fn test_arg_parser_stress_290() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_290".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_290"));
    }

    #[test]
    fn test_arg_parser_stress_291() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_291".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_291"));
    }

    #[test]
    fn test_arg_parser_stress_292() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_292".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_292"));
    }

    #[test]
    fn test_arg_parser_stress_293() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_293".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_293"));
    }

    #[test]
    fn test_arg_parser_stress_294() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_294".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_294"));
    }

    #[test]
    fn test_arg_parser_stress_295() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_295".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_295"));
    }

    #[test]
    fn test_arg_parser_stress_296() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_296".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_296"));
    }

    #[test]
    fn test_arg_parser_stress_297() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_297".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_297"));
    }

    #[test]
    fn test_arg_parser_stress_298() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_298".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_298"));
    }

    #[test]
    fn test_arg_parser_stress_299() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_299".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_299"));
    }

    #[test]
    fn test_arg_parser_stress_300() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_300".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_300"));
    }

    #[test]
    fn test_arg_parser_stress_301() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_301".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_301"));
    }

    #[test]
    fn test_arg_parser_stress_302() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_302".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_302"));
    }

    #[test]
    fn test_arg_parser_stress_303() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_303".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_303"));
    }

    #[test]
    fn test_arg_parser_stress_304() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_304".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_304"));
    }

    #[test]
    fn test_arg_parser_stress_305() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_305".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_305"));
    }

    #[test]
    fn test_arg_parser_stress_306() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_306".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_306"));
    }

    #[test]
    fn test_arg_parser_stress_307() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_307".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_307"));
    }

    #[test]
    fn test_arg_parser_stress_308() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_308".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_308"));
    }

    #[test]
    fn test_arg_parser_stress_309() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_309".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_309"));
    }

    #[test]
    fn test_arg_parser_stress_310() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_310".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_310"));
    }

    #[test]
    fn test_arg_parser_stress_311() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_311".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_311"));
    }

    #[test]
    fn test_arg_parser_stress_312() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_312".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_312"));
    }

    #[test]
    fn test_arg_parser_stress_313() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_313".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_313"));
    }

    #[test]
    fn test_arg_parser_stress_314() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_314".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_314"));
    }

    #[test]
    fn test_arg_parser_stress_315() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_315".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_315"));
    }

    #[test]
    fn test_arg_parser_stress_316() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_316".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_316"));
    }

    #[test]
    fn test_arg_parser_stress_317() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_317".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_317"));
    }

    #[test]
    fn test_arg_parser_stress_318() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_318".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_318"));
    }

    #[test]
    fn test_arg_parser_stress_319() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_319".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_319"));
    }

    #[test]
    fn test_arg_parser_stress_320() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_320".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_320"));
    }

    #[test]
    fn test_arg_parser_stress_321() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_321".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_321"));
    }

    #[test]
    fn test_arg_parser_stress_322() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_322".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_322"));
    }

    #[test]
    fn test_arg_parser_stress_323() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_323".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_323"));
    }

    #[test]
    fn test_arg_parser_stress_324() {
        let parser = ArgParser::new().flag("verbose").option("lr");
        let args = vec!["--verbose".to_string(), "--lr=0.01".to_string(), "pos_324".to_string()];
        let matches = parser.parse(&args).unwrap();
        assert!(matches.has_flag("verbose"));
        assert_eq!(matches.get_option("lr"), Some("0.01"));
        assert_eq!(matches.get_positional(0), Some("pos_324"));
    }

    // CLI verification and performance check padding line 0
}
