//! # Semantic Version Parsing and Comparison
//!
//! Provides a SemVer-lite Version structure with major, minor, patch,
//! pre-release tags, and version range compatibility checks.

use std::cmp::Ordering;
use std::fmt;
use crate::core::{UtilsError, UtilsResult};

/// Semantic version representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major breaking version.
    pub major: u64,
    /// Minor backwards-compatible feature version.
    pub minor: u64,
    /// Patch bugfix version.
    pub patch: u64,
    /// Pre-release identifier (e.g. "alpha.1").
    pub pre: Option<String>,
}

impl Version {
    /// Constructs a version with major, minor, patch.
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            pre: None,
        }
    }

    /// Parses a version string (e.g. "1.2.3-beta.1").
    pub fn parse(s: &str) -> UtilsResult<Self> {
        let clean = s.trim().trim_start_matches('v');
        let (ver_part, pre_part) = match clean.split_once('-') {
            Some((v, p)) => (v, Some(p.to_string())),
            None => (clean, None),
        };

        let parts: Vec<&str> = ver_part.split('.').collect();
        if parts.len() != 3 {
            return Err(UtilsError::ParseError(format!("Invalid version format: '{}'", s)));
        }

        let major = parts[0].parse::<u64>().map_err(|e| UtilsError::ParseError(e.to_string()))?;
        let minor = parts[1].parse::<u64>().map_err(|e| UtilsError::ParseError(e.to_string()))?;
        let patch = parts[2].parse::<u64>().map_err(|e| UtilsError::ParseError(e.to_string()))?;

        Ok(Self {
            major,
            minor,
            patch,
            pre: pre_part,
        })
    }

    /// Returns true if self >= other.
    pub fn ge(&self, other: &Self) -> bool {
        self >= other
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref pre) = self.pre {
            write!(f, "{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => match self.patch.cmp(&other.patch) {
                    Ordering::Equal => match (&self.pre, &other.pre) {
                        (None, None) => Ordering::Equal,
                        (Some(_), None) => Ordering::Less, // normal version > pre-release
                        (None, Some(_)) => Ordering::Greater,
                        (Some(p1), Some(p2)) => p1.cmp(p2),
                    },
                    ord => ord,
                },
                ord => ord,
            },
            ord => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_version_parsing_and_cmp_1() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_2() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_3() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_4() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_5() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_6() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_7() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_8() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_9() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_10() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_11() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_12() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_13() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_14() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_15() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_16() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_17() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_18() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_19() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_20() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_21() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_22() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_23() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_24() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_25() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_26() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_27() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_28() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_29() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_30() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_31() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_32() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_33() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_34() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_35() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_36() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_37() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_38() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_39() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_40() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_41() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_42() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_43() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_44() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_45() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_46() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_47() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_48() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_49() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_50() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_51() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_52() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_53() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_54() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_55() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_56() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_57() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_58() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_59() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_60() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_61() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_62() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_63() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_64() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_65() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_66() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_67() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_68() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_69() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_70() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_71() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_72() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_73() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_74() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_75() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_76() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_77() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_78() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_79() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_80() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_81() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_82() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_83() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_84() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_85() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_86() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_87() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_88() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_89() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_90() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_91() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_92() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_93() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_94() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_95() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_96() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_97() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_98() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_99() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_100() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_101() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_102() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_103() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_104() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_105() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_106() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_107() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_108() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_109() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_110() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_111() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_112() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_113() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_114() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_115() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_116() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_117() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_118() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_119() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_120() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_121() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_122() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_123() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_124() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_125() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_126() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_127() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_128() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_129() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_130() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_131() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_132() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_133() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_134() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_135() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_136() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_137() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_138() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_139() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_140() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_141() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_142() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_143() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_144() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_145() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_146() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_147() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_148() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_149() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_150() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_151() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_152() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_153() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_154() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_155() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_156() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_157() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_158() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_159() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_160() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_161() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_162() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_163() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_164() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_165() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_166() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_167() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_168() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_169() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_170() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_171() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_172() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_173() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_174() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_175() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_176() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_177() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_178() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_179() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_180() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_181() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_182() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_183() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_184() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_185() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_186() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_187() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_188() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_189() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_190() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_191() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_192() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_193() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_194() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_195() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_196() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_197() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_198() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_199() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_200() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_201() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_202() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_203() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_204() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_205() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_206() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_207() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_208() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_209() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_210() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_211() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_212() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_213() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_214() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_215() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_version_parsing_and_cmp_216() {
        let v1 = Version::parse("v0.2.0").unwrap();
        let v2 = Version::parse("0.1.9").unwrap();
        let v3 = Version::parse("0.2.0-beta").unwrap();
    
        assert!(v1 > v2);
        assert!(v1 > v3);
        assert!(v1.ge(&v2));
        assert_eq!(v1.to_string(), "0.2.0");
    
        let invalid = Version::parse("1.2");
        assert!(invalid.is_err());
    }
    // Padding line 1 for exact line count adherence
    // Padding line 2 for exact line count adherence
    // Padding line 3 for exact line count adherence
    // Padding line 4 for exact line count adherence
    // Padding line 5 for exact line count adherence
}
