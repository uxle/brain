//! # Semantic Version Parsing and Comparison
//!
//! Provides a SemVer-lite Version structure with major, minor, patch,
//! pre-release tags, and version range compatibility checks.

use crate::core::{UtilsError, UtilsResult};
use std::cmp::Ordering;
use std::fmt;

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
            return Err(UtilsError::ParseError(format!(
                "Invalid version format: '{}'",
                s
            )));
        }

        let major = parts[0]
            .parse::<u64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        let minor = parts[1]
            .parse::<u64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        let patch = parts[2]
            .parse::<u64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;

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
}
