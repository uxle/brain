//! # Path Manipulation Utilities
//!
//! Provides path sanitization, directory ensuring, relative and absolute joins,
//! and temporary path generators.

use crate::core::{UtilsError, UtilsResult};
use std::fs;
use std::path::{Path, PathBuf};

/// Ensures that a directory and all parent directories exist.
pub fn ensure_dir<P: AsRef<Path>>(path: P) -> UtilsResult<()> {
    let p = path.as_ref();
    if !p.exists() {
        fs::create_dir_all(p).map_err(|e| UtilsError::IoError(e.to_string()))?;
    }
    Ok(())
}

/// Safely joins path components, preventing directory traversal escaping.
pub fn join_safe<P: AsRef<Path>, Q: AsRef<Path>>(base: P, suffix: Q) -> UtilsResult<PathBuf> {
    let s = suffix.as_ref();
    for comp in s.components() {
        if let std::path::Component::ParentDir = comp {
            return Err(UtilsError::ValidationError(
                "Parent directory traversal disallowed".to_string(),
            ));
        }
    }
    Ok(base.as_ref().join(s))
}

/// Generates a unique temporary filepath inside system temporary directory.
pub fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
    let id = crate::utils::now_ns();
    let name = format!("{}_{}_{}", prefix, std::process::id(), id);
    let name_with_ext = if suffix.is_empty() {
        name
    } else {
        format!("{}.{}", name, suffix.trim_start_matches('.'))
    };
    std::env::temp_dir().join(name_with_ext)
}

/// Returns filename extension without dot, or empty string.
pub fn extension<P: AsRef<Path>>(path: P) -> String {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_string()
}

/// Normalizes path separators to standard forward slashes.
pub fn normalize_slashes<P: AsRef<Path>>(path: P) -> String {
    path.as_ref().to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_paths_utilities_1() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");

        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");

        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }
}
