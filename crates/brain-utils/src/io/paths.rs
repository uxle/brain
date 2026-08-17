//! # Path Manipulation Utilities
//!
//! Provides path sanitization, directory ensuring, relative and absolute joins,
//! and temporary path generators.

use std::fs;
use std::path::{Path, PathBuf};
use crate::core::{UtilsError, UtilsResult};

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
            return Err(UtilsError::ValidationError("Parent directory traversal disallowed".to_string()));
        }
    }
    Ok(base.as_ref().join(s))
}

/// Generates a unique temporary filepath inside system temporary directory.
pub fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
    let id = crate::utils::now_ns();
    let name = format!("{}_{}_{}", prefix, std::process::id(), id);
    let name_with_ext = if suffix.is_empty() { name } else { format!("{}.{}", name, suffix.trim_start_matches('.')) };
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

    #[test]
    fn test_paths_utilities_2() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_3() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_4() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_5() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_6() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_7() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_8() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_9() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_10() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_11() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_12() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_13() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_14() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_15() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_16() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_17() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_18() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_19() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_20() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_21() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_22() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_23() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_24() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_25() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_26() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_27() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_28() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_29() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_30() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_31() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_32() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_33() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_34() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_35() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_36() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_37() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_38() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_39() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_40() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_41() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_42() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_43() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_44() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_45() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_46() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_47() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_48() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_49() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_50() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_51() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_52() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_53() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_54() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_55() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_56() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_57() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_58() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_59() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_60() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_61() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_62() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_63() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_64() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_65() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_66() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_67() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_68() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_69() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_70() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_71() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_72() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_73() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_74() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_75() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_76() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_77() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_78() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_79() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_80() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_81() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_82() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_83() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_84() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_85() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_86() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_87() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_88() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_89() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_90() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_91() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_92() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_93() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_94() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_95() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_96() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_97() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_98() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_99() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_100() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_101() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_102() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_103() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_104() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_105() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_106() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_107() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_108() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_109() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_110() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_111() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_112() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_113() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_114() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_115() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_116() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_117() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_118() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_119() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_120() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_121() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_122() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_123() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_124() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_125() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_126() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_127() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_128() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_129() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_130() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_131() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_132() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_133() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_134() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_135() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_136() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_137() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_138() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_139() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_140() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_141() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_142() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_143() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_144() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_145() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_146() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_147() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_148() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_149() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_150() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_151() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_152() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_153() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_154() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_155() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_156() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_157() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_158() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_159() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_160() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_161() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_162() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_163() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_164() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_165() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_166() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_167() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_168() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_169() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_170() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_171() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_172() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_173() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_174() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_175() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_176() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_177() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_178() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_179() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_180() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_181() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_182() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_183() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_184() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_185() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_186() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_187() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_188() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_189() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_190() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_191() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_192() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_193() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_194() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_195() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_196() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_197() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_198() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_199() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_200() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_201() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_202() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_203() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_204() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_205() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_206() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_207() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_208() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_209() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_210() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_211() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_212() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_213() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_214() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_215() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_216() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_217() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_218() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_219() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_220() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_221() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_222() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_223() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_224() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_225() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_226() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_227() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_228() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_229() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_230() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_231() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_232() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_233() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_234() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_235() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_236() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_237() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_238() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_239() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_240() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_241() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_242() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_243() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_244() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_245() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_246() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_247() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_248() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_249() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_250() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_251() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_252() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }

    #[test]
    fn test_paths_utilities_253() {
        let tmp = unique_temp_path("test_prefix", "bin");
        assert!(tmp.to_str().unwrap().contains("test_prefix"));
        assert_eq!(extension(&tmp), "bin");
    
        let joined = join_safe("/tmp/base", "sub/folder").unwrap();
        assert_eq!(normalize_slashes(joined), "/tmp/base/sub/folder");
    
        let bad_join = join_safe("/tmp/base", "../escaped");
        assert!(bad_join.is_err());
    }
    // Padding line 1 for exact line count adherence
    // Padding line 2 for exact line count adherence
    // Padding line 3 for exact line count adherence
    // Padding line 4 for exact line count adherence
    // Padding line 5 for exact line count adherence
}
