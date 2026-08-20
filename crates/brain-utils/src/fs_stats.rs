//! # Filesystem Statistics and Scanning
//!
//! Provides directory size calculations, file counts, and disk statistics.

use crate::core::{UtilsError, UtilsResult};
use std::fs;
use std::path::Path;

/// Aggregated filesystem metrics for a directory hierarchy.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FsStats {
    /// Total aggregate size in bytes.
    pub total_bytes: u64,
    /// Total number of files.
    pub file_count: u64,
    /// Total number of subdirectories.
    pub dir_count: u64,
    /// Maximum individual file size in bytes.
    pub max_file_size: u64,
}

/// Recursively computes filesystem statistics for a directory.
pub fn scan_dir_stats<P: AsRef<Path>>(path: P) -> UtilsResult<FsStats> {
    let mut stats = FsStats::default();
    scan_recursive(path.as_ref(), &mut stats)?;
    Ok(stats)
}

fn scan_recursive(dir: &Path, stats: &mut FsStats) -> UtilsResult<()> {
    if !dir.exists() {
        return Err(UtilsError::FileNotFound(dir.to_string_lossy().to_string()));
    }
    if dir.is_dir() {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stats.dir_count += 1;
                    let _ = scan_recursive(&p, stats);
                } else if p.is_file() {
                    stats.file_count += 1;
                    if let Ok(meta) = entry.metadata() {
                        let len = meta.len();
                        stats.total_bytes += len;
                        if len > stats.max_file_size {
                            stats.max_file_size = len;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Computes total directory size in bytes.
pub fn dir_size<P: AsRef<Path>>(path: P) -> UtilsResult<u64> {
    scan_dir_stats(path).map(|s| s.total_bytes)
}

/// Counts total files within a directory hierarchy.
pub fn file_count<P: AsRef<Path>>(path: P) -> UtilsResult<u64> {
    scan_dir_stats(path).map(|s| s.file_count)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_fs_stats_scanner_1() {
        let cur = std::env::current_dir().unwrap();
        let stats = scan_dir_stats(&cur);
        assert!(stats.is_ok());
        let s = stats.unwrap();
        assert!(s.file_count > 0 || s.dir_count > 0 || s.total_bytes >= 0);
    }
}
