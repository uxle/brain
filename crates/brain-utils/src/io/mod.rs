//! # I/O Utilities
//!
//! Provides atomic file writes, stream utilities, buffered file reading,
//! and filesystem helper routines.

pub mod csv;
pub mod ini;
pub mod json;
pub mod paths;

use crate::core::{UtilsError, UtilsResult};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

/// I/O configuration options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IoConfig {
    /// Buffer size for read/write streams.
    pub buffer_size: usize,
    /// Whether to force file sync on write.
    pub sync_on_write: bool,
}

/// Reads entire file content as a UTF-8 string.
pub fn read_file_str<P: AsRef<Path>>(path: P) -> UtilsResult<String> {
    fs::read_to_string(path.as_ref()).map_err(|e| UtilsError::IoError(e.to_string()))
}

/// Reads entire file content as raw bytes.
pub fn read_file_bytes<P: AsRef<Path>>(path: P) -> UtilsResult<Vec<u8>> {
    fs::read(path.as_ref()).map_err(|e| UtilsError::IoError(e.to_string()))
}

/// Writes byte slice to file, replacing if exists.
pub fn write_file<P: AsRef<Path>>(path: P, data: &[u8]) -> UtilsResult<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| UtilsError::IoError(e.to_string()))?;
        }
    }
    let mut file = File::create(path.as_ref()).map_err(|e| UtilsError::IoError(e.to_string()))?;
    file.write_all(data)
        .map_err(|e| UtilsError::IoError(e.to_string()))
}

/// Appends byte slice to file.
pub fn append_file<P: AsRef<Path>>(path: P, data: &[u8]) -> UtilsResult<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| UtilsError::IoError(e.to_string()))?;
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_ref())
        .map_err(|e| UtilsError::IoError(e.to_string()))?;
    file.write_all(data)
        .map_err(|e| UtilsError::IoError(e.to_string()))
}

/// Atomically writes data to a temporary file and renames to target.
pub fn atomic_write_file<P: AsRef<Path>>(path: P, data: &[u8]) -> UtilsResult<()> {
    let p = path.as_ref();
    let parent = p.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent).map_err(|e| UtilsError::IoError(e.to_string()))?;
    }
    let tmp_path = parent.join(format!(
        ".tmp_{}_{}",
        std::process::id(),
        crate::utils::now_ns()
    ));
    {
        let mut f = File::create(&tmp_path).map_err(|e| UtilsError::IoError(e.to_string()))?;
        f.write_all(data)
            .map_err(|e| UtilsError::IoError(e.to_string()))?;
        f.sync_all()
            .map_err(|e| UtilsError::IoError(e.to_string()))?;
    }
    fs::rename(&tmp_path, p).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        UtilsError::IoError(e.to_string())
    })
}

/// Deletes a file if it exists.
pub fn delete_file<P: AsRef<Path>>(path: P) -> UtilsResult<bool> {
    let p = path.as_ref();
    if p.exists() {
        fs::remove_file(p)
            .map(|_| true)
            .map_err(|e| UtilsError::IoError(e.to_string()))
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_io_operations_1() {
        let tmp_path = format!("/tmp/brain_test_io_{}.txt", 1);
        let data = b"hello world from brain io";
        assert!(atomic_write_file(&tmp_path, data).is_ok());

        let content = read_file_str(&tmp_path).unwrap();
        assert_eq!(content, "hello world from brain io");

        let bytes = read_file_bytes(&tmp_path).unwrap();
        assert_eq!(bytes.len(), data.len());

        assert!(append_file(&tmp_path, b" appended").is_ok());
        let appended = read_file_str(&tmp_path).unwrap();
        assert_eq!(appended, "hello world from brain io appended");

        assert!(delete_file(&tmp_path).unwrap());
    }
}
