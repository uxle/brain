//! Serialization for tensors in the Brain deep learning framework.
//!
//! This module provides binary and JSON serialization for tensors, including
//! save/load functions, a binary format with magic headers, and NPZ-like
//! multi-tensor archives.
//!
//! # Binary Format
//!
//! The binary format consists of:
//! - Magic header: 8 bytes ("BRAIN\0\0\0\0")
//! - Version: 4 bytes (u32 LE)
//! - DType code: 1 byte (0=f64, 1=f32, 2=i32, etc.)
//! - ndim: 4 bytes (u32 LE)
//! - shape: ndim * 8 bytes (u64 LE per dimension)
//! - Data: numel * 8 bytes (f64 LE per element)

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{BrainError, BrainResult};
use crate::tensor::Tensor;
use std::io::{Read, Write, Cursor};
use std::fs;
use std::path::Path;

// =============================================================================
// Constants
// =============================================================================

/// Magic header for the Brain binary format.
pub const MAGIC_HEADER: &[u8; 8] = b"BRAIN\0\0\0\0";

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;

/// DType codes for binary serialization.
pub const DTYPE_F64: u8 = 0;
pub const DTYPE_F32: u8 = 1;
pub const DTYPE_I64: u8 = 2;
pub const DTYPE_I32: u8 = 3;
pub const DTYPE_I16: u8 = 4;
pub const DTYPE_I8: u8 = 5;
pub const DTYPE_U8: u8 = 6;

// =============================================================================
// Binary Serialization
// =============================================================================

/// Saves a tensor to a binary file.
pub fn save(tensor: &Tensor, path: &str) -> BrainResult<()> {
    let bytes = serialize_tensor(tensor)?;
    let mut file = fs::File::create(path).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    file.write_all(&bytes).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    Ok(())
}

/// Loads a tensor from a binary file.
pub fn load(path: &str) -> BrainResult<Tensor> {
    let bytes = fs::read(path).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    deserialize_tensor(&bytes)
}

/// Serializes a tensor to a byte vector.
pub fn serialize_tensor(tensor: &Tensor) -> BrainResult<Vec<u8>> {
    let mut buf = Vec::with_capacity(64 + tensor.numel() * 8);

    // Magic header
    buf.extend_from_slice(MAGIC_HEADER);

    // Version
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

    // DType code
    let dtype_code = dtype_to_code(tensor.dtype());
    buf.push(dtype_code);

    // ndim
    let ndim = tensor.ndim() as u32;
    buf.extend_from_slice(&ndim.to_le_bytes());

    // Shape
    for &dim in tensor.shape() {
        buf.extend_from_slice(&(dim as u64).to_le_bytes());
    }

    // Data (as f64 LE for simplicity)
    for &v in tensor.data() {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    Ok(buf)
}

/// Deserializes a tensor from a byte slice.
pub fn deserialize_tensor(bytes: &[u8]) -> BrainResult<Tensor> {
    let mut cursor = Cursor::new(bytes);

    // Read and verify magic header
    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    if &magic != MAGIC_HEADER {
        return Err(BrainError::serialization_error(
            &format!("Invalid magic header: expected {:?}, got {:?}", MAGIC_HEADER, magic),
        ));
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    cursor.read_exact(&mut version_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    let version = u32::from_le_bytes(version_bytes);
    if version > FORMAT_VERSION {
        return Err(BrainError::serialization_error(
            &format!("Unsupported version: {} (max supported: {})", version, FORMAT_VERSION),
        ));
    }

    // Read dtype code
    let mut dtype_byte = [0u8; 1];
    cursor.read_exact(&mut dtype_byte).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    let dtype = code_to_dtype(dtype_byte[0])?;

    // Read ndim
    let mut ndim_bytes = [0u8; 4];
    cursor.read_exact(&mut ndim_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    let ndim = u32::from_le_bytes(ndim_bytes) as usize;

    // Read shape
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        cursor.read_exact(&mut dim_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    // Compute element count
    let numel: usize = shape.iter().product();

    // Read data
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let mut val_bytes = [0u8; 8];
        cursor.read_exact(&mut val_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
        data.push(f64::from_le_bytes(val_bytes));
    }

    // Construct tensor
    let mut tensor = Tensor::new(data, shape);
    tensor.set_dtype(dtype);
    Ok(tensor)
}

fn dtype_to_code(dtype: DType) -> u8 {
    match dtype {
        DType::F64 => DTYPE_F64,
        DType::F32 => DTYPE_F32,
        DType::I64 => DTYPE_I64,
        DType::I32 => DTYPE_I32,
        DType::I16 => DTYPE_I16,
        DType::I8 => DTYPE_I8,
        DType::U8 => DTYPE_U8,
        _ => DTYPE_F64, // Default to f64 for unsupported types
    }
}

fn code_to_dtype(code: u8) -> BrainResult<DType> {
    match code {
        DTYPE_F64 => Ok(DType::F64),
        DTYPE_F32 => Ok(DType::F32),
        DTYPE_I64 => Ok(DType::I64),
        DTYPE_I32 => Ok(DType::I32),
        DTYPE_I16 => Ok(DType::I16),
        DTYPE_I8 => Ok(DType::I8),
        DTYPE_U8 => Ok(DType::U8),
        _ => Err(BrainError::serialization_error(&format!("Unknown dtype code: {}", code))),
    }
}

// =============================================================================
// NPZ-like Multi-Tensor Archive (Simplified)
// =============================================================================

/// Saves multiple tensors to a simplified archive format.
pub fn save_npz(tensors: &[(String, Tensor)], path: &str) -> BrainResult<()> {
    let mut file = fs::File::create(path).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;

    // Global header
    file.write_all(MAGIC_HEADER).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    let count = tensors.len() as u32;
    file.write_all(&count.to_le_bytes()).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;

    // Each tensor: name_len(4) | name(N) | tensor_data
    for (name, tensor) in tensors {
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        file.write_all(&name_len.to_le_bytes()).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
        file.write_all(name_bytes).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;

        let tensor_bytes = serialize_tensor(tensor)?;
        let tensor_len = tensor_bytes.len() as u32;
        file.write_all(&tensor_len.to_le_bytes()).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
        file.write_all(&tensor_bytes).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    }

    Ok(())
}

/// Loads multiple tensors from a simplified archive format.
pub fn load_npz(path: &str) -> BrainResult<Vec<(String, Tensor)>> {
    let bytes = fs::read(path).map_err(|e| BrainError::io_error_with_path(&e.to_string(), path))?;
    let mut cursor = Cursor::new(&bytes);

    // Read global header
    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    if &magic != MAGIC_HEADER {
        return Err(BrainError::serialization_error("Invalid NPZ magic header"));
    }

    let mut count_bytes = [0u8; 4];
    cursor.read_exact(&mut count_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let mut name_len_bytes = [0u8; 4];
        cursor.read_exact(&mut name_len_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        cursor.read_exact(&mut name_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
        let name = String::from_utf8(name_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;

        let mut tensor_len_bytes = [0u8; 4];
        cursor.read_exact(&mut tensor_len_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;
        let tensor_len = u32::from_le_bytes(tensor_len_bytes) as usize;

        let mut tensor_bytes = vec![0u8; tensor_len];
        cursor.read_exact(&mut tensor_bytes).map_err(|e| BrainError::serialization_error(&e.to_string()))?;

        let tensor = deserialize_tensor(&tensor_bytes)?;
        result.push((name, tensor));
    }

    Ok(result)
}

// =============================================================================
// Vec Serialization
// =============================================================================

/// Converts tensor data to a Vec<f64>.
pub fn to_vec(tensor: &Tensor) -> Vec<f64> {
    tensor.data().to_vec()
}

/// Creates a tensor from a Vec<f64>.
pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(data, shape)
}

// =============================================================================
// JSON Format
// =============================================================================

/// Serializes a small tensor to a JSON string.
pub fn to_json(tensor: &Tensor) -> String {
    let data = tensor.data();
    let shape = tensor.shape();
    let dtype = tensor.dtype();

    let data_str = if data.len() <= 20 {
        format!("[{}]", data.iter().map(|v| format!("{:.6}", v).parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>().join(", "))
    } else {
        format!("[{}..{} elements, first 20: [{}]]",
            data.len(), data.len(),
            data.iter().take(20).map(|v| format!("{:.6}", v).parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>().join(", "))
    };

    format!(r#"{{"shape": {:?}, "dtype": "{}", "data": {}, "numel": {}}}"#,
        shape, dtype.short_name(), data_str, data.len())
}

/// Deserializes a tensor from a JSON string (limited support).
pub fn from_json(json: &str) -> BrainResult<Tensor> {
    // Simple JSON parser for the format produced by to_json
    let json = json.trim();

    // Extract shape
    let shape = extract_json_array(json, "shape")?;
    // Extract data
    let data = extract_json_data(json)?;

    Tensor::new(data, shape)
}

/// Extracts a JSON array of numbers.
fn extract_json_array(json: &str, key: &str) -> BrainResult<Vec<usize>> {
    let key_pattern = format!("\"{}\"", key);
    let start = json.find(&key_pattern).ok_or_else(|| BrainError::parse_error(json, "array", "key not found"))?;
    let rest = &json[start + key_pattern.len()..];
    let bracket_start = rest.find('[').ok_or_else(|| BrainError::parse_error(json, "array", "[ not found"))?;
    let rest = &rest[bracket_start + 1..];
    let bracket_end = rest.find(']').ok_or_else(|| BrainError::parse_error(json, "array", "] not found"))?;
    let inner = &rest[..bracket_end];

    if inner.trim().is_empty() {
        return Ok(vec![]);
    }

    inner.split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|e| BrainError::parse_error(s, "usize", &e.to_string())))
        .collect()
}

/// Extracts the data array from JSON.
fn extract_json_data(json: &str) -> BrainResult<Vec<f64>> {
    let key_pattern = "\"data\"";
    let start = json.find(key_pattern).ok_or_else(|| BrainError::parse_error(json, "data", "not found"))?;
    let rest = &json[start + key_pattern.len()..];
    let bracket_start = rest.find('[').ok_or_else(|| BrainError::parse_error(json, "data", "[ not found"))?;
    let rest = &rest[bracket_start + 1..];
    let bracket_end = rest.find(']').ok_or_else(|| BrainError::parse_error(json, "data", "] not found"))?;
    let inner = &rest[..bracket_end];

    if inner.trim().is_empty() {
        return Ok(vec![]);
    }

    inner.split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| BrainError::parse_error(s, "f64", &e.to_string())))
        .collect()
}

// =============================================================================
// Version Compatibility
// =============================================================================

/// Checks version compatibility between two format versions.
pub fn check_version(file_version: u32, current_version: u32) -> BrainResult<()> {
    if file_version > current_version {
        return Err(BrainError::NotImplemented {
            feature: format!("Format version {}. Maximum supported: {}", file_version, current_version),
            alternative: "Upgrade the library version",
        });
    }
    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Returns the expected file size for a tensor in the binary format.
pub fn estimate_file_size(tensor: &Tensor) -> usize {
    let header_size = 8 + 4 + 1 + 4; // magic + version + dtype + ndim
    let shape_size = tensor.ndim() * 8;
    let data_size = tensor.numel() * 8;
    header_size + shape_size + data_size
}

/// Validates that a byte slice contains a valid tensor header.
pub fn validate_header(bytes: &[u8]) -> BrainResult<(u32, u8, usize, Vec<usize>)> {
    if bytes.len() < 17 {
        return Err(BrainError::serialization_error("File too short for tensor header"));
    }
    if &bytes[0..8] != MAGIC_HEADER {
        return Err(BrainError::serialization_error("Invalid magic header"));
    }

    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let dtype_code = bytes[12];
    let ndim = u32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]) as usize;

    if bytes.len() < 17 + ndim * 8 {
        return Err(BrainError::serialization_error("File too short for shape"));
    }

    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let offset = 17 + i * 8;
        shape.push(u64::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
        ]) as usize);
    }

    Ok((version, dtype_code, ndim, shape))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(data.clone(), vec![2, 3]);
        let bytes = serialize_tensor(&tensor).unwrap();
        let loaded = deserialize_tensor(&bytes).unwrap();
        assert_eq!(loaded.shape(), tensor.shape());
        for i in 0..6 { assert!((loaded.get(i) - tensor.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_serialize_scalar() {
        let tensor = Tensor::scalar(42.0);
        let bytes = serialize_tensor(&tensor).unwrap();
        let loaded = deserialize_tensor(&bytes).unwrap();
        assert!(loaded.is_scalar());
        assert!((loaded.get(0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_serialize_1d() {
        let tensor = Tensor::arange(0.0, 100.0, 1.0);
        let bytes = serialize_tensor(&tensor).unwrap();
        let loaded = deserialize_tensor(&bytes).unwrap();
        assert_eq!(loaded.shape(), &[100]);
        for i in 0..100 { assert!((loaded.get(i) - i as f64).abs() < 1e-10); }
    }

    #[test]
    fn test_serialize_empty_shape() {
        let tensor = Tensor::zeros(vec![]);
        let bytes = serialize_tensor(&tensor).unwrap();
        let loaded = deserialize_tensor(&bytes).unwrap();
        assert_eq!(loaded.shape(), &[]);
    }

    #[test]
    fn test_magic_header() {
        assert_eq!(MAGIC_HEADER.len(), 8);
        assert_eq!(&MAGIC_HEADER[0..5], b"BRAIN");
    }

    #[test]
    fn test_version_compatibility() {
        assert!(check_version(1, 1).is_ok());
        assert!(check_version(0, 1).is_ok());
        assert!(check_version(2, 1).is_err());
    }

    #[test]
    fn test_dtype_code_roundtrip() {
        let codes = vec![
            (DType::F64, DTYPE_F64),
            (DType::F32, DTYPE_F32),
            (DType::I64, DTYPE_I64),
            (DType::I32, DTYPE_I32),
            (DType::I16, DTYPE_I16),
            (DType::I8, DTYPE_I8),
            (DType::U8, DTYPE_U8),
        ];
        for (dtype, code) in codes {
            assert_eq!(dtype_to_code(dtype), code);
            assert_eq!(code_to_dtype(code).unwrap(), dtype);
        }
    }

    #[test]
    fn test_unknown_dtype_code() {
        assert!(code_to_dtype(255).is_err());
    }

    #[test]
    fn test_to_vec_from_vec() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let v = to_vec(&tensor);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
        let t2 = from_vec(v.clone(), vec![3]);
        assert_eq!(t2, tensor);
    }

    #[test]
    fn test_to_json() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let json = to_json(&tensor);
        assert!(json.contains("shape"));
        assert!(json.contains("f64"));
        assert!(json.contains("data"));
    }

    #[test]
    fn test_to_json_small() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let json = to_json(&tensor);
        assert!(json.contains("[1.000000, 2.000000, 3.000000]"));
    }

    #[test]
    fn test_from_json_roundtrip() {
        let tensor = Tensor::from_slice(&[1.5, 2.5, 3.5], vec![3]);
        let json = to_json(&tensor);
        let loaded = from_json(&json).unwrap();
        assert_eq!(loaded.shape(), vec![3]);
        for i in 0..3 { assert!((loaded.get(i) - tensor.get(i)).abs() < 1e-10); }
    }

    #[test]
    fn test_from_json_empty() {
        let json = r#"{"shape": [], "dtype": "f64", "data": [], "numel": 0}"#;
        let loaded = from_json(json).unwrap();
        assert_eq!(loaded.shape(), &[]);
    }

    #[test]
    fn test_estimate_file_size() {
        let tensor = Tensor::zeros(vec![100, 100]);
        let size = estimate_file_size(&tensor);
        assert!(size > 100 * 100 * 8);
        assert!(size < 100 * 100 * 8 + 100);
    }

    #[test]
    fn test_validate_header() {
        let tensor = Tensor::from_slice(&[1.0], vec![1]);
        let bytes = serialize_tensor(&tensor).unwrap();
        let (version, dtype, ndim, shape) = validate_header(&bytes).unwrap();
        assert_eq!(version, 1);
        assert_eq!(ndim, 1);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn test_validate_header_too_short() {
        assert!(validate_header(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_validate_header_bad_magic() {
        let bad_magic = b"BADHD\0\0\0\0".to_vec();
        let mut full = bad_magic;
        full.extend_from_slice(&[0u8; 100]);
        assert!(validate_header(&full).is_err());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("brain_test_save.bin");
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        save(&tensor, path.to_str().unwrap()).unwrap();
        let loaded = load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape(), vec![2, 3]);
        for i in 0..6 { assert!((loaded.get(i) - tensor.get(i)).abs() < 1e-10); }
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_npz() {
        let dir = std::env::temp_dir();
        let path = dir.join("brain_test.npz");
        let t1 = Tensor::ones(vec![3]);
        let t2 = Tensor::zeros(vec![2, 2]);
        save_npz(&[("tensor1".into(), t1), ("tensor2".into(), t2)], path.to_str().unwrap()).unwrap();
        let loaded = load_npz(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "tensor1");
        assert_eq!(loaded[1].0, "tensor2");
        assert_eq!(loaded[0].1.get(0), 1.0);
        assert_eq!(loaded[1].1.get(0), 0.0);
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_nonexistent() {
        let result = load("/nonexistent/path.bin");
        assert!(result.is_err());
    }
}
