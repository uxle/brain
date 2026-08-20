//! # Zero-Dependency HuggingFace / PyTorch Safetensors Reader & Writer
//!
//! Direct binary interop to load and save weights conforming to the Hugging Face `safetensors` standard.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use brain_core::Tensor;

/// Metadata information for a single tensor inside a safetensors archive.
#[derive(Debug, Clone)]
pub struct SafetensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Safetensors file representation.
#[derive(Debug, Clone, Default)]
pub struct SafetensorsArchive {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

impl SafetensorsArchive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.tensors.insert(name.into(), tensor);
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Loads a safetensors archive from a file path.
    pub fn load_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parses a safetensors archive from in-memory byte slice.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        if bytes.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Safetensors buffer too small for header length",
            ));
        }

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Safetensors buffer shorter than declared header length",
            ));
        }

        let header_json = std::str::from_utf8(&bytes[8..8 + header_len]).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid UTF-8 header: {}", e))
        })?;

        let buffer_start = 8 + header_len;
        let data_buffer = &bytes[buffer_start..];

        let mut archive = SafetensorsArchive::new();
        let parsed_entries = parse_safetensors_header_json(header_json)?;

        for (name, info) in parsed_entries {
            if name == "__metadata__" {
                continue;
            }
            let (start, end) = info.data_offsets;
            if end > data_buffer.len() || start > end {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid data offsets for tensor '{}': {:?}", name, info.data_offsets),
                ));
            }
            let raw_tensor_bytes = &data_buffer[start..end];
            let numel: usize = info.shape.iter().product();

            let data: Vec<f64> = match info.dtype.as_str() {
                "F32" => {
                    let mut vec = Vec::with_capacity(numel);
                    for chunk in raw_tensor_bytes.chunks_exact(4) {
                        let val = f32::from_le_bytes(chunk.try_into().unwrap()) as f64;
                        vec.push(val);
                    }
                    vec
                }
                "F64" => {
                    let mut vec = Vec::with_capacity(numel);
                    for chunk in raw_tensor_bytes.chunks_exact(8) {
                        let val = f64::from_le_bytes(chunk.try_into().unwrap());
                        vec.push(val);
                    }
                    vec
                }
                "I32" => {
                    let mut vec = Vec::with_capacity(numel);
                    for chunk in raw_tensor_bytes.chunks_exact(4) {
                        let val = i32::from_le_bytes(chunk.try_into().unwrap()) as f64;
                        vec.push(val);
                    }
                    vec
                }
                "I64" => {
                    let mut vec = Vec::with_capacity(numel);
                    for chunk in raw_tensor_bytes.chunks_exact(8) {
                        let val = i64::from_le_bytes(chunk.try_into().unwrap()) as f64;
                        vec.push(val);
                    }
                    vec
                }
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported safetensors dtype: {}", info.dtype),
                    ));
                }
            };

            let tensor = Tensor::from_vec(data, info.shape);
            archive.insert(name, tensor);
        }

        Ok(archive)
    }

    /// Serializes the archive into valid safetensors bytes (with F32 payload).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header_json = String::from("{");
        let mut data_buffer = Vec::new();
        let mut current_offset = 0usize;

        let mut sorted_keys: Vec<&String> = self.tensors.keys().collect();
        sorted_keys.sort();

        for (i, name) in sorted_keys.iter().enumerate() {
            let tensor = &self.tensors[*name];
            let shape_str: Vec<String> = tensor.shape().iter().map(|d| d.to_string()).collect();
            let start = current_offset;
            let byte_len = tensor.numel() * 4; // F32 = 4 bytes
            let end = start + byte_len;

            // Serialize tensor data as f32 little-endian
            for &val in tensor.data() {
                data_buffer.extend_from_slice(&(val as f32).to_le_bytes());
            }
            current_offset = end;

            if i > 0 {
                header_json.push(',');
            }
            header_json.push_str(&format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                shape_str.join(","),
                start,
                end
            ));
        }

        header_json.push('}');
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut output = Vec::with_capacity(8 + header_bytes.len() + data_buffer.len());
        output.extend_from_slice(&header_len.to_le_bytes());
        output.extend_from_slice(header_bytes);
        output.extend_from_slice(&data_buffer);

        output
    }

    /// Saves the archive to a `.safetensors` file path.
    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let bytes = self.to_bytes();
        file.write_all(&bytes)?;
        Ok(())
    }
}

/// Minimal zero-dependency parser for safetensors header JSON.
fn parse_safetensors_header_json(json: &str) -> std::io::Result<HashMap<String, SafetensorInfo>> {
    let mut map = HashMap::new();
    let trimmed = json.trim().trim_start_matches('{').trim_end_matches('}');

    for entry in split_top_level_entries(trimmed) {
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = parts[0].trim().trim_matches('"');
        let body = parts[1].trim();

        if key == "__metadata__" {
            continue;
        }

        let mut dtype = "F32".to_string();
        let mut shape = Vec::new();
        let mut offsets = (0, 0);

        // Parse dtype
        if let Some(pos) = body.find("\"dtype\":") {
            let rest = &body[pos + 8..];
            if let Some(q1) = rest.find('"') {
                if let Some(q2) = rest[q1 + 1..].find('"') {
                    dtype = rest[q1 + 1..q1 + 1 + q2].to_string();
                }
            }
        }

        // Parse shape
        if let Some(pos) = body.find("\"shape\":") {
            let rest = &body[pos + 8..];
            if let (Some(s), Some(e)) = (rest.find('['), rest.find(']')) {
                let dims_str = &rest[s + 1..e];
                shape = dims_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
            }
        }

        // Parse data_offsets
        if let Some(pos) = body.find("\"data_offsets\":") {
            let rest = &body[pos + 15..];
            if let (Some(s), Some(e)) = (rest.find('['), rest.find(']')) {
                let off_str = &rest[s + 1..e];
                let nums: Vec<usize> = off_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
                if nums.len() == 2 {
                    offsets = (nums[0], nums[1]);
                }
            }
        }

        map.insert(key.to_string(), SafetensorInfo {
            dtype,
            shape,
            data_offsets: offsets,
        });
    }

    Ok(map)
}

fn split_top_level_entries(json: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_str = false;

    for ch in json.chars() {
        match ch {
            '"' => {
                in_str = !in_str;
                current.push(ch);
            }
            '{' | '[' if !in_str => {
                depth += 1;
                current.push(ch);
            }
            '}' | ']' if !in_str => {
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_str && depth == 0 => {
                if !current.trim().is_empty() {
                    entries.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        entries.push(current.trim().to_string());
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_roundtrip() {
        let mut archive = SafetensorsArchive::new();
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        archive.insert("weight", t1.clone());
        archive.insert("bias", t2.clone());

        let bytes = archive.to_bytes();
        let restored = SafetensorsArchive::from_bytes(&bytes).unwrap();

        assert_eq!(restored.get("weight").unwrap().shape(), &[2, 2]);
        assert_eq!(restored.get("weight").unwrap().to_vec(), t1.to_vec());

        assert_eq!(restored.get("bias").unwrap().shape(), &[3]);
        assert_eq!(restored.get("bias").unwrap().to_vec(), t2.to_vec());
    }
}
