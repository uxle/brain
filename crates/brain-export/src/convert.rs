//! # Format-to-Format Model Conversion
//!
//! Inter-format conversion routines between Brain Model (`.bn`) and HuggingFace (`.safetensors`).

use crate::safetensors::SafetensorsArchive;
use brain_core::BrainModelFile;

/// Conversion report summarizing graph transformations.
#[derive(Debug, Clone, Default)]
pub struct ConversionReport {
    pub num_tensors_converted: usize,
    pub format_from: String,
    pub format_to: String,
}

impl ConversionReport {
    /// Creates a new `ConversionReport`.
    pub fn new(num_tensors: usize, format_from: &str, format_to: &str) -> Self {
        Self {
            num_tensors_converted: num_tensors,
            format_from: format_from.to_string(),
            format_to: format_to.to_string(),
        }
    }
}

/// Converts a `BrainModelFile` (.bn) container into a HuggingFace `SafetensorsArchive`.
pub fn convert_bn_to_safetensors(model: &BrainModelFile) -> (SafetensorsArchive, ConversionReport) {
    let mut archive = SafetensorsArchive::new();
    let mut count = 0;

    for (name, tensor) in model.archive.iter() {
        archive.insert(name, tensor.clone());
        count += 1;
    }

    for (k, v) in &model.metadata {
        archive.metadata.insert(k.clone(), v.clone());
    }

    let report = ConversionReport::new(count, "brain.bn", "safetensors");
    (archive, report)
}

/// Converts a HuggingFace `SafetensorsArchive` into a `BrainModelFile` (.bn).
pub fn convert_safetensors_to_bn(
    archive: &SafetensorsArchive,
    model_name: &str,
) -> (BrainModelFile, ConversionReport) {
    let mut model = BrainModelFile::new(model_name);
    let mut count = 0;

    for (name, tensor) in &archive.tensors {
        model.add_tensor(name, tensor.clone(), None);
        count += 1;
    }

    for (k, v) in &archive.metadata {
        model = model.with_meta(k, v);
    }

    let report = ConversionReport::new(count, "safetensors", "brain.bn");
    (model, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_bn_to_safetensors_conversion() {
        let mut model = BrainModelFile::new("transformer_block")
            .with_meta("author", "lion")
            .with_meta("layers", "12");

        let w_attn = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        model.add_tensor("attn.weight", w_attn, None);

        let (archive, report) = convert_bn_to_safetensors(&model);
        assert_eq!(report.num_tensors_converted, 1);
        assert_eq!(report.format_from, "brain.bn");
        assert_eq!(report.format_to, "safetensors");

        let converted_tensor = archive.get("attn.weight").unwrap();
        assert_eq!(converted_tensor.data(), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(archive.metadata.get("author").unwrap(), "lion");

        let (restored_model, _) = convert_safetensors_to_bn(&archive, "restored_transformer");
        assert_eq!(restored_model.name, "restored_transformer");
        assert_eq!(
            restored_model.archive.get("attn.weight").unwrap().data(),
            &[0.1, 0.2, 0.3, 0.4]
        );
    }
}
