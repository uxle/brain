//! # Model Export Utilities for brain-vit
//!
//! Supports:
//! - [`JsonCheckpoint`] — save/load model weights as JSON-like key:value text
//! - [`OnnxStub`] — minimal ONNX stub describing the model graph structure
//! - [`TorchScriptStub`] — TorchScript-compatible description of the model
//! - [`ModelCard`] — structured model metadata card

use std::collections::HashMap;
use std::fmt;
use crate::core::{VitError, VitResult};

/// JSON-style checkpoint format (text-based, portable).
///
/// Format: one `key:val1,val2,...` per line, with metadata header.
pub struct JsonCheckpoint {
    /// Model weights.
    pub weights: HashMap<String, Vec<f64>>,
    /// Version string.
    pub version: String,
    /// Model name.
    pub model_name: String,
}

impl JsonCheckpoint {
    /// Create a new checkpoint.
    pub fn new(model_name: &str, weights: HashMap<String, Vec<f64>>) -> Self {
        Self {
            weights,
            version: "brain-vit-1.0".to_string(),
            model_name: model_name.to_string(),
        }
    }

    /// Serialize to a portable text format.
    ///
    /// Header: `# brain-vit checkpoint: <name> v<version>\n`
    /// Then one `key:v1,v2,...` per line.
    pub fn serialize(&self) -> String {
        let mut lines = vec![
            format!("# brain-vit checkpoint: {} v{}", self.model_name, self.version),
        ];
        let mut keys: Vec<&String> = self.weights.keys().collect();
        keys.sort();
        for key in keys {
            let vals: Vec<String> = self.weights[key].iter().map(|v| format!("{:.10e}", v)).collect();
            lines.push(format!("{}:{}", key, vals.join(",")));
        }
        lines.join("\n") + "\n"
    }

    /// Deserialize from the text format.
    pub fn deserialize(text: &str) -> VitResult<Self> {
        let mut weights = HashMap::new();
        let mut model_name = String::new();
        let mut version = String::new();

        for line in text.lines() {
            if line.starts_with("# brain-vit checkpoint:") {
                let rest = line.trim_start_matches("# brain-vit checkpoint:").trim();
                let parts: Vec<&str> = rest.splitn(2, " v").collect();
                if !parts.is_empty() { model_name = parts[0].trim().to_string(); }
                if parts.len() > 1 { version = parts[1].trim().to_string(); }
                continue;
            }
            if line.starts_with('#') || line.is_empty() { continue; }
            let colon = line.find(':').ok_or_else(|| VitError::Checkpoint(
                format!("Malformed line (no colon): {}", &line[..line.len().min(40)])
            ))?;
            let key = line[..colon].to_string();
            let vals_str = &line[colon + 1..];
            let vals: Result<Vec<f64>, _> = vals_str.split(',').map(|s| s.parse::<f64>()).collect();
            let vals = vals.map_err(|e| VitError::Checkpoint(format!("Parse error in key '{}': {}", key, e)))?;
            weights.insert(key, vals);
        }

        Ok(Self { weights, version, model_name })
    }

    /// Number of parameters total.
    pub fn total_params(&self) -> usize { self.weights.values().map(|v| v.len()).sum() }

    /// List all keys.
    pub fn keys(&self) -> Vec<&String> {
        let mut ks: Vec<&String> = self.weights.keys().collect();
        ks.sort();
        ks
    }

    /// Merge another checkpoint (overwrite conflicting keys).
    pub fn merge(&mut self, other: &JsonCheckpoint) {
        for (k, v) in &other.weights {
            self.weights.insert(k.clone(), v.clone());
        }
    }
}

/// ONNX stub: textual representation of the model graph.
///
/// Not a real ONNX file, but a human-readable description suitable for
/// feeding into external ONNX-generating tools.
#[derive(Debug, Clone)]
pub struct OnnxStub {
    /// Model name.
    pub name: String,
    /// Input shape `[batch, channels, height, width]`.
    pub input_shape: Vec<Option<usize>>,
    /// Output shape `[batch, num_classes]`.
    pub output_shape: Vec<Option<usize>>,
    /// Layer descriptions.
    pub layers: Vec<String>,
    /// OpSet version.
    pub opset: usize,
}

impl OnnxStub {
    /// Create an ONNX stub from basic model info.
    pub fn from_vit(
        name: &str,
        image_size: usize,
        in_channels: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
    ) -> Self {
        let mut layers = vec![];
        layers.push(format!("PatchEmbedding: Conv2d(in={}, out={}, ks=16, stride=16)", in_channels, embed_dim));
        layers.push(format!("CLSTokenPrepend: embed_dim={}", embed_dim));
        layers.push(format!("PositionalEncoding: learnable [{}+1, {}]", (image_size/16).pow(2), embed_dim));
        for d in 0..depth {
            layers.push(format!("TransformerBlock({}): MHSA(heads={}) + FFN(dim={})", d, num_heads, embed_dim * 4));
        }
        layers.push(format!("LayerNorm: dim={}", embed_dim));
        layers.push(format!("Linear: [{}, {}] (head)", embed_dim, num_classes));

        Self {
            name: name.to_string(),
            input_shape: vec![None, Some(in_channels), Some(image_size), Some(image_size)],
            output_shape: vec![None, Some(num_classes)],
            layers,
            opset: 17,
        }
    }

    /// Render as a human-readable text description.
    pub fn describe(&self) -> String {
        let mut lines = vec![
            format!("ONNX Model Stub: {} (opset={})", self.name, self.opset),
            format!("Input: {:?}", self.input_shape),
            format!("Output: {:?}", self.output_shape),
            "Layers:".to_string(),
        ];
        for l in &self.layers { lines.push(format!("  {}", l)); }
        lines.join("\n")
    }

    /// Total number of layer descriptions.
    pub fn num_layers(&self) -> usize { self.layers.len() }
}

impl fmt::Display for OnnxStub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// Model card for documentation and reproducibility.
#[derive(Debug, Clone, Default)]
pub struct ModelCard {
    /// Model name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Architecture summary.
    pub architecture: String,
    /// Dataset info.
    pub dataset: String,
    /// Reported metrics.
    pub metrics: Vec<(String, f64)>,
    /// Limitations.
    pub limitations: Vec<String>,
    /// License.
    pub license: String,
    /// Training configuration notes.
    pub training_notes: String,
}

impl ModelCard {
    /// Create a model card for a ViT model.
    pub fn for_vit(
        name: &str,
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Vision Transformer ({}) for image classification", name),
            architecture: format!(
                "ViT-{}: image_size={}, patch_size={}, embed_dim={}, depth={}, heads={}, classes={}",
                name, image_size, patch_size, embed_dim, depth, num_heads, num_classes
            ),
            dataset: "ImageNet-1k (proxy)".to_string(),
            metrics: vec![],
            limitations: vec!["Requires square images".to_string(), "No data augmentation built-in".to_string()],
            license: "Apache-2.0".to_string(),
            training_notes: String::new(),
        }
    }

    /// Add a metric to the card.
    pub fn add_metric(&mut self, name: &str, value: f64) -> &mut Self {
        self.metrics.push((name.to_string(), value));
        self
    }

    /// Render card as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut lines = vec![
            format!("# Model Card: {}", self.name),
            String::new(),
            format!("**Description**: {}", self.description),
            String::new(),
            format!("**Architecture**: {}", self.architecture),
            String::new(),
            format!("**Dataset**: {}", self.dataset),
            String::new(),
            format!("**License**: {}", self.license),
        ];
        if !self.metrics.is_empty() {
            lines.push(String::new());
            lines.push("## Metrics".to_string());
            for (name, val) in &self.metrics {
                lines.push(format!("- **{}**: {:.4}", name, val));
            }
        }
        if !self.limitations.is_empty() {
            lines.push(String::new());
            lines.push("## Limitations".to_string());
            for lim in &self.limitations {
                lines.push(format!("- {}", lim));
            }
        }
        if !self.training_notes.is_empty() {
            lines.push(String::new());
            lines.push("## Training Notes".to_string());
            lines.push(self.training_notes.clone());
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> HashMap<String, Vec<f64>> {
        let mut m = HashMap::new();
        m.insert("weight".to_string(), vec![1.0, 2.0, 3.0]);
        m.insert("bias".to_string(), vec![0.1, 0.2]);
        m
    }

    #[test]
    fn test_checkpoint_new() {
        let ckpt = JsonCheckpoint::new("test_model", sample_weights());
        assert_eq!(ckpt.model_name, "test_model");
        assert_eq!(ckpt.total_params(), 5);
    }

    #[test]
    fn test_checkpoint_serialize_header() {
        let ckpt = JsonCheckpoint::new("vit_tiny", sample_weights());
        let s = ckpt.serialize();
        assert!(s.contains("# brain-vit checkpoint: vit_tiny"));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let original = JsonCheckpoint::new("vit", sample_weights());
        let text = original.serialize();
        let loaded = JsonCheckpoint::deserialize(&text).unwrap();
        assert_eq!(loaded.model_name, "vit");
        let w = loaded.weights.get("weight").unwrap();
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!((w[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_keys() {
        let ckpt = JsonCheckpoint::new("m", sample_weights());
        let keys = ckpt.keys();
        assert!(keys.contains(&&"weight".to_string()));
        assert!(keys.contains(&&"bias".to_string()));
    }

    #[test]
    fn test_checkpoint_malformed_line() {
        let bad = "no_colon_here\n";
        assert!(JsonCheckpoint::deserialize(bad).is_err());
    }

    #[test]
    fn test_checkpoint_empty_weights() {
        let ckpt = JsonCheckpoint::new("empty", HashMap::new());
        let text = ckpt.serialize();
        let loaded = JsonCheckpoint::deserialize(&text).unwrap();
        assert_eq!(loaded.total_params(), 0);
    }

    #[test]
    fn test_checkpoint_merge() {
        let mut c1 = JsonCheckpoint::new("m1", sample_weights());
        let mut extra = HashMap::new();
        extra.insert("new_key".to_string(), vec![9.0]);
        let c2 = JsonCheckpoint::new("m2", extra);
        c1.merge(&c2);
        assert!(c1.weights.contains_key("new_key"));
        assert!(c1.weights.contains_key("weight"));
    }

    #[test]
    fn test_onnx_stub_from_vit() {
        let stub = OnnxStub::from_vit("ViT-B", 224, 3, 1000, 768, 12, 12);
        assert_eq!(stub.name, "ViT-B");
        assert_eq!(stub.opset, 17);
        assert!(stub.num_layers() > 0);
    }

    #[test]
    fn test_onnx_stub_describe() {
        let stub = OnnxStub::from_vit("ViT-T", 32, 1, 4, 16, 2, 2);
        let desc = stub.describe();
        assert!(desc.contains("ViT-T"));
        assert!(desc.contains("Layers:"));
    }

    #[test]
    fn test_onnx_stub_display() {
        let stub = OnnxStub::from_vit("ViT-S", 64, 3, 10, 384, 6, 6);
        let s = format!("{}", stub);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_onnx_stub_layers_depth() {
        let depth = 6;
        let stub = OnnxStub::from_vit("ViT-S", 224, 3, 1000, 384, depth, 6);
        // Should have: 1 patch + 1 cls + 1 pos + depth blocks + 1 LN + 1 head = depth + 5
        assert_eq!(stub.num_layers(), depth + 5);
    }

    #[test]
    fn test_model_card_for_vit() {
        let card = ModelCard::for_vit("ViT-B/16", 224, 16, 768, 12, 12, 1000);
        assert_eq!(card.name, "ViT-B/16");
        assert!(!card.architecture.is_empty());
    }

    #[test]
    fn test_model_card_add_metric() {
        let mut card = ModelCard::for_vit("ViT-T", 32, 4, 16, 2, 2, 4);
        card.add_metric("top1_acc", 0.812);
        assert_eq!(card.metrics.len(), 1);
    }

    #[test]
    fn test_model_card_to_markdown() {
        let mut card = ModelCard::for_vit("ViT-B", 224, 16, 768, 12, 12, 1000);
        card.add_metric("top1_acc", 0.812);
        let md = card.to_markdown();
        assert!(md.contains("# Model Card"));
        assert!(md.contains("top1_acc"));
    }

    #[test]
    fn test_model_card_default() {
        let card = ModelCard::default();
        assert!(card.name.is_empty());
        assert!(card.metrics.is_empty());
    }

    #[test]
    fn test_model_card_limitations() {
        let card = ModelCard::for_vit("ViT-B", 224, 16, 768, 12, 12, 1000);
        assert!(!card.limitations.is_empty());
    }

    #[test]
    fn test_checkpoint_version() {
        let ckpt = JsonCheckpoint::new("m", sample_weights());
        let text = ckpt.serialize();
        let loaded = JsonCheckpoint::deserialize(&text).unwrap();
        assert!(!loaded.version.is_empty());
    }

    #[test]
    fn test_checkpoint_large_values() {
        let mut m = HashMap::new();
        m.insert("w".to_string(), vec![f64::MAX * 0.5, f64::MIN_POSITIVE, -1e-300]);
        let ckpt = JsonCheckpoint::new("big", m);
        let text = ckpt.serialize();
        let loaded = JsonCheckpoint::deserialize(&text).unwrap();
        let w = loaded.weights.get("w").unwrap();
        assert_eq!(w.len(), 3);
        assert!(w[0].is_finite());
    }

    #[test]
    fn test_onnx_stub_input_shape() {
        let stub = OnnxStub::from_vit("ViT-B", 224, 3, 1000, 768, 12, 12);
        assert_eq!(stub.input_shape[1], Some(3)); // in_channels
        assert_eq!(stub.input_shape[2], Some(224)); // H
    }

    #[test]
    fn test_onnx_stub_output_shape() {
        let stub = OnnxStub::from_vit("ViT-B", 224, 3, 1000, 768, 12, 12);
        assert_eq!(stub.output_shape[1], Some(1000));
    }
}
