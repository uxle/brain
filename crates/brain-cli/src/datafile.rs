//! # Dataset loading for the CLI
//!
//! A tiny CSV/text loader. Each non-empty, non-`#` line is one sample; columns
//! are separated by commas or whitespace and parsed as `f64`. When a dataset is
//! *labeled*, the last column is an integer class label and the preceding
//! columns are the feature vector.

use brain_core::Tensor;

/// A loaded dataset: feature rows plus optional integer labels.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// One feature vector per sample.
    pub features: Vec<Vec<f64>>,
    /// Class label per sample (empty when the dataset is unlabeled).
    pub labels: Vec<usize>,
    /// Number of features per sample.
    pub n_features: usize,
}

impl Dataset {
    /// Stacks the feature rows into a `[n, n_features]` tensor.
    pub fn feature_matrix(&self) -> Tensor {
        let n = self.features.len();
        let mut data = Vec::with_capacity(n * self.n_features);
        for row in &self.features {
            data.extend_from_slice(row);
        }
        Tensor::from_vec(data, vec![n, self.n_features])
    }

    /// Parses a single sample row (whitespace/comma separated f64).
    pub fn parse_sample(text: &str) -> Result<Vec<f64>, String> {
        let tokens: Vec<&str> = text
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(|t| t.trim())
            .collect();
        let mut vals = Vec::with_capacity(tokens.len());
        for tok in tokens {
            if tok.is_empty() {
                continue;
            }
            vals.push(
                tok.parse::<f64>()
                    .map_err(|err| format!("failed to parse '{}' as number: {}", tok, err))?,
            );
        }
        Ok(vals)
    }
}

/// Loads a dataset from a text file. `labeled` controls whether the final
/// column of each row is interpreted as an integer class label.
pub fn load(path: &str, labeled: bool) -> Result<Dataset, String> {
    let text = std::fs::read_to_string(std::path::Path::new(path))
        .map_err(|err| format!("could not read dataset '{}': {}", path, err))?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    let mut n_features = 0usize;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let vals = Dataset::parse_sample(line)?;
        if vals.is_empty() {
            continue;
        }
        if labeled {
            if vals.len() < 2 {
                return Err(format!("line {}: labeled datasets need at least 2 columns", lineno + 1));
            }
            let label_val = vals[vals.len() - 1];
            if label_val.fract() != 0.0 {
                return Err(format!("line {}: label column must be an integer, got {}", lineno + 1, label_val));
            }
            let label = label_val as usize;
            let row = vals[..vals.len() - 1].to_vec();
            if n_features == 0 {
                n_features = row.len();
            } else if row.len() != n_features {
                return Err(format!("line {}: expected {} features, got {}", lineno + 1, n_features, row.len()));
            }
            features.push(row);
            labels.push(label);
        } else {
            if n_features == 0 {
                n_features = vals.len();
            } else if vals.len() != n_features {
                return Err(format!("line {}: expected {} features, got {}", lineno + 1, n_features, vals.len()));
            }
            features.push(vals);
        }
    }

    if features.is_empty() {
        return Err(format!("dataset '{}' contains no samples", path));
    }

    Ok(Dataset {
        n_features,
        features,
        labels,
    })
}
