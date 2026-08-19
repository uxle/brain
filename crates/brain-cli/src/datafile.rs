//! Dataset loading for the Brain CLI.
//!
//! Supported formats:
//!
//! Classification:
//!     1.0,2.0,0
//!     3.0,4.0,1
//!
//! Regression:
//!     1.0,1.0,2.0
//!     2.0,1.0,3.0
//!
//! In regression mode the final column is a floating-point target.

use brain_core::Tensor;

#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature vector for every sample.
    pub features: Vec<Vec<f64>>,

    /// Integer labels for classification.
    pub labels: Vec<usize>,

    /// Floating-point targets for regression.
    pub targets: Vec<f64>,

    /// Number of input features.
    pub n_features: usize,
}

impl Dataset {
    pub fn feature_matrix(&self) -> Tensor {
        let n = self.features.len();

        let mut data =
            Vec::with_capacity(n * self.n_features);

        for row in &self.features {
            data.extend_from_slice(row);
        }

        Tensor::from_vec(
            data,
            vec![n, self.n_features],
        )
    }

    pub fn parse_sample(text: &str) -> Result<Vec<f64>, String> {
        let tokens = text
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(str::trim);

        let mut values = Vec::new();

        for token in tokens {
            if token.is_empty() {
                continue;
            }

            values.push(
                token
                    .parse::<f64>()
                    .map_err(|err| {
                        format!(
                            "failed to parse '{}' as number: {}",
                            token, err
                        )
                    })?,
            );
        }

        Ok(values)
    }
}

/// Load classification dataset.
///
/// Last column = integer class.
/// Previous columns = features.
pub fn load(
    path: &str,
    labeled: bool,
) -> Result<Dataset, String> {
    load_task(path, if labeled {
        DatasetTask::Classification
    } else {
        DatasetTask::Unlabeled
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetTask {
    Classification,
    Regression,
    Unlabeled,
}

/// Load dataset according to task.
pub fn load_task(
    path: &str,
    task: DatasetTask,
) -> Result<Dataset, String> {
    let text = std::fs::read_to_string(
        std::path::Path::new(path),
    )
    .map_err(|err| {
        format!(
            "could not read dataset '{}': {}",
            path, err
        )
    })?;

    let mut features = Vec::<Vec<f64>>::new();
    let mut labels = Vec::<usize>::new();
    let mut targets = Vec::<f64>::new();

    let mut n_features = 0usize;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let values = Dataset::parse_sample(line)?;

        if values.is_empty() {
            continue;
        }

        match task {
            DatasetTask::Classification => {
                if values.len() < 2 {
                    return Err(format!(
                        "line {}: classification dataset needs at least 2 columns",
                        lineno + 1
                    ));
                }

                let label_value =
                    values[values.len() - 1];

                if !label_value.is_finite()
                    || label_value.fract() != 0.0
                    || label_value < 0.0
                {
                    return Err(format!(
                        "line {}: label must be a non-negative integer, got {}",
                        lineno + 1,
                        label_value
                    ));
                }

                let row =
                    values[..values.len() - 1].to_vec();

                let label = label_value as usize;

                validate_feature_count(
                    &mut n_features,
                    row.len(),
                    lineno + 1,
                )?;

                features.push(row);
                labels.push(label);
            }

            DatasetTask::Regression => {
                if values.len() < 2 {
                    return Err(format!(
                        "line {}: regression dataset needs at least 2 columns",
                        lineno + 1
                    ));
                }

                let target =
                    values[values.len() - 1];

                if !target.is_finite() {
                    return Err(format!(
                        "line {}: target must be finite, got {}",
                        lineno + 1,
                        target
                    ));
                }

                let row =
                    values[..values.len() - 1].to_vec();

                validate_feature_count(
                    &mut n_features,
                    row.len(),
                    lineno + 1,
                )?;

                features.push(row);
                targets.push(target);
            }

            DatasetTask::Unlabeled => {
                validate_feature_count(
                    &mut n_features,
                    values.len(),
                    lineno + 1,
                )?;

                features.push(values);
            }
        }
    }

    if features.is_empty() {
        return Err(format!(
            "dataset '{}' contains no samples",
            path
        ));
    }

    Ok(Dataset {
        features,
        labels,
        targets,
        n_features,
    })
}

fn validate_feature_count(
    n_features: &mut usize,
    count: usize,
    line: usize,
) -> Result<(), String> {
    if *n_features == 0 {
        *n_features = count;
        return Ok(());
    }

    if count != *n_features {
        return Err(format!(
            "line {}: expected {} features, got {}",
            line,
            *n_features,
            count
        ));
    }

    Ok(())
}