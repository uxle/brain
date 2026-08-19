//! # Utilities for brain-vit
//!
//! Shared helper functions: 2D interpolation, attention rollout,
//! image cropping/resizing, numerical utilities.

use crate::core::{VitError, VitResult, Tensor2D};

/// Bilinear interpolation of a 2D grid to a new size.
///
/// - `grid`: flat `[H, W]` values.
/// - Returns flat `[new_h, new_w]` values.
///
/// # Example
/// ```rust
/// use brain_vit::utils::interpolate_2d;
/// let grid = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
/// let out = interpolate_2d(&grid, 2, 2, 4, 4).unwrap();
/// assert_eq!(out.len(), 16);
/// ```
pub fn interpolate_2d(
    grid: &[f64],
    h: usize,
    w: usize,
    new_h: usize,
    new_w: usize,
) -> VitResult<Vec<f64>> {
    if grid.len() != h * w {
        return Err(VitError::Shape(format!(
            "interpolate_2d: grid len {} != h*w={}", grid.len(), h * w
        )));
    }
    if new_h == 0 || new_w == 0 {
        return Err(VitError::Shape("interpolate_2d: new dimensions must be > 0".to_string()));
    }
    let mut out = vec![0.0f64; new_h * new_w];
    for ny in 0..new_h {
        for nx in 0..new_w {
            let fy = ny as f64 * (h as f64 - 1.0) / (new_h as f64 - 1.0).max(1.0);
            let fx = nx as f64 * (w as f64 - 1.0) / (new_w as f64 - 1.0).max(1.0);
            let y0 = fy.floor() as usize;
            let x0 = fx.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let ty = fy - y0 as f64;
            let tx = fx - x0 as f64;
            let v00 = grid[y0 * w + x0];
            let v01 = grid[y0 * w + x1];
            let v10 = grid[y1 * w + x0];
            let v11 = grid[y1 * w + x1];
            out[ny * new_w + nx] =
                v00 * (1.0 - ty) * (1.0 - tx)
                + v01 * (1.0 - ty) * tx
                + v10 * ty * (1.0 - tx)
                + v11 * ty * tx;
        }
    }
    Ok(out)
}

/// Simulate a simple center crop + resize of an image patch.
///
/// Crops `[crop_h, crop_w]` from the center of a `[H, W]` image,
/// then resizes to `[out_h, out_w]` using bilinear interpolation.
pub fn crop_resize(
    image: &[f64],
    h: usize,
    w: usize,
    crop_h: usize,
    crop_w: usize,
    out_h: usize,
    out_w: usize,
) -> VitResult<Vec<f64>> {
    if h < crop_h || w < crop_w {
        return Err(VitError::Config(format!(
            "crop_resize: image {}x{} smaller than crop {}x{}", h, w, crop_h, crop_w
        )));
    }
    let y0 = (h - crop_h) / 2;
    let x0 = (w - crop_w) / 2;
    let mut cropped = vec![0.0f64; crop_h * crop_w];
    for cy in 0..crop_h {
        for cx in 0..crop_w {
            cropped[cy * crop_w + cx] = image[(y0 + cy) * w + (x0 + cx)];
        }
    }
    interpolate_2d(&cropped, crop_h, crop_w, out_h, out_w)
}

/// Attention rollout: aggregate multi-layer attention maps.
///
/// Given per-layer attention matrices `attns: Vec<[N, N]>` flat,
/// compute the rollout (product of residual + attn for each layer).
///
/// # Arguments
/// - `attns`: attention maps per layer, each `[N, N]` flat (all in one Vec).
/// - `n`: sequence length N (including CLS).
/// - `num_layers`: number of layers.
///
/// # Returns
/// - `[N, N]` rollout matrix.
pub fn attention_rollout(
    attns: &[f64],
    n: usize,
    num_layers: usize,
) -> VitResult<Vec<f64>> {
    let expected = num_layers * n * n;
    if attns.len() != expected {
        return Err(VitError::Shape(format!(
            "attention_rollout: expected {} elements, got {}", expected, attns.len()
        )));
    }
    if num_layers == 0 || n == 0 {
        return Ok(vec![]);
    }

    // Identity matrix
    let identity: Vec<f64> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
    let mut result = identity.clone();

    for layer in 0..num_layers {
        let layer_attn = &attns[layer * n * n..(layer + 1) * n * n];
        // A_hat = 0.5 * attn + 0.5 * I (residual connection approximation)
        let a_hat: Vec<f64> = layer_attn.iter().enumerate().map(|(i, &a)| {
            a * 0.5 + if i / n == i % n { 0.5 } else { 0.0 }
        }).collect();
        // result = a_hat @ result (matrix multiply)
        let a_hat_mat = Tensor2D::from_data(n, n, a_hat)?;
        let result_mat = Tensor2D::from_data(n, n, result)?;
        result = a_hat_mat.matmul(&result_mat)?.data;
    }
    Ok(result)
}

/// Compute per-row entropy of an attention matrix `[N, N]`.
pub fn attention_entropy(attn: &[f64], n: usize) -> VitResult<Vec<f64>> {
    if attn.len() != n * n {
        return Err(VitError::Shape("attention_entropy: shape mismatch".to_string()));
    }
    let mut entropies = vec![0.0f64; n];
    for r in 0..n {
        let mut h = 0.0f64;
        for c in 0..n {
            let p = attn[r * n + c];
            if p > 1e-12 {
                h -= p * p.ln();
            }
        }
        entropies[r] = h;
    }
    Ok(entropies)
}

/// Compute top-k accuracy given logits `[B, C]` and labels `[B]`.
pub fn top_k_accuracy(
    logits: &[f64],
    labels: &[usize],
    batch: usize,
    num_classes: usize,
    k: usize,
) -> VitResult<f64> {
    if logits.len() != batch * num_classes {
        return Err(VitError::Shape("top_k_accuracy: logits shape mismatch".to_string()));
    }
    if labels.len() != batch {
        return Err(VitError::Shape("top_k_accuracy: labels length mismatch".to_string()));
    }
    let mut correct = 0usize;
    for b in 0..batch {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let mut indexed: Vec<(f64, usize)> = row.iter().copied().enumerate()
            .map(|(i, v)| (v, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<usize> = indexed.iter().take(k).map(|&(_, i)| i).collect();
        if top_k.contains(&labels[b]) { correct += 1; }
    }
    Ok(correct as f64 / batch as f64)
}

/// Softmax over last dimension for a flat `[B, C]` logits array.
pub fn softmax_logits(logits: &[f64], batch: usize, num_classes: usize) -> VitResult<Vec<f64>> {
    if logits.len() != batch * num_classes {
        return Err(VitError::Shape("softmax_logits: shape mismatch".to_string()));
    }
    let mut out = logits.to_vec();
    for b in 0..batch {
        let start = b * num_classes;
        let slice = &mut out[start..start + num_classes];
        let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = slice.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for (i, &e) in exps.iter().enumerate() {
            slice[i] = e / sum;
        }
    }
    Ok(out)
}

/// Generate sinusoidal position encoding for a sequence.
///
/// Returns `[seq_len, embed_dim]` flat, following the original Transformer paper.
pub fn sinusoidal_encoding(seq_len: usize, embed_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; seq_len * embed_dim];
    let _pi = std::f64::consts::PI;
    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle = pos as f64 / 10000f64.powf(2.0 * (i / 2) as f64 / embed_dim as f64);
            out[pos * embed_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    out
}

/// Compute mean squared error loss.
pub fn mse_loss(pred: &[f64], target: &[f64]) -> VitResult<f64> {
    if pred.len() != target.len() {
        return Err(VitError::Shape("mse_loss: length mismatch".to_string()));
    }
    if pred.is_empty() {
        return Ok(0.0);
    }
    let loss = pred.iter().zip(target.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>() / pred.len() as f64;
    Ok(loss)
}

/// Compute cross-entropy loss from logits (log-sum-exp stable).
pub fn cross_entropy_loss(logits: &[f64], labels: &[usize], batch: usize, num_classes: usize) -> VitResult<f64> {
    if logits.len() != batch * num_classes {
        return Err(VitError::Shape("cross_entropy_loss: logits shape mismatch".to_string()));
    }
    let mut total = 0.0f64;
    for b in 0..batch {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max_val + row.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
        let label = labels[b];
        if label >= num_classes {
            return Err(VitError::Shape(format!("cross_entropy_loss: label {} >= num_classes {}", label, num_classes)));
        }
        total += log_sum_exp - row[label];
    }
    Ok(total / batch as f64)
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> VitResult<f64> {
    if a.len() != b.len() {
        return Err(VitError::Shape("cosine_similarity: length mismatch".to_string()));
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return Ok(0.0);
    }
    Ok(dot / (na * nb))
}

/// Format a float as a percentage string.
pub fn fmt_pct(v: f64) -> String { format!("{:.2}%", v * 100.0) }

/// Compute NT-Xent (contrastive) loss for a batch of embeddings.
///
/// `z1`, `z2`: normalized embeddings, each `[B, D]`.
/// Returns average NT-Xent loss.
pub fn nt_xent_loss(z1: &[f64], z2: &[f64], batch: usize, dim: usize, temperature: f64) -> VitResult<f64> {
    if z1.len() != batch * dim || z2.len() != batch * dim {
        return Err(VitError::Shape("nt_xent_loss: shape mismatch".to_string()));
    }
    // Build 2B x 2B similarity matrix
    let n = 2 * batch;
    let mut z = vec![0.0f64; n * dim];
    z[..batch * dim].copy_from_slice(z1);
    z[batch * dim..].copy_from_slice(z2);

    let mut sim = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..dim).map(|d| z[i * dim + d] * z[j * dim + d]).sum();
            sim[i * n + j] = dot / temperature;
        }
    }

    // Cross-entropy over positive pairs
    let mut loss = 0.0f64;
    for i in 0..batch {
        let j = i + batch; // positive pair
        // i's loss: i vs j
        let max_i = (0..n).filter(|&k| k != i).map(|k| sim[i * n + k])
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum_i = max_i + (0..n).filter(|&k| k != i)
            .map(|k| (sim[i * n + k] - max_i).exp()).sum::<f64>().ln();
        loss += log_sum_i - sim[i * n + j];
        // j's loss: j vs i
        let max_j = (0..n).filter(|&k| k != j).map(|k| sim[j * n + k])
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum_j = max_j + (0..n).filter(|&k| k != j)
            .map(|k| (sim[j * n + k] - max_j).exp()).sum::<f64>().ln();
        loss += log_sum_j - sim[j * n + i];
    }
    Ok(loss / (2.0 * batch as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_2d_same_size() {
        let grid = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2
        let out = interpolate_2d(&grid, 2, 2, 2, 2).unwrap();
        for (a, b) in grid.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_interpolate_2d_upsample() {
        let grid = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2
        let out = interpolate_2d(&grid, 2, 2, 4, 4).unwrap();
        assert_eq!(out.len(), 16);
        // Corner values should be preserved
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[3] - 2.0).abs() < 1e-9);
        assert!((out[12] - 3.0).abs() < 1e-9);
        assert!((out[15] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_2d_shape_err() {
        let grid = vec![1.0f64; 5];
        assert!(interpolate_2d(&grid, 2, 2, 4, 4).is_err());
    }

    #[test]
    fn test_interpolate_2d_zero_dim() {
        let grid = vec![1.0f64; 4];
        assert!(interpolate_2d(&grid, 2, 2, 0, 4).is_err());
    }

    #[test]
    fn test_crop_resize_basic() {
        let img: Vec<f64> = (0..8 * 8).map(|x| x as f64).collect();
        let out = crop_resize(&img, 8, 8, 4, 4, 8, 8).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_crop_resize_too_large() {
        let img = vec![0.0f64; 4 * 4];
        assert!(crop_resize(&img, 4, 4, 8, 4, 8, 8).is_err());
    }

    #[test]
    fn test_attention_rollout_identity() {
        // Single identity layer should produce identity rollout
        let n = 3;
        let attn: Vec<f64> = (0..n*n).map(|i| if i/n == i%n { 1.0 } else { 0.0 }).collect();
        let rollout = attention_rollout(&attn, n, 1).unwrap();
        assert_eq!(rollout.len(), n * n);
        // Should remain identity-like after residual weighting
        let diag: f64 = (0..n).map(|i| rollout[i * n + i]).sum::<f64>() / n as f64;
        assert!(diag > 0.5);
    }

    #[test]
    fn test_attention_rollout_shape_err() {
        assert!(attention_rollout(&[0.0f64; 5], 3, 1).is_err());
    }

    #[test]
    fn test_attention_rollout_empty() {
        let out = attention_rollout(&[], 0, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_attention_entropy() {
        // Uniform attention → maximum entropy
        let n = 4;
        let attn: Vec<f64> = vec![0.25f64; n * n];
        let entropies = attention_entropy(&attn, n).unwrap();
        assert_eq!(entropies.len(), n);
        let max_entropy = (n as f64).ln();
        for &e in &entropies {
            assert!((e - max_entropy).abs() < 1e-6);
        }
    }

    #[test]
    fn test_top_k_accuracy_top1() {
        let logits = vec![0.1, 0.9, 0.0, 0.0, // sample 0 → class 1 highest
                          0.0, 0.0, 0.8, 0.2]; // sample 1 → class 2 highest
        let labels = vec![1usize, 2];
        let acc = top_k_accuracy(&logits, &labels, 2, 4, 1).unwrap();
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_accuracy_wrong() {
        let logits = vec![0.1, 0.9]; // 1 sample, 2 classes, pred=class 1
        let labels = vec![0usize]; // correct is class 0
        let acc = top_k_accuracy(&logits, &labels, 1, 2, 1).unwrap();
        assert!((acc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_accuracy_top2() {
        let logits = vec![0.1, 0.8, 0.5]; // 1 sample, 3 classes
        let labels = vec![2usize]; // class 2 is 2nd highest
        let acc = top_k_accuracy(&logits, &labels, 1, 3, 2).unwrap();
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_logits() {
        let logits = vec![1.0f64, 2.0, 3.0];
        let probs = softmax_logits(&logits, 1, 3).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let enc = sinusoidal_encoding(10, 64);
        assert_eq!(enc.len(), 10 * 64);
    }

    #[test]
    fn test_sinusoidal_encoding_bounds() {
        let enc = sinusoidal_encoding(100, 128);
        for &v in &enc {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_mse_loss_zero() {
        let pred = vec![1.0f64, 2.0, 3.0];
        let target = vec![1.0f64, 2.0, 3.0];
        let loss = mse_loss(&pred, &target).unwrap();
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = vec![0.0f64; 4];
        let target = vec![1.0f64; 4];
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_mismatch() {
        assert!(mse_loss(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_cross_entropy_loss_perfect() {
        // logits heavily favor correct class
        let logits = vec![0.0f64, 100.0, 0.0];
        let labels = vec![1usize];
        let loss = cross_entropy_loss(&logits, &labels, 1, 3).unwrap();
        assert!(loss < 0.01);
    }

    #[test]
    fn test_cross_entropy_loss_wrong() {
        // logits favor class 0 but label is class 1
        let logits = vec![100.0f64, 0.0, 0.0];
        let labels = vec![1usize];
        let loss = cross_entropy_loss(&logits, &labels, 1, 3).unwrap();
        assert!(loss > 1.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let cs = cosine_similarity(&a, &a).unwrap();
        assert!((cs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cs = cosine_similarity(&a, &b).unwrap();
        assert!(cs.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        assert!(cosine_similarity(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_fmt_pct() {
        assert_eq!(fmt_pct(0.9512), "95.12%");
    }

    #[test]
    fn test_nt_xent_loss_shape() {
        let mut rng = crate::core::SimpleRng::new(42);
        let z1: Vec<f64> = rng.gen_vec(2 * 8, 0.0, 1.0);
        let z2: Vec<f64> = rng.gen_vec(2 * 8, 0.0, 1.0);
        let loss = nt_xent_loss(&z1, &z2, 2, 8, 0.5).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_nt_xent_loss_shape_mismatch() {
        assert!(nt_xent_loss(&[0.0; 4], &[0.0; 8], 2, 4, 0.5).is_err());
    }

    #[test]
    fn test_interpolate_2d_constant_field() {
        let grid = vec![5.0f64; 9]; // 3x3 all 5s
        let out = interpolate_2d(&grid, 3, 3, 6, 6).unwrap();
        for &v in &out { assert!((v - 5.0).abs() < 1e-9); }
    }

    #[test]
    fn test_attention_entropy_concentrated() {
        // Delta distribution → entropy = 0
        let n = 4;
        let mut attn = vec![0.0f64; n * n];
        for r in 0..n { attn[r * n] = 1.0; } // all weight on token 0
        let entropies = attention_entropy(&attn, n).unwrap();
        for &e in &entropies { assert!(e.abs() < 1e-9); }
    }

    #[test]
    fn test_top_k_accuracy_shape_err() {
        assert!(top_k_accuracy(&[0.0f64; 6], &[0, 1], 2, 4, 1).is_err());
    }
}
