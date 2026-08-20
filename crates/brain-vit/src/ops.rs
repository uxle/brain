//! # ViT Operations for brain-vit
//!
//! Core tensor operations for Vision Transformers including patch extraction,
//! reconstruction, token manipulation, and interpolation utilities.
//!
//! ## Key operations
//! - [`extract_patches`] — unfold an image into non-overlapping patches
//! - [`reconstruct_patches`] — fold patches back to image space
//! - [`patchify`] / [`unpatchify`] — normalize patch ↔ pixel format
//! - [`add_cls_token`] / [`split_cls_token`] — CLS token management
//! - [`mask_patches`] — apply binary mask to token sequences
//! - [`position_ids`] — generate position index tensors
//! - [`token_interpolate`] — interpolate token sequences to new length

use crate::core::{SimpleRng, Tensor2D, VitError, VitResult};

/// Extract non-overlapping patches from a batch of images.
///
/// # Arguments
/// - `images`: flat image data `[B, C, H, W]` in CHW order.
/// - `batch`: batch size B.
/// - `channels`: C.
/// - `height`: H.
/// - `width`: W.
/// - `patch_h`: patch height.
/// - `patch_w`: patch width.
///
/// # Returns
/// Flat data `[B, num_patches, patch_h * patch_w * C]`.
///
/// # Example
/// ```rust
/// use brain_vit::ops::extract_patches;
/// let img = vec![0.0f64; 1 * 3 * 16 * 16]; // 1 image, 3 channels, 16x16
/// let patches = extract_patches(&img, 1, 3, 16, 16, 8, 8).unwrap();
/// assert_eq!(patches.len(), 1 * 4 * (8 * 8 * 3)); // 4 patches of size 8*8*3
/// ```
pub fn extract_patches(
    images: &[f64],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
) -> VitResult<Vec<f64>> {
    if batch == 0 {
        return Err(VitError::EmptyBatch);
    }
    if !height.is_multiple_of(patch_h) {
        return Err(VitError::InvalidPatchSize {
            image_dim: height,
            patch_size: patch_h,
        });
    }
    if !width.is_multiple_of(patch_w) {
        return Err(VitError::InvalidPatchSize {
            image_dim: width,
            patch_size: patch_w,
        });
    }
    let expected = batch * channels * height * width;
    if images.len() != expected {
        return Err(VitError::Shape(format!(
            "extract_patches: expected {} elements, got {}",
            expected,
            images.len()
        )));
    }

    let gh = height / patch_h;
    let gw = width / patch_w;
    let num_patches = gh * gw;
    let patch_dim = patch_h * patch_w * channels;
    let mut out = vec![0.0f64; batch * num_patches * patch_dim];

    for b in 0..batch {
        for ph in 0..gh {
            for pw in 0..gw {
                let patch_idx = ph * gw + pw;
                for c in 0..channels {
                    for dy in 0..patch_h {
                        for dx in 0..patch_w {
                            let iy = ph * patch_h + dy;
                            let ix = pw * patch_w + dx;
                            // Input index: [b, c, iy, ix]
                            let in_idx = b * channels * height * width
                                + c * height * width
                                + iy * width
                                + ix;
                            // Output index: [b, patch_idx, c * patch_h * patch_w + dy * patch_w + dx]
                            let feat_idx = c * patch_h * patch_w + dy * patch_w + dx;
                            let out_idx =
                                b * num_patches * patch_dim + patch_idx * patch_dim + feat_idx;
                            out[out_idx] = images[in_idx];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Reconstruct images from patches (inverse of extract_patches).
///
/// # Arguments
/// - `patches`: flat data `[B, N, patch_dim]`.
/// - `batch`, `channels`, `height`, `width`: target shape.
/// - `patch_h`, `patch_w`: patch dimensions.
///
/// # Returns
/// Flat image data `[B, C, H, W]`.
pub fn reconstruct_patches(
    patches: &[f64],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
) -> VitResult<Vec<f64>> {
    if batch == 0 {
        return Err(VitError::EmptyBatch);
    }
    let gh = height / patch_h;
    let gw = width / patch_w;
    let num_patches = gh * gw;
    let patch_dim = patch_h * patch_w * channels;
    let expected = batch * num_patches * patch_dim;
    if patches.len() != expected {
        return Err(VitError::Shape(format!(
            "reconstruct_patches: expected {} elements, got {}",
            expected,
            patches.len()
        )));
    }

    let mut out = vec![0.0f64; batch * channels * height * width];
    for b in 0..batch {
        for ph in 0..gh {
            for pw in 0..gw {
                let patch_idx = ph * gw + pw;
                for c in 0..channels {
                    for dy in 0..patch_h {
                        for dx in 0..patch_w {
                            let iy = ph * patch_h + dy;
                            let ix = pw * patch_w + dx;
                            let feat_idx = c * patch_h * patch_w + dy * patch_w + dx;
                            let in_idx =
                                b * num_patches * patch_dim + patch_idx * patch_dim + feat_idx;
                            let out_idx = b * channels * height * width
                                + c * height * width
                                + iy * width
                                + ix;
                            out[out_idx] = patches[in_idx];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Normalize patches by subtracting per-channel mean and dividing by std.
///
/// Operates on `[B, N, patch_dim]` format where `patch_dim = C * Ph * Pw`.
pub fn patchify(
    patches: &[f64],
    batch: usize,
    num_patches: usize,
    patch_dim: usize,
) -> VitResult<Vec<f64>> {
    let expected = batch * num_patches * patch_dim;
    if patches.len() != expected {
        return Err(VitError::Shape(format!(
            "patchify: expected {} elements, got {}",
            expected,
            patches.len()
        )));
    }
    let mut out = patches.to_vec();
    // Normalize each patch independently
    for b in 0..batch {
        for p in 0..num_patches {
            let start = (b * num_patches + p) * patch_dim;
            let slice = &mut out[start..start + patch_dim];
            let mean: f64 = slice.iter().sum::<f64>() / patch_dim as f64;
            let var: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / patch_dim as f64;
            let std = (var + 1e-6).sqrt();
            for x in slice.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
    }
    Ok(out)
}

/// Denormalize patches (identity for now — actual denorm requires stored stats).
pub fn unpatchify(patches: &[f64]) -> Vec<f64> {
    patches.to_vec()
}

/// Prepend a CLS token to a token sequence.
///
/// - `tokens`: `[B, N, D]` flat.
/// - `cls`: `[D]` token vector repeated for each batch item.
/// - Returns `[B, N+1, D]` with CLS prepended.
pub fn add_cls_token(
    tokens: &[f64],
    cls: &[f64],
    batch: usize,
    num_tokens: usize,
    dim: usize,
) -> VitResult<Vec<f64>> {
    let expected = batch * num_tokens * dim;
    if tokens.len() != expected {
        return Err(VitError::Shape(
            "add_cls_token: tokens shape mismatch".to_string(),
        ));
    }
    if cls.len() != dim {
        return Err(VitError::DimMismatch {
            expected: dim,
            got: cls.len(),
        });
    }
    let new_len = batch * (num_tokens + 1) * dim;
    let mut out = vec![0.0f64; new_len];
    for b in 0..batch {
        // Write CLS token
        let cls_start = b * (num_tokens + 1) * dim;
        out[cls_start..cls_start + dim].copy_from_slice(cls);
        // Copy remaining tokens
        let tok_src = b * num_tokens * dim;
        let tok_dst = cls_start + dim;
        out[tok_dst..tok_dst + num_tokens * dim]
            .copy_from_slice(&tokens[tok_src..tok_src + num_tokens * dim]);
    }
    Ok(out)
}

/// Split CLS token from a sequence.
///
/// Returns `(cls: [B, D], rest: [B, N, D])`.
pub fn split_cls_token(
    tokens: &[f64],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> VitResult<(Vec<f64>, Vec<f64>)> {
    let expected = batch * seq_len * dim;
    if tokens.len() != expected {
        return Err(VitError::Shape(format!(
            "split_cls_token: expected {} got {}",
            expected,
            tokens.len()
        )));
    }
    if seq_len == 0 {
        return Err(VitError::Shape(
            "split_cls_token: seq_len must be > 0".to_string(),
        ));
    }
    let n_rest = seq_len - 1;
    let mut cls_out = vec![0.0f64; batch * dim];
    let mut rest_out = vec![0.0f64; batch * n_rest * dim];
    for b in 0..batch {
        let src_start = b * seq_len * dim;
        cls_out[b * dim..b * dim + dim].copy_from_slice(&tokens[src_start..src_start + dim]);
        let rest_src = src_start + dim;
        let rest_dst = b * n_rest * dim;
        rest_out[rest_dst..rest_dst + n_rest * dim]
            .copy_from_slice(&tokens[rest_src..rest_src + n_rest * dim]);
    }
    Ok((cls_out, rest_out))
}

/// Apply a binary mask to a token sequence.
///
/// - `tokens`: `[B, N, D]` flat.
/// - `mask`: `[N]` boolean mask (true = keep, false = remove).
/// - Returns `[B, N_kept, D]` and the kept indices.
pub fn mask_patches(
    tokens: &[f64],
    mask: &[bool],
    batch: usize,
    num_tokens: usize,
    dim: usize,
) -> VitResult<(Vec<f64>, Vec<usize>)> {
    let expected = batch * num_tokens * dim;
    if tokens.len() != expected {
        return Err(VitError::Shape(
            "mask_patches: tokens shape mismatch".to_string(),
        ));
    }
    if mask.len() != num_tokens {
        return Err(VitError::Shape(format!(
            "mask_patches: mask length {} != num_tokens {}",
            mask.len(),
            num_tokens
        )));
    }
    let kept: Vec<usize> = (0..num_tokens).filter(|&i| mask[i]).collect();
    let n_kept = kept.len();
    let mut out = vec![0.0f64; batch * n_kept * dim];
    for b in 0..batch {
        for (out_i, &tok_i) in kept.iter().enumerate() {
            let src = b * num_tokens * dim + tok_i * dim;
            let dst = b * n_kept * dim + out_i * dim;
            out[dst..dst + dim].copy_from_slice(&tokens[src..src + dim]);
        }
    }
    Ok((out, kept))
}

/// Generate position IDs `[0, 1, ..., seq_len-1]` as f64.
pub fn position_ids(seq_len: usize) -> Vec<f64> {
    (0..seq_len).map(|i| i as f64).collect()
}

/// Linearly interpolate a 1D token sequence to a new length.
///
/// - `tokens`: `[seq_len, dim]` flat.
/// - `new_len`: target sequence length.
/// - Returns `[new_len, dim]` flat.
pub fn token_interpolate(
    tokens: &[f64],
    seq_len: usize,
    dim: usize,
    new_len: usize,
) -> VitResult<Vec<f64>> {
    if tokens.len() != seq_len * dim {
        return Err(VitError::Shape(
            "token_interpolate: shape mismatch".to_string(),
        ));
    }
    if new_len == 0 {
        return Err(VitError::Shape(
            "token_interpolate: new_len must be > 0".to_string(),
        ));
    }
    if seq_len == 0 {
        return Err(VitError::Shape(
            "token_interpolate: seq_len must be > 0".to_string(),
        ));
    }
    let mut out = vec![0.0f64; new_len * dim];
    for j in 0..new_len {
        let frac = j as f64 * (seq_len - 1) as f64 / (new_len - 1).max(1) as f64;
        let lo = frac.floor() as usize;
        let hi = (lo + 1).min(seq_len - 1);
        let t = frac - lo as f64;
        for d in 0..dim {
            out[j * dim + d] = tokens[lo * dim + d] * (1.0 - t) + tokens[hi * dim + d] * t;
        }
    }
    Ok(out)
}

/// Add learned position embedding to token sequence (in-place).
///
/// - `tokens`: `[B, N, D]` flat.
/// - `pos_embed`: `[N, D]` flat.
pub fn add_pos_embed(
    tokens: &mut [f64],
    pos_embed: &[f64],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> VitResult<()> {
    let expected_tok = batch * seq_len * dim;
    let expected_pos = seq_len * dim;
    if tokens.len() != expected_tok {
        return Err(VitError::Shape(
            "add_pos_embed: tokens shape mismatch".to_string(),
        ));
    }
    if pos_embed.len() != expected_pos {
        return Err(VitError::Shape(format!(
            "add_pos_embed: pos_embed expected {} got {}",
            expected_pos,
            pos_embed.len()
        )));
    }
    for b in 0..batch {
        let base = b * seq_len * dim;
        for s in 0..seq_len {
            for d in 0..dim {
                tokens[base + s * dim + d] += pos_embed[s * dim + d];
            }
        }
    }
    Ok(())
}

/// Apply dropout to a slice (training mode only).
/// Uses simple threshold: if `rng.next_f64() < rate`, set to 0; else scale by 1/(1-rate).
pub fn apply_dropout(data: &mut [f64], rate: f64, rng: &mut SimpleRng) {
    if rate <= 0.0 {
        return;
    }
    let scale = 1.0 / (1.0 - rate).max(1e-7);
    for x in data.iter_mut() {
        if rng.next_f64() < rate {
            *x = 0.0;
        } else {
            *x *= scale;
        }
    }
}

/// Compute scaled dot-product attention.
///
/// Q, K, V are all `[seq, head_dim]` (single head, single batch).
/// Returns attention output `[seq, head_dim]` and attention weights `[seq, seq]`.
pub fn scaled_dot_product_attention(
    q: &Tensor2D,
    k: &Tensor2D,
    v: &Tensor2D,
) -> VitResult<(Tensor2D, Tensor2D)> {
    if q.cols != k.cols || q.cols != v.cols {
        return Err(VitError::DimMismatch {
            expected: q.cols,
            got: k.cols,
        });
    }
    if k.rows != v.rows {
        return Err(VitError::Shape(
            "SDPA: K and V must have same seq length".to_string(),
        ));
    }
    let scale = (q.cols as f64).sqrt();
    // scores = Q @ K^T / sqrt(d_k)
    let kt = k.transpose();
    let scores = q.matmul(&kt)?.scale(1.0 / scale);
    // softmax over keys
    let attn = scores.softmax_rows();
    // output = attn @ V
    let out = attn.matmul(v)?;
    Ok((out, attn))
}

/// Apply layer normalization to each row of a 2D tensor.
pub fn layer_norm_2d(t: &Tensor2D, eps: f64) -> Tensor2D {
    t.layer_norm(eps)
}

/// Apply a simple feed-forward MLP to each row: Linear → Activation → Linear.
///
/// - `input`: `[seq, d_in]`
/// - `w1`, `b1`: first layer weights `[d_hidden, d_in]` and bias `[d_hidden]`
/// - `w2`, `b2`: second layer weights `[d_out, d_hidden]` and bias `[d_out]`
#[allow(clippy::needless_range_loop)]
pub fn mlp_forward(
    input: &Tensor2D,
    w1: &Tensor2D,
    b1: &[f64],
    w2: &Tensor2D,
    b2: &[f64],
    act: &crate::config::Activation,
) -> VitResult<Tensor2D> {
    // h = input @ w1^T + b1
    let w1t = w1.transpose();
    let mut h = input.matmul(&w1t)?;
    // add bias
    for r in 0..h.rows {
        for c in 0..h.cols {
            let old = h.get(r, c);
            h.set(r, c, old + b1[c]);
        }
    }
    // activation
    for x in h.data.iter_mut() {
        *x = act.apply(*x);
    }
    // out = h @ w2^T + b2
    let w2t = w2.transpose();
    let mut out = h.matmul(&w2t)?;
    for r in 0..out.rows {
        for c in 0..out.cols {
            let old = out.get(r, c);
            out.set(r, c, old + b2[c]);
        }
    }
    Ok(out)
}

/// Linear projection: `output = input @ weight^T + bias`.
///
/// - `input`: `[seq, d_in]`
/// - `weight`: `[d_out, d_in]`
/// - `bias`: optional `[d_out]`
#[allow(clippy::needless_range_loop)]
pub fn linear(input: &Tensor2D, weight: &Tensor2D, bias: Option<&[f64]>) -> VitResult<Tensor2D> {
    let wt = weight.transpose();
    let mut out = input.matmul(&wt)?;
    if let Some(b) = bias {
        if b.len() != out.cols {
            return Err(VitError::DimMismatch {
                expected: out.cols,
                got: b.len(),
            });
        }
        for r in 0..out.rows {
            for c in 0..out.cols {
                let old = out.get(r, c);
                out.set(r, c, old + b[c]);
            }
        }
    }
    Ok(out)
}

/// Compute stochastic depth scaling factor (for training with drop path).
/// Returns 1.0 if not dropping; 0.0 if dropping this sample.
pub fn stochastic_depth_scale(rate: f64, training: bool, rng: &mut SimpleRng) -> f64 {
    if !training || rate <= 0.0 {
        return 1.0;
    }
    if rng.next_f64() < rate {
        0.0
    } else {
        1.0 / (1.0 - rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;
    use crate::core::SimpleRng;

    #[test]
    fn test_extract_patches_basic() {
        // 1 image, 1 channel, 4x4 -> 2x2 patches of 2x2
        let img: Vec<f64> = (0..16).map(|x| x as f64).collect();
        let patches = extract_patches(&img, 1, 1, 4, 4, 2, 2).unwrap();
        assert_eq!(patches.len(), 1 * 4 * 4); // 4 patches, each 4 elements
                                              // First patch: pixels (0,0),(0,1),(1,0),(1,1) = 0,1,4,5
        assert_eq!(patches[0], 0.0);
        assert_eq!(patches[1], 1.0);
        assert_eq!(patches[2], 4.0);
        assert_eq!(patches[3], 5.0);
    }

    #[test]
    fn test_extract_patches_empty_batch() {
        let img = vec![1.0f64; 16];
        assert!(extract_patches(&img, 0, 1, 4, 4, 2, 2).is_err());
    }

    #[test]
    fn test_extract_patches_invalid_size() {
        let img = vec![1.0f64; 15]; // wrong size
        assert!(extract_patches(&img, 1, 1, 5, 3, 2, 1).is_err());
    }

    #[test]
    fn test_reconstruct_patches_roundtrip() {
        let img: Vec<f64> = (0..1 * 3 * 16 * 16).map(|x| x as f64).collect();
        let patches = extract_patches(&img, 1, 3, 16, 16, 4, 4).unwrap();
        let reconstructed = reconstruct_patches(&patches, 1, 3, 16, 16, 4, 4).unwrap();
        assert_eq!(img.len(), reconstructed.len());
        for (a, b) in img.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_add_cls_token() {
        let tokens = vec![1.0f64; 2 * 4 * 8]; // B=2, N=4, D=8
        let cls = vec![99.0f64; 8];
        let out = add_cls_token(&tokens, &cls, 2, 4, 8).unwrap();
        assert_eq!(out.len(), 2 * 5 * 8);
        // First 8 elements of batch 0 should be cls
        for i in 0..8 {
            assert_eq!(out[i], 99.0);
        }
        // Next 32 should be the original tokens
        for i in 8..40 {
            assert_eq!(out[i], 1.0);
        }
    }

    #[test]
    fn test_add_cls_token_dim_mismatch() {
        let tokens = vec![1.0f64; 2 * 4 * 8];
        let cls = vec![0.0f64; 4]; // wrong dim
        assert!(add_cls_token(&tokens, &cls, 2, 4, 8).is_err());
    }

    #[test]
    fn test_split_cls_token() {
        // Construct tokens: CLS=99, rest=1
        let tokens = vec![1.0f64; 2 * 5 * 8];
        let mut with_cls = vec![0.0f64; 2 * 5 * 8];
        with_cls[..8].copy_from_slice(&vec![99.0f64; 8]);
        with_cls[8..40].copy_from_slice(&vec![1.0f64; 32]);
        with_cls[40..48].copy_from_slice(&vec![99.0f64; 8]);
        with_cls[48..].copy_from_slice(&vec![1.0f64; 32]);
        let (cls, rest) = split_cls_token(&with_cls, 2, 5, 8).unwrap();
        assert_eq!(cls.len(), 2 * 8);
        assert_eq!(rest.len(), 2 * 4 * 8);
        for i in 0..8 {
            assert_eq!(cls[i], 99.0);
        }
        let _ = tokens; // suppress unused warning
    }

    #[test]
    fn test_split_cls_token_zero_seq() {
        assert!(split_cls_token(&[], 1, 0, 8).is_err());
    }

    #[test]
    fn test_mask_patches() {
        let tokens = vec![1.0f64; 2 * 6 * 4]; // B=2, N=6, D=4
        let mask = vec![true, false, true, false, true, false]; // keep 3
        let (out, kept) = mask_patches(&tokens, &mask, 2, 6, 4).unwrap();
        assert_eq!(kept.len(), 3);
        assert_eq!(out.len(), 2 * 3 * 4);
    }

    #[test]
    fn test_mask_patches_bad_mask_len() {
        let tokens = vec![1.0f64; 2 * 6 * 4];
        let mask = vec![true; 4]; // wrong length
        assert!(mask_patches(&tokens, &mask, 2, 6, 4).is_err());
    }

    #[test]
    fn test_position_ids() {
        let ids = position_ids(5);
        assert_eq!(ids, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_token_interpolate_same_len() {
        let tokens = vec![1.0f64; 4 * 3];
        let out = token_interpolate(&tokens, 4, 3, 4).unwrap();
        assert_eq!(out.len(), 4 * 3);
        for (a, b) in tokens.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_token_interpolate_upsample() {
        let tokens: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let out = token_interpolate(&tokens, 4, 1, 7).unwrap();
        assert_eq!(out.len(), 7);
        // First value should be 0, last should be 3
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[6] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_interpolate_zero_new_len() {
        let tokens = vec![1.0f64; 4];
        assert!(token_interpolate(&tokens, 4, 1, 0).is_err());
    }

    #[test]
    fn test_add_pos_embed() {
        let mut tokens = vec![1.0f64; 2 * 4 * 8]; // B=2, N=4, D=8
        let pos = vec![0.5f64; 4 * 8];
        add_pos_embed(&mut tokens, &pos, 2, 4, 8).unwrap();
        assert!((tokens[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_add_pos_embed_wrong_shape() {
        let mut tokens = vec![1.0f64; 2 * 4 * 8];
        let pos = vec![0.5f64; 5 * 8]; // wrong seq_len
        assert!(add_pos_embed(&mut tokens, &pos, 2, 4, 8).is_err());
    }

    #[test]
    fn test_patchify() {
        let patches = vec![1.0f64; 2 * 4 * 12];
        let out = patchify(&patches, 2, 4, 12).unwrap();
        assert_eq!(out.len(), patches.len());
        // Output should be normalized (mean ~0 since all same)
        let mean: f64 = out[..12].iter().sum::<f64>() / 12.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = Tensor2D::from_data(
            3,
            4,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let k = q.clone();
        let v = Tensor2D::from_data(
            3,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let (out, attn) = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 4);
        // Attention matrix rows should sum to 1
        for r in 0..3 {
            let sum: f64 = (0..3).map(|c| attn.get(r, c)).sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_linear() {
        let input = Tensor2D::from_data(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let weight = Tensor2D::from_data(4, 3, (0..12).map(|x| x as f64).collect()).unwrap();
        let out = linear(&input, &weight, None).unwrap();
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_linear_with_bias() {
        let input = Tensor2D::from_data(1, 2, vec![1.0, 0.0]).unwrap();
        let weight = Tensor2D::from_data(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let bias = vec![10.0, 20.0];
        let out = linear(&input, &weight, Some(&bias)).unwrap();
        assert!((out.get(0, 0) - 11.0).abs() < 1e-9);
        assert!((out.get(0, 1) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_layer_norm_2d() {
        let t = Tensor2D::from_data(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = layer_norm_2d(&t, 1e-5);
        let mean: f64 = n.data.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_apply_dropout_zero_rate() {
        let mut data = vec![1.0f64; 10];
        let mut rng = SimpleRng::new(0);
        apply_dropout(&mut data, 0.0, &mut rng);
        assert!(data.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_apply_dropout_nonzero() {
        let mut data = vec![1.0f64; 1000];
        let mut rng = SimpleRng::new(42);
        apply_dropout(&mut data, 0.5, &mut rng);
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        // Roughly 50% should be zero
        assert!(zeros > 300 && zeros < 700);
    }

    #[test]
    fn test_stochastic_depth_no_drop_eval() {
        let mut rng = SimpleRng::new(0);
        let s = stochastic_depth_scale(0.5, false, &mut rng);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn test_stochastic_depth_zero_rate() {
        let mut rng = SimpleRng::new(0);
        let s = stochastic_depth_scale(0.0, true, &mut rng);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn test_extract_patches_multichannel() {
        // 1 image, 2 channels, 4x4, patch 2x2
        let img: Vec<f64> = (0..2 * 4 * 4).map(|x| x as f64).collect();
        let patches = extract_patches(&img, 1, 2, 4, 4, 2, 2).unwrap();
        // 4 patches, each 2*2*2=8 elements
        assert_eq!(patches.len(), 1 * 4 * 8);
    }

    #[test]
    fn test_mask_patches_all_kept() {
        let tokens = vec![2.0f64; 1 * 5 * 3];
        let mask = vec![true; 5];
        let (out, kept) = mask_patches(&tokens, &mask, 1, 5, 3).unwrap();
        assert_eq!(kept.len(), 5);
        assert_eq!(out, tokens);
    }

    #[test]
    fn test_mask_patches_none_kept() {
        let tokens = vec![2.0f64; 1 * 5 * 3];
        let mask = vec![false; 5];
        let (out, kept) = mask_patches(&tokens, &mask, 1, 5, 3).unwrap();
        assert_eq!(kept.len(), 0);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_split_then_add_cls_roundtrip() {
        let cls = vec![9.0f64; 8];
        let patches = vec![1.0f64; 2 * 4 * 8];
        let full = add_cls_token(&patches, &cls, 2, 4, 8).unwrap();
        let (cls_out, rest) = split_cls_token(&full, 2, 5, 8).unwrap();
        // CLS values preserved
        for i in 0..8 {
            assert_eq!(cls_out[i], 9.0);
        }
        // Rest values preserved
        for &v in rest.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_sdpa_dim_mismatch() {
        let q = Tensor2D::zeros(3, 4);
        let k = Tensor2D::zeros(3, 5); // different head_dim
        let v = Tensor2D::zeros(3, 4);
        assert!(scaled_dot_product_attention(&q, &k, &v).is_err());
    }

    #[test]
    fn test_mlp_forward_identity() {
        // With identity weight matrices and zero biases
        let input =
            Tensor2D::from_data(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        // w1: identity-like [8, 4] will project to 8 dims, then w2 projects back to 4
        let w1 = Tensor2D::from_data(8, 4, {
            let mut data = vec![0.0f64; 32];
            for i in 0..4 {
                data[i * 4 + i] = 1.0;
            } // first 4 rows are identity
            data
        })
        .unwrap();
        let b1 = vec![0.0f64; 8];
        let w2 = Tensor2D::from_data(4, 8, {
            let mut data = vec![0.0f64; 32];
            for i in 0..4 {
                data[i * 8 + i] = 1.0;
            } // first 4 cols are identity
            data
        })
        .unwrap();
        let b2 = vec![0.0f64; 4];
        let act = Activation::Relu;
        let out = mlp_forward(&input, &w1, &b1, &w2, &b2, &act).unwrap();
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, 4);
    }
}
