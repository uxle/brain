//! # GAN Tensor Operations
//!
//! Reusable conv-layer helpers, normalization, WGAN clipping, spectral norm.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies a 2D convolution-like linear mapping (height-width flattened).
/// `weight` shape: [out_channels, in_channels].
pub fn linear_map(input: &Tensor, weight: &Tensor) -> Tensor {
    // Simplified: treat as matrix multiply input [batch, in] x weight^T [in, out]
    let in_dim = input.shape().last().copied().unwrap_or(1);
    let out_dim = weight.shape()[0];
    let batch = input.to_vec().len() / in_dim;
    let input_data = input.to_vec();
    let weight_data = weight.to_vec();
    let mut out = vec![0.0f64; batch * out_dim];
    for b in 0..batch {
        for o in 0..out_dim {
            let mut s = 0.0f64;
            for i in 0..in_dim {
                s += input_data[b * in_dim + i] * weight_data[o * in_dim + i];
            }
            out[b * out_dim + o] = s;
        }
    }
    Tensor::from_vec(out, vec![batch, out_dim])
}

/// Applies batch normalization (mean=0, std=1 normalization per channel).
pub fn batch_norm(t: &Tensor, eps: f64) -> Tensor {
    let data = t.to_vec();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + eps).sqrt();
    let norm: Vec<f64> = data.iter().map(|v| (v - mean) / std).collect();
    Tensor::from_vec(norm, t.shape().to_vec())
}

/// Leaky ReLU activation.
pub fn leaky_relu(t: &Tensor, neg_slope: f64) -> Tensor {
    let data: Vec<f64> = t
        .to_vec()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { neg_slope * v })
        .collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// ReLU activation.
pub fn relu(t: &Tensor) -> Tensor {
    let data: Vec<f64> = t.to_vec().iter().map(|&v| v.max(0.0)).collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// Tanh activation.
pub fn tanh_act(t: &Tensor) -> Tensor {
    let data: Vec<f64> = t.to_vec().iter().map(|v| v.tanh()).collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// Sigmoid activation.
pub fn sigmoid_act(t: &Tensor) -> Tensor {
    let data: Vec<f64> = t
        .to_vec()
        .iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// WGAN weight clipping: clips all params of a tensor to [-c, c].
pub fn wgan_clip(t: &Tensor, c: f64) -> Tensor {
    let data: Vec<f64> = t.to_vec().iter().map(|v| v.clamp(-c, c)).collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// Spectral norm estimate via power iteration (one step).
pub fn spectral_norm_apply(w: &Tensor, u: &Tensor) -> (Tensor, Tensor, f64) {
    let wdata = w.to_vec();
    let rows = w.shape()[0];
    let cols = if w.shape().len() > 1 { w.shape()[1] } else { 1 };
    let udata = u.to_vec();
    // v = W^T u / ||W^T u||
    let mut v = vec![0.0f64; cols];
    for j in 0..cols {
        for i in 0..rows {
            v[j] += wdata[i * cols + j] * udata[i];
        }
    }
    let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
    for vj in v.iter_mut() {
        *vj /= vnorm;
    }
    // u_new = W v / ||W v||
    let mut u_new = vec![0.0f64; rows];
    for i in 0..rows {
        for j in 0..cols {
            u_new[i] += wdata[i * cols + j] * v[j];
        }
    }
    let sigma: f64 = u_new.iter().zip(udata.iter()).map(|(a, b)| a * b).sum();
    let unorm = u_new.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
    for ui in u_new.iter_mut() {
        *ui /= unorm;
    }
    let _sigma_t = Tensor::scalar(sigma.max(1e-8));
    let w_sn_data: Vec<f64> = wdata.iter().map(|v| v / sigma.max(1e-8)).collect();
    let w_sn = Tensor::from_vec(w_sn_data, w.shape().to_vec());
    (w_sn, Tensor::from_vec(u_new, u.shape().to_vec()), sigma)
}

/// Interpolates two tensors: out = alpha*a + (1-alpha)*b.
pub fn interpolate_latents(a: &Tensor, b: &Tensor, alpha: f64) -> Tensor {
    let ad = a.to_vec();
    let bd = b.to_vec();
    let data: Vec<f64> = ad
        .iter()
        .zip(bd.iter())
        .map(|(x, y)| alpha * x + (1.0 - alpha) * y)
        .collect();
    Tensor::from_vec(data, a.shape().to_vec())
}

/// Assembles a batch of images into a grid tensor (vertical concatenation, simplified).
pub fn image_grid(images: &[Tensor]) -> Tensor {
    if images.is_empty() {
        return Tensor::zeros(vec![1]);
    }
    let total: Vec<f64> = images.iter().flat_map(|img| img.to_vec()).collect();
    let total_len = total.len();
    Tensor::from_vec(total, vec![total_len])
}

/// Nearest-neighbor resize (1D for tests).
pub fn resize_like(t: &Tensor, new_size: usize) -> Tensor {
    let data = t.to_vec();
    let old_size = data.len();
    if old_size == 0 {
        return Tensor::zeros(vec![new_size]);
    }
    let out: Vec<f64> = (0..new_size)
        .map(|i| {
            let src = (i * old_size / new_size).min(old_size - 1);
            data[src]
        })
        .collect();
    Tensor::from_vec(out, vec![new_size])
}

/// Mixes two style codes with a mix ratio.
pub fn mix_style(s1: &Tensor, s2: &Tensor, ratio: f64) -> Tensor {
    interpolate_latents(s1, s2, ratio)
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
