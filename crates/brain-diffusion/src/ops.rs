//! # Diffusion Mathematical Operations
//!
//! Noise perturbation (q-sample), denoising updates, and Classifier-Free Guidance (CFG).

use brain_core::Tensor;

/// Applies noise to a clean image `x0` given timestep cumulative alpha `alpha_cumprod`.
pub fn add_noise(x0: &Tensor, noise: &Tensor, alpha_cumprod: f64) -> Tensor {
    let sqrt_alpha = alpha_cumprod.sqrt();
    let sqrt_one_minus_alpha = (1.0 - alpha_cumprod).sqrt();
    let t_alpha = Tensor::scalar(sqrt_alpha);
    let t_noise = Tensor::scalar(sqrt_one_minus_alpha);
    &(x0 * &t_alpha) + &(noise * &t_noise)
}

/// Combines conditional and unconditional predictions with guidance scale (CFG).
pub fn apply_cfg(uncond_pred: &Tensor, cond_pred: &Tensor, guidance_scale: f64) -> Tensor {
    let diff = cond_pred - uncond_pred;
    let t_scale = Tensor::scalar(guidance_scale);
    let guided = &diff * &t_scale;
    uncond_pred + &guided
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
