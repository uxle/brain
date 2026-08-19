//! # GAN Configuration
//!
//! Master configuration covering generator, discriminator, training and evaluation.
#![allow(missing_docs)]

/// Latent space type for the generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatentType {
    #[default]
    Gaussian,
    Uniform,
    Spherical,
}

/// Output activation for the generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputActivation {
    #[default]
    Tanh,
    Sigmoid,
    Linear,
}

/// Type of GAN loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossVariant {
    #[default]
    Classic,
    LeastSquares,
    Hinge,
    Wasserstein,
    Relativistic,
}

/// Architecture of the generator/discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArchVariant {
    #[default]
    Dcgan,
    Resnet,
    Conditional,
    Patch,
    StyleLite,
}

/// Generator configuration.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub latent_dim: usize,
    pub base_channels: usize,
    pub num_layers: usize,
    pub image_size: usize,
    pub output_channels: usize,
    pub latent_type: LatentType,
    pub output_activation: OutputActivation,
    pub num_classes: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            base_channels: 64,
            num_layers: 4,
            image_size: 64,
            output_channels: 3,
            latent_type: LatentType::Gaussian,
            output_activation: OutputActivation::Tanh,
            num_classes: 0,
        }
    }
}

/// Discriminator configuration.
#[derive(Debug, Clone)]
pub struct DiscriminatorConfig {
    pub base_channels: usize,
    pub num_layers: usize,
    pub image_size: usize,
    pub input_channels: usize,
    pub patch: bool,
    pub num_classes: usize,
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            base_channels: 64,
            num_layers: 4,
            image_size: 64,
            input_channels: 3,
            patch: false,
            num_classes: 0,
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct GanTrainConfig {
    pub n_critic: usize,
    pub learning_rate_g: f64,
    pub learning_rate_d: f64,
    pub batch_size: usize,
    pub gradient_penalty: bool,
    pub gp_lambda: f64,
    pub label_smoothing: f64,
    pub clip_value: f64,
    pub ema_decay: f64,
}

impl Default for GanTrainConfig {
    fn default() -> Self {
        Self {
            n_critic: 1,
            learning_rate_g: 2e-4,
            learning_rate_d: 2e-4,
            batch_size: 64,
            gradient_penalty: false,
            gp_lambda: 10.0,
            label_smoothing: 0.0,
            clip_value: 0.01,
            ema_decay: 0.999,
        }
    }
}

/// Master GAN configuration.
#[derive(Debug, Clone, Default)]
pub struct GanConfig {
    pub generator: GeneratorConfig,
    pub discriminator: DiscriminatorConfig,
    pub training: GanTrainConfig,
    pub loss: LossVariant,
    pub arch: ArchVariant,
}

impl GanConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.generator.latent_dim == 0 {
            return Err("latent_dim must be > 0".into());
        }
        if self.generator.base_channels == 0 {
            return Err("base_channels must be > 0".into());
        }
        if self.training.batch_size == 0 {
            return Err("batch_size must be > 0".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "GAN[arch={:?} loss={:?} latent={} img={}x{}]",
            self.arch, self.loss,
            self.generator.latent_dim,
            self.generator.image_size,
            self.generator.image_size,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
