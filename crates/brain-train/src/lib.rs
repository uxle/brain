//! Integrated training primitives for Brain.
//!
//! This crate intentionally contains the first mutable model surface in the
//! workspace. Existing `brain-nn::Module` implementations expose cloned
//! parameters, so `brain-train` provides trainable layers whose state can be
//! updated by `brain-optim` after loss gradients are computed.

use brain_autograd::Value;
use brain_core::Tensor;
use brain_loss::classification::{ClassificationLoss, CrossEntropyLoss};
use brain_metric::classification::accuracy_score;
use brain_nn::Module;
use brain_optim::optimizer::{Optimizer, ParamGroup, StepInfo};
use brain_optim::sgd::{Sgd, SgdConfig};
use std::collections::HashMap;
use std::fmt;

pub mod callbacks;

pub use callbacks::{CallbackAction, EarlyStopping, MetricHistoryLogger, TrainingCallback};

/// Result type used by integrated training APIs.
pub type TrainResult<T> = Result<T, TrainError>;

/// Error type for training, state, and adapter operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainError {
    /// A tensor had an unexpected rank or dimension.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// A target class index was invalid for the model output.
    InvalidTarget { target: usize, classes: usize },
    /// A model has not been supplied to a builder.
    MissingModel,
    /// An optimizer error was raised.
    Optimizer(String),
    /// A module adapter error was raised.
    Module(String),
    /// A loss error was raised.
    Loss(String),
    /// Serialized model state could not be parsed.
    State(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TrainError::InvalidTarget { target, classes } => {
                write!(f, "invalid target class {} for {} classes", target, classes)
            }
            TrainError::MissingModel => write!(f, "trainer builder is missing a model"),
            TrainError::Optimizer(msg) => write!(f, "optimizer error: {}", msg),
            TrainError::Module(msg) => write!(f, "module error: {}", msg),
            TrainError::Loss(msg) => write!(f, "loss error: {}", msg),
            TrainError::State(msg) => write!(f, "state error: {}", msg),
        }
    }
}

impl std::error::Error for TrainError {}

/// Converts a `brain-core::Tensor` into a `brain-autograd::Value`.
pub fn tensor_to_value(tensor: &Tensor, requires_grad: bool) -> Value {
    Value::new(tensor.clone(), requires_grad)
}

/// Clones the tensor payload from a `brain-autograd::Value`.
pub fn value_to_tensor(value: &Value) -> Tensor {
    value.data().clone()
}

/// A boundary adapter for tensor-only `brain-nn::Module` implementations.
///
/// This adapter lets a module accept and return `Value` payloads, but it does
/// not synthesize gradients for the wrapped module internals. Use trainable
/// layers in this crate for optimizer-backed training.
pub struct TensorModuleAdapter<M> {
    module: M,
}

impl<M> TensorModuleAdapter<M> {
    /// Creates a new adapter around an existing tensor-only module.
    pub fn new(module: M) -> Self {
        Self { module }
    }

    /// Returns the wrapped module.
    pub fn into_inner(self) -> M {
        self.module
    }
}

impl<M: Module> TensorModuleAdapter<M> {
    /// Runs the wrapped module using the tensor payload from a `Value`.
    pub fn forward_value(&self, input: &Value) -> TrainResult<Value> {
        let output = self
            .module
            .forward(input.data())
            .map_err(|err| TrainError::Module(err.to_string()))?;
        Ok(tensor_to_value(&output, input.requires_grad()))
    }
}

/// One mini-batch of dense inputs and integer class targets.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Input tensor, conventionally shaped `[batch, features]`.
    pub inputs: Tensor,
    /// Integer class labels shaped `[batch]`.
    pub targets: Vec<usize>,
}

impl Batch {
    /// Creates a batch.
    pub fn new(inputs: Tensor, targets: Vec<usize>) -> TrainResult<Self> {
        if inputs.ndim() == 0 || inputs.shape()[0] != targets.len() {
            return Err(TrainError::ShapeMismatch {
                expected: vec![targets.len()],
                got: inputs.shape().to_vec(),
            });
        }
        Ok(Self { inputs, targets })
    }

    /// Returns targets as a tensor for APIs that consume tensor targets.
    pub fn target_tensor(&self) -> Tensor {
        Tensor::from_vec(
            self.targets.iter().map(|&target| target as f64).collect(),
            vec![self.targets.len()],
        )
    }
}

/// Deterministic synthetic classification data for examples and smoke tests.
#[derive(Debug, Clone)]
pub struct SyntheticClassification {
    inputs: Tensor,
    targets: Vec<usize>,
}

impl SyntheticClassification {
    /// Builds a two-class, two-feature linearly separable dataset.
    pub fn two_class_points(samples_per_class: usize) -> Self {
        let mut data = Vec::with_capacity(samples_per_class * 4);
        let mut targets = Vec::with_capacity(samples_per_class * 2);

        for i in 0..samples_per_class {
            let jitter_x = (i % 7) as f64 * 0.03;
            let jitter_y = (i % 5) as f64 * 0.04;
            data.push(-1.0 + jitter_x);
            data.push(-0.9 + jitter_y);
            targets.push(0);
        }

        for i in 0..samples_per_class {
            let jitter_x = (i % 7) as f64 * 0.03;
            let jitter_y = (i % 5) as f64 * 0.04;
            data.push(0.9 + jitter_x);
            data.push(1.0 + jitter_y);
            targets.push(1);
        }

        Self {
            inputs: Tensor::from_vec(data, vec![samples_per_class * 2, 2]),
            targets,
        }
    }

    /// Splits the dataset into deterministic contiguous mini-batches.
    pub fn batches(&self, batch_size: usize) -> Vec<Batch> {
        let rows = self.targets.len();
        let cols = self.inputs.shape()[1];
        let data = self.inputs.data();
        let mut batches = Vec::new();
        let mut start = 0;

        while start < rows {
            let end = (start + batch_size).min(rows);
            let mut batch_data = Vec::with_capacity((end - start) * cols);
            for row in start..end {
                batch_data.extend_from_slice(&data[row * cols..(row + 1) * cols]);
            }
            batches.push(Batch {
                inputs: Tensor::from_vec(batch_data, vec![end - start, cols]),
                targets: self.targets[start..end].to_vec(),
            });
            start = end;
        }

        batches
    }
}

/// A trainable dense layer: `y = x W^T + b`.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Creates a deterministic trainable linear layer.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let scale = (2.0 / in_features.max(1) as f64).sqrt() * 0.25;
        let mut values = Vec::with_capacity(in_features * out_features);
        for o in 0..out_features {
            for i in 0..in_features {
                let sign = if (o + i) % 2 == 0 { 1.0 } else { -1.0 };
                values.push(sign * scale * (1.0 + ((o + i) % 3) as f64 * 0.1));
            }
        }

        Self {
            weight: Tensor::from_vec(values, vec![out_features, in_features]),
            bias: bias.then(|| Tensor::zeros(vec![out_features])),
            in_features,
            out_features,
        }
    }

    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        ensure_2d(input, self.in_features)?;
        let batch = input.shape()[0];
        let in_data = input.data();
        let weights = self.weight.data();
        let bias = self.bias.as_ref().map(Tensor::data);
        let mut out = vec![0.0; batch * self.out_features];

        for b in 0..batch {
            for o in 0..self.out_features {
                let mut sum = bias.map(|values| values[o]).unwrap_or(0.0);
                for i in 0..self.in_features {
                    sum += in_data[b * self.in_features + i] * weights[o * self.in_features + i];
                }
                out[b * self.out_features + o] = sum;
            }
        }

        Ok(Tensor::from_vec(out, vec![batch, self.out_features]))
    }

    fn backward(&self, input: &Tensor, grad_output: &Tensor) -> TrainResult<(Tensor, Vec<Tensor>)> {
        ensure_2d(input, self.in_features)?;
        ensure_2d(grad_output, self.out_features)?;
        let batch = input.shape()[0];
        let input_data = input.data();
        let grad_data = grad_output.data();
        let weights = self.weight.data();
        let mut grad_input = vec![0.0; batch * self.in_features];
        let mut grad_weight = vec![0.0; self.out_features * self.in_features];
        let mut grad_bias = vec![0.0; self.out_features];

        for b in 0..batch {
            for o in 0..self.out_features {
                let go = grad_data[b * self.out_features + o];
                grad_bias[o] += go;
                for i in 0..self.in_features {
                    grad_weight[o * self.in_features + i] +=
                        go * input_data[b * self.in_features + i];
                    grad_input[b * self.in_features + i] += go * weights[o * self.in_features + i];
                }
            }
        }

        let mut grads = vec![Tensor::from_vec(
            grad_weight,
            vec![self.out_features, self.in_features],
        )];
        if self.bias.is_some() {
            grads.push(Tensor::from_vec(grad_bias, vec![self.out_features]));
        }

        Ok((
            Tensor::from_vec(grad_input, vec![batch, self.in_features]),
            grads,
        ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn load_parameters(&mut self, params: &[Tensor]) -> TrainResult<usize> {
        if params.is_empty() {
            return Err(TrainError::State(
                "linear layer missing weight tensor".into(),
            ));
        }
        if params[0].shape() != self.weight.shape() {
            return Err(TrainError::ShapeMismatch {
                expected: self.weight.shape().to_vec(),
                got: params[0].shape().to_vec(),
            });
        }
        self.weight = params[0].clone();
        let mut consumed = 1;
        if let Some(bias) = &self.bias {
            let found = params
                .get(1)
                .ok_or_else(|| TrainError::State("linear layer missing bias tensor".into()))?;
            if found.shape() != bias.shape() {
                return Err(TrainError::ShapeMismatch {
                    expected: bias.shape().to_vec(),
                    got: found.shape().to_vec(),
                });
            }
            self.bias = Some(found.clone());
            consumed += 1;
        }
        Ok(consumed)
    }
}

/// ReLU activation layer.
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl ReLU {
    /// Creates a ReLU layer.
    pub fn new() -> Self {
        Self
    }
}

/// Alias with the conventional Rust spelling.
pub type Relu = ReLU;

/// Trainable 2D Convolution layer.
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    /// Creates a trainable Conv2d layer with default stride (1, 1) and same padding.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, bias: bool) -> Self {
        Self::with_config(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (kernel_size / 2, kernel_size / 2),
            bias,
        )
    }

    /// Returns the kernel size (height, width).
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Returns the stride (height, width).
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Returns the padding (height, width).
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// Creates a trainable Conv2d layer with explicit configuration.
    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let scale = (2.0 / fan_in.max(1) as f64).sqrt() * 0.25;
        let num_weights = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        let mut values = Vec::with_capacity(num_weights);
        for i in 0..num_weights {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            values.push(sign * scale * (1.0 + (i % 5) as f64 * 0.05));
        }

        Self {
            weight: Tensor::from_vec(
                values,
                vec![out_channels, in_channels, kernel_size.0, kernel_size.1],
            ),
            bias: bias.then(|| Tensor::zeros(vec![out_channels])),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        if input.ndim() != 4 || input.shape()[1] != self.in_channels {
            return Err(TrainError::ShapeMismatch {
                expected: vec![input.shape().first().copied().unwrap_or(1), self.in_channels, 0, 0],
                got: input.shape().to_vec(),
            });
        }
        let out = brain_core::tensor::conv::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
        );
        Ok(out)
    }

    fn backward(&self, input: &Tensor, grad_output: &Tensor) -> TrainResult<(Tensor, Vec<Tensor>)> {
        let (dinput, dweight, dbias) = brain_autograd::ops::conv_grad::grad_conv2d(
            input,
            &self.weight,
            grad_output,
            self.stride,
            self.padding,
        )
        .map_err(|e| TrainError::Module(e.to_string()))?;

        let mut grads = vec![dweight];
        if self.bias.is_some() {
            grads.push(dbias.unwrap_or_else(|| Tensor::zeros(vec![self.out_channels])));
        }
        Ok((dinput, grads))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn load_parameters(&mut self, params: &[Tensor]) -> TrainResult<usize> {
        if params.is_empty() {
            return Err(TrainError::State("conv2d layer missing weight tensor".into()));
        }
        if params[0].shape() != self.weight.shape() {
            return Err(TrainError::ShapeMismatch {
                expected: self.weight.shape().to_vec(),
                got: params[0].shape().to_vec(),
            });
        }
        self.weight = params[0].clone();
        let mut consumed = 1;
        if let Some(bias) = &self.bias {
            let found = params
                .get(1)
                .ok_or_else(|| TrainError::State("conv2d layer missing bias tensor".into()))?;
            if found.shape() != bias.shape() {
                return Err(TrainError::ShapeMismatch {
                    expected: bias.shape().to_vec(),
                    got: found.shape().to_vec(),
                });
            }
            self.bias = Some(found.clone());
            consumed += 1;
        }
        Ok(consumed)
    }
}

/// 2D Max Pooling layer.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    /// Creates a 2D Max Pooling layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (0, 0),
        }
    }

    /// Creates a 2D Max Pooling layer with explicit padding.
    pub fn with_padding(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        if input.ndim() != 4 {
            return Err(TrainError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                got: input.shape().to_vec(),
            });
        }
        let out = brain_core::tensor::pool::max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
        );
        Ok(out)
    }

    fn backward(&self, input: &Tensor, grad_output: &Tensor) -> TrainResult<Tensor> {
        brain_autograd::ops::pool_grad::grad_max_pool2d(
            input,
            grad_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        .map_err(|e| TrainError::Module(e.to_string()))
    }
}

/// 2D Average Pooling layer.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    /// Creates a 2D Average Pooling layer.
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (0, 0),
        }
    }

    /// Creates a 2D Average Pooling layer with explicit padding.
    pub fn with_padding(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        if input.ndim() != 4 {
            return Err(TrainError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                got: input.shape().to_vec(),
            });
        }
        let out = brain_core::tensor::pool::avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
        );
        Ok(out)
    }

    fn backward(&self, input: &Tensor, grad_output: &Tensor) -> TrainResult<Tensor> {
        brain_autograd::ops::pool_grad::grad_avg_pool2d_ext(
            input.shape(),
            grad_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        .map_err(|e| TrainError::Module(e.to_string()))
    }
}

/// Spatial flattening layer: `[batch, c, h, w]` -> `[batch, c * h * w]`.
#[derive(Debug, Clone, Default)]
pub struct Flatten;

impl Flatten {
    /// Creates a Flatten layer.
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        if input.ndim() < 2 {
            return Err(TrainError::ShapeMismatch {
                expected: vec![1, 1],
                got: input.shape().to_vec(),
            });
        }
        let batch = input.shape()[0];
        let features = input.numel() / batch;
        Ok(input.reshape(vec![batch, features]))
    }

    fn backward(&self, original_shape: &[usize], grad_output: &Tensor) -> TrainResult<Tensor> {
        Ok(grad_output.reshape(original_shape.to_vec()))
    }
}

/// A layer that can be stored in a trainable sequential model.
#[derive(Debug, Clone)]
pub enum Layer {
    /// Dense linear layer.
    Linear(Linear),
    /// ReLU activation.
    ReLU(ReLU),
    /// 2D Convolution layer.
    Conv2d(Conv2d),
    /// 2D Max Pooling layer.
    MaxPool2d(MaxPool2d),
    /// 2D Average Pooling layer.
    AvgPool2d(AvgPool2d),
    /// Spatial Flatten layer.
    Flatten(Flatten),
}

impl From<Linear> for Layer {
    fn from(value: Linear) -> Self {
        Layer::Linear(value)
    }
}

impl From<ReLU> for Layer {
    fn from(value: ReLU) -> Self {
        Layer::ReLU(value)
    }
}

impl From<Conv2d> for Layer {
    fn from(value: Conv2d) -> Self {
        Layer::Conv2d(value)
    }
}

impl From<MaxPool2d> for Layer {
    fn from(value: MaxPool2d) -> Self {
        Layer::MaxPool2d(value)
    }
}

impl From<AvgPool2d> for Layer {
    fn from(value: AvgPool2d) -> Self {
        Layer::AvgPool2d(value)
    }
}

impl From<Flatten> for Layer {
    fn from(value: Flatten) -> Self {
        Layer::Flatten(value)
    }
}

#[derive(Debug, Clone)]
enum LayerCache {
    Linear(Tensor),
    ReLU(Tensor),
    Conv2d(Tensor),
    MaxPool2d(Tensor),
    AvgPool2d(Tensor),
    Flatten(Vec<usize>),
}

/// Trait for mutable modules that expose parameter state and gradients.
pub trait TrainableModule {
    /// Runs a forward pass.
    fn forward(&self, input: &Tensor) -> TrainResult<Tensor>;
    /// Returns a cloned parameter vector in optimizer order.
    fn parameters(&self) -> Vec<Tensor>;
    /// Loads parameters in optimizer order.
    fn load_parameters(&mut self, params: &[Tensor]) -> TrainResult<()>;
    /// Returns stable parameter names in optimizer order.
    fn parameter_names(&self) -> Vec<String>;
}

/// Sequential trainable model.
#[derive(Debug, Clone, Default)]
pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Creates an empty sequential model.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a layer and returns the model for chaining.
    pub fn add<L: Into<Layer>>(mut self, layer: L) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Number of contained layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true when the model contains no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    fn forward_with_cache(&self, input: &Tensor) -> TrainResult<(Tensor, Vec<LayerCache>)> {
        let mut current = input.clone();
        let mut caches = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                Layer::Linear(linear) => {
                    let layer_input = current.clone();
                    current = linear.forward(&current)?;
                    caches.push(LayerCache::Linear(layer_input));
                }
                Layer::ReLU(_) => {
                    let layer_input = current.clone();
                    current = Tensor::from_vec(
                        current.data().iter().map(|value| value.max(0.0)).collect(),
                        current.shape().to_vec(),
                    );
                    caches.push(LayerCache::ReLU(layer_input));
                }
                Layer::Conv2d(conv) => {
                    let layer_input = current.clone();
                    current = conv.forward(&current)?;
                    caches.push(LayerCache::Conv2d(layer_input));
                }
                Layer::MaxPool2d(mp) => {
                    let layer_input = current.clone();
                    current = mp.forward(&current)?;
                    caches.push(LayerCache::MaxPool2d(layer_input));
                }
                Layer::AvgPool2d(ap) => {
                    let layer_input = current.clone();
                    current = ap.forward(&current)?;
                    caches.push(LayerCache::AvgPool2d(layer_input));
                }
                Layer::Flatten(flat) => {
                    let orig_shape = current.shape().to_vec();
                    current = flat.forward(&current)?;
                    caches.push(LayerCache::Flatten(orig_shape));
                }
            }
        }

        Ok((current, caches))
    }

    fn backward_from_cache(
        &self,
        caches: &[LayerCache],
        grad_output: &Tensor,
    ) -> TrainResult<Vec<Tensor>> {
        let mut grad = grad_output.clone();
        let mut param_grads_rev = Vec::new();

        for (layer, cache) in self.layers.iter().zip(caches.iter()).rev() {
            match (layer, cache) {
                (Layer::Linear(linear), LayerCache::Linear(input)) => {
                    let (grad_input, grads) = linear.backward(input, &grad)?;
                    param_grads_rev.extend(grads.into_iter().rev());
                    grad = grad_input;
                }
                (Layer::ReLU(_), LayerCache::ReLU(input)) => {
                    let masked: Vec<f64> = grad
                        .data()
                        .iter()
                        .zip(input.data())
                        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                        .collect();
                    grad = Tensor::from_vec(masked, grad.shape().to_vec());
                }
                (Layer::Conv2d(conv), LayerCache::Conv2d(input)) => {
                    let (grad_input, grads) = conv.backward(input, &grad)?;
                    param_grads_rev.extend(grads.into_iter().rev());
                    grad = grad_input;
                }
                (Layer::MaxPool2d(mp), LayerCache::MaxPool2d(input)) => {
                    grad = mp.backward(input, &grad)?;
                }
                (Layer::AvgPool2d(ap), LayerCache::AvgPool2d(input)) => {
                    grad = ap.backward(input, &grad)?;
                }
                (Layer::Flatten(flat), LayerCache::Flatten(orig_shape)) => {
                    grad = flat.backward(orig_shape, &grad)?;
                }
                _ => return Err(TrainError::State("layer cache did not match model".into())),
            }
        }

        param_grads_rev.reverse();
        Ok(param_grads_rev)
    }

    /// Exports parameter state for checkpointing, distribution, or export adapters.
    pub fn state(&self) -> ModelState {
        let names = self.parameter_names();
        let tensors = self
            .parameters()
            .into_iter()
            .zip(names)
            .map(|(tensor, name)| NamedTensor { name, tensor })
            .collect();

        ModelState {
            tensors,
            metadata: HashMap::from([
                ("format".to_string(), "brain-train-state-v1".to_string()),
                ("layers".to_string(), self.layers.len().to_string()),
            ]),
        }
    }
}

impl TrainableModule for Sequential {
    fn forward(&self, input: &Tensor) -> TrainResult<Tensor> {
        self.forward_with_cache(input).map(|(output, _)| output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| match layer {
                Layer::Linear(linear) => linear.parameters(),
                Layer::Conv2d(conv) => conv.parameters(),
                Layer::ReLU(_)
                | Layer::MaxPool2d(_)
                | Layer::AvgPool2d(_)
                | Layer::Flatten(_) => Vec::new(),
            })
            .collect()
    }

    fn load_parameters(&mut self, params: &[Tensor]) -> TrainResult<()> {
        let mut cursor = 0;
        for layer in &mut self.layers {
            match layer {
                Layer::Linear(linear) => {
                    cursor += linear.load_parameters(&params[cursor..])?;
                }
                Layer::Conv2d(conv) => {
                    cursor += conv.load_parameters(&params[cursor..])?;
                }
                Layer::ReLU(_)
                | Layer::MaxPool2d(_)
                | Layer::AvgPool2d(_)
                | Layer::Flatten(_) => {}
            }
        }
        if cursor != params.len() {
            return Err(TrainError::State(format!(
                "unused parameter tensors: expected {}, got {}",
                cursor,
                params.len()
            )));
        }
        Ok(())
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Linear(linear) => {
                    names.push(format!("layers.{}.weight", idx));
                    if linear.bias.is_some() {
                        names.push(format!("layers.{}.bias", idx));
                    }
                }
                Layer::Conv2d(conv) => {
                    names.push(format!("layers.{}.weight", idx));
                    if conv.bias.is_some() {
                        names.push(format!("layers.{}.bias", idx));
                    }
                }
                Layer::ReLU(_)
                | Layer::MaxPool2d(_)
                | Layer::AvgPool2d(_)
                | Layer::Flatten(_) => {}
            }
        }
        names
    }
}

/// One named tensor in a model state dictionary.
#[derive(Debug, Clone)]
pub struct NamedTensor {
    /// Stable parameter name.
    pub name: String,
    /// Tensor payload.
    pub tensor: Tensor,
}

/// Serializable model parameter state.
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Named tensors in optimizer order.
    pub tensors: Vec<NamedTensor>,
    /// Metadata available to checkpoint/export/distributed systems.
    pub metadata: HashMap<String, String>,
}

impl ModelState {
    /// Encodes state as a small deterministic Brain text checkpoint.
    pub fn to_brain_bytes(&self) -> Vec<u8> {
        let mut out = String::from("BRAIN_STATE_V1\n");
        for (key, value) in &self.metadata {
            out.push_str("meta|");
            out.push_str(key);
            out.push('|');
            out.push_str(value);
            out.push('\n');
        }
        for named in &self.tensors {
            out.push_str("tensor|");
            out.push_str(&named.name);
            out.push('|');
            out.push_str(
                &named
                    .tensor
                    .shape()
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
            );
            out.push('|');
            out.push_str(
                &named
                    .tensor
                    .data()
                    .iter()
                    .map(f64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push('\n');
        }
        out.into_bytes()
    }

    /// Extracts parameter tensors in order.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.tensors.iter().map(|nt| nt.tensor.clone()).collect()
    }

    /// Serializes model state to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_brain_bytes()
    }

    /// Deserializes model state from bytes.
    pub fn from_bytes(bytes: &[u8]) -> TrainResult<Self> {
        Self::from_brain_bytes(bytes)
    }

    /// Decodes a Brain text checkpoint produced by [`ModelState::to_brain_bytes`].
    pub fn from_brain_bytes(bytes: &[u8]) -> TrainResult<Self> {
        let text = std::str::from_utf8(bytes).map_err(|err| TrainError::State(err.to_string()))?;
        let mut lines = text.lines();
        if lines.next() != Some("BRAIN_STATE_V1") {
            return Err(TrainError::State("missing BRAIN_STATE_V1 header".into()));
        }

        let mut metadata = HashMap::new();
        let mut tensors = Vec::new();
        for line in lines {
            let parts = line.split('|').collect::<Vec<_>>();
            match parts.as_slice() {
                ["meta", key, value] => {
                    metadata.insert((*key).to_string(), (*value).to_string());
                }
                ["tensor", name, shape, values] => {
                    let shape = if shape.is_empty() {
                        Vec::new()
                    } else {
                        shape
                            .split('x')
                            .map(|part| {
                                part.parse::<usize>()
                                    .map_err(|err| TrainError::State(err.to_string()))
                            })
                            .collect::<TrainResult<Vec<_>>>()?
                    };
                    let data = if values.is_empty() {
                        Vec::new()
                    } else {
                        values
                            .split(',')
                            .map(|part| {
                                part.parse::<f64>()
                                    .map_err(|err| TrainError::State(err.to_string()))
                            })
                            .collect::<TrainResult<Vec<_>>>()?
                    };
                    tensors.push(NamedTensor {
                        name: (*name).to_string(),
                        tensor: Tensor::from_vec(data, shape),
                    });
                }
                _ => return Err(TrainError::State(format!("invalid state line: {}", line))),
            }
        }

        Ok(Self { tensors, metadata })
    }

    /// Returns cloned tensors in optimizer/model order.
    pub fn tensors(&self) -> Vec<Tensor> {
        self.tensors
            .iter()
            .map(|named| named.tensor.clone())
            .collect()
    }
}

/// A regularization hook that contributes additional parameter gradients.
pub trait Regularizer: Send + Sync {
    /// Returns additive gradients in model parameter order.
    fn gradients(&self, model: &Sequential) -> Vec<Tensor>;
}

/// L2 weight decay as a composable training hook.
#[derive(Debug, Clone, Copy)]
pub struct L2Regularization {
    /// L2 gradient multiplier.
    pub lambda: f64,
}

impl L2Regularization {
    /// Creates an L2 regularizer.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Regularizer for L2Regularization {
    fn gradients(&self, model: &Sequential) -> Vec<Tensor> {
        model
            .parameters()
            .into_iter()
            .map(|param| {
                Tensor::from_vec(
                    param
                        .data()
                        .iter()
                        .map(|value| self.lambda * value)
                        .collect(),
                    param.shape().to_vec(),
                )
            })
            .collect()
    }
}

/// Result from a single training step.
#[derive(Debug, Clone)]
pub struct TrainStep {
    /// Cross-entropy loss value.
    pub loss: f64,
    /// Batch accuracy.
    pub accuracy: f64,
    /// Optimizer step metadata.
    pub optimizer: StepInfo,
}

/// Aggregate training metrics.
#[derive(Debug, Clone, Default)]
pub struct TrainingSummary {
    /// Number of optimizer steps.
    pub steps: usize,
    /// Mean loss over recorded steps.
    pub loss: f64,
    /// Accuracy over all examples seen.
    pub accuracy: f64,
}

/// Builder for [`Trainer`].
pub struct TrainerBuilder {
    model: Option<Sequential>,
    loss: CrossEntropyLoss,
    optimizer: Option<Sgd>,
    learning_rate: f64,
    regularizers: Vec<Box<dyn Regularizer>>,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self {
            model: None,
            loss: CrossEntropyLoss::default(),
            optimizer: None,
            learning_rate: 0.1,
            regularizers: Vec::new(),
        }
    }
}

impl TrainerBuilder {
    /// Supplies the trainable model.
    pub fn model(mut self, model: Sequential) -> Self {
        self.model = Some(model);
        self
    }

    /// Supplies a cross-entropy loss implementation from `brain-loss`.
    pub fn loss(mut self, loss: CrossEntropyLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Supplies an SGD optimizer from `brain-optim`.
    pub fn optimizer(mut self, optimizer: Sgd) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Sets the learning rate used by the default SGD optimizer.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Adds a regularization hook.
    pub fn regularizer<R: Regularizer + 'static>(mut self, regularizer: R) -> Self {
        self.regularizers.push(Box::new(regularizer));
        self
    }

    /// Builds a trainer.
    pub fn build(self) -> TrainResult<Trainer> {
        let model = self.model.ok_or(TrainError::MissingModel)?;
        let params = model.parameters();
        let param_ids = (0..params.len()).collect::<Vec<_>>();
        let optimizer = self.optimizer.unwrap_or_else(|| {
            Sgd::new(
                vec![ParamGroup::new(param_ids, self.learning_rate)],
                SgdConfig {
                    lr: self.learning_rate,
                    ..SgdConfig::default()
                },
            )
        });

        Ok(Trainer {
            model,
            loss: self.loss,
            optimizer,
            regularizers: self.regularizers,
            seen_preds: Vec::new(),
            seen_targets: Vec::new(),
            steps: Vec::new(),
        })
    }
}

/// A concrete trainer for deterministic dense classification.
pub struct Trainer {
    /// Trainable model.
    pub model: Sequential,
    loss: CrossEntropyLoss,
    optimizer: Sgd,
    regularizers: Vec<Box<dyn Regularizer>>,
    seen_preds: Vec<usize>,
    seen_targets: Vec<usize>,
    steps: Vec<TrainStep>,
}

impl Trainer {
    /// Starts building a trainer.
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::default()
    }

    /// Runs one optimizer-backed training step.
    pub fn train_batch(&mut self, batch: &Batch) -> TrainResult<TrainStep> {
        let (logits, caches) = self.model.forward_with_cache(&batch.inputs)?;
        let loss_tensor = self
            .loss
            .compute(&logits, &batch.targets)
            .map_err(|err| TrainError::Loss(err.to_string()))?;
        let loss = loss_tensor.get(0);
        let grad_logits = cross_entropy_grad(&logits, &batch.targets)?;
        let mut grads = self.model.backward_from_cache(&caches, &grad_logits)?;

        for reg_grads in self
            .regularizers
            .iter()
            .map(|reg| reg.gradients(&self.model))
        {
            for (grad, reg_grad) in grads.iter_mut().zip(reg_grads) {
                for (value, add) in grad.data_mut().iter_mut().zip(reg_grad.data()) {
                    *value += add;
                }
            }
        }

        let mut params = self.model.parameters();
        let optimizer = self
            .optimizer
            .step(&mut params, &grads)
            .map_err(|err| TrainError::Optimizer(err.to_string()))?;
        self.model.load_parameters(&params)?;

        let preds = argmax_rows(&logits)?;
        let accuracy = accuracy_score(&preds, &batch.targets);
        self.seen_preds.extend_from_slice(&preds);
        self.seen_targets.extend_from_slice(&batch.targets);

        let step = TrainStep {
            loss,
            accuracy,
            optimizer,
        };
        self.steps.push(step.clone());
        Ok(step)
    }

    /// Runs multiple epochs over a fixed set of batches.
    pub fn fit(&mut self, batches: &[Batch], epochs: usize) -> TrainResult<TrainingSummary> {
        for _ in 0..epochs {
            for batch in batches {
                self.train_batch(batch)?;
            }
        }
        Ok(self.summary())
    }

    /// Runs multiple epochs with gradient accumulation over micro-batches.
    pub fn fit_accumulated(
        &mut self,
        batches: &[Batch],
        epochs: usize,
        accum_steps: usize,
    ) -> TrainResult<TrainingSummary> {
        let accum = accum_steps.max(1);
        for _ in 0..epochs {
            let mut accum_grads: Option<Vec<Tensor>> = None;
            for (idx, batch) in batches.iter().enumerate() {
                let (logits, caches) = self.model.forward_with_cache(&batch.inputs)?;
                let loss_tensor = self
                    .loss
                    .compute(&logits, &batch.targets)
                    .map_err(|err| TrainError::Loss(err.to_string()))?;
                let loss = loss_tensor.get(0);
                let grad_logits = cross_entropy_grad(&logits, &batch.targets)?;
                let grads = self.model.backward_from_cache(&caches, &grad_logits)?;

                let ag = accum_grads.get_or_insert_with(|| {
                    grads.iter().map(|g| Tensor::zeros(g.shape().to_vec())).collect()
                });
                for (acc, g) in ag.iter_mut().zip(&grads) {
                    for (acc_val, g_val) in acc.data_mut().iter_mut().zip(g.data()) {
                        *acc_val += g_val / (accum as f64);
                    }
                }

                let preds = argmax_rows(&logits)?;
                self.seen_preds.extend_from_slice(&preds);
                self.seen_targets.extend_from_slice(&batch.targets);

                if (idx + 1) % accum == 0 || idx + 1 == batches.len() {
                    let mut params = self.model.parameters();
                    let opt_info = self
                        .optimizer
                        .step(&mut params, ag)
                        .map_err(|err| TrainError::Optimizer(err.to_string()))?;
                    self.model.load_parameters(&params)?;
                    for g in ag.iter_mut() {
                        for v in g.data_mut().iter_mut() {
                            *v = 0.0;
                        }
                    }
                    self.steps.push(TrainStep {
                        loss,
                        accuracy: accuracy_score(&preds, &batch.targets),
                        optimizer: opt_info,
                    });
                }
            }
        }
        Ok(self.summary())
    }

    /// Evaluates batches without updating parameters.
    pub fn evaluate(&self, batches: &[Batch]) -> TrainResult<TrainingSummary> {
        let mut preds = Vec::new();
        let mut targets = Vec::new();
        let mut losses = Vec::new();

        for batch in batches {
            let logits = self.model.forward(&batch.inputs)?;
            let loss = self
                .loss
                .compute(&logits, &batch.targets)
                .map_err(|err| TrainError::Loss(err.to_string()))?
                .get(0);
            losses.push(loss);
            preds.extend(argmax_rows(&logits)?);
            targets.extend_from_slice(&batch.targets);
        }

        Ok(TrainingSummary {
            steps: batches.len(),
            loss: mean(&losses),
            accuracy: accuracy_score(&preds, &targets),
        })
    }

    /// Returns an aggregate summary of training so far.
    pub fn summary(&self) -> TrainingSummary {
        TrainingSummary {
            steps: self.steps.len(),
            loss: mean(&self.steps.iter().map(|step| step.loss).collect::<Vec<_>>()),
            accuracy: accuracy_score(&self.seen_preds, &self.seen_targets),
        }
    }

    /// Exports model parameters and training metadata.
    pub fn state(&self) -> ModelState {
        let mut state = self.model.state();
        state.metadata.insert(
            "optimizer_steps".into(),
            self.optimizer.get_step_count().to_string(),
        );
        state
    }

    /// Loads model parameter state into this trainer's model.
    pub fn load_state(&mut self, state: &ModelState) -> TrainResult<()> {
        let params = state.parameters();
        self.model.load_parameters(&params)
    }
}

/// Computes row-wise argmax predictions.
pub fn argmax_rows(logits: &Tensor) -> TrainResult<Vec<usize>> {
    if logits.ndim() != 2 {
        return Err(TrainError::ShapeMismatch {
            expected: vec![0, 0],
            got: logits.shape().to_vec(),
        });
    }
    let rows = logits.shape()[0];
    let cols = logits.shape()[1];
    let mut preds = Vec::with_capacity(rows);
    for row in 0..rows {
        let offset = row * cols;
        let mut best_idx = 0;
        let mut best_value = logits.get(offset);
        for col in 1..cols {
            let value = logits.get(offset + col);
            if value > best_value {
                best_value = value;
                best_idx = col;
            }
        }
        preds.push(best_idx);
    }
    Ok(preds)
}

fn ensure_2d(tensor: &Tensor, last_dim: usize) -> TrainResult<()> {
    if tensor.ndim() != 2 || tensor.shape()[1] != last_dim {
        return Err(TrainError::ShapeMismatch {
            expected: vec![0, last_dim],
            got: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

fn cross_entropy_grad(logits: &Tensor, targets: &[usize]) -> TrainResult<Tensor> {
    if logits.ndim() != 2 || logits.shape()[0] != targets.len() {
        return Err(TrainError::ShapeMismatch {
            expected: vec![targets.len(), 0],
            got: logits.shape().to_vec(),
        });
    }

    let rows = logits.shape()[0];
    let cols = logits.shape()[1];
    let mut grad = vec![0.0; rows * cols];

    for row in 0..rows {
        let target = targets[row];
        if target >= cols {
            return Err(TrainError::InvalidTarget {
                target,
                classes: cols,
            });
        }

        let offset = row * cols;
        let row_values = &logits.data()[offset..offset + cols];
        let max_value = row_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_values = row_values
            .iter()
            .map(|value| (value - max_value).exp())
            .collect::<Vec<_>>();
        let denom = exp_values.iter().sum::<f64>();

        for col in 0..cols {
            let mut prob = exp_values[col] / denom;
            if col == target {
                prob -= 1.0;
            }
            grad[offset + col] = prob / rows as f64;
        }
    }

    Ok(Tensor::from_vec(grad, vec![rows, cols]))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Common imports for training Brain models.
pub mod prelude {
    pub use crate::{
        argmax_rows, tensor_to_value, value_to_tensor, AvgPool2d, Batch, Conv2d, Flatten,
        L2Regularization, Layer, Linear, MaxPool2d, ModelState, NamedTensor, ReLU, Relu, Sequential,
        SyntheticClassification, TensorModuleAdapter, TrainError, TrainResult, TrainStep,
        TrainableModule, Trainer, TrainerBuilder, TrainingSummary,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_mlp_trains_and_serializes_state() {
        let data = SyntheticClassification::two_class_points(8);
        let batches = data.batches(4);
        let model = Sequential::new()
            .add(Linear::new(2, 4, true))
            .add(ReLU::new())
            .add(Linear::new(4, 2, true));
        let mut trainer = Trainer::builder()
            .model(model)
            .learning_rate(0.2)
            .regularizer(L2Regularization::new(1e-5))
            .build()
            .unwrap();

        let before = trainer.evaluate(&batches).unwrap();
        let after = trainer.fit(&batches, 8).unwrap();
        assert!(after.loss <= before.loss);
        assert!(after.accuracy >= 0.5);

        let encoded = trainer.state().to_brain_bytes();
        let decoded = ModelState::from_brain_bytes(&encoded).unwrap();
        assert_eq!(decoded.tensors.len(), trainer.model.parameters().len());
    }

    #[test]
    fn synthetic_cnn_trains_end_to_end() {
        // Create synthetic 4D images: 8 samples of [1, 6, 6]
        let mut data_vec = Vec::with_capacity(8 * 1 * 6 * 6);
        let mut targets = Vec::with_capacity(8);

        // Class 0: top-left bright
        for i in 0..4 {
            let mut img = vec![0.1; 36];
            img[0] = 2.0 + i as f64 * 0.1;
            img[1] = 2.0;
            img[6] = 2.0;
            data_vec.extend(img);
            targets.push(0);
        }

        // Class 1: bottom-right bright
        for i in 0..4 {
            let mut img = vec![0.1; 36];
            img[35] = 2.0 + i as f64 * 0.1;
            img[34] = 2.0;
            img[29] = 2.0;
            data_vec.extend(img);
            targets.push(1);
        }

        let inputs = Tensor::from_vec(data_vec, vec![8, 1, 6, 6]);
        let batch = Batch::new(inputs, targets).unwrap();
        let batches = vec![batch];

        // CNN architecture: Conv2d(1->4, 3x3) -> ReLU -> MaxPool2d(2x2) -> Flatten -> Linear(4*3*3 -> 2)
        let model = Sequential::new()
            .add(Conv2d::new(1, 4, 3, true))
            .add(ReLU::new())
            .add(MaxPool2d::new(2, 2))
            .add(Flatten::new())
            .add(Linear::new(4 * 3 * 3, 2, true));

        let mut trainer = Trainer::builder()
            .model(model)
            .learning_rate(0.1)
            .build()
            .unwrap();

        let before = trainer.evaluate(&batches).unwrap();
        let after = trainer.fit(&batches, 15).unwrap();

        assert!(
            after.loss < before.loss,
            "CNN loss should strictly decrease: before={}, after={}",
            before.loss,
            after.loss
        );
        assert!(after.accuracy >= 0.85, "CNN should achieve high accuracy: got {}", after.accuracy);
    }

    #[test]
    fn tensor_value_conversion_preserves_payload() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let value = tensor_to_value(&tensor, true);
        assert!(value.requires_grad());
        assert_eq!(value_to_tensor(&value).to_vec(), vec![1.0, 2.0]);
    }
}
