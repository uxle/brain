//! Neural network modules in Python.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use brain_nn::{Linear, Conv2d, LayerNorm, Module};
use crate::tensor::PyTensor;
use crate::autograd::PyValue;

#[cfg_attr(feature = "extension-module", pyclass(name = "Linear"))]
pub struct PyLinear {
    pub inner: Linear,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            inner: Linear::new(in_features, out_features, bias),
        }
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward_tensor(&x.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Linear forward error: {:?}", e)))?;
        Ok(PyTensor::new(out))
    }

    pub fn parameters(&self) -> Vec<PyValue> {
        self.inner.parameters().into_iter().map(PyValue::new).collect()
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "Conv2d"))]
pub struct PyConv2d {
    pub inner: Conv2d,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, bias=true))]
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, bias: bool) -> Self {
        Self {
            inner: Conv2d::new(in_channels, out_channels, kernel_size, bias),
        }
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward_tensor(&x.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Conv2d forward error: {:?}", e)))?;
        Ok(PyTensor::new(out))
    }

    pub fn parameters(&self) -> Vec<PyValue> {
        self.inner.parameters().into_iter().map(PyValue::new).collect()
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "LayerNorm"))]
pub struct PyLayerNorm {
    pub inner: LayerNorm,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Self {
        Self {
            inner: LayerNorm::new(normalized_shape, eps),
        }
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let out = self.inner.forward(&x.inner);
        Ok(PyTensor::new(out))
    }

    pub fn parameters(&self) -> Vec<PyValue> {
        self.inner.parameters().into_iter().map(PyValue::new).collect()
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "MSELoss"))]
#[derive(Default)]
pub struct PyMSELoss {}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyMSELoss {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward(&self, pred: &PyValue, target: &PyValue) -> PyValue {
        let diff = pred.__sub__(target);
        let sq = diff.__mul__(&diff);
        sq.sum()
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "CrossEntropyLoss"))]
#[derive(Default)]
pub struct PyCrossEntropyLoss {}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward(&self, logits: &PyValue, target: &PyValue) -> PyValue {
        let diff = logits.__sub__(target);
        let sq = diff.__mul__(&diff);
        sq.sum()
    }
}
