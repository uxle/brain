//! Python wrapper for autograd and gradient tape operations.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use brain_autograd::Value;
use brain_core::Tensor;
use crate::tensor::PyTensor;

#[cfg_attr(feature = "extension-module", pyclass(name = "Value"))]
#[derive(Clone)]
pub struct PyValue {
    pub inner: Value,
}

impl PyValue {
    pub fn new(inner: Value) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(signature = (data, shape=None, requires_grad=true))]
    pub fn py_new(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data elements ({}) does not match shape total ({})",
                data.len(), total
            )));
        }
        let t = Tensor::from_vec(data, shape);
        Ok(Self::new(Value::new(t, requires_grad)))
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.data().shape().to_vec()
    }

    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    pub fn data(&self) -> PyTensor {
        PyTensor::new((*self.inner.data()).clone())
    }

    #[getter]
    pub fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::new)
    }

    pub fn backward(&self) -> PyResult<()> {
        self.inner.backward()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Autograd backward error: {:?}", e)))
    }

    pub fn backward_with_grad(&self, seed: &PyTensor) -> PyResult<()> {
        self.inner.backward_with_grad(&seed.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Autograd backward error: {:?}", e)))
    }

    pub fn item(&self) -> PyResult<f64> {
        if self.inner.data().numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("item() only valid for scalar values"));
        }
        Ok(self.inner.data().item())
    }

    pub fn to_list(&self) -> Vec<f64> {
        self.inner.data().to_vec()
    }

    pub fn relu(&self) -> Self {
        Self::new(self.inner.relu())
    }

    pub fn sigmoid(&self) -> Self {
        Self::new(self.inner.sigmoid())
    }

    pub fn tanh(&self) -> Self {
        Self::new(self.inner.tanh())
    }

    pub fn exp(&self) -> Self {
        Self::new(self.inner.exp())
    }

    pub fn log(&self) -> Self {
        Self::new(self.inner.log())
    }

    pub fn sqrt(&self) -> Self {
        Self::new(self.inner.sqrt())
    }

    pub fn sum(&self) -> Self {
        Self::new(self.inner.sum())
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        Self::new(self.inner.reshape(new_shape))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        Self::new(self.inner.transpose(dim0, dim1))
    }

    pub fn __add__(&self, other: &PyValue) -> Self {
        Self::new(self.inner.add(&other.inner))
    }

    pub fn __sub__(&self, other: &PyValue) -> Self {
        Self::new(self.inner.sub(&other.inner))
    }

    pub fn __mul__(&self, other: &PyValue) -> Self {
        Self::new(self.inner.mul(&other.inner))
    }

    pub fn __truediv__(&self, other: &PyValue) -> Self {
        Self::new(self.inner.div(&other.inner))
    }

    pub fn __matmul__(&self, other: &PyValue) -> Self {
        Self::new(self.inner.matmul(&other.inner))
    }

    pub fn __repr__(&self) -> String {
        format!("brain.Value(shape={:?}, requires_grad={}, data={:?})",
            self.inner.data().shape(),
            self.inner.requires_grad(),
            self.inner.data().to_vec()
        )
    }
}
