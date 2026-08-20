//! Python wrapper around brain_core::Tensor.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use brain_core::Tensor;
use brain_core::tensor::arithmetic as arith;
use brain_core::tensor::reduction as red;

#[cfg_attr(feature = "extension-module", pyclass(name = "Tensor"))]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor,
    pub grad: Option<Box<PyTensor>>,
    pub requires_grad: bool,
}

impl PyTensor {
    pub fn new(inner: Tensor) -> Self {
        Self {
            inner,
            grad: None,
            requires_grad: false,
        }
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape=None))]
    pub fn py_new(data: Vec<f64>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data elements ({}) does not match shape total ({})",
                data.len(), total
            )));
        }
        Ok(Self::new(Tensor::from_vec(data, shape)))
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[setter]
    pub fn set_requires_grad(&mut self, req: bool) {
        self.requires_grad = req;
    }

    #[getter]
    pub fn grad(&self) -> Option<PyTensor> {
        self.grad.as_deref().cloned()
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn to_list(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    pub fn item(&self) -> PyResult<f64> {
        if self.inner.numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("item() only valid for single-element tensors"));
        }
        Ok(self.inner.item())
    }

    pub fn __getitem__(&self, idx: isize) -> PyResult<f64> {
        let n = self.inner.numel() as isize;
        let actual_idx = if idx < 0 { n + idx } else { idx };
        if actual_idx < 0 || actual_idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of bounds for tensor with {} elements", idx, n
            )));
        }
        Ok(self.inner.data()[actual_idx as usize])
    }

    pub fn backward(&mut self) -> PyResult<()> {
        let val = brain_autograd::Value::new(self.inner.clone(), true);
        val.backward()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Autograd backward error: {:?}", e)))?;
        if let Some(g) = val.grad() {
            self.grad = Some(Box::new(PyTensor::new(g)));
        }
        Ok(())
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::new(self.inner.reshape(new_shape)))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        Self::new(self.inner.transpose(dim0, dim1))
    }

    pub fn sum(&self) -> Self {
        Self::new(Tensor::from_slice(&[red::sum(&self.inner)], vec![]))
    }

    pub fn mean(&self) -> Self {
        Self::new(Tensor::from_slice(&[red::mean(&self.inner)], vec![]))
    }

    // Operator Overloading
    pub fn __add__(&self, other: &PyTensor) -> Self {
        Self::new(arith::add(&self.inner, &other.inner))
    }

    pub fn __sub__(&self, other: &PyTensor) -> Self {
        Self::new(arith::sub(&self.inner, &other.inner))
    }

    pub fn __mul__(&self, other: &PyTensor) -> Self {
        Self::new(arith::mul(&self.inner, &other.inner))
    }

    pub fn __matmul__(&self, other: &PyTensor) -> Self {
        Self::new(arith::matmul(&self.inner, &other.inner))
    }

    pub fn __repr__(&self) -> String {
        format!("brain.Tensor(shape={:?}, data={:?})", self.inner.shape(), self.inner.to_vec())
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (shape))]
pub fn zeros(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(Tensor::zeros(shape))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (shape))]
pub fn ones(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(Tensor::ones(shape))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (data, shape=None))]
pub fn tensor(data: Vec<f64>, shape: Option<Vec<usize>>) -> PyResult<PyTensor> {
    PyTensor::py_new(data, shape)
}
