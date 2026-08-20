//! Python wrapper for Brain optimizers.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use brain_optim::{Adam, AdamConfig, Sgd, SgdConfig, ParamGroup, Optimizer};
use brain_core::Tensor;
use crate::autograd::PyValue;

#[cfg_attr(feature = "extension-module", pyclass(name = "Adam"))]
pub struct PyAdam {
    pub inner: Adam,
    pub params: Vec<Py<PyValue>>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
    pub fn new(
        params: Vec<Py<PyValue>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let cfg = AdamConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad: false,
            decoupled_weight_decay: false,
        };
        let param_indices: Vec<usize> = (0..params.len()).collect();
        let group = ParamGroup::new(param_indices, lr);
        Self {
            inner: Adam::new(vec![group], cfg),
            params,
        }
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut param_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());
        let mut grad_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());

        for (idx, p) in self.params.iter().enumerate() {
            let val = p.borrow(py);
            param_tensors.push((*val.inner.data()).clone());
            match val.inner.grad() {
                Some(g) => grad_tensors.push(g),
                None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Param {} has no gradient (did you call loss.backward()?)",
                        idx
                    )));
                }
            }
        }

        self.inner
            .step(&mut param_tensors, &grad_tensors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Optimizer step error: {:?}", e)))?;

        for (idx, p) in self.params.iter().enumerate() {
            let mut val = p.borrow_mut(py);
            val.inner.set_data(param_tensors[idx].clone());
        }

        Ok(())
    }

    pub fn zero_grad(&mut self, py: Python<'_>) {
        for p in &self.params {
            p.borrow(py).inner.zero_grad();
        }
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "AdamW"))]
pub struct PyAdamW {
    pub inner: Adam,
    pub params: Vec<Py<PyValue>>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01))]
    pub fn new(
        params: Vec<Py<PyValue>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let cfg = AdamConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad: false,
            decoupled_weight_decay: true,
        };
        let param_indices: Vec<usize> = (0..params.len()).collect();
        let group = ParamGroup::new(param_indices, lr);
        Self {
            inner: Adam::new(vec![group], cfg),
            params,
        }
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut param_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());
        let mut grad_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());

        for (idx, p) in self.params.iter().enumerate() {
            let val = p.borrow(py);
            param_tensors.push((*val.inner.data()).clone());
            match val.inner.grad() {
                Some(g) => grad_tensors.push(g),
                None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Param {} has no gradient (did you call loss.backward()?)",
                        idx
                    )));
                }
            }
        }

        self.inner
            .step(&mut param_tensors, &grad_tensors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Optimizer step error: {:?}", e)))?;

        for (idx, p) in self.params.iter().enumerate() {
            let mut val = p.borrow_mut(py);
            val.inner.set_data(param_tensors[idx].clone());
        }

        Ok(())
    }

    pub fn zero_grad(&mut self, py: Python<'_>) {
        for p in &self.params {
            p.borrow(py).inner.zero_grad();
        }
    }
}

#[cfg_attr(feature = "extension-module", pyclass(name = "SGD"))]
pub struct PySgd {
    pub inner: Sgd,
    pub params: Vec<Py<PyValue>>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySgd {
    #[new]
    #[pyo3(signature = (params, lr=1e-2, momentum=0.9, weight_decay=0.0, nesterov=false))]
    pub fn new(
        params: Vec<Py<PyValue>>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Self {
        let cfg = SgdConfig {
            lr,
            momentum,
            dampening: 0.0,
            weight_decay,
            nesterov,
            decoupled_weight_decay: false,
        };
        let param_indices: Vec<usize> = (0..params.len()).collect();
        let group = ParamGroup::new(param_indices, lr);
        Self {
            inner: Sgd::new(vec![group], cfg),
            params,
        }
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut param_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());
        let mut grad_tensors: Vec<Tensor> = Vec::with_capacity(self.params.len());

        for (idx, p) in self.params.iter().enumerate() {
            let val = p.borrow(py);
            param_tensors.push((*val.inner.data()).clone());
            match val.inner.grad() {
                Some(g) => grad_tensors.push(g),
                None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Param {} has no gradient (did you call loss.backward()?)",
                        idx
                    )));
                }
            }
        }

        self.inner
            .step(&mut param_tensors, &grad_tensors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Optimizer step error: {:?}", e)))?;

        for (idx, p) in self.params.iter().enumerate() {
            let mut val = p.borrow_mut(py);
            val.inner.set_data(param_tensors[idx].clone());
        }

        Ok(())
    }

    pub fn zero_grad(&mut self, py: Python<'_>) {
        for p in &self.params {
            p.borrow(py).inner.zero_grad();
        }
    }
}
