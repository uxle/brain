//! Python bindings for the Brain Framework.
#![allow(missing_docs)]

pub mod autograd;
pub mod error;
pub mod nn;
pub mod optim;
pub mod tensor;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pymodule]
fn brain_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<autograd::PyValue>()?;
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyLayerNorm>()?;
    m.add_class::<nn::PyMSELoss>()?;
    m.add_class::<nn::PyCrossEntropyLoss>()?;
    m.add_class::<optim::PyAdam>()?;
    m.add_class::<optim::PyAdamW>()?;
    m.add_class::<optim::PySgd>()?;
    m.add_function(wrap_pyfunction!(tensor::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::tensor, m)?)?;
    Ok(())
}
