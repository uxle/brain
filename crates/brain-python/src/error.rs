//! Error conversion for Python exceptions.

#[cfg(feature = "extension-module")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "extension-module")]
use pyo3::PyErr;
use brain_core::BrainError;

#[cfg(feature = "extension-module")]
pub fn to_py_err(err: BrainError) -> PyErr {
    match err {
        BrainError::ShapeMismatch { .. } => PyValueError::new_err(err.to_string()),
        BrainError::IndexOutOfBounds { .. } => PyValueError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}
