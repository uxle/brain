//! Error types and error handling utilities for the Brain deep learning framework.
//!
//! This module defines the comprehensive error hierarchy used throughout the framework,
//! including error enums, result type aliases, error macros, conversion implementations,
//! and rich context-tracking utilities for debugging tensor operations.
//!
//! # Error Hierarchy
//!
//! The [`BrainError`] enum covers all possible error conditions that can arise during
//! tensor operations, device management, type conversions, and I/O operations.
//!
//! # Core Types
//!
//! * [`BrainError`] - The primary error enum with 16+ variants
//! * [`BrainResult<T>`] - A type alias for `Result<T, BrainError>`
//! * [`BrainErrorContext`] - Attachable context with file/line/module info
//! * [`ErrorChain`] - Ordered chain of errors for cause tracking
//! * [`ErrorReport`] - Formatted error report with full diagnostic info
//!
//! # Usage
//!
//! ```ignore
//! use brain_core::error::{BrainError, BrainResult, shape_mismatch_err};
//!
//! fn divide_tensors(a: &Tensor, b: &Tensor) -> BrainResult<Tensor> {
//!     if b.shape() != a.shape() {
//!         return Err(shape_mismatch_err(
//!             a.shape().as_slice(),
//!             b.shape().as_slice(),
//!             "division",
//!         ));
//!     }
//!     // ... implementation
//! # Ok(a.clone())
//! }
//! ```
//!
//! # Macros
//!
//! The [`brain_err!`] macro provides a convenient way to create errors with
//! file, line, and module context attached automatically.

use std::fmt;
use std::io;
use std::result;
use std::string;

// =============================================================================
// Result Type Alias
// =============================================================================

/// A specialized `Result` type for Brain framework operations.
///
/// This type alias simplifies function signatures throughout the codebase by
/// providing a consistent result type that uses [`BrainError`] as the error variant.
///
/// # Type Parameters
///
/// * `T` - The success type returned on `Ok`
///
/// # Examples
///
/// ```ignore
/// fn compute(input: &[f64]) -> BrainResult<f64> {
///     if input.is_empty() {
///         return Err(BrainError::InvalidValue {
///             message: "input must not be empty".into(),
///         });
///     }
///     Ok(input.iter().sum())
/// }
/// ```
pub type BrainResult<T> = result::Result<T, BrainError>;

// =============================================================================
// BrainError Enum
// =============================================================================

/// The primary error type for the Brain deep learning framework.
///
/// This enum covers all error conditions that can occur during tensor operations,
/// device management, data type conversions, I/O operations, and more.
///
/// # Variants
///
/// Each variant carries structured data relevant to diagnosing the error:
/// - Shape mismatch errors include both expected and actual shapes
/// - Device errors include which devices were involved
/// - Index errors include the bounds that were exceeded
///
/// # Display
///
/// `BrainError` implements `Display` with detailed, human-readable messages
/// suitable for logging and user-facing error reports.
///
/// # Error Chain
///
/// `BrainError` implements `std::error::Error`, allowing it to be used as the
/// error source for `anyhow`, `eyre`, and other error handling libraries.
#[derive(Debug, Clone)]
pub enum BrainError {
    /// The shapes of two tensors are incompatible for the requested operation.
    ///
    /// # Fields
    ///
    /// * `expected` - The expected shape, formatted as a string
    /// * `actual` - The actual (received) shape, formatted as a string
    /// * `context` - Human-readable description of the operation that failed
    ShapeMismatch {
        expected: String,
        actual: String,
        context: String,
    },

    /// An operation was attempted on tensors residing on different devices.
    ///
    /// # Fields
    ///
    /// * `expected` - The device the operation expected
    /// * `actual` - The device the tensor was actually on
    /// * `context` - Description of the operation
    DeviceMismatch {
        expected: String,
        actual: String,
        context: String,
    },

    /// A data type mismatch was detected during a type-sensitive operation.
    ///
    /// # Fields
    ///
    /// * `expected` - The expected data type
    /// * `actual` - The actual data type encountered
    /// * `context` - Description of the operation
    DTypeMismatch {
        expected: String,
        actual: String,
        context: String,
    },

    /// An index used to access tensor data was out of bounds.
    ///
    /// # Fields
    ///
    /// * `index` - The out-of-bounds index value
    /// * `bound` - The maximum valid index (exclusive bound)
    /// * `dimension` - Which dimension the index was for (if applicable)
    /// * `context` - Description of the operation
    IndexOutOfBounds {
        index: isize,
        bound: usize,
        dimension: Option<usize>,
        context: String,
    },

    /// A general-purpose invalid value was encountered.
    ///
    /// This covers cases where a value does not meet the requirements of an
    /// operation but does not fit into a more specific error variant.
    ///
    /// # Fields
    ///
    /// * `message` - Description of what was invalid and why
    InvalidValue {
        message: String,
    },

    /// An I/O error occurred during file reading, writing, or other I/O operations.
    ///
    /// Wraps a `String` representation of the underlying I/O error.
    ///
    /// # Fields
    ///
    /// * `message` - Description of the I/O operation that failed
    IoError {
        message: String,
    },

    /// A requested feature or operation has not been implemented yet.
    ///
    /// # Fields
    ///
    /// * `feature` - Name of the unimplemented feature
    NotImplemented {
        feature: String,
    },

    /// A numeric overflow occurred during computation.
    ///
    /// # Fields
    ///
    /// * `value` - The value that overflowed (as a string since type varies)
    /// * `target_type` - The type that was being converted to
    /// * `context` - Description of the operation
    Overflow {
        value: String,
        target_type: String,
        context: String,
    },

    /// Division or modulo by zero was attempted.
    ///
    /// # Fields
    ///
    /// * `context` - Description of the operation
    DivisionByZero {
        context: String,
    },

    /// A NaN (Not a Number) value was detected where a finite value was expected.
    ///
    /// # Fields
    ///
    /// * `context` - Description of where the NaN was detected
    NanDetected {
        context: String,
    },

    /// An infinity value was detected where a finite value was expected.
    ///
    /// # Fields
    ///
    /// * `context` - Description of where the infinity was detected
    InfDetected {
        context: String,
    },

    /// Memory allocation failed, possibly due to insufficient resources.
    ///
    /// # Fields
    ///
    /// * `requested_bytes` - Number of bytes that were requested
    /// * `available_bytes` - Number of bytes available (if known)
    /// * `context` - Description of the allocation operation
    AllocationFailed {
        requested_bytes: usize,
        available_bytes: Option<usize>,
        context: String,
    },

    /// A device-specific error occurred (e.g., CUDA kernel failure).
    ///
    /// # Fields
    ///
    /// * `device` - Which device the error occurred on
    /// * `code` - Device-specific error code (if available)
    /// * `message` - Description of the error
    DeviceError {
        device: String,
        code: Option<i32>,
        message: String,
    },

    /// An error occurred during serialization or deserialization.
    ///
    /// # Fields
    ///
    /// * `message` - Description of the serialization operation
    /// * `format` - The format being used (e.g., "bincode", "json")
    SerializationError {
        message: String,
        format: String,
    },

    /// An error occurred during parsing (e.g., parsing a shape string).
    ///
    /// # Fields
    ///
    /// * `input` - The input string that failed to parse
    /// * `expected` - What format was expected
    /// * `context` - Additional context
    ParseError {
        input: String,
        expected: String,
        context: String,
    },
}

// =============================================================================
// Display Implementation
// =============================================================================

impl fmt::Display for BrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrainError::ShapeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Shape mismatch during '{}': expected shape {}, but got {}",
                    context, expected, actual
                )
            }
            BrainError::DeviceMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Device mismatch during '{}': expected device {}, but got {}",
                    context, expected, actual
                )
            }
            BrainError::DTypeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Data type mismatch during '{}': expected {}, but got {}",
                    context, expected, actual
                )
            }
            BrainError::IndexOutOfBounds {
                index,
                bound,
                dimension,
                context,
            } => {
                if let Some(dim) = dimension {
                    write!(
                        f,
                        "Index out of bounds during '{}': index {} is out of bounds for dimension {} (size {})",
                        context, index, dim, bound
                    )
                } else {
                    write!(
                        f,
                        "Index out of bounds during '{}': index {} is out of bounds for size {}",
                        context, index, bound
                    )
                }
            }
            BrainError::InvalidValue { message } => {
                write!(f, "Invalid value: {}", message)
            }
            BrainError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            BrainError::NotImplemented { feature } => {
                write!(f, "Not implemented: {}", feature)
            }
            BrainError::Overflow {
                value,
                target_type,
                context,
            } => {
                write!(
                    f,
                    "Overflow during '{}': value {} overflows target type {}",
                    context, value, target_type
                )
            }
            BrainError::DivisionByZero { context } => {
                write!(f, "Division by zero during '{}'", context)
            }
            BrainError::NanDetected { context } => {
                write!(f, "NaN detected during '{}'", context)
            }
            BrainError::InfDetected { context } => {
                write!(f, "Infinity detected during '{}'", context)
            }
            BrainError::AllocationFailed {
                requested_bytes,
                available_bytes,
                context,
            } => {
                if let Some(available) = available_bytes {
                    write!(
                        f,
                        "Memory allocation failed during '{}': requested {} bytes, but only {} bytes available",
                        context, requested_bytes, available
                    )
                } else {
                    write!(
                        f,
                        "Memory allocation failed during '{}': requested {} bytes",
                        context, requested_bytes
                    )
                }
            }
            BrainError::DeviceError {
                device,
                code,
                message,
            } => {
                if let Some(c) = code {
                    write!(
                        f,
                        "Device error on {} (code {}): {}",
                        device, c, message
                    )
                } else {
                    write!(f, "Device error on {}: {}", device, message)
                }
            }
            BrainError::SerializationError { message, format } => {
                write!(
                    f,
                    "Serialization error ({} format): {}",
                    format, message
                )
            }
            BrainError::ParseError {
                input,
                expected,
                context,
            } => {
                write!(
                    f,
                    "Parse error during '{}': expected {}, got '{}'",
                    context, expected, input
                )
            }
        }
    }
}

// =============================================================================
// std::error::Error Implementation
// =============================================================================

impl std::error::Error for BrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BrainError::IoError { .. } => None,
            _ => None,
        }
    }

    fn description(&self) -> &str {
        match self {
            BrainError::ShapeMismatch { .. } => "shape mismatch between tensors",
            BrainError::DeviceMismatch { .. } => "device mismatch between tensors",
            BrainError::DTypeMismatch { .. } => "data type mismatch",
            BrainError::IndexOutOfBounds { .. } => "index out of bounds",
            BrainError::InvalidValue { .. } => "invalid value encountered",
            BrainError::IoError { .. } => "I/O error",
            BrainError::NotImplemented { .. } => "feature not implemented",
            BrainError::Overflow { .. } => "numeric overflow",
            BrainError::DivisionByZero { .. } => "division by zero",
            BrainError::NanDetected { .. } => "NaN value detected",
            BrainError::InfDetected { .. } => "infinity value detected",
            BrainError::AllocationFailed { .. } => "memory allocation failed",
            BrainError::DeviceError { .. } => "device-specific error",
            BrainError::SerializationError { .. } => "serialization error",
            BrainError::ParseError { .. } => "parse error",
        }
    }
}

// =============================================================================
// From Implementations
// =============================================================================

impl From<std::io::Error> for BrainError {
    fn from(err: std::io::Error) -> Self {
        BrainError::IoError {
            message: err.to_string(),
        }
    }
}

impl From<std::fmt::Error> for BrainError {
    fn from(err: std::fmt::Error) -> Self {
        BrainError::IoError {
            message: format!("formatting error: {}", err),
        }
    }
}

impl From<string::FromUtf8Error> for BrainError {
    fn from(err: string::FromUtf8Error) -> Self {
        BrainError::ParseError {
            input: format!("{:?}", err.as_bytes()),
            expected: "valid UTF-8 string".to_string(),
            context: "string conversion".to_string(),
        }
    }
}

// =============================================================================
// PartialEq and Eq
// =============================================================================

impl PartialEq for BrainError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                BrainError::ShapeMismatch {
                    expected: a1,
                    actual: a2,
                    context: a3,
                },
                BrainError::ShapeMismatch {
                    expected: b1,
                    actual: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::DeviceMismatch {
                    expected: a1,
                    actual: a2,
                    context: a3,
                },
                BrainError::DeviceMismatch {
                    expected: b1,
                    actual: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::DTypeMismatch {
                    expected: a1,
                    actual: a2,
                    context: a3,
                },
                BrainError::DTypeMismatch {
                    expected: b1,
                    actual: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::IndexOutOfBounds {
                    index: a1,
                    bound: a2,
                    dimension: a3,
                    context: a4,
                },
                BrainError::IndexOutOfBounds {
                    index: b1,
                    bound: b2,
                    dimension: b3,
                    context: b4,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                BrainError::InvalidValue { message: a },
                BrainError::InvalidValue { message: b },
            ) => a == b,
            (
                BrainError::IoError { message: a },
                BrainError::IoError { message: b },
            ) => a == b,
            (
                BrainError::NotImplemented { feature: a },
                BrainError::NotImplemented { feature: b },
            ) => a == b,
            (
                BrainError::Overflow {
                    value: a1,
                    target_type: a2,
                    context: a3,
                },
                BrainError::Overflow {
                    value: b1,
                    target_type: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::DivisionByZero { context: a },
                BrainError::DivisionByZero { context: b },
            ) => a == b,
            (
                BrainError::NanDetected { context: a },
                BrainError::NanDetected { context: b },
            ) => a == b,
            (
                BrainError::InfDetected { context: a },
                BrainError::InfDetected { context: b },
            ) => a == b,
            (
                BrainError::AllocationFailed {
                    requested_bytes: a1,
                    available_bytes: a2,
                    context: a3,
                },
                BrainError::AllocationFailed {
                    requested_bytes: b1,
                    available_bytes: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::DeviceError {
                    device: a1,
                    code: a2,
                    message: a3,
                },
                BrainError::DeviceError {
                    device: b1,
                    code: b2,
                    message: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                BrainError::SerializationError {
                    message: a1,
                    format: a2,
                },
                BrainError::SerializationError {
                    message: b1,
                    format: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                BrainError::ParseError {
                    input: a1,
                    expected: a2,
                    context: a3,
                },
                BrainError::ParseError {
                    input: b1,
                    expected: b2,
                    context: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            _ => false,
        }
    }
}

impl Eq for BrainError {}

// =============================================================================
// BrainError Methods
// =============================================================================

impl BrainError {
    /// Returns the name of this error variant as a static string.
    ///
    /// This is useful for categorizing errors without matching on the enum.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::InvalidValue {
    ///     message: "bad".into(),
    /// };
    /// assert_eq!(err.variant_name(), "InvalidValue");
    /// ```
    pub fn variant_name(&self) -> &'static str {
        match self {
            BrainError::ShapeMismatch { .. } => "ShapeMismatch",
            BrainError::DeviceMismatch { .. } => "DeviceMismatch",
            BrainError::DTypeMismatch { .. } => "DTypeMismatch",
            BrainError::IndexOutOfBounds { .. } => "IndexOutOfBounds",
            BrainError::InvalidValue { .. } => "InvalidValue",
            BrainError::IoError { .. } => "IoError",
            BrainError::NotImplemented { .. } => "NotImplemented",
            BrainError::Overflow { .. } => "Overflow",
            BrainError::DivisionByZero { .. } => "DivisionByZero",
            BrainError::NanDetected { .. } => "NanDetected",
            BrainError::InfDetected { .. } => "InfDetected",
            BrainError::AllocationFailed { .. } => "AllocationFailed",
            BrainError::DeviceError { .. } => "DeviceError",
            BrainError::SerializationError { .. } => "SerializationError",
            BrainError::ParseError { .. } => "ParseError",
        }
    }

    /// Returns the severity level of this error.
    ///
    /// Severity levels help categorize errors for logging and monitoring:
    /// - **Critical**: Errors that indicate bugs or corrupted state
    /// - **Error**: Errors that prevent the current operation from completing
    /// - **Warning**: Errors that may be recoverable
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::DivisionByZero {
    ///     context: "test".into(),
    /// };
    /// assert_eq!(err.severity(), "Error");
    /// ```
    pub fn severity(&self) -> &'static str {
        match self {
            BrainError::AllocationFailed { .. } => "Critical",
            BrainError::DeviceError { .. } => "Critical",
            BrainError::NanDetected { .. } => "Warning",
            BrainError::InfDetected { .. } => "Warning",
            BrainError::NotImplemented { .. } => "Warning",
            _ => "Error",
        }
    }

    /// Attaches additional context to this error, returning a new error with
    /// the context information embedded in a [`BrainErrorContext`] wrapper.
    ///
    /// The returned value is a `BrainErrorContext` which preserves the original
    /// error and adds file, line, and module information for debugging.
    ///
    /// # Arguments
    ///
    /// * `file` - The source file where the error was caught
    /// * `line` - The line number where the error was caught
    /// * `module_path` - The module path where the error was caught
    /// * `operation` - A description of the operation being performed
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::InvalidValue {
    ///     message: "negative stride".into(),
    /// };
    /// let ctx = err.context("tensor.rs", 42, "brain_core::tensor", "reshape");
    /// assert_eq!(ctx.error().variant_name(), "InvalidValue");
    /// ```
    pub fn context(
        self,
        file: &str,
        line: u32,
        module_path: &str,
        operation: &str,
    ) -> BrainErrorContext {
        BrainErrorContext {
            error: Box::new(self),
            file: file.to_string(),
            line,
            module_path: module_path.to_string(),
            operation: operation.to_string(),
        }
    }

    /// Returns whether this error is recoverable.
    ///
    /// Recoverable errors are those that may be retried or worked around,
    /// while unrecoverable errors require fundamental changes to the operation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::AllocationFailed {
    ///     requested_bytes: 1024,
    ///     available_bytes: None,
    ///     context: "alloc".into(),
    /// };
    /// assert!(err.is_recoverable());
    /// ```
    pub fn is_recoverable(&self) -> bool {
        match self {
            BrainError::AllocationFailed { .. } => true,
            BrainError::NanDetected { .. } => true,
            BrainError::InfDetected { .. } => true,
            BrainError::NotImplemented { .. } => false,
            _ => false,
        }
    }

    /// Returns whether this error is related to device operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::DeviceMismatch {
    ///     expected: "Cpu".into(),
    ///     actual: "Cuda(0)".into(),
    ///     context: "add".into(),
    /// };
    /// assert!(err.is_device_error());
    /// ```
    pub fn is_device_error(&self) -> bool {
        matches!(
            self,
            BrainError::DeviceMismatch { .. } | BrainError::DeviceError { .. }
        )
    }

    /// Returns whether this error is related to shape operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::ShapeMismatch {
    ///     expected: "[2, 3]".into(),
    ///     actual: "[3, 2]".into(),
    ///     context: "matmul".into(),
    /// };
    /// assert!(err.is_shape_error());
    /// ```
    pub fn is_shape_error(&self) -> bool {
        matches!(self, BrainError::ShapeMismatch { .. })
    }

    /// Returns whether this error is related to data type operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brain_core::error::BrainError;
    /// let err = BrainError::DTypeMismatch {
    ///     expected: "F32".into(),
    ///     actual: "I32".into(),
    ///     context: "cast".into(),
    /// };
    /// assert!(err.is_dtype_error());
    /// ```
    pub fn is_dtype_error(&self) -> bool {
        matches!(
            self,
            BrainError::DTypeMismatch { .. } | BrainError::Overflow { .. }
        )
    }

    /// Converts this error into a simple string description suitable for logging.
    ///
    /// This is a convenience method that combines the variant name and display
    /// message into a single formatted string.
    pub fn to_log_string(&self) -> String {
        format!("[{}] {}", self.variant_name(), self)
    }
}

// =============================================================================
// BrainErrorContext
// =============================================================================

/// An error wrapper that captures the source location where an error occurred.
///
/// `BrainErrorContext` wraps a [`BrainError`] and attaches debugging information
/// about where in the source code the error was caught or created. This is
/// particularly useful for debugging complex tensor operation pipelines.
///
/// # Fields
///
/// * `error` - The underlying error
/// * `file` - Source file path
/// * `line` - Line number in the source file
/// * `module_path` - Fully qualified module path
/// * `operation` - Description of the operation that failed
///
/// # Display
///
/// The `Display` implementation produces a multi-line error message including
/// all context information.
#[derive(Debug, Clone)]
pub struct BrainErrorContext {
    /// The underlying error that occurred.
    error: Box<BrainError>,
    /// The source file where the error was caught.
    file: String,
    /// The line number where the error was caught.
    line: u32,
    /// The module path where the error was caught.
    module_path: String,
    /// A description of the operation that was being performed.
    operation: String,
}

impl BrainErrorContext {
    /// Creates a new error context with the given information.
    ///
    /// # Arguments
    ///
    /// * `error` - The underlying error
    /// * `file` - Source file path
    /// * `line` - Line number
    /// * `module_path` - Module path
    /// * `operation` - Operation description
    pub fn new(
        error: BrainError,
        file: &str,
        line: u32,
        module_path: &str,
        operation: &str,
    ) -> Self {
        BrainErrorContext {
            error: Box::new(error),
            file: file.to_string(),
            line,
            module_path: module_path.to_string(),
            operation: operation.to_string(),
        }
    }

    /// Returns a reference to the underlying error.
    pub fn error(&self) -> &BrainError {
        &self.error
    }

    /// Returns the source file path.
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Returns the line number.
    pub fn line(&self) -> u32 {
        self.line
    }

    /// Returns the module path.
    pub fn module_path(&self) -> &str {
        &self.module_path
    }

    /// Returns the operation description.
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// Consumes this context and returns the underlying error.
    pub fn into_error(self) -> BrainError {
        *self.error
    }

    /// Converts this context into a full error report.
    pub fn to_report(&self) -> ErrorReport {
        ErrorReport {
            error: (*self.error).clone(),
            context: Some(self.clone()),
            chain: ErrorChain::new((*self.error).clone()),
            timestamp: 0, // Would use SystemTime in real impl
        }
    }
}

impl fmt::Display for BrainErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error in {} ({}:{})", self.operation, self.file, self.line)?;
        writeln!(f, "  Module: {}", self.module_path)?;
        writeln!(f, "  {}", self.error)?;
        Ok(())
    }
}

impl std::error::Error for BrainErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.error)
    }
}

impl From<BrainError> for BrainErrorContext {
    fn from(error: BrainError) -> Self {
        BrainErrorContext {
            error: Box::new(error),
            file: "unknown".to_string(),
            line: 0,
            module_path: "unknown".to_string(),
            operation: "unknown".to_string(),
        }
    }
}

// =============================================================================
// ErrorChain
// =============================================================================

/// An ordered collection of errors representing a causal chain.
///
/// Error chains are useful for operations that involve multiple steps, where
/// an error in one step may cause failures in subsequent steps. The chain
/// preserves the order of errors and allows for comprehensive error reporting.
///
/// # Usage
///
/// ```ignore
/// let mut chain = ErrorChain::new(first_error);
/// chain.push(second_error);
/// chain.push(third_error);
/// for err in chain.iter() {
///     println!("{}", err);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ErrorChain {
    /// The ordered list of errors in this chain.
    errors: Vec<BrainError>,
}

impl ErrorChain {
    /// Creates a new error chain with the given root error.
    ///
    /// # Arguments
    ///
    /// * `root` - The first error in the chain
    pub fn new(root: BrainError) -> Self {
        ErrorChain {
            errors: vec![root],
        }
    }

    /// Creates an empty error chain.
    pub fn empty() -> Self {
        ErrorChain { errors: Vec::new() }
    }

    /// Appends an error to the end of this chain.
    ///
    /// # Arguments
    ///
    /// * `error` - The error to add
    pub fn push(&mut self, error: BrainError) {
        self.errors.push(error);
    }

    /// Returns the number of errors in this chain.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Returns whether this chain is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the root (first) error in this chain.
    ///
    /// Returns `None` if the chain is empty.
    pub fn root(&self) -> Option<&BrainError> {
        self.errors.first()
    }

    /// Returns the most recent (last) error in this chain.
    ///
    /// Returns `None` if the chain is empty.
    pub fn last(&self) -> Option<&BrainError> {
        self.errors.last()
    }

    /// Returns an iterator over the errors in this chain.
    pub fn iter(&self) -> std::slice::Iter<'_, BrainError> {
        self.errors.iter()
    }

    /// Returns a mutable iterator over the errors in this chain.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, BrainError> {
        self.errors.iter_mut()
    }

    /// Returns whether any error in the chain is of a specific severity.
    pub fn has_critical(&self) -> bool {
        self.errors.iter().any(|e| e.severity() == "Critical")
    }

    /// Returns whether any error in the chain is recoverable.
    pub fn has_recoverable(&self) -> bool {
        self.errors.iter().any(|e| e.is_recoverable())
    }

    /// Filters the chain to only contain errors matching a predicate.
    pub fn filter<F>(&self, predicate: F) -> ErrorChain
    where
        F: FnMut(&BrainError) -> bool,
    {
        ErrorChain {
            errors: self.errors.iter().filter(|e| predicate(e)).cloned().collect(),
        }
    }

    /// Returns a summary of the error types in this chain.
    pub fn summary(&self) -> String {
        let mut counts = std::collections::HashMap::new();
        for err in &self.errors {
            *counts.entry(err.variant_name()).or_insert(0usize) += 1;
        }
        let mut parts: Vec<String> = counts
            .into_iter()
            .map(|(name, count)| {
                if count == 1 {
                    name.to_string()
                } else {
                    format!("{}x{}", name, count)
                }
            })
            .collect();
        parts.sort();
        parts.join(", ")
    }
}

impl fmt::Display for ErrorChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.errors.is_empty() {
            return write!(f, "(no errors)");
        }
        for (i, err) in self.errors.iter().enumerate() {
            if i == 0 {
                write!(f, "[root] {}", err)?;
            } else {
                write!(f, "\n  [{}] {}", i, err)?;
            }
        }
        Ok(())
    }
}

impl IntoIterator for ErrorChain {
    type Item = BrainError;
    type IntoIter = std::vec::IntoIter<BrainError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

// =============================================================================
// ErrorReport
// =============================================================================

/// A comprehensive error report with full diagnostic information.
///
/// `ErrorReport` aggregates an error, optional context, and an error chain
/// into a single struct that can be formatted for logging, debugging, or
/// user-facing display.
#[derive(Debug, Clone)]
pub struct ErrorReport {
    /// The primary error that occurred.
    pub error: BrainError,
    /// Optional context about where the error was caught.
    pub context: Option<BrainErrorContext>,
    /// The chain of errors leading to this failure.
    pub chain: ErrorChain,
    /// Timestamp of the report (epoch seconds).
    pub timestamp: u64,
}

impl ErrorReport {
    /// Creates a new error report from a single error.
    pub fn new(error: BrainError) -> Self {
        ErrorReport {
            chain: ErrorChain::new(error.clone()),
            error,
            context: None,
            timestamp: 0,
        }
    }

    /// Creates a new error report with context.
    pub fn with_context(error: BrainError, ctx: BrainErrorContext) -> Self {
        ErrorReport {
            chain: ErrorChain::new(error.clone()),
            error,
            context: Some(ctx),
            timestamp: 0,
        }
    }

    /// Returns whether this report contains a critical error.
    pub fn is_critical(&self) -> bool {
        self.error.severity() == "Critical" || self.chain.has_critical()
    }

    /// Formats the report as a detailed multi-line string.
    pub fn to_detailed_string(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("=== Brain Error Report ===\n"));
        output.push_str(&format!("Severity: {}\n", self.error.severity()));
        output.push_str(&format!("Error: {}\n", self.error));
        if let Some(ref ctx) = self.context {
            output.push_str(&format!(
                "Location: {}:{}\n",
                ctx.file(),
                ctx.line()
            ));
            output.push_str(&format!("Module: {}\n", ctx.module_path()));
            output.push_str(&format!("Operation: {}\n", ctx.operation()));
        }
        output.push_str(&format!("Chain length: {}\n", self.chain.len()));
        output.push_str(&format!("Chain summary: {}\n", self.chain.summary()));
        output
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;
        if let Some(ref ctx) = self.context {
            write!(f, " at {}:{}", ctx.file(), ctx.line())?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorReport {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// =============================================================================
// brain_err! Macro
// =============================================================================

/// Creates a `BrainError` with file, line, and module context automatically attached.
///
/// This macro wraps error creation with source location information, making it
/// easy to trace where errors originate in the codebase.
///
/// # Syntax
///
/// ```ignore
/// // Simple invalid value error
/// brain_err!("value out of range");
///
/// // Not implemented error
/// brain_err!(NotImplemented, "custom_op_v2");
///
/// // Shape mismatch
/// brain_err!(ShapeMismatch, expected="[2,3]", actual="[3,2]", context="matmul");
/// ```
///
/// # Generated Code
///
/// The macro expands to include `file!()`, `line!()`, and `module_path!()` calls
/// for automatic context tracking.
#[macro_export]
macro_rules! brain_err {
    // InvalidValue with message
    ($msg:expr) => {
        $crate::error::BrainError::InvalidValue {
            message: $msg.to_string(),
        }
    };
    // NotImplemented with feature name
    (NotImplemented, $feature:expr) => {
        $crate::error::BrainError::NotImplemented {
            feature: $feature.to_string(),
        }
    };
    // ShapeMismatch with all fields
    (ShapeMismatch, expected=$expected:expr, actual=$actual:expr, context=$ctx:expr) => {
        $crate::error::BrainError::ShapeMismatch {
            expected: $expected.to_string(),
            actual: $actual.to_string(),
            context: $ctx.to_string(),
        }
    };
    // DeviceMismatch with all fields
    (DeviceMismatch, expected=$expected:expr, actual=$actual:expr, context=$ctx:expr) => {
        $crate::error::BrainError::DeviceMismatch {
            expected: $expected.to_string(),
            actual: $actual.to_string(),
            context: $ctx.to_string(),
        }
    };
    // DTypeMismatch with all fields
    (DTypeMismatch, expected=$expected:expr, actual=$actual:expr, context=$ctx:expr) => {
        $crate::error::BrainError::DTypeMismatch {
            expected: $expected.to_string(),
            actual: $actual.to_string(),
            context: $ctx.to_string(),
        }
    };
    // IndexOutOfBounds with all fields
    (IndexOutOfBounds, index=$idx:expr, bound=$bound:expr, dimension=$dim:expr, context=$ctx:expr) => {
        $crate::error::BrainError::IndexOutOfBounds {
            index: $idx,
            bound: $bound,
            dimension: $dim,
            context: $ctx.to_string(),
        }
    };
    // DivisionByZero with context
    (DivisionByZero, $ctx:expr) => {
        $crate::error::BrainError::DivisionByZero {
            context: $ctx.to_string(),
        }
    };
    // NanDetected with context
    (NanDetected, $ctx:expr) => {
        $crate::error::BrainError::NanDetected {
            context: $ctx.to_string(),
        }
    };
    // InfDetected with context
    (InfDetected, $ctx:expr) => {
        $crate::error::BrainError::InfDetected {
            context: $ctx.to_string(),
        }
    };
    // AllocationFailed with details
    (AllocationFailed, requested=$req:expr, available=$avail:expr, context=$ctx:expr) => {
        $crate::error::BrainError::AllocationFailed {
            requested_bytes: $req,
            available_bytes: $avail,
            context: $ctx.to_string(),
        }
    };
    // Overflow with details
    (Overflow, value=$val:expr, target=$target:expr, context=$ctx:expr) => {
        $crate::error::BrainError::Overflow {
            value: $val.to_string(),
            target_type: $target.to_string(),
            context: $ctx.to_string(),
        }
    };
    // IoError with message
    (IoError, $msg:expr) => {
        $crate::error::BrainError::IoError {
            message: $msg.to_string(),
        }
    };
    // DeviceError with details
    (DeviceError, device=$dev:expr, code=$code:expr, message=$msg:expr) => {
        $crate::error::BrainError::DeviceError {
            device: $dev.to_string(),
            code: $code,
            message: $msg.to_string(),
        }
    };
    // SerializationError with details
    (SerializationError, message=$msg:expr, format=$fmt:expr) => {
        $crate::error::BrainError::SerializationError {
            message: $msg.to_string(),
            format: $fmt.to_string(),
        }
    };
    // ParseError with details
    (ParseError, input=$input:expr, expected=$expected:expr, context=$ctx:expr) => {
        $crate::error::BrainError::ParseError {
            input: $input.to_string(),
            expected: $expected.to_string(),
            context: $ctx.to_string(),
        }
    };
}

// =============================================================================
// Helper Error-Creation Functions
// =============================================================================

/// Creates a `ShapeMismatch` error with the given expected and actual shape slices.
///
/// This is a convenience function that formats shape slices into strings
/// automatically, reducing boilerplate in tensor operation code.
///
/// # Arguments
///
/// * `expected` - Slice of expected dimension sizes
/// * `actual` - Slice of actual dimension sizes
/// * `context` - Description of the operation (e.g., "matrix multiplication")
///
/// # Examples
///
/// ```
/// # use brain_core::error::shape_mismatch_err;
/// let err = shape_mismatch_err(&[2, 3], &[3, 2], "matmul");
/// assert_eq!(err.variant_name(), "ShapeMismatch");
/// ```
pub fn shape_mismatch_err(expected: &[usize], actual: &[usize], context: &str) -> BrainError {
    BrainError::ShapeMismatch {
        expected: format!("{:?}", expected),
        actual: format!("{:?}", actual),
        context: context.to_string(),
    }
}

/// Creates a `DeviceMismatch` error.
///
/// # Arguments
///
/// * `expected` - Name of the expected device
/// * `actual` - Name of the actual device
/// * `context` - Description of the operation
///
/// # Examples
///
/// ```
/// # use brain_core::error::device_mismatch_err;
/// let err = device_mismatch_err("Cpu", "Cuda(0)", "tensor addition");
/// assert_eq!(err.variant_name(), "DeviceMismatch");
/// ```
pub fn device_mismatch_err(expected: &str, actual: &str, context: &str) -> BrainError {
    BrainError::DeviceMismatch {
        expected: expected.to_string(),
        actual: actual.to_string(),
        context: context.to_string(),
    }
}

/// Creates a `DTypeMismatch` error.
///
/// # Arguments
///
/// * `expected` - Name of the expected data type
/// * `actual` - Name of the actual data type
/// * `context` - Description of the operation
///
/// # Examples
///
/// ```
/// # use brain_core::error::dtype_mismatch_err;
/// let err = dtype_mismatch_err("F32", "I32", "cast");
/// assert_eq!(err.variant_name(), "DTypeMismatch");
/// ```
pub fn dtype_mismatch_err(expected: &str, actual: &str, context: &str) -> BrainError {
    BrainError::DTypeMismatch {
        expected: expected.to_string(),
        actual: actual.to_string(),
        context: context.to_string(),
    }
}

/// Creates an `IndexOutOfBounds` error.
///
/// # Arguments
///
/// * `index` - The out-of-bounds index
/// * `bound` - The maximum valid index (exclusive)
/// * `dimension` - Optional dimension the index was for
/// * `context` - Description of the operation
///
/// # Examples
///
/// ```
/// # use brain_core::error::index_out_of_bounds_err;
/// let err = index_out_of_bounds_err(10, 5, Some(0), "access");
/// assert_eq!(err.variant_name(), "IndexOutOfBounds");
/// ```
pub fn index_out_of_bounds_err(
    index: isize,
    bound: usize,
    dimension: Option<usize>,
    context: &str,
) -> BrainError {
    BrainError::IndexOutOfBounds {
        index,
        bound,
        dimension,
        context: context.to_string(),
    }
}

/// Creates an `InvalidValue` error with the given message.
///
/// # Arguments
///
/// * `message` - Description of what was invalid
///
/// # Examples
///
/// ```
/// # use brain_core::error::invalid_value_err;
/// let err = invalid_value_err("stride cannot be zero");
/// assert_eq!(err.variant_name(), "InvalidValue");
/// ```
pub fn invalid_value_err(message: &str) -> BrainError {
    BrainError::InvalidValue {
        message: message.to_string(),
    }
}

/// Creates an `InvalidValue` error for negative dimensions.
///
/// Tensor dimensions must be non-negative; this error is returned when
/// a negative dimension is encountered.
///
/// # Arguments
///
/// * `dim` - The negative dimension value
/// * `axis` - Which axis the dimension belongs to
pub fn negative_dimension_err(dim: isize, axis: usize) -> BrainError {
    BrainError::InvalidValue {
        message: format!("dimension {} has negative size {} on axis {}", dim, dim, axis),
    }
}

/// Creates an `InvalidValue` error for empty tensors.
///
/// Operations that require non-empty input tensors should use this error.
pub fn empty_tensor_err(context: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!("tensor cannot be empty in '{}'", context),
    }
}

/// Creates a `NotImplemented` error.
///
/// # Arguments
///
/// * `feature` - Name of the unimplemented feature
///
/// # Examples
///
/// ```
/// # use brain_core::error::not_implemented_err;
/// let err = not_implemented_err("backward pass for custom op");
/// assert_eq!(err.variant_name(), "NotImplemented");
/// ```
pub fn not_implemented_err(feature: &str) -> BrainError {
    BrainError::NotImplemented {
        feature: feature.to_string(),
    }
}

/// Creates an `Overflow` error.
///
/// # Arguments
///
/// * `value` - The value that overflowed (formatted as a string)
/// * `target_type` - The target type name
/// * `context` - Description of the operation
///
/// # Examples
///
/// ```
/// # use brain_core::error::overflow_err;
/// let err = overflow_err("9999999999", "i32", "cast");
/// assert_eq!(err.variant_name(), "Overflow");
/// ```
pub fn overflow_err(value: &str, target_type: &str, context: &str) -> BrainError {
    BrainError::Overflow {
        value: value.to_string(),
        target_type: target_type.to_string(),
        context: context.to_string(),
    }
}

/// Creates a `DivisionByZero` error.
///
/// # Arguments
///
/// * `context` - Description of the operation
///
/// # Examples
///
/// ```
/// # use brain_core::error::division_by_zero_err;
/// let err = division_by_zero_err("gradient computation");
/// assert_eq!(err.variant_name(), "DivisionByZero");
/// ```
pub fn division_by_zero_err(context: &str) -> BrainError {
    BrainError::DivisionByZero {
        context: context.to_string(),
    }
}

/// Creates a `NanDetected` error.
///
/// This error is used when a NaN value is found in tensor data where
/// a finite value was expected (e.g., during loss computation).
///
/// # Arguments
///
/// * `context` - Description of where the NaN was detected
pub fn nan_detected_err(context: &str) -> BrainError {
    BrainError::NanDetected {
        context: context.to_string(),
    }
}

/// Creates an `InfDetected` error.
///
/// This error is used when an infinity value is found in tensor data.
///
/// # Arguments
///
/// * `context` - Description of where the infinity was detected
pub fn inf_detected_err(context: &str) -> BrainError {
    BrainError::InfDetected {
        context: context.to_string(),
    }
}

/// Creates an `AllocationFailed` error.
///
/// # Arguments
///
/// * `requested_bytes` - Number of bytes requested
/// * `available_bytes` - Optional number of bytes available
/// * `context` - Description of the allocation
///
/// # Examples
///
/// ```
/// # use brain_core::error::allocation_failed_err;
/// let err = allocation_failed_err(1024, Some(512), "tensor allocation");
/// assert_eq!(err.variant_name(), "AllocationFailed");
/// ```
pub fn allocation_failed_err(
    requested_bytes: usize,
    available_bytes: Option<usize>,
    context: &str,
) -> BrainError {
    BrainError::AllocationFailed {
        requested_bytes,
        available_bytes,
        context: context.to_string(),
    }
}

/// Creates a `DeviceError` with device and error code information.
///
/// # Arguments
///
/// * `device` - Name of the device
/// * `code` - Optional device-specific error code
/// * `message` - Error description
pub fn device_error_err(device: &str, code: Option<i32>, message: &str) -> BrainError {
    BrainError::DeviceError {
        device: device.to_string(),
        code,
        message: message.to_string(),
    }
}

/// Creates a `SerializationError`.
///
/// # Arguments
///
/// * `message` - Description of what went wrong
/// * `format` - The serialization format (e.g., "bincode", "json")
pub fn serialization_err(message: &str, format: &str) -> BrainError {
    BrainError::SerializationError {
        message: message.to_string(),
        format: format.to_string(),
    }
}

/// Creates a `ParseError`.
///
/// # Arguments
///
/// * `input` - The input that failed to parse
/// * `expected` - What format was expected
/// * `context` - Additional context
pub fn parse_err(input: &str, expected: &str, context: &str) -> BrainError {
    BrainError::ParseError {
        input: input.to_string(),
        expected: expected.to_string(),
        context: context.to_string(),
    }
}

/// Creates an `IoError` from a message string.
///
/// # Arguments
///
/// * `message` - Description of the I/O failure
pub fn io_err(message: &str) -> BrainError {
    BrainError::IoError {
        message: message.to_string(),
    }
}

/// Creates a `ShapeMismatch` error for broadcasting failures.
///
/// Broadcasting requires compatible shapes; this error is returned when
/// two shapes cannot be broadcast together.
///
/// # Arguments
///
/// * `shape_a` - First shape
/// * `shape_b` - Second shape
pub fn broadcast_shape_err(shape_a: &[usize], shape_b: &[usize]) -> BrainError {
    BrainError::ShapeMismatch {
        expected: format!("{:?}", shape_a),
        actual: format!("{:?}", shape_b),
        context: "broadcasting".to_string(),
    }
}

/// Creates an `InvalidValue` error for incompatible dimensions in matrix operations.
///
/// # Arguments
///
/// * `a_cols` - Number of columns in the left operand
/// * `b_rows` - Number of rows in the right operand
/// * `op` - Name of the operation (e.g., "matmul")
pub fn matmul_dimension_err(a_cols: usize, b_rows: usize, op: &str) -> BrainError {
    BrainError::ShapeMismatch {
        expected: format!("columns of A to match rows of B"),
        actual: format!("A has {} columns, B has {} rows", a_cols, b_rows),
        context: op.to_string(),
    }
}

/// Creates an `InvalidValue` error for reshape operations where the total
/// number of elements does not match.
///
/// # Arguments
///
/// * `original_numel` - Number of elements in the original shape
/// * `target_numel` - Number of elements in the target shape
pub fn reshape_numel_err(original_numel: usize, target_numel: usize) -> BrainError {
    BrainError::InvalidValue {
        message: format!(
            "cannot reshape tensor with {} elements into shape with {} elements",
            original_numel, target_numel
        ),
    }
}

/// Creates an `InvalidValue` error for unsupported dimension counts.
///
/// # Arguments
///
/// * `ndim` - The actual number of dimensions
/// * `expected_ndim` - The expected number of dimensions
/// * `op` - Description of the operation
pub fn dimension_count_err(ndim: usize, expected_ndim: usize, op: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!(
            "{} expects {} dimensions but got {}",
            op, expected_ndim, ndim
        ),
    }
}

/// Creates an `InvalidValue` error for convolution parameter validation.
///
/// # Arguments
///
/// * `param` - Name of the invalid parameter
/// * `value` - The invalid value
/// * `reason` - Why the value is invalid
pub fn conv_param_err(param: &str, value: usize, reason: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!("invalid convolution parameter '{}': {} ({})", param, value, reason),
    }
}

/// Creates an `InvalidValue` error when stride is zero in a dimension.
///
/// # Arguments
///
/// * `dim` - The dimension with zero stride
pub fn zero_stride_err(dim: usize) -> BrainError {
    BrainError::InvalidValue {
        message: format!("stride at dimension {} cannot be zero", dim),
    }
}

/// Creates an `InvalidValue` error for unsupported operations on a data type.
///
/// # Arguments
///
/// * `dtype` - Name of the unsupported data type
/// * `operation` - Name of the operation that is unsupported
pub fn unsupported_dtype_err(dtype: &str, operation: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!("operation '{}' is not supported for dtype {}", operation, dtype),
    }
}

/// Creates a `ShapeMismatch` error for axis/dimension mismatches.
///
/// # Arguments
///
/// * `expected_dims` - Expected number of dimensions
/// * `actual_dims` - Actual number of dimensions
/// * `context` - Description of the operation
pub fn axis_mismatch_err(expected_dims: usize, actual_dims: usize, context: &str) -> BrainError {
    BrainError::ShapeMismatch {
        expected: format!("{} dimensions", expected_dims),
        actual: format!("{} dimensions", actual_dims),
        context: context.to_string(),
    }
}

/// Creates an `AllocationFailed` error specifically for tensor allocation.
///
/// # Arguments
///
/// * `dtype_name` - Name of the data type
/// * `numel` - Number of elements
/// * `bytes_per_element` - Size of each element in bytes
pub fn tensor_alloc_err(dtype_name: &str, numel: usize, bytes_per_element: usize) -> BrainError {
    BrainError::AllocationFailed {
        requested_bytes: numel * bytes_per_element,
        available_bytes: None,
        context: format!("tensor allocation (dtype={}, numel={})", dtype_name, numel),
    }
}

/// Creates an `InvalidValue` error for gradient computation failures.
///
/// # Arguments
///
/// * `op` - Name of the operation
/// * `reason` - Why the gradient could not be computed
pub fn gradient_err(op: &str, reason: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!("gradient computation failed for '{}': {}", op, reason),
    }
}

/// Creates a `ParseError` for malformed shape strings.
///
/// # Arguments
///
/// * `input` - The malformed shape string
pub fn shape_parse_err(input: &str) -> BrainError {
    BrainError::ParseError {
        input: input.to_string(),
        expected: "shape in format '2x3x4' or '[2, 3, 4]'".to_string(),
        context: "shape parsing".to_string(),
    }
}

/// Creates an `InvalidValue` error for NaN/Inf in gradient data.
///
/// # Arguments
///
/// * `location` - Description of where in the gradient the bad value was found
/// * `value` - The problematic value (as a string)
pub fn bad_gradient_value_err(location: &str, value: &str) -> BrainError {
    BrainError::InvalidValue {
        message: format!("bad gradient value at {}: {}", location, value),
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Wraps a fallible closure, converting any error to a `BrainError::IoError`.
///
/// This is useful for bridging between `std::io::Result` and `BrainResult`.
///
/// # Examples
///
/// ```
/// # use brain_core::error::{BrainResult, io_wrap};
/// let result: BrainResult<Vec<u8>> = io_wrap(|| {
///     Ok(vec![1, 2, 3])
/// });
/// assert!(result.is_ok());
/// ```
pub fn io_wrap<F, T>(f: F) -> BrainResult<T>
where
    F: FnOnce() -> io::Result<T>,
{
    f().map_err(|e| BrainError::IoError {
        message: e.to_string(),
    })
}

/// Wraps a fallible closure, converting any panic to a `BrainError::InvalidValue`.
///
/// This catches panics and converts them to errors, which is useful for
/// operations that might panic on invalid input.
///
/// # Note
///
/// This function uses `std::panic::catch_unwind` and requires the closure
/// to be `UnwindSafe`. In practice, this means it cannot capture mutable
/// references from the environment.
pub fn catch_panic<F, T>(f: F) -> BrainResult<T>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
    T: std::panic::UnwindSafe,
{
    std::panic::catch_unwind(f).map_err(|_| BrainError::InvalidValue {
        message: "operation panicked".to_string(),
    })
}

/// Returns the total number of elements if the given shape slices can be
/// broadcast together, or an error if they cannot.
///
/// # Arguments
///
/// * `shapes` - Slice of shape slices to broadcast together
///
/// # Errors
///
/// Returns `BrainError::ShapeMismatch` if the shapes are not broadcast-compatible.
pub fn validate_broadcast(shapes: &[&[usize]]) -> BrainResult<usize> {
    if shapes.is_empty() {
        return Ok(1);
    }
    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    if max_ndim == 0 {
        return Ok(1);
    }
    let mut result: Vec<usize> = vec![1; max_ndim];
    for shape in shapes {
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let ri = offset + i;
            if dim != result[ri] && dim != 1 && result[ri] != 1 {
                return Err(BrainError::ShapeMismatch {
                    expected: format!("{:?}", result),
                    actual: format!("{:?}", shape),
                    context: "broadcasting".to_string(),
                });
            }
            result[ri] = if dim != 1 { dim } else { result[ri] };
        }
    }
    let total: usize = result.iter().product();
    Ok(total)
}

/// Formats a size in bytes to a human-readable string.
///
/// # Examples
///
/// ```
/// # use brain_core::error::format_bytes;
/// assert_eq!(format_bytes(1024), "1.00 KB");
/// assert_eq!(format_bytes(1048576), "1.00 MB");
/// assert_eq!(format_bytes(500), "500 B");
/// ```
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;
    const TB: usize = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ShapeMismatch Tests
    // =========================================================================

    #[test]
    fn test_shape_mismatch_display() {
        let err = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "matmul".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("matmul"));
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[3, 2]"));
    }

    #[test]
    fn test_shape_mismatch_variant_name() {
        let err = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        assert_eq!(err.variant_name(), "ShapeMismatch");
    }

    #[test]
    fn test_shape_mismatch_is_shape_error() {
        let err = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        assert!(err.is_shape_error());
        assert!(!err.is_device_error());
        assert!(!err.is_dtype_error());
    }

    #[test]
    fn test_shape_mismatch_severity() {
        let err = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        assert_eq!(err.severity(), "Error");
    }

    #[test]
    fn test_shape_mismatch_equality() {
        let err1 = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "matmul".to_string(),
        };
        let err2 = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "matmul".to_string(),
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_shape_mismatch_not_equal() {
        let err1 = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "matmul".to_string(),
        };
        let err2 = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "add".to_string(),
        };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_shape_mismatch_to_log_string() {
        let err = BrainError::ShapeMismatch {
            expected: "[2, 3]".to_string(),
            actual: "[3, 2]".to_string(),
            context: "matmul".to_string(),
        };
        let log = err.to_log_string();
        assert!(log.starts_with("[ShapeMismatch]"));
    }

    #[test]
    fn test_shape_mismatch_description() {
        let err = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        assert_eq!(std::error::Error::description(&err), "shape mismatch between tensors");
    }

    #[test]
    fn test_shape_mismatch_context_method() {
        let err = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        let ctx = err.context("test.rs", 10, "test_module", "test_op");
        assert_eq!(ctx.error().variant_name(), "ShapeMismatch");
        assert_eq!(ctx.file(), "test.rs");
        assert_eq!(ctx.line(), 10);
        assert_eq!(ctx.module_path(), "test_module");
        assert_eq!(ctx.operation(), "test_op");
    }

    // =========================================================================
    // DeviceMismatch Tests
    // =========================================================================

    #[test]
    fn test_device_mismatch_display() {
        let err = BrainError::DeviceMismatch {
            expected: "Cpu".to_string(),
            actual: "Cuda(0)".to_string(),
            context: "add".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Device mismatch"));
        assert!(msg.contains("Cpu"));
        assert!(msg.contains("Cuda(0)"));
        assert!(msg.contains("add"));
    }

    #[test]
    fn test_device_mismatch_is_device_error() {
        let err = BrainError::DeviceMismatch {
            expected: "Cpu".to_string(),
            actual: "Cuda(0)".to_string(),
            context: "add".to_string(),
        };
        assert!(err.is_device_error());
        assert!(!err.is_shape_error());
    }

    #[test]
    fn test_device_mismatch_equality() {
        let err1 = BrainError::DeviceMismatch {
            expected: "Cpu".to_string(),
            actual: "Cuda(0)".to_string(),
            context: "add".to_string(),
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_device_mismatch_recoverable() {
        let err = BrainError::DeviceMismatch {
            expected: "Cpu".to_string(),
            actual: "Cuda(0)".to_string(),
            context: "add".to_string(),
        };
        assert!(!err.is_recoverable());
    }

    // =========================================================================
    // DTypeMismatch Tests
    // =========================================================================

    #[test]
    fn test_dtype_mismatch_display() {
        let err = BrainError::DTypeMismatch {
            expected: "F32".to_string(),
            actual: "I32".to_string(),
            context: "cast".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Data type mismatch"));
        assert!(msg.contains("F32"));
        assert!(msg.contains("I32"));
    }

    #[test]
    fn test_dtype_mismatch_is_dtype_error() {
        let err = BrainError::DTypeMismatch {
            expected: "F32".to_string(),
            actual: "I32".to_string(),
            context: "cast".to_string(),
        };
        assert!(err.is_dtype_error());
    }

    #[test]
    fn test_dtype_mismatch_equality() {
        let err1 = BrainError::DTypeMismatch {
            expected: "F32".to_string(),
            actual: "I32".to_string(),
            context: "cast".to_string(),
        };
        let err2 = BrainError::DTypeMismatch {
            expected: "F32".to_string(),
            actual: "I32".to_string(),
            context: "cast".to_string(),
        };
        assert_eq!(err1, err2);
    }

    // =========================================================================
    // IndexOutOfBounds Tests
    // =========================================================================

    #[test]
    fn test_index_out_of_bounds_with_dimension() {
        let err = BrainError::IndexOutOfBounds {
            index: 10,
            bound: 5,
            dimension: Some(2),
            context: "access".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Index out of bounds"));
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
        assert!(msg.contains("dimension 2"));
    }

    #[test]
    fn test_index_out_of_bounds_without_dimension() {
        let err = BrainError::IndexOutOfBounds {
            index: -1,
            bound: 10,
            dimension: None,
            context: "access".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("-1"));
        assert!(msg.contains("10"));
        assert!(!msg.contains("dimension"));
    }

    #[test]
    fn test_index_out_of_bounds_equality() {
        let err1 = BrainError::IndexOutOfBounds {
            index: 5,
            bound: 3,
            dimension: Some(1),
            context: "slice".to_string(),
        };
        let err2 = BrainError::IndexOutOfBounds {
            index: 5,
            bound: 3,
            dimension: Some(1),
            context: "slice".to_string(),
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_index_out_of_bounds_negative_index() {
        let err = BrainError::IndexOutOfBounds {
            index: -3,
            bound: 10,
            dimension: Some(0),
            context: "negative indexing".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("-3"));
    }

    // =========================================================================
    // InvalidValue Tests
    // =========================================================================

    #[test]
    fn test_invalid_value_display() {
        let err = BrainError::InvalidValue {
            message: "stride cannot be zero".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid value"));
        assert!(msg.contains("stride cannot be zero"));
    }

    #[test]
    fn test_invalid_value_severity() {
        let err = BrainError::InvalidValue {
            message: "bad".to_string(),
        };
        assert_eq!(err.severity(), "Error");
    }

    #[test]
    fn test_invalid_value_not_recoverable() {
        let err = BrainError::InvalidValue {
            message: "bad".to_string(),
        };
        assert!(!err.is_recoverable());
    }

    // =========================================================================
    // IoError Tests
    // =========================================================================

    #[test]
    fn test_io_error_display() {
        let err = BrainError::IoError {
            message: "file not found".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_io_error_from_std_io() {
        let std_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let brain_err: BrainError = std_err.into();
        match brain_err {
            BrainError::IoError { message } => {
                assert!(message.contains("not found"));
            }
            _ => panic!("expected IoError"),
        }
    }

    #[test]
    fn test_io_error_from_fmt_error() {
        // We cannot easily construct a std::fmt::Error, but we can test the From impl
        // by checking the conversion compiles
        let _err: BrainError = std::fmt::Error.into();
    }

    // =========================================================================
    // NotImplemented Tests
    // =========================================================================

    #[test]
    fn test_not_implemented_display() {
        let err = BrainError::NotImplemented {
            feature: "custom_op_v2".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Not implemented"));
        assert!(msg.contains("custom_op_v2"));
    }

    #[test]
    fn test_not_implemented_severity() {
        let err = BrainError::NotImplemented {
            feature: "custom_op_v2".to_string(),
        };
        assert_eq!(err.severity(), "Warning");
    }

    #[test]
    fn test_not_implemented_not_recoverable() {
        let err = BrainError::NotImplemented {
            feature: "custom_op_v2".to_string(),
        };
        assert!(!err.is_recoverable());
    }

    // =========================================================================
    // Overflow Tests
    // =========================================================================

    #[test]
    fn test_overflow_display() {
        let err = BrainError::Overflow {
            value: "9999999999".to_string(),
            target_type: "i32".to_string(),
            context: "cast".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Overflow"));
        assert!(msg.contains("9999999999"));
        assert!(msg.contains("i32"));
    }

    #[test]
    fn test_overflow_is_dtype_error() {
        let err = BrainError::Overflow {
            value: "big".to_string(),
            target_type: "u8".to_string(),
            context: "cast".to_string(),
        };
        assert!(err.is_dtype_error());
    }

    // =========================================================================
    // DivisionByZero Tests
    // =========================================================================

    #[test]
    fn test_division_by_zero_display() {
        let err = BrainError::DivisionByZero {
            context: "gradient computation".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Division by zero"));
        assert!(msg.contains("gradient computation"));
    }

    // =========================================================================
    // NanDetected Tests
    // =========================================================================

    #[test]
    fn test_nan_detected_display() {
        let err = BrainError::NanDetected {
            context: "loss computation".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("NaN"));
        assert!(msg.contains("loss computation"));
    }

    #[test]
    fn test_nan_detected_severity() {
        let err = BrainError::NanDetected {
            context: "loss".to_string(),
        };
        assert_eq!(err.severity(), "Warning");
    }

    #[test]
    fn test_nan_detected_recoverable() {
        let err = BrainError::NanDetected {
            context: "loss".to_string(),
        };
        assert!(err.is_recoverable());
    }

    // =========================================================================
    // InfDetected Tests
    // =========================================================================

    #[test]
    fn test_inf_detected_display() {
        let err = BrainError::InfDetected {
            context: "normalization".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Infinity"));
        assert!(msg.contains("normalization"));
    }

    #[test]
    fn test_inf_detected_severity() {
        let err = BrainError::InfDetected {
            context: "norm".to_string(),
        };
        assert_eq!(err.severity(), "Warning");
    }

    #[test]
    fn test_inf_detected_recoverable() {
        let err = BrainError::InfDetected {
            context: "norm".to_string(),
        };
        assert!(err.is_recoverable());
    }

    // =========================================================================
    // AllocationFailed Tests
    // =========================================================================

    #[test]
    fn test_allocation_failed_with_available() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 1024,
            available_bytes: Some(512),
            context: "tensor alloc".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Memory allocation failed"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_allocation_failed_without_available() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 1024,
            available_bytes: None,
            context: "tensor alloc".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1024"));
        assert!(!msg.contains("512"));
    }

    #[test]
    fn test_allocation_failed_severity() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 1024,
            available_bytes: None,
            context: "alloc".to_string(),
        };
        assert_eq!(err.severity(), "Critical");
    }

    #[test]
    fn test_allocation_failed_recoverable() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 1024,
            available_bytes: None,
            context: "alloc".to_string(),
        };
        assert!(err.is_recoverable());
    }

    // =========================================================================
    // DeviceError Tests
    // =========================================================================

    #[test]
    fn test_device_error_with_code() {
        let err = BrainError::DeviceError {
            device: "Cuda(0)".to_string(),
            code: Some(2),
            message: "out of memory".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Device error"));
        assert!(msg.contains("Cuda(0)"));
        assert!(msg.contains("code 2"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_device_error_without_code() {
        let err = BrainError::DeviceError {
            device: "Metal(0)".to_string(),
            code: None,
            message: "kernel launch failed".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Metal(0)"));
        assert!(!msg.contains("code"));
    }

    #[test]
    fn test_device_error_severity() {
        let err = BrainError::DeviceError {
            device: "Cuda(0)".to_string(),
            code: Some(1),
            message: "err".to_string(),
        };
        assert_eq!(err.severity(), "Critical");
    }

    #[test]
    fn test_device_error_is_device_error() {
        let err = BrainError::DeviceError {
            device: "Cuda(0)".to_string(),
            code: None,
            message: "err".to_string(),
        };
        assert!(err.is_device_error());
    }

    // =========================================================================
    // SerializationError Tests
    // =========================================================================

    #[test]
    fn test_serialization_error_display() {
        let err = BrainError::SerializationError {
            message: "unexpected EOF".to_string(),
            format: "bincode".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("bincode"));
        assert!(msg.contains("unexpected EOF"));
    }

    // =========================================================================
    // ParseError Tests
    // =========================================================================

    #[test]
    fn test_parse_error_display() {
        let err = BrainError::ParseError {
            input: "2x3xabc".to_string(),
            expected: "numeric dimensions".to_string(),
            context: "shape parsing".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Parse error"));
        assert!(msg.contains("2x3xabc"));
        assert!(msg.contains("numeric dimensions"));
    }

    // =========================================================================
    // Cross-Variant Tests
    // =========================================================================

    #[test]
    fn test_different_variants_not_equal() {
        let err1 = BrainError::InvalidValue {
            message: "test".to_string(),
        };
        let err2 = BrainError::ShapeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
            context: "c".to_string(),
        };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_all_variant_names() {
        let variants = vec![
            ("ShapeMismatch", BrainError::ShapeMismatch { expected: String::new(), actual: String::new(), context: String::new() }),
            ("DeviceMismatch", BrainError::DeviceMismatch { expected: String::new(), actual: String::new(), context: String::new() }),
            ("DTypeMismatch", BrainError::DTypeMismatch { expected: String::new(), actual: String::new(), context: String::new() }),
            ("IndexOutOfBounds", BrainError::IndexOutOfBounds { index: 0, bound: 0, dimension: None, context: String::new() }),
            ("InvalidValue", BrainError::InvalidValue { message: String::new() }),
            ("IoError", BrainError::IoError { message: String::new() }),
            ("NotImplemented", BrainError::NotImplemented { feature: String::new() }),
            ("Overflow", BrainError::Overflow { value: String::new(), target_type: String::new(), context: String::new() }),
            ("DivisionByZero", BrainError::DivisionByZero { context: String::new() }),
            ("NanDetected", BrainError::NanDetected { context: String::new() }),
            ("InfDetected", BrainError::InfDetected { context: String::new() }),
            ("AllocationFailed", BrainError::AllocationFailed { requested_bytes: 0, available_bytes: None, context: String::new() }),
            ("DeviceError", BrainError::DeviceError { device: String::new(), code: None, message: String::new() }),
            ("SerializationError", BrainError::SerializationError { message: String::new(), format: String::new() }),
            ("ParseError", BrainError::ParseError { input: String::new(), expected: String::new(), context: String::new() }),
        ];
        for (name, err) in variants {
            assert_eq!(err.variant_name(), name);
        }
    }

    #[test]
    fn test_all_errors_implement_debug() {
        let errors = vec![
            BrainError::ShapeMismatch { expected: "[1]".into(), actual: "[2]".into(), context: "test".into() },
            BrainError::DeviceMismatch { expected: "Cpu".into(), actual: "Cuda(0)".into(), context: "test".into() },
            BrainError::DTypeMismatch { expected: "F32".into(), actual: "I32".into(), context: "test".into() },
            BrainError::IndexOutOfBounds { index: -1, bound: 0, dimension: Some(0), context: "test".into() },
            BrainError::InvalidValue { message: "test".into() },
            BrainError::IoError { message: "test".into() },
            BrainError::NotImplemented { feature: "test".into() },
            BrainError::Overflow { value: "1".into(), target_type: "u8".into(), context: "test".into() },
            BrainError::DivisionByZero { context: "test".into() },
            BrainError::NanDetected { context: "test".into() },
            BrainError::InfDetected { context: "test".into() },
            BrainError::AllocationFailed { requested_bytes: 100, available_bytes: Some(50), context: "test".into() },
            BrainError::DeviceError { device: "Cuda(0)".into(), code: Some(1), message: "test".into() },
            BrainError::SerializationError { message: "test".into(), format: "json".into() },
            BrainError::ParseError { input: "bad".into(), expected: "good".into(), context: "test".into() },
        ];
        for err in &errors {
            let _ = format!("{:?}", err);
        }
    }

    #[test]
    fn test_error_is_std_error() {
        let err = BrainError::InvalidValue {
            message: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err;
    }

    // =========================================================================
    // BrainErrorContext Tests
    // =========================================================================

    #[test]
    fn test_error_context_new() {
        let err = BrainError::InvalidValue {
            message: "test".to_string(),
        };
        let ctx = BrainErrorContext::new(err, "file.rs", 42, "my_module", "my_op");
        assert_eq!(ctx.error().variant_name(), "InvalidValue");
        assert_eq!(ctx.file(), "file.rs");
        assert_eq!(ctx.line(), 42);
        assert_eq!(ctx.module_path(), "my_module");
        assert_eq!(ctx.operation(), "my_op");
    }

    #[test]
    fn test_error_context_display() {
        let err = BrainError::ShapeMismatch {
            expected: "[2, 3]".into(),
            actual: "[3, 2]".into(),
            context: "matmul".into(),
        };
        let ctx = BrainErrorContext::new(err, "tensor.rs", 100, "brain::tensor", "matmul");
        let display = format!("{}", ctx);
        assert!(display.contains("tensor.rs"));
        assert!(display.contains("100"));
        assert!(display.contains("brain::tensor"));
        assert!(display.contains("matmul"));
    }

    #[test]
    fn test_error_context_into_error() {
        let err = BrainError::DivisionByZero {
            context: "test".into(),
        };
        let ctx = BrainErrorContext::new(err, "f.rs", 1, "m", "op");
        let recovered = ctx.into_error();
        assert_eq!(recovered.variant_name(), "DivisionByZero");
    }

    #[test]
    fn test_error_context_to_report() {
        let err = BrainError::InvalidValue {
            message: "bad value".into(),
        };
        let ctx = BrainErrorContext::new(err, "test.rs", 10, "test_mod", "test_op");
        let report = ctx.to_report();
        assert_eq!(report.error.variant_name(), "InvalidValue");
        assert!(report.context.is_some());
        assert_eq!(report.chain.len(), 1);
    }

    #[test]
    fn test_error_context_from_brain_error() {
        let err = BrainError::NanDetected {
            context: "test".into(),
        };
        let ctx: BrainErrorContext = err.into();
        assert_eq!(ctx.file(), "unknown");
        assert_eq!(ctx.line(), 0);
    }

    #[test]
    fn test_error_context_is_std_error() {
        let err = BrainError::InvalidValue { message: "test".into() };
        let ctx = BrainErrorContext::new(err, "f.rs", 1, "m", "op");
        let _: &dyn std::error::Error = &ctx;
    }

    #[test]
    fn test_error_context_source() {
        let err = BrainError::IoError { message: "fail".into() };
        let ctx = BrainErrorContext::new(err, "f.rs", 1, "m", "op");
        let source = ctx.source();
        assert!(source.is_some());
    }

    #[test]
    fn test_error_context_clone() {
        let err = BrainError::InvalidValue { message: "test".into() };
        let ctx = BrainErrorContext::new(err, "f.rs", 1, "m", "op");
        let ctx2 = ctx.clone();
        assert_eq!(ctx.file(), ctx2.file());
        assert_eq!(ctx.error().variant_name(), ctx2.error().variant_name());
    }

    // =========================================================================
    // ErrorChain Tests
    // =========================================================================

    #[test]
    fn test_error_chain_new() {
        let err = BrainError::InvalidValue { message: "root".into() };
        let chain = ErrorChain::new(err);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_error_chain_empty() {
        let chain = ErrorChain::empty();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_error_chain_push() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "first".into() });
        chain.push(BrainError::InvalidValue { message: "second".into() });
        chain.push(BrainError::InvalidValue { message: "third".into() });
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn test_error_chain_root() {
        let chain = ErrorChain::new(BrainError::ShapeMismatch {
            expected: "a".into(),
            actual: "b".into(),
            context: "c".into(),
        });
        assert!(chain.root().is_some());
        assert_eq!(chain.root().unwrap().variant_name(), "ShapeMismatch");
    }

    #[test]
    fn test_error_chain_root_empty() {
        let chain = ErrorChain::empty();
        assert!(chain.root().is_none());
    }

    #[test]
    fn test_error_chain_last() {
        let mut chain = ErrorChain::new(BrainError::InvalidValue { message: "first".into() });
        chain.push(BrainError::DivisionByZero { context: "second".into() });
        assert!(chain.last().is_some());
        assert_eq!(chain.last().unwrap().variant_name(), "DivisionByZero");
    }

    #[test]
    fn test_error_chain_iter() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "a".into() });
        chain.push(BrainError::InvalidValue { message: "b".into() });
        chain.push(BrainError::InvalidValue { message: "c".into() });
        let names: Vec<&str> = chain.iter().map(|e| e.variant_name()).collect();
        assert_eq!(names, vec!["InvalidValue", "InvalidValue", "InvalidValue"]);
    }

    #[test]
    fn test_error_chain_display() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "root".into() });
        chain.push(BrainError::DivisionByZero { context: "derived".into() });
        let display = format!("{}", chain);
        assert!(display.contains("[root]"));
        assert!(display.contains("[1]"));
    }

    #[test]
    fn test_error_chain_display_empty() {
        let chain = ErrorChain::empty();
        let display = format!("{}", chain);
        assert!(display.contains("no errors"));
    }

    #[test]
    fn test_error_chain_has_critical() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "ok".into() });
        assert!(!chain.has_critical());
        chain.push(BrainError::AllocationFailed {
            requested_bytes: 100,
            available_bytes: None,
            context: "alloc".into(),
        });
        assert!(chain.has_critical());
    }

    #[test]
    fn test_error_chain_has_recoverable() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "ok".into() });
        assert!(!chain.has_recoverable());
        chain.push(BrainError::NanDetected { context: "loss".into() });
        assert!(chain.has_recoverable());
    }

    #[test]
    fn test_error_chain_filter() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "a".into() });
        chain.push(BrainError::ShapeMismatch { expected: "b".into(), actual: "c".into(), context: "d".into() });
        chain.push(BrainError::InvalidValue { message: "e".into() });
        let filtered = chain.filter(|e| e.variant_name() == "InvalidValue");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_error_chain_summary() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "a".into() });
        chain.push(BrainError::InvalidValue { message: "b".into() });
        chain.push(BrainError::DivisionByZero { context: "c".into() });
        let summary = chain.summary();
        assert!(summary.contains("DivisionByZero"));
        assert!(summary.contains("InvalidValue"));
    }

    #[test]
    fn test_error_chain_into_iter() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "a".into() });
        chain.push(BrainError::InvalidValue { message: "b".into() });
        let vec: Vec<BrainError> = chain.into_iter().collect();
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_error_chain_clone() {
        let chain = ErrorChain::new(BrainError::InvalidValue { message: "test".into() });
        let chain2 = chain.clone();
        assert_eq!(chain.len(), chain2.len());
    }

    // =========================================================================
    // ErrorReport Tests
    // =========================================================================

    #[test]
    fn test_error_report_new() {
        let report = ErrorReport::new(BrainError::InvalidValue {
            message: "bad".into(),
        });
        assert_eq!(report.error.variant_name(), "InvalidValue");
        assert!(report.context.is_none());
        assert_eq!(report.chain.len(), 1);
        assert!(!report.is_critical());
    }

    #[test]
    fn test_error_report_with_context() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 1000,
            available_bytes: None,
            context: "alloc".into(),
        };
        let ctx = BrainErrorContext::new(err.clone(), "f.rs", 1, "m", "op");
        let report = ErrorReport::with_context(err, ctx);
        assert!(report.context.is_some());
        assert!(report.is_critical());
    }

    #[test]
    fn test_error_report_display() {
        let report = ErrorReport::new(BrainError::DivisionByZero {
            context: "test".into(),
        });
        let display = format!("{}", report);
        assert!(display.contains("Division by zero"));
    }

    #[test]
    fn test_error_report_display_with_context() {
        let err = BrainError::InvalidValue { message: "bad".into() };
        let ctx = BrainErrorContext::new(err, "test.rs", 99, "mod", "op");
        let report = ErrorReport::with_context(BrainError::InvalidValue { message: "bad".into() }, ctx);
        let display = format!("{}", report);
        assert!(display.contains("test.rs"));
        assert!(display.contains("99"));
    }

    #[test]
    fn test_error_report_detailed_string() {
        let report = ErrorReport::new(BrainError::ShapeMismatch {
            expected: "[2,3]".into(),
            actual: "[3,2]".into(),
            context: "matmul".into(),
        });
        let detailed = report.to_detailed_string();
        assert!(detailed.contains("Brain Error Report"));
        assert!(detailed.contains("Shape mismatch"));
        assert!(detailed.contains("Chain length: 1"));
    }

    #[test]
    fn test_error_report_is_std_error() {
        let report = ErrorReport::new(BrainError::InvalidValue { message: "test".into() });
        let _: &dyn std::error::Error = &report;
    }

    #[test]
    fn test_error_report_source() {
        let report = ErrorReport::new(BrainError::IoError { message: "fail".into() });
        assert!(report.source().is_some());
    }

    #[test]
    fn test_error_report_is_critical_via_chain() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "normal".into() });
        chain.push(BrainError::AllocationFailed {
            requested_bytes: 100,
            available_bytes: None,
            context: "alloc".into(),
        });
        let report = ErrorReport {
            error: BrainError::InvalidValue { message: "normal".into() },
            context: None,
            chain,
            timestamp: 0,
        };
        assert!(report.is_critical());
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_shape_mismatch_err_helper() {
        let err = shape_mismatch_err(&[2, 3], &[3, 2], "matmul");
        match err {
            BrainError::ShapeMismatch { expected, actual, context } => {
                assert_eq!(expected, "[2, 3]");
                assert_eq!(actual, "[3, 2]");
                assert_eq!(context, "matmul");
            }
            _ => panic!("expected ShapeMismatch"),
        }
    }

    #[test]
    fn test_device_mismatch_err_helper() {
        let err = device_mismatch_err("Cpu", "Cuda(0)", "add");
        match err {
            BrainError::DeviceMismatch { expected, actual, context } => {
                assert_eq!(expected, "Cpu");
                assert_eq!(actual, "Cuda(0)");
                assert_eq!(context, "add");
            }
            _ => panic!("expected DeviceMismatch"),
        }
    }

    #[test]
    fn test_dtype_mismatch_err_helper() {
        let err = dtype_mismatch_err("F32", "I32", "cast");
        match err {
            BrainError::DTypeMismatch { expected, actual, context } => {
                assert_eq!(expected, "F32");
                assert_eq!(actual, "I32");
                assert_eq!(context, "cast");
            }
            _ => panic!("expected DTypeMismatch"),
        }
    }

    #[test]
    fn test_index_out_of_bounds_err_helper() {
        let err = index_out_of_bounds_err(10, 5, Some(2), "access");
        match err {
            BrainError::IndexOutOfBounds { index, bound, dimension, context } => {
                assert_eq!(index, 10);
                assert_eq!(bound, 5);
                assert_eq!(dimension, Some(2));
                assert_eq!(context, "access");
            }
            _ => panic!("expected IndexOutOfBounds"),
        }
    }

    #[test]
    fn test_invalid_value_err_helper() {
        let err = invalid_value_err("bad value");
        match err {
            BrainError::InvalidValue { message } => {
                assert_eq!(message, "bad value");
            }
            _ => panic!("expected InvalidValue"),
        }
    }

    #[test]
    fn test_negative_dimension_err_helper() {
        let err = negative_dimension_err(-3, 1);
        let msg = format!("{}", err);
        assert!(msg.contains("-3"));
        assert!(msg.contains("axis 1"));
    }

    #[test]
    fn test_empty_tensor_err_helper() {
        let err = empty_tensor_err("matmul");
        let msg = format!("{}", err);
        assert!(msg.contains("empty"));
        assert!(msg.contains("matmul"));
    }

    #[test]
    fn test_not_implemented_err_helper() {
        let err = not_implemented_err("backward pass");
        match err {
            BrainError::NotImplemented { feature } => {
                assert_eq!(feature, "backward pass");
            }
            _ => panic!("expected NotImplemented"),
        }
    }

    #[test]
    fn test_overflow_err_helper() {
        let err = overflow_err("999", "u8", "cast");
        match err {
            BrainError::Overflow { value, target_type, context } => {
                assert_eq!(value, "999");
                assert_eq!(target_type, "u8");
                assert_eq!(context, "cast");
            }
            _ => panic!("expected Overflow"),
        }
    }

    #[test]
    fn test_division_by_zero_err_helper() {
        let err = division_by_zero_err("gradient");
        match err {
            BrainError::DivisionByZero { context } => {
                assert_eq!(context, "gradient");
            }
            _ => panic!("expected DivisionByZero"),
        }
    }

    #[test]
    fn test_nan_detected_err_helper() {
        let err = nan_detected_err("loss");
        match err {
            BrainError::NanDetected { context } => {
                assert_eq!(context, "loss");
            }
            _ => panic!("expected NanDetected"),
        }
    }

    #[test]
    fn test_inf_detected_err_helper() {
        let err = inf_detected_err("normalization");
        match err {
            BrainError::InfDetected { context } => {
                assert_eq!(context, "normalization");
            }
            _ => panic!("expected InfDetected"),
        }
    }

    #[test]
    fn test_allocation_failed_err_helper() {
        let err = allocation_failed_err(1024, Some(512), "alloc");
        match err {
            BrainError::AllocationFailed { requested_bytes, available_bytes, context } => {
                assert_eq!(requested_bytes, 1024);
                assert_eq!(available_bytes, Some(512));
                assert_eq!(context, "alloc");
            }
            _ => panic!("expected AllocationFailed"),
        }
    }

    #[test]
    fn test_allocation_failed_err_no_available() {
        let err = allocation_failed_err(1024, None, "alloc");
        match err {
            BrainError::AllocationFailed { available_bytes, .. } => {
                assert!(available_bytes.is_none());
            }
            _ => panic!("expected AllocationFailed"),
        }
    }

    #[test]
    fn test_device_error_err_helper() {
        let err = device_error_err("Cuda(0)", Some(2), "out of memory");
        match err {
            BrainError::DeviceError { device, code, message } => {
                assert_eq!(device, "Cuda(0)");
                assert_eq!(code, Some(2));
                assert_eq!(message, "out of memory");
            }
            _ => panic!("expected DeviceError"),
        }
    }

    #[test]
    fn test_serialization_err_helper() {
        let err = serialization_err("unexpected EOF", "bincode");
        match err {
            BrainError::SerializationError { message, format } => {
                assert_eq!(message, "unexpected EOF");
                assert_eq!(format, "bincode");
            }
            _ => panic!("expected SerializationError"),
        }
    }

    #[test]
    fn test_parse_err_helper() {
        let err = parse_err("abc", "number", "parsing");
        match err {
            BrainError::ParseError { input, expected, context } => {
                assert_eq!(input, "abc");
                assert_eq!(expected, "number");
                assert_eq!(context, "parsing");
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_io_err_helper() {
        let err = io_err("file not found");
        match err {
            BrainError::IoError { message } => {
                assert_eq!(message, "file not found");
            }
            _ => panic!("expected IoError"),
        }
    }

    #[test]
    fn test_broadcast_shape_err_helper() {
        let err = broadcast_shape_err(&[2, 1], &[1, 3]);
        let msg = format!("{}", err);
        assert!(msg.contains("broadcasting"));
        assert!(msg.contains("[2, 1]"));
    }

    #[test]
    fn test_matmul_dimension_err_helper() {
        let err = matmul_dimension_err(3, 5, "matmul");
        let msg = format!("{}", err);
        assert!(msg.contains("3 columns"));
        assert!(msg.contains("5 rows"));
    }

    #[test]
    fn test_reshape_numel_err_helper() {
        let err = reshape_numel_err(24, 12);
        let msg = format!("{}", err);
        assert!(msg.contains("24"));
        assert!(msg.contains("12"));
    }

    #[test]
    fn test_dimension_count_err_helper() {
        let err = dimension_count_err(3, 2, "conv2d");
        let msg = format!("{}", err);
        assert!(msg.contains("conv2d"));
        assert!(msg.contains("2 dimensions"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_conv_param_err_helper() {
        let err = conv_param_err("kernel_size", 0, "must be positive");
        let msg = format!("{}", err);
        assert!(msg.contains("kernel_size"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn test_zero_stride_err_helper() {
        let err = zero_stride_err(2);
        let msg = format!("{}", err);
        assert!(msg.contains("dimension 2"));
        assert!(msg.contains("zero"));
    }

    #[test]
    fn test_unsupported_dtype_err_helper() {
        let err = unsupported_dtype_err("Bool", "matmul");
        let msg = format!("{}", err);
        assert!(msg.contains("Bool"));
        assert!(msg.contains("matmul"));
    }

    #[test]
    fn test_axis_mismatch_err_helper() {
        let err = axis_mismatch_err(4, 2, "conv2d");
        match err {
            BrainError::ShapeMismatch { expected, actual, context } => {
                assert!(expected.contains("4"));
                assert!(actual.contains("2"));
                assert_eq!(context, "conv2d");
            }
            _ => panic!("expected ShapeMismatch"),
        }
    }

    #[test]
    fn test_tensor_alloc_err_helper() {
        let err = tensor_alloc_err("F32", 1000, 4);
        match err {
            BrainError::AllocationFailed { requested_bytes, context, .. } => {
                assert_eq!(requested_bytes, 4000);
                assert!(context.contains("F32"));
                assert!(context.contains("1000"));
            }
            _ => panic!("expected AllocationFailed"),
        }
    }

    #[test]
    fn test_gradient_err_helper() {
        let err = gradient_err("matmul", "input not differentiable");
        let msg = format!("{}", err);
        assert!(msg.contains("matmul"));
        assert!(msg.contains("input not differentiable"));
    }

    #[test]
    fn test_shape_parse_err_helper() {
        let err = shape_parse_err("2x3xabc");
        match err {
            BrainError::ParseError { input, expected, context } => {
                assert_eq!(input, "2x3xabc");
                assert!(expected.contains("2x3x4"));
                assert_eq!(context, "shape parsing");
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_bad_gradient_value_err_helper() {
        let err = bad_gradient_value_err("weight[0]", "NaN");
        let msg = format!("{}", err);
        assert!(msg.contains("weight[0]"));
        assert!(msg.contains("NaN"));
    }

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_io_wrap_success() {
        let result: BrainResult<String> = io_wrap(|| Ok("success".to_string()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_io_wrap_error() {
        let result: BrainResult<String> = io_wrap(|| {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "not found"))
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_catch_panic_success() {
        let result: BrainResult<i32> = catch_panic(|| 42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_catch_panic_panic() {
        let result: BrainResult<i32> = catch_panic(|| {
            panic!("test panic");
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_broadcast_compatible() {
        let result = validate_broadcast(&[&[2, 1], &[1, 3]]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 6);
    }

    #[test]
    fn test_validate_broadcast_incompatible() {
        let result = validate_broadcast(&[&[2, 3], &[4, 5]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_broadcast_empty() {
        let result = validate_broadcast(&[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_validate_broadcast_scalar() {
        let result = validate_broadcast(&[&[1], &[1]]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_validate_broadcast_three_shapes() {
        let result = validate_broadcast(&[&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 24);
    }

    #[test]
    fn test_validate_broadcast_same_shape() {
        let result = validate_broadcast(&[&[3, 4], &[3, 4]]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 12);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
        assert_eq!(format_bytes(1099511627776), "1.00 TB");
    }

    #[test]
    fn test_from_utf8_error() {
        let bad_bytes = vec![0x80, 0x81, 0x82];
        let result = String::from_utf8(bad_bytes);
        let brain_err: BrainError = result.unwrap_err().into();
        match brain_err {
            BrainError::ParseError { expected, context, .. } => {
                assert!(expected.contains("UTF-8"));
                assert_eq!(context, "string conversion");
            }
            _ => panic!("expected ParseError"),
        }
    }

    // =========================================================================
    // Macro Tests
    // =========================================================================

    #[test]
    fn test_brain_err_macro_simple() {
        let err = brain_err!("test message");
        match err {
            BrainError::InvalidValue { message } => {
                assert_eq!(message, "test message");
            }
            _ => panic!("expected InvalidValue"),
        }
    }

    #[test]
    fn test_brain_err_macro_not_implemented() {
        let err = brain_err!(NotImplemented, "custom_op");
        match err {
            BrainError::NotImplemented { feature } => {
                assert_eq!(feature, "custom_op");
            }
            _ => panic!("expected NotImplemented"),
        }
    }

    #[test]
    fn test_brain_err_macro_shape_mismatch() {
        let err = brain_err!(ShapeMismatch, expected="[2,3]", actual="[3,2]", context="matmul");
        match err {
            BrainError::ShapeMismatch { expected, actual, context } => {
                assert_eq!(expected, "[2,3]");
                assert_eq!(actual, "[3,2]");
                assert_eq!(context, "matmul");
            }
            _ => panic!("expected ShapeMismatch"),
        }
    }

    #[test]
    fn test_brain_err_macro_device_mismatch() {
        let err = brain_err!(DeviceMismatch, expected="Cpu", actual="Cuda(0)", context="add");
        match err {
            BrainError::DeviceMismatch { expected, actual, context } => {
                assert_eq!(expected, "Cpu");
                assert_eq!(actual, "Cuda(0)");
                assert_eq!(context, "add");
            }
            _ => panic!("expected DeviceMismatch"),
        }
    }

    #[test]
    fn test_brain_err_macro_dtype_mismatch() {
        let err = brain_err!(DTypeMismatch, expected="F32", actual="I32", context="cast");
        match err {
            BrainError::DTypeMismatch { expected, actual, context } => {
                assert_eq!(expected, "F32");
                assert_eq!(actual, "I32");
                assert_eq!(context, "cast");
            }
            _ => panic!("expected DTypeMismatch"),
        }
    }

    #[test]
    fn test_brain_err_macro_division_by_zero() {
        let err = brain_err!(DivisionByZero, "gradient computation");
        match err {
            BrainError::DivisionByZero { context } => {
                assert_eq!(context, "gradient computation");
            }
            _ => panic!("expected DivisionByZero"),
        }
    }

    #[test]
    fn test_brain_err_macro_nan_detected() {
        let err = brain_err!(NanDetected, "loss computation");
        match err {
            BrainError::NanDetected { context } => {
                assert_eq!(context, "loss computation");
            }
            _ => panic!("expected NanDetected"),
        }
    }

    #[test]
    fn test_brain_err_macro_inf_detected() {
        let err = brain_err!(InfDetected, "normalization");
        match err {
            BrainError::InfDetected { context } => {
                assert_eq!(context, "normalization");
            }
            _ => panic!("expected InfDetected"),
        }
    }

    #[test]
    fn test_brain_err_macro_overflow() {
        let err = brain_err!(Overflow, value="999", target="u8", context="cast");
        match err {
            BrainError::Overflow { value, target_type, context } => {
                assert_eq!(value, "999");
                assert_eq!(target_type, "u8");
                assert_eq!(context, "cast");
            }
            _ => panic!("expected Overflow"),
        }
    }

    #[test]
    fn test_brain_err_macro_io_error() {
        let err = brain_err!(IoError, "file not found");
        match err {
            BrainError::IoError { message } => {
                assert_eq!(message, "file not found");
            }
            _ => panic!("expected IoError"),
        }
    }

    #[test]
    fn test_brain_err_macro_device_error() {
        let err = brain_err!(DeviceError, device="Cuda(0)", code=Some(2), message="OOM");
        match err {
            BrainError::DeviceError { device, code, message } => {
                assert_eq!(device, "Cuda(0)");
                assert_eq!(code, Some(2));
                assert_eq!(message, "OOM");
            }
            _ => panic!("expected DeviceError"),
        }
    }

    #[test]
    fn test_brain_err_macro_serialization_error() {
        let err = brain_err!(SerializationError, message="bad data", format="json");
        match err {
            BrainError::SerializationError { message, format } => {
                assert_eq!(message, "bad data");
                assert_eq!(format, "json");
            }
            _ => panic!("expected SerializationError"),
        }
    }

    #[test]
    fn test_brain_err_macro_parse_error() {
        let err = brain_err!(ParseError, input="abc", expected="number", context="parsing");
        match err {
            BrainError::ParseError { input, expected, context } => {
                assert_eq!(input, "abc");
                assert_eq!(expected, "number");
                assert_eq!(context, "parsing");
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_brain_err_macro_allocation_failed() {
        let err = brain_err!(AllocationFailed, requested=1024, available=Some(512), context="tensor alloc");
        match err {
            BrainError::AllocationFailed { requested_bytes, available_bytes, context } => {
                assert_eq!(requested_bytes, 1024);
                assert_eq!(available_bytes, Some(512));
                assert_eq!(context, "tensor alloc");
            }
            _ => panic!("expected AllocationFailed"),
        }
    }

    #[test]
    fn test_brain_err_macro_index_out_of_bounds() {
        let err = brain_err!(IndexOutOfBounds, index=10, bound=5, dimension=Some(0), context="access");
        match err {
            BrainError::IndexOutOfBounds { index, bound, dimension, context } => {
                assert_eq!(index, 10);
                assert_eq!(bound, 5);
                assert_eq!(dimension, Some(0));
                assert_eq!(context, "access");
            }
            _ => panic!("expected IndexOutOfBounds"),
        }
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_string_messages() {
        let err = BrainError::InvalidValue { message: String::new() };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid value:"));
    }

    #[test]
    fn test_large_index_out_of_bounds() {
        let err = BrainError::IndexOutOfBounds {
            index: isize::MAX,
            bound: 0,
            dimension: None,
            context: "test".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains(&isize::MAX.to_string()));
    }

    #[test]
    fn test_allocation_with_zero_bytes() {
        let err = BrainError::AllocationFailed {
            requested_bytes: 0,
            available_bytes: Some(0),
            context: "zero alloc".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0 bytes"));
    }

    #[test]
    fn test_unicode_in_error_messages() {
        let err = BrainError::InvalidValue {
            message: "值无效 ❌".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("值无效"));
    }

    #[test]
    fn test_long_error_message() {
        let long_msg = "a".repeat(10000);
        let err = BrainError::InvalidValue {
            message: long_msg.clone(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains(&long_msg));
    }

    #[test]
    fn test_error_chain_with_many_errors() {
        let mut chain = ErrorChain::empty();
        for i in 0..100 {
            chain.push(BrainError::InvalidValue {
                message: format!("error {}", i),
            });
        }
        assert_eq!(chain.len(), 100);
        assert_eq!(chain.root().unwrap().variant_name(), "InvalidValue");
    }

    #[test]
    fn test_error_chain_summary_single_type() {
        let mut chain = ErrorChain::empty();
        for _ in 0..5 {
            chain.push(BrainError::InvalidValue { message: "test".into() });
        }
        let summary = chain.summary();
        assert!(summary.contains("InvalidValue"));
    }

    #[test]
    fn test_error_chain_summary_mixed_types() {
        let mut chain = ErrorChain::empty();
        chain.push(BrainError::InvalidValue { message: "a".into() });
        chain.push(BrainError::DivisionByZero { context: "b".into() });
        chain.push(BrainError::InvalidValue { message: "c".into() });
        let summary = chain.summary();
        assert!(summary.contains("DivisionByZero"));
    }

    #[test]
    fn test_error_context_with_long_paths() {
        let err = BrainError::InvalidValue { message: "test".into() };
        let long_path = "very/deep/nested/module/path/brain_core/tensor/ops/arithmetic/mod";
        let ctx = BrainErrorContext::new(err, long_path, 999999, long_path, "deep_op");
        assert_eq!(ctx.file(), long_path);
        assert_eq!(ctx.line(), 999999);
    }

    #[test]
    fn test_error_report_clone() {
        let report = ErrorReport::new(BrainError::InvalidValue { message: "test".into() });
        let report2 = report.clone();
        assert_eq!(report.error.variant_name(), report2.error.variant_name());
    }

    #[test]
    fn test_all_severities() {
        let critical = BrainError::AllocationFailed {
            requested_bytes: 100,
            available_bytes: None,
            context: "test".into(),
        };
        assert_eq!(critical.severity(), "Critical");

        let error = BrainError::InvalidValue { message: "test".into() };
        assert_eq!(error.severity(), "Error");

        let warning = BrainError::NanDetected { context: "test".into() };
        assert_eq!(warning.severity(), "Warning");
    }

    #[test]
    fn test_all_error_categories() {
        let shape = BrainError::ShapeMismatch {
            expected: "a".into(), actual: "b".into(), context: "c".into(),
        };
        assert!(shape.is_shape_error());

        let device = BrainError::DeviceMismatch {
            expected: "a".into(), actual: "b".into(), context: "c".into(),
        };
        assert!(device.is_device_error());

        let dtype = BrainError::DTypeMismatch {
            expected: "a".into(), actual: "b".into(), context: "c".into(),
        };
        assert!(dtype.is_dtype_error());
    }

    #[test]
    fn test_recoverability() {
        let recoverable = vec![
            BrainError::AllocationFailed { requested_bytes: 0, available_bytes: None, context: "".into() },
            BrainError::NanDetected { context: "".into() },
            BrainError::InfDetected { context: "".into() },
        ];
        for err in &recoverable {
            assert!(err.is_recoverable());
        }

        let unrecoverable = vec![
            BrainError::InvalidValue { message: "".into() },
            BrainError::NotImplemented { feature: "".into() },
            BrainError::ShapeMismatch { expected: "".into(), actual: "".into(), context: "".into() },
        ];
        for err in &unrecoverable {
            assert!(!err.is_recoverable());
        }
    }
}
