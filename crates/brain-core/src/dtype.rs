//! Data type definitions and conversions for the Brain deep learning framework.
//!
//! This module defines the [`DType`] enum which represents all supported numeric
//! and boolean data types. It provides conversion utilities, size queries,
//! type classification methods, and safe/lossless casting infrastructure.
//!
//! # Supported Types
//!
//! | Category    | Types                                                        |
//! |-------------|--------------------------------------------------------------|
//! | Floating    | `F16`, `BF16`, `F32`, `F64`                                  |
//! | Signed Int  | `I8`, `I16`, `I32`, `I64`                                   |
//! | Unsigned    | `U8`, `U16`, `U32`, `U64`                                   |
//! | Boolean     | `Bool`                                                       |
//! | Complex     | `Complex64`, `Complex128`                                   |
//!
//! # Type Casting
//!
//! The module provides comprehensive casting utilities:
//! - [`DType::can_cast_from`] - checks if a cast is valid
//! - [`DType::is_lossless_cast`] - checks if a cast preserves all values
//! - [`cast_slice_f64_to_f32`] - performs actual value conversion on slices
//! - [`promote_dtypes`] - finds the common type for binary operations
//!
//! # Examples
//!
//! ```
//! use brain_core::dtype::DType;
//!
//! let dt = DType::F32;
//! assert!(dt.is_float());
//! assert_eq!(dt.size_bytes(), 4);
//! assert_eq!(dt.bit_width(), 32);
//!
//! let promoted = DType::promote(DType::F16, DType::F32);
//! assert_eq!(promoted, DType::F32);
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

// =============================================================================
// DType Enum
// =============================================================================

/// Represents a numeric data type in the Brain framework.
///
/// Each variant corresponds to a specific bit width and representation,
/// matching common types used in deep learning and scientific computing.
///
/// # Type Categories
///
/// * **Floating point**: `F16`, `BF16`, `F32`, `F64` - for real-valued computation
/// * **Signed integer**: `I8`, `I16`, `I32`, `I64` - for index and label data
/// * **Unsigned integer**: `U8`, `U16`, `U32`, `U64` - for pixel and flag data
/// * **Boolean**: `Bool` - for masks and predicates
/// * **Complex**: `Complex64`, `Complex128` - for frequency domain operations
///
/// # Ordering
///
/// The variants are ordered by bit width, with signed types before unsigned
/// types at the same width. This ordering is consistent and deterministic
/// but does not carry semantic meaning.
///
/// # Serialization
///
/// DType implements `FromStr` for parsing type names (case-insensitive)
/// and `Display` for formatting. This makes it suitable for configuration
/// files and command-line arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    /// 16-bit IEEE 754 half-precision floating point (1 sign, 5 exp, 10 mantissa).
    /// Range: ±65504, ~3.6 decimal digits. Used in mixed-precision training
    /// and inference to reduce memory bandwidth.
    F16,

    /// 16-bit bfloat16 floating point (Google Brain / Intel format).
    /// 1 sign, 8 exponent, 7 mantissa. Same dynamic range as F32 but
    /// reduced precision (~2.4 decimal digits). Excellent for deep learning
    /// due to reduced risk of underflow/overflow compared to F16.
    BF16,

    /// 32-bit IEEE 754 single-precision floating point (1 sign, 8 exp, 23 mantissa).
    /// Range: ±3.4e38, ~7.2 decimal digits. The standard type for most
    /// deep learning training and inference.
    F32,

    /// 64-bit IEEE 754 double-precision floating point (1 sign, 11 exp, 52 mantissa).
    /// Range: ±1.8e308, ~15.9 decimal digits. Used for numerical precision
    /// in loss computation, Hessian calculation, and scientific computing.
    F64,

    /// 8-bit signed integer. Range: -128 to 127.
    /// Used for quantized models (e.g., INT8 inference) and label data.
    I8,

    /// 16-bit signed integer. Range: -32768 to 32767.
    /// Used for audio samples (PCM-16) and quantized weights.
    I16,

    /// 32-bit signed integer. Range: -2147483648 to 2147483647.
    /// Used for indices, offsets, and integer computation.
    I32,

    /// 64-bit signed integer. Range: -9223372036854775808 to 9223372036854775807.
    /// Used for large indices and timestamp data.
    I64,

    /// 8-bit unsigned integer. Range: 0 to 255.
    /// Used for pixel values (uint8 images) and quantized inference.
    U8,

    /// 16-bit unsigned integer. Range: 0 to 65535.
    /// Used for 16-bit pixel depths and index values.
    U16,

    /// 32-bit unsigned integer. Range: 0 to 4294967295.
    /// Used for large index values and color values.
    U32,

    /// 64-bit unsigned integer. Range: 0 to 18446744073709551615.
    /// Used for memory offsets and bit flags.
    U64,

    /// Boolean type (1 bit, stored as 1 byte).
    /// Used for masks, predicates, and conditional operations.
    Bool,

    /// 64-bit complex number (32-bit real + 32-bit imaginary parts).
    /// Used for FFT, frequency-domain operations, and signal processing.
    Complex64,

    /// 128-bit complex number (64-bit real + 64-bit imaginary parts).
    /// Used for high-precision frequency domain operations.
    Complex128,
}

// =============================================================================
// DType Constants
// =============================================================================

impl DType {
    /// Total number of data type variants.
    pub const VARIANT_COUNT: usize = 16;

    /// All floating-point types in order of increasing precision.
    pub const FLOAT_TYPES: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];

    /// All signed integer types in order of increasing size.
    pub const SIGNED_INT_TYPES: [DType; 4] = [DType::I8, DType::I16, DType::I32, DType::I64];

    /// All unsigned integer types in order of increasing size.
    pub const UNSIGNED_INT_TYPES: [DType; 4] = [DType::U8, DType::U16, DType::U32, DType::U64];

    /// All integer types (signed + unsigned) in order.
    pub const INT_TYPES: [DType; 8] = [
        DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
    ];

    /// All complex types in order of increasing size.
    pub const COMPLEX_TYPES: [DType; 2] = [DType::Complex64, DType::Complex128];

    /// All numeric types (float + int + complex) in order.
    pub const ALL_NUMERIC: [DType; 14] = [
        DType::F16, DType::BF16, DType::F32, DType::F64,
        DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::Complex64, DType::Complex128,
    ];

    /// All data types.
    pub const ALL: [DType; 16] = [
        DType::F16, DType::BF16, DType::F32, DType::F64,
        DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::Bool, DType::Complex64, DType::Complex128,
    ];
}

// =============================================================================
// DType Core Methods
// =============================================================================

impl DType {
    /// Returns the size of this data type in bytes.
    ///
    /// For complex types, the returned size includes both real and imaginary
    /// components (e.g., Complex64 returns 8, not 4).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.size_bytes(), 4);
    /// assert_eq!(DType::F64.size_bytes(), 8);
    /// assert_eq!(DType::Complex64.size_bytes(), 8);
    /// assert_eq!(DType::Complex128.size_bytes(), 16);
    /// assert_eq!(DType::Bool.size_bytes(), 1);
    /// assert_eq!(DType::I8.size_bytes(), 1);
    /// ```
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Bool => 1,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// Returns `true` if this is a floating-point type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.is_float());
    /// assert!(DType::F16.is_float());
    /// assert!(!DType::I32.is_float());
    /// ```
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Returns `true` if this is a signed integer type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::I32.is_signed_int());
    /// assert!(!DType::U32.is_signed_int());
    /// ```
    pub fn is_signed_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// Returns `true` if this is an integer type (signed or unsigned).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::I32.is_int());
    /// assert!(DType::U32.is_int());
    /// assert!(!DType::F32.is_int());
    /// ```
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64
                | DType::U8 | DType::U16 | DType::U32 | DType::U64
        )
    }

    /// Returns `true` if this is an unsigned integer type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::U32.is_unsigned());
    /// assert!(!DType::I32.is_unsigned());
    /// ```
    pub fn is_unsigned(&self) -> bool {
        matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Returns `true` if this is a complex number type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::Complex64.is_complex());
    /// assert!(!DType::F32.is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    /// Returns the total bit width of this data type.
    ///
    /// For complex types, this includes both real and imaginary components.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.bit_width(), 32);
    /// assert_eq!(DType::I64.bit_width(), 64);
    /// assert_eq!(DType::Complex64.bit_width(), 64);
    /// assert_eq!(DType::Complex128.bit_width(), 128);
    /// ```
    pub fn bit_width(&self) -> u32 {
        match self {
            DType::F16 => 16,
            DType::BF16 => 16,
            DType::F32 => 32,
            DType::F64 => 64,
            DType::I8 => 8,
            DType::I16 => 16,
            DType::I32 => 32,
            DType::I64 => 64,
            DType::U8 => 8,
            DType::U16 => 16,
            DType::U32 => 32,
            DType::U64 => 64,
            DType::Bool => 8,
            DType::Complex64 => 64,
            DType::Complex128 => 128,
        }
    }

    /// Returns the bit width of the real part of this data type.
    ///
    /// For non-complex types, this is the same as `bit_width()`.
    /// For complex types, this is half the total bit width.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.real_bit_width(), 32);
    /// assert_eq!(DType::Complex64.real_bit_width(), 32);
    /// ```
    pub fn real_bit_width(&self) -> u32 {
        if self.is_complex() {
            self.bit_width() / 2
        } else {
            self.bit_width()
        }
    }

    /// Returns `true` if values of this type can represent negative numbers.
    ///
    /// This includes floating-point types, signed integers, and complex types.
    /// Unsigned integers and booleans cannot represent negative values.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.is_signed());
    /// assert!(DType::I32.is_signed());
    /// assert!(!DType::U32.is_signed());
    /// assert!(!DType::Bool.is_signed());
    /// ```
    pub fn is_signed(&self) -> bool {
        self.is_float() || self.is_signed_int() || self.is_complex()
    }

    /// Returns `true` if this type is numeric (not Bool).
    ///
    /// All types except Bool are considered numeric.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.is_numeric());
    /// assert!(DType::I32.is_numeric());
    /// assert!(!DType::Bool.is_numeric());
    /// ```
    pub fn is_numeric(&self) -> bool {
        !matches!(self, DType::Bool)
    }

    /// Returns `true` if this type supports arithmetic operations.
    ///
    /// Bool does not support most arithmetic operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.supports_arithmetic());
    /// assert!(!DType::Bool.supports_arithmetic());
    /// ```
    pub fn supports_arithmetic(&self) -> bool {
        !matches!(self, DType::Bool)
    }

    /// Returns `true` if this type supports comparison operations
    /// that return a Bool (e.g., <, >, ==, !=).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.supports_comparison());
    /// assert!(DType::Bool.supports_comparison());
    /// ```
    pub fn supports_comparison(&self) -> bool {
        true
    }

    /// Returns the category of this data type as a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.category(), "float");
    /// assert_eq!(DType::I32.category(), "signed_int");
    /// assert_eq!(DType::U8.category(), "unsigned_int");
    /// assert_eq!(DType::Bool.category(), "bool");
    /// assert_eq!(DType::Complex64.category(), "complex");
    /// ```
    pub fn category(&self) -> &'static str {
        match self {
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => "float",
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => "signed_int",
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => "unsigned_int",
            DType::Bool => "bool",
            DType::Complex64 | DType::Complex128 => "complex",
        }
    }

    /// Returns a short human-readable name for this data type.
    ///
    /// This is the same as the Display output but available as a `&str`.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.short_name(), "F32");
    /// assert_eq!(DType::Complex64.short_name(), "Complex64");
    /// ```
    pub fn short_name(&self) -> &'static str {
        match self {
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::U32 => "U32",
            DType::U64 => "U64",
            DType::Bool => "Bool",
            DType::Complex64 => "Complex64",
            DType::Complex128 => "Complex128",
        }
    }

    /// Returns a Rust-style type name for this data type.
    ///
    /// This returns the name as it would appear in Rust source code.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.rust_type_name(), "f32");
    /// assert_eq!(DType::I64.rust_type_name(), "i64");
    /// ```
    pub fn rust_type_name(&self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::Bool => "bool",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }

    /// Returns a C-style type name for this data type.
    ///
    /// This returns the name as it would appear in C source code.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.c_type_name(), "float");
    /// assert_eq!(DType::I64.c_type_name(), "int64_t");
    /// ```
    pub fn c_type_name(&self) -> &'static str {
        match self {
            DType::F16 => "_Float16",
            DType::BF16 => "__bf16",
            DType::F32 => "float",
            DType::F64 => "double",
            DType::I8 => "int8_t",
            DType::I16 => "int16_t",
            DType::I32 => "int32_t",
            DType::I64 => "int64_t",
            DType::U8 => "uint8_t",
            DType::U16 => "uint16_t",
            DType::U32 => "uint32_t",
            DType::U64 => "uint64_t",
            DType::Bool => "bool",
            DType::Complex64 => "float _Complex",
            DType::Complex128 => "double _Complex",
        }
    }

    /// Returns a NumPy-style type string for this data type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.numpy_name(), "float32");
    /// assert_eq!(DType::I32.numpy_name(), "int32");
    /// ```
    pub fn numpy_name(&self) -> &'static str {
        match self {
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::Bool => "bool",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }

    /// Returns the minimum value representable by this type (as a string).
    ///
    /// For floating-point types, returns the most negative finite value.
    /// For Bool, returns "false".
    /// For complex types, returns the minimum real component.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::I8.min_value(), "-128");
    /// assert_eq!(DType::U8.min_value(), "0");
    /// ```
    pub fn min_value(&self) -> &'static str {
        match self {
            DType::F16 => "-65504.0",
            DType::BF16 => "-3.389531389251535e+38",
            DType::F32 => "-3.4028235e+38",
            DType::F64 => "-1.7976931348623157e+308",
            DType::I8 => "-128",
            DType::I16 => "-32768",
            DType::I32 => "-2147483648",
            DType::I64 => "-9223372036854775808",
            DType::U8 => "0",
            DType::U16 => "0",
            DType::U32 => "0",
            DType::U64 => "0",
            DType::Bool => "false",
            DType::Complex64 => "-3.4028235e+38",
            DType::Complex128 => "-1.7976931348623157e+308",
        }
    }

    /// Returns the maximum value representable by this type (as a string).
    ///
    /// For floating-point types, returns the most positive finite value.
    /// For Bool, returns "true".
    /// For complex types, returns the maximum real component.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::U8.max_value(), "255");
    /// assert_eq!(DType::I8.max_value(), "127");
    /// ```
    pub fn max_value(&self) -> &'static str {
        match self {
            DType::F16 => "65504.0",
            DType::BF16 => "3.389531389251535e+38",
            DType::F32 => "3.4028235e+38",
            DType::F64 => "1.7976931348623157e+308",
            DType::I8 => "127",
            DType::I16 => "32767",
            DType::I32 => "2147483647",
            DType::I64 => "9223372036854775807",
            DType::U8 => "255",
            DType::U16 => "65535",
            DType::U32 => "4294967295",
            DType::U64 => "18446744073709551615",
            DType::Bool => "true",
            DType::Complex64 => "3.4028235e+38",
            DType::Complex128 => "1.7976931348623157e+308",
        }
    }

    /// Returns the machine epsilon (smallest representable difference
    /// between 1.0 and the next representable value) for this type.
    ///
    /// Only defined for floating-point and complex types.
    /// Returns `None` for integer and boolean types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.epsilon().is_some());
    /// assert!(DType::I32.epsilon().is_none());
    /// ```
    pub fn epsilon(&self) -> Option<f64> {
        match self {
            DType::F16 => Some(9.77e-4),
            DType::BF16 => Some(7.18e-2),
            DType::F32 => Some(1.1920929e-7),
            DType::F64 => Some(2.220446049250313e-16),
            DType::Complex64 => Some(1.1920929e-7),
            DType::Complex128 => Some(2.220446049250313e-16),
            _ => None,
        }
    }

    /// Returns the radix (base) of this floating-point type's representation.
    ///
    /// Returns `None` for non-floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.radix(), Some(2));
    /// assert_eq!(DType::I32.radix(), None);
    /// ```
    pub fn radix(&self) -> Option<u32> {
        if self.is_float() || self.is_complex() {
            Some(2)
        } else {
            None
        }
    }

    /// Returns the number of mantissa digits for this floating-point type.
    ///
    /// Returns `None` for non-floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.mantissa_digits(), Some(24));
    /// assert_eq!(DType::F16.mantissa_digits(), Some(11));
    /// ```
    pub fn mantissa_digits(&self) -> Option<u32> {
        match self {
            DType::F16 => Some(11),
            DType::BF16 => Some(8),
            DType::F32 => Some(24),
            DType::F64 => Some(53),
            DType::Complex64 => Some(24),
            DType::Complex128 => Some(53),
            _ => None,
        }
    }

    /// Returns the maximum exponent for this floating-point type.
    ///
    /// Returns `None` for non-floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.max_exponent(), Some(128));
    /// ```
    pub fn max_exponent(&self) -> Option<i32> {
        match self {
            DType::F16 => Some(16),
            DType::BF16 => Some(128),
            DType::F32 => Some(128),
            DType::F64 => Some(1024),
            DType::Complex64 => Some(128),
            DType::Complex128 => Some(1024),
            _ => None,
        }
    }

    /// Returns the minimum exponent for this floating-point type.
    ///
    /// Returns `None` for non-floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.min_exponent(), Some(-125));
    /// ```
    pub fn min_exponent(&self) -> Option<i32> {
        match self {
            DType::F16 => Some(-14),
            DType::BF16 => Some(-126),
            DType::F32 => Some(-125),
            DType::F64 => Some(-1021),
            DType::Complex64 => Some(-125),
            DType::Complex128 => Some(-1021),
            _ => None,
        }
    }

    /// Parses a data type from a string (case-insensitive).
    ///
    /// # Supported Names
    ///
    /// * Short names: "F32", "f32", "F16", "bf16", etc.
    /// * NumPy names: "float32", "int32", "uint8", etc.
    /// * Aliases: "half", "bfloat16", "single", "double", "float", "int", "long"
    ///
    /// # Errors
    ///
    /// Returns an error if the string does not match any known type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// use std::str::FromStr;
    /// assert_eq!(DType::from_str("F32"), Ok(DType::F32));
    /// assert_eq!(DType::from_str("float32"), Ok(DType::F32));
    /// assert_eq!(DType::from_str("int64"), Ok(DType::I64));
    /// assert!(DType::from_str("invalid").is_err());
    /// ```
    pub fn from_str(s: &str) -> Result<DType, String> {
        match s.to_lowercase().as_str() {
            "f16" | "float16" | "half" => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::BF16),
            "f32" | "float32" | "single" | "float" => Ok(DType::F32),
            "f64" | "float64" | "double" => Ok(DType::F64),
            "i8" | "int8" => Ok(DType::I8),
            "i16" | "int16" | "short" => Ok(DType::I16),
            "i32" | "int32" | "int" => Ok(DType::I32),
            "i64" | "int64" | "long" => Ok(DType::I64),
            "u8" | "uint8" => Ok(DType::U8),
            "u16" | "uint16" => Ok(DType::U16),
            "u32" | "uint32" => Ok(DType::U32),
            "u64" | "uint64" => Ok(DType::U64),
            "bool" | "boolean" => Ok(DType::Bool),
            "complex64" | "complex32" => Ok(DType::Complex64),
            "complex128" | "complex64_f64" => Ok(DType::Complex128),
            _ => Err(format!(
                "Unknown dtype '{}'. Supported: f16, bf16, f32, f64, i8, i16, i32, i64, \
                 u8, u16, u32, u64, bool, complex64, complex128",
                s
            )),
        }
    }

    /// Returns whether a cast from the given source type to this type is valid.
    ///
    /// All casts between numeric types are valid. Casting from Bool to any
    /// numeric type is valid. Casting from any type to Bool is also valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.can_cast_from(DType::I32));
    /// assert!(DType::I32.can_cast_from(DType::F32));
    /// assert!(DType::F64.can_cast_from(DType::Bool));
    /// ```
    pub fn can_cast_from(&self, from: DType) -> bool {
        // All numeric types can cast between each other
        if self.is_numeric() && from.is_numeric() {
            return true;
        }
        // Bool can cast to/from any numeric type
        if *self == DType::Bool || from == DType::Bool {
            return true;
        }
        // Same type is always valid
        *self == from
    }

    /// Returns whether a cast from the given source type to this type is lossless.
    ///
    /// A lossless cast preserves all representable values of the source type
    /// in the target type. For example, I8 -> I32 is lossless, but I32 -> I8
    /// is not (values outside the I8 range would be lost).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::I32.is_lossless_cast(DType::I8)); // I8 fits in I32
    /// assert!(!DType::I8.is_lossless_cast(DType::I32)); // I32 may not fit in I8
    /// assert!(DType::F64.is_lossless_cast(DType::F32)); // F32 fits in F64
    /// assert!(!DType::F32.is_lossless_cast(DType::F64)); // F64 may lose precision
    /// ```
    pub fn is_lossless_cast(&self, from: DType) -> bool {
        if *self == from {
            return true;
        }
        match (*self, from) {
            // Same category widening
            (DType::I8, DType::Bool) => true,
            (DType::I16, DType::Bool) => true,
            (DType::I32, DType::Bool) => true,
            (DType::I64, DType::Bool) => true,
            (DType::U8, DType::Bool) => true,
            (DType::U16, DType::Bool) => true,
            (DType::U32, DType::Bool) => true,
            (DType::U64, DType::Bool) => true,
            (DType::F32, DType::Bool) => true,
            (DType::F64, DType::Bool) => true,

            // Integer widening (signed)
            (DType::I16, DType::I8) => true,
            (DType::I32, DType::I8) => true,
            (DType::I64, DType::I8) => true,
            (DType::I32, DType::I16) => true,
            (DType::I64, DType::I16) => true,
            (DType::I64, DType::I32) => true,

            // Integer widening (unsigned)
            (DType::U16, DType::U8) => true,
            (DType::U32, DType::U8) => true,
            (DType::U64, DType::U8) => true,
            (DType::U32, DType::U16) => true,
            (DType::U64, DType::U16) => true,
            (DType::U64, DType::U32) => true,

            // Unsigned to wider signed (lossless because values are positive)
            (DType::I16, DType::U8) => true,
            (DType::I32, DType::U8) => true,
            (DType::I64, DType::U8) => true,
            (DType::I32, DType::U16) => true,
            (DType::I64, DType::U16) => true,
            (DType::I64, DType::U32) => true,

            // Float widening
            (DType::F32, DType::F16) => true,
            (DType::BF16, DType::F16) => true,
            (DType::F64, DType::F16) => true,
            (DType::F32, DType::BF16) => true,
            (DType::F64, DType::BF16) => true,
            (DType::F64, DType::F32) => true,

            // Float from integer (integers fit exactly in float up to 2^24)
            (DType::F64, DType::I8) => true,
            (DType::F64, DType::I16) => true,
            (DType::F64, DType::I32) => true,
            (DType::F64, DType::U8) => true,
            (DType::F64, DType::U16) => true,
            (DType::F64, DType::U32) => true,
            (DType::F32, DType::I8) => true,
            (DType::F32, DType::U8) => true,
            (DType::F32, DType::U16) => true,

            // Complex widening
            (DType::Complex128, DType::Complex64) => true,
            (DType::Complex64, DType::F32) => true,
            (DType::Complex128, DType::F32) => true,
            (DType::Complex128, DType::F64) => true,

            _ => false,
        }
    }

    /// Returns the "promoted" dtype when two dtypes are combined in a binary operation.
    ///
    /// Type promotion follows these rules:
    /// 1. If types are the same, return that type
    /// 2. Complex types promote to Complex64 or Complex128
    /// 3. Floating types promote to the wider float
    /// 4. Float + Integer promotes to float
    /// 5. Integer types promote to the wider integer (unsigned > signed at same width)
    /// 6. Bool promotes to I8 when combined with integers, F32 with floats
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::promote(DType::F16, DType::F32), DType::F32);
    /// assert_eq!(DType::promote(DType::I8, DType::F32), DType::F32);
    /// assert_eq!(DType::promote(DType::I16, DType::I32), DType::I32);
    /// assert_eq!(DType::promote(DType::U32, DType::I32), DType::U32);
    /// ```
    pub fn promote(a: DType, b: DType) -> DType {
        if a == b {
            return a;
        }

        // Bool promotion
        if a == DType::Bool {
            return if b.is_float() { DType::F32 } else { DType::I8 };
        }
        if b == DType::Bool {
            return if a.is_float() { DType::F32 } else { DType::I8 };
        }

        // Complex promotion
        if a.is_complex() || b.is_complex() {
            return match (a, b) {
                (DType::Complex128, _) | (_, DType::Complex128) => DType::Complex128,
                _ => DType::Complex64,
            };
        }

        // Float promotion
        if a.is_float() && b.is_float() {
            return if a > b { a } else { b };
        }
        if a.is_float() {
            return a;
        }
        if b.is_float() {
            return b;
        }

        // Integer promotion: wider type wins, unsigned > signed at same width
        if a.is_int() && b.is_int() {
            return if a > b { a } else { b };
        }

        // Default fallback
        DType::F32
    }

    /// Returns the common dtype for a list of dtypes.
    ///
    /// This is equivalent to repeatedly applying `promote` to all pairs.
    /// Returns an error if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// let common = DType::common_dtype(&[DType::I8, DType::I16, DType::F32]).unwrap();
    /// assert_eq!(common, DType::F32);
    /// ```
    pub fn common_dtype(types: &[DType]) -> Result<DType, String> {
        if types.is_empty() {
            return Err("cannot compute common dtype of empty list".to_string());
        }
        let mut result = types[0];
        for &dt in &types[1..] {
            result = DType::promote(result, dt);
        }
        Ok(result)
    }

    /// Returns the number of decimal digits of precision for this type.
    ///
    /// Only meaningful for floating-point types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.decimal_digits(), Some(7));
    /// assert_eq!(DType::F64.decimal_digits(), Some(16));
    /// assert_eq!(DType::I32.decimal_digits(), None);
    /// ```
    pub fn decimal_digits(&self) -> Option<u32> {
        match self {
            DType::F16 => Some(3),
            DType::BF16 => Some(2),
            DType::F32 => Some(7),
            DType::F64 => Some(16),
            DType::Complex64 => Some(7),
            DType::Complex128 => Some(16),
            _ => None,
        }
    }

    /// Returns whether this type can represent infinity and NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.has_infinity());
    /// assert!(!DType::I32.has_infinity());
    /// ```
    pub fn has_infinity(&self) -> bool {
        self.is_float() || self.is_complex()
    }

    /// Returns whether this type can represent subnormal (denormalized) values.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.has_subnormal());
    /// assert!(!DType::I32.has_subnormal());
    /// ```
    pub fn has_subnormal(&self) -> bool {
        self.is_float() || self.is_complex()
    }

    /// Returns whether this type is commonly used for inference (quantized types).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::I8.is_quantized_type());
    /// assert!(DType::U8.is_quantized_type());
    /// assert!(!DType::F32.is_quantized_type());
    /// ```
    pub fn is_quantized_type(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::BF16
        )
    }

    /// Returns whether this type is suitable for training gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert!(DType::F32.is_gradient_type());
    /// assert!(!DType::I32.is_gradient_type());
    /// assert!(!DType::Bool.is_gradient_type());
    /// ```
    pub fn is_gradient_type(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::BF16)
    }

    /// Returns an iterator over all DType variants.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// let count = DType::all_types().count();
    /// assert_eq!(count, 16);
    /// ```
    pub fn all_types() -> std::slice::Iter<'static, DType> {
        DType::ALL.iter()
    }

    /// Returns the alignment requirement for this type in bytes.
    ///
    /// This is the same as `size_bytes()` for all current types.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::DType;
    /// assert_eq!(DType::F32.alignment(), 4);
    /// assert_eq!(DType::I64.alignment(), 8);
    /// ```
    pub fn alignment(&self) -> usize {
        self.size_bytes()
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

// =============================================================================
// Default Implementation
// =============================================================================

impl Default for DType {
    /// Returns `DType::F32` as the default data type.
    ///
    /// F32 is the standard precision for deep learning operations
    /// and is the most commonly used type throughout the framework.
    fn default() -> Self {
        DType::F32
    }
}

// =============================================================================
// FromStr Implementation
// =============================================================================

impl FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DType::from_str(s)
    }
}

// =============================================================================
// Cast Functions
// =============================================================================

/// Converts a slice of `f64` values to a `Vec<f32>` using truncation to single precision.
///
/// This is a pure-Rust implementation that converts each f64 value to f32.
/// Values outside the f32 range will be clamped to f32::MAX or f32::MIN.
/// NaN and Inf values are preserved.
///
/// # Arguments
///
/// * `data` - Slice of f64 values to convert
///
/// # Returns
///
/// A Vec<f32> with the converted values.
///
/// # Examples
///
/// ```
/// use brain_core::dtype::cast_slice_f64_to_f32;
/// let input = vec![1.0_f64, 2.5, -3.7, 0.0];
/// let output = cast_slice_f64_to_f32(&input);
/// assert_eq!(output, vec![1.0_f32, 2.5, -3.7, 0.0]);
/// ```
pub fn cast_slice_f64_to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

/// Converts a slice of `f32` values to a `Vec<f64>` with zero loss.
///
/// # Arguments
///
/// * `data` - Slice of f32 values to convert
///
/// # Returns
///
/// A Vec<f64> with the converted values.
///
/// # Examples
///
/// ```
/// use brain_core::dtype::cast_slice_f32_to_f64;
/// let input = vec![1.0_f32, 2.5, -3.7];
/// let output = cast_slice_f32_to_f64(&input);
/// assert_eq!(output.len(), 3);
/// assert!((output[0] - 1.0).abs() < 1e-10);
/// ```
pub fn cast_slice_f32_to_f64(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&v| v as f64).collect()
}

/// Converts a slice of `i32` values to a `Vec<f32>`.
///
/// # Arguments
///
/// * `data` - Slice of i32 values to convert
///
/// # Returns
///
/// A Vec<f32> with the converted values.
pub fn cast_slice_i32_to_f32(data: &[i32]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

/// Converts a slice of `i64` values to a `Vec<f64>`.
///
/// # Arguments
///
/// * `data` - Slice of i64 values to convert
///
/// # Returns
///
/// A Vec<f64> with the converted values.
pub fn cast_slice_i64_to_f64(data: &[i64]) -> Vec<f64> {
    data.iter().map(|&v| v as f64).collect()
}

/// Converts a slice of `f32` values to a `Vec<i32>` using truncation.
///
/// Values outside the i32 range will saturate to i32::MAX or i32::MIN.
/// NaN values are converted to 0.
///
/// # Arguments
///
/// * `data` - Slice of f32 values to convert
///
/// # Returns
///
/// A Vec<i32> with the converted values.
pub fn cast_slice_f32_to_i32(data: &[f32]) -> Vec<i32> {
    data.iter()
        .map(|&v| {
            if v.is_nan() {
                0
            } else if v > i32::MAX as f32 {
                i32::MAX
            } else if v < i32::MIN as f32 {
                i32::MIN
            } else {
                v as i32
            }
        })
        .collect()
}

/// Converts a slice of `bool` values to a `Vec<f32>` (false -> 0.0, true -> 1.0).
///
/// # Arguments
///
/// * `data` - Slice of bool values to convert
///
/// # Returns
///
/// A Vec<f32> with 0.0 for false and 1.0 for true.
pub fn cast_slice_bool_to_f32(data: &[bool]) -> Vec<f32> {
    data.iter().map(|&v| if v { 1.0f32 } else { 0.0f32 }).collect()
}

/// Converts a slice of `f32` values to a `Vec<bool>` (0.0 -> false, non-zero -> true).
///
/// # Arguments
///
/// * `data` - Slice of f32 values to convert
///
/// # Returns
///
/// A Vec<bool> with false for zero values and true for non-zero values.
pub fn cast_slice_f32_to_bool(data: &[f32]) -> Vec<bool> {
    data.iter().map(|&v| v != 0.0).collect()
}

/// Converts a slice of `u8` values to a `Vec<f32>`.
///
/// # Arguments
///
/// * `data` - Slice of u8 values to convert
///
/// # Returns
///
/// A Vec<f32> with the converted values in range [0.0, 255.0].
pub fn cast_slice_u8_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&v| v as f32).collect()
}

/// Converts a slice of `f32` values to a `Vec<u8>` with clamping.
///
/// Values are clamped to [0, 255] and truncated to integers.
///
/// # Arguments
///
/// * `data` - Slice of f32 values to convert
///
/// # Returns
///
/// A Vec<u8> with the converted values.
pub fn cast_slice_f32_to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|&v| {
            if v.is_nan() {
                0
            } else if v > 255.0 {
                255
            } else if v < 0.0 {
                0
            } else {
                v as u8
            }
        })
        .collect()
}

/// Converts a slice of `i64` values to a `Vec<i32>` with saturation.
///
/// Values outside the i32 range are saturated.
///
/// # Arguments
///
/// * `data` - Slice of i64 values to convert
///
/// # Returns
///
/// A Vec<i32> with the converted values.
pub fn cast_slice_i64_to_i32(data: &[i64]) -> Vec<i32> {
    data.iter()
        .map(|&v| {
            if v > i32::MAX as i64 {
                i32::MAX
            } else if v < i32::MIN as i64 {
                i32::MIN
            } else {
                v as i32
            }
        })
        .collect()
}

// =============================================================================
// DTypeInfo Struct
// =============================================================================

/// Comprehensive metadata about a specific data type.
///
/// `DTypeInfo` provides a single struct that aggregates all properties of a
/// data type, useful for introspection, serialization, and documentation.
///
/// # Examples
///
/// ```
/// use brain_core::dtype::{DType, DTypeInfo};
/// let info = DTypeInfo::for_dtype(DType::F32);
/// assert_eq!(info.name, "F32");
/// assert_eq!(info.size_bytes, 4);
/// assert!(info.is_float);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DTypeInfo {
    /// The canonical name of this data type (e.g., "F32").
    pub name: &'static str,
    /// The size in bytes of a single element.
    pub size_bytes: usize,
    /// The total bit width.
    pub bit_width: u32,
    /// Whether this is a floating-point type.
    pub is_float: bool,
    /// Whether this is a signed integer type.
    pub is_signed_int: bool,
    /// Whether this is an unsigned integer type.
    pub is_unsigned_int: bool,
    /// Whether this is a boolean type.
    pub is_bool: bool,
    /// Whether this is a complex number type.
    pub is_complex: bool,
    /// Whether this is any numeric type.
    pub is_numeric: bool,
    /// Whether this type can represent negative values.
    pub is_signed: bool,
    /// The type category name.
    pub category: &'static str,
    /// The NumPy-compatible name.
    pub numpy_name: &'static str,
    /// The C-compatible type name.
    pub c_type_name: &'static str,
    /// The Rust type name.
    pub rust_type_name: &'static str,
    /// Minimum representable value (as a string).
    pub min_value: &'static str,
    /// Maximum representable value (as a string).
    pub max_value: &'static str,
    /// Whether the type supports infinity/NaN.
    pub has_infinity: bool,
    /// Whether the type supports subnormal values.
    pub has_subnormal: bool,
    /// Machine epsilon for floating-point types (None for integers).
    pub epsilon: Option<f64>,
    /// Mantissa digits for floating-point types.
    pub mantissa_digits: Option<u32>,
    /// Maximum exponent for floating-point types.
    pub max_exponent: Option<i32>,
    /// Minimum exponent for floating-point types.
    pub min_exponent: Option<i32>,
    /// Number of decimal digits of precision.
    pub decimal_digits: Option<u32>,
    /// Alignment requirement in bytes.
    pub alignment: usize,
}

impl DTypeInfo {
    /// Creates a `DTypeInfo` for the given data type.
    ///
    /// This is the primary constructor for obtaining type metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::{DType, DTypeInfo};
    /// let info = DTypeInfo::for_dtype(DType::F64);
    /// assert_eq!(info.name, "F64");
    /// assert_eq!(info.size_bytes, 8);
    /// ```
    pub fn for_dtype(dt: DType) -> Self {
        DTypeInfo {
            name: dt.short_name(),
            size_bytes: dt.size_bytes(),
            bit_width: dt.bit_width(),
            is_float: dt.is_float(),
            is_signed_int: dt.is_signed_int(),
            is_unsigned_int: dt.is_unsigned(),
            is_bool: *dt == DType::Bool,
            is_complex: dt.is_complex(),
            is_numeric: dt.is_numeric(),
            is_signed: dt.is_signed(),
            category: dt.category(),
            numpy_name: dt.numpy_name(),
            c_type_name: dt.c_type_name(),
            rust_type_name: dt.rust_type_name(),
            min_value: dt.min_value(),
            max_value: dt.max_value(),
            has_infinity: dt.has_infinity(),
            has_subnormal: dt.has_subnormal(),
            epsilon: dt.epsilon(),
            mantissa_digits: dt.mantissa_digits(),
            max_exponent: dt.max_exponent(),
            min_exponent: dt.min_exponent(),
            decimal_digits: dt.decimal_digits(),
            alignment: dt.alignment(),
        }
    }

    /// Returns a reference to the DType corresponding to this info.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::{DType, DTypeInfo};
    /// let info = DTypeInfo::for_dtype(DType::F32);
    /// assert_eq!(info.as_dtype(), DType::F32);
    /// ```
    pub fn as_dtype(&self) -> DType {
        match self.name {
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            "F64" => DType::F64,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" => DType::U8,
            "U16" => DType::U16,
            "U32" => DType::U32,
            "U64" => DType::U64,
            "Bool" => DType::Bool,
            "Complex64" => DType::Complex64,
            "Complex128" => DType::Complex128,
            _ => DType::F32,
        }
    }

    /// Returns a formatted summary of this type's properties.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::dtype::{DType, DTypeInfo};
    /// let info = DTypeInfo::for_dtype(DType::F32);
    /// let summary = info.summary();
    /// assert!(summary.contains("F32"));
    /// assert!(summary.contains("4 bytes"));
    /// ```
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("{} ({} bits, {} bytes)", self.name, self.bit_width, self.size_bytes));
        lines.push(format!("  Category: {}", self.category));
        lines.push(format!("  NumPy: {}", self.numpy_name));
        lines.push(format!("  C type: {}", self.c_type_name));
        lines.push(format!("  Rust type: {}", self.rust_type_name));
        lines.push(format!("  Range: {} to {}", self.min_value, self.max_value));
        if let Some(eps) = self.epsilon {
            lines.push(format!("  Epsilon: {:.2e}", eps));
        }
        if let Some(digits) = self.decimal_digits {
            lines.push(format!("  Decimal digits: {}", digits));
        }
        lines.join("\n")
    }

    /// Returns DTypeInfo for all data types.
    pub fn all() -> Vec<DTypeInfo> {
        DType::ALL.iter().map(|dt| DTypeInfo::for_dtype(*dt)).collect()
    }
}

impl fmt::Display for DTypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// =============================================================================
// DTypeMap - Generic container for per-dtype values
// =============================================================================

/// A container that maps each DType to a value of type T.
///
/// This is useful for operations that need different behavior per data type,
/// such as allocation size calculators or operation dispatch tables.
///
/// # Examples
///
/// ```
/// use brain_core::dtype::{DType, DTypeMap};
/// let map = DTypeMap::from_fn(|dt| dt.size_bytes() * 2);
/// assert_eq!(map.get(DType::F32), 8);
/// assert_eq!(map.get(DType::I64), 16);
/// ```
#[derive(Debug, Clone)]
pub struct DTypeMap<T> {
    f16: T,
    bf16: T,
    f32: T,
    f64: T,
    i8: T,
    i16: T,
    i32: T,
    i64: T,
    u8: T,
    u16: T,
    u32: T,
    u64: T,
    bool_: T,
    complex64: T,
    complex128: T,
}

impl<T> DTypeMap<T> {
    /// Creates a new DTypeMap by applying a function to each DType.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(DType) -> T,
    {
        DTypeMap {
            f16: f(DType::F16),
            bf16: f(DType::BF16),
            f32: f(DType::F32),
            f64: f(DType::F64),
            i8: f(DType::I8),
            i16: f(DType::I16),
            i32: f(DType::I32),
            i64: f(DType::I64),
            u8: f(DType::U8),
            u16: f(DType::U16),
            u32: f(DType::U32),
            u64: f(DType::U64),
            bool_: f(DType::Bool),
            complex64: f(DType::Complex64),
            complex128: f(DType::Complex128),
        }
    }

    /// Creates a new DTypeMap where all values are the same.
    pub fn fill(value: T) -> Self
    where
        T: Clone,
    {
        DTypeMap {
            f16: value.clone(),
            bf16: value.clone(),
            f32: value.clone(),
            f64: value.clone(),
            i8: value.clone(),
            i16: value.clone(),
            i32: value.clone(),
            i64: value.clone(),
            u8: value.clone(),
            u16: value.clone(),
            u32: value.clone(),
            u64: value.clone(),
            bool_: value.clone(),
            complex64: value.clone(),
            complex128: value,
        }
    }

    /// Returns the value associated with the given DType.
    pub fn get(&self, dt: DType) -> &T {
        match dt {
            DType::F16 => &self.f16,
            DType::BF16 => &self.bf16,
            DType::F32 => &self.f32,
            DType::F64 => &self.f64,
            DType::I8 => &self.i8,
            DType::I16 => &self.i16,
            DType::I32 => &self.i32,
            DType::I64 => &self.i64,
            DType::U8 => &self.u8,
            DType::U16 => &self.u16,
            DType::U32 => &self.u32,
            DType::U64 => &self.u64,
            DType::Bool => &self.bool_,
            DType::Complex64 => &self.complex64,
            DType::Complex128 => &self.complex128,
        }
    }

    /// Returns a mutable reference to the value associated with the given DType.
    pub fn get_mut(&mut self, dt: DType) -> &mut T {
        match dt {
            DType::F16 => &mut self.f16,
            DType::BF16 => &mut self.bf16,
            DType::F32 => &mut self.f32,
            DType::F64 => &mut self.f64,
            DType::I8 => &mut self.i8,
            DType::I16 => &mut self.i16,
            DType::I32 => &mut self.i32,
            DType::I64 => &mut self.i64,
            DType::U8 => &mut self.u8,
            DType::U16 => &mut self.u16,
            DType::U32 => &mut self.u32,
            DType::U64 => &mut self.u64,
            DType::Bool => &mut self.bool_,
            DType::Complex64 => &mut self.complex64,
            DType::Complex128 => &mut self.complex128,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Size Tests
    // =========================================================================

    #[test]
    fn test_f16_size_bytes() {
        assert_eq!(DType::F16.size_bytes(), 2);
    }

    #[test]
    fn test_bf16_size_bytes() {
        assert_eq!(DType::BF16.size_bytes(), 2);
    }

    #[test]
    fn test_f32_size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
    }

    #[test]
    fn test_f64_size_bytes() {
        assert_eq!(DType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_i8_size_bytes() {
        assert_eq!(DType::I8.size_bytes(), 1);
    }

    #[test]
    fn test_i16_size_bytes() {
        assert_eq!(DType::I16.size_bytes(), 2);
    }

    #[test]
    fn test_i32_size_bytes() {
        assert_eq!(DType::I32.size_bytes(), 4);
    }

    #[test]
    fn test_i64_size_bytes() {
        assert_eq!(DType::I64.size_bytes(), 8);
    }

    #[test]
    fn test_u8_size_bytes() {
        assert_eq!(DType::U8.size_bytes(), 1);
    }

    #[test]
    fn test_u16_size_bytes() {
        assert_eq!(DType::U16.size_bytes(), 2);
    }

    #[test]
    fn test_u32_size_bytes() {
        assert_eq!(DType::U32.size_bytes(), 4);
    }

    #[test]
    fn test_u64_size_bytes() {
        assert_eq!(DType::U64.size_bytes(), 8);
    }

    #[test]
    fn test_bool_size_bytes() {
        assert_eq!(DType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_complex64_size_bytes() {
        assert_eq!(DType::Complex64.size_bytes(), 8);
    }

    #[test]
    fn test_complex128_size_bytes() {
        assert_eq!(DType::Complex128.size_bytes(), 16);
    }

    // =========================================================================
    // Bit Width Tests
    // =========================================================================

    #[test]
    fn test_bit_width_all() {
        assert_eq!(DType::F16.bit_width(), 16);
        assert_eq!(DType::BF16.bit_width(), 16);
        assert_eq!(DType::F32.bit_width(), 32);
        assert_eq!(DType::F64.bit_width(), 64);
        assert_eq!(DType::I8.bit_width(), 8);
        assert_eq!(DType::I16.bit_width(), 16);
        assert_eq!(DType::I32.bit_width(), 32);
        assert_eq!(DType::I64.bit_width(), 64);
        assert_eq!(DType::U8.bit_width(), 8);
        assert_eq!(DType::U16.bit_width(), 16);
        assert_eq!(DType::U32.bit_width(), 32);
        assert_eq!(DType::U64.bit_width(), 64);
        assert_eq!(DType::Bool.bit_width(), 8);
        assert_eq!(DType::Complex64.bit_width(), 64);
        assert_eq!(DType::Complex128.bit_width(), 128);
    }

    #[test]
    fn test_real_bit_width_complex() {
        assert_eq!(DType::Complex64.real_bit_width(), 32);
        assert_eq!(DType::Complex128.real_bit_width(), 64);
    }

    #[test]
    fn test_real_bit_width_non_complex() {
        assert_eq!(DType::F32.real_bit_width(), 32);
        assert_eq!(DType::I64.real_bit_width(), 64);
    }

    // =========================================================================
    // Type Classification Tests
    // =========================================================================

    #[test]
    fn test_is_float() {
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::U32.is_float());
        assert!(!DType::Bool.is_float());
        assert!(!DType::Complex64.is_float());
    }

    #[test]
    fn test_is_int() {
        assert!(DType::I8.is_int());
        assert!(DType::I16.is_int());
        assert!(DType::I32.is_int());
        assert!(DType::I64.is_int());
        assert!(DType::U8.is_int());
        assert!(DType::U16.is_int());
        assert!(DType::U32.is_int());
        assert!(DType::U64.is_int());
        assert!(!DType::F32.is_int());
        assert!(!DType::Bool.is_int());
        assert!(!DType::Complex64.is_int());
    }

    #[test]
    fn test_is_signed_int() {
        assert!(DType::I8.is_signed_int());
        assert!(DType::I16.is_signed_int());
        assert!(DType::I32.is_signed_int());
        assert!(DType::I64.is_signed_int());
        assert!(!DType::U8.is_signed_int());
        assert!(!DType::I32.is_unsigned());
        assert!(!DType::F32.is_signed_int());
    }

    #[test]
    fn test_is_unsigned() {
        assert!(DType::U8.is_unsigned());
        assert!(DType::U16.is_unsigned());
        assert!(DType::U32.is_unsigned());
        assert!(DType::U64.is_unsigned());
        assert!(!DType::I8.is_unsigned());
        assert!(!DType::F32.is_unsigned());
    }

    #[test]
    fn test_is_complex() {
        assert!(DType::Complex64.is_complex());
        assert!(DType::Complex128.is_complex());
        assert!(!DType::F32.is_complex());
        assert!(!DType::I32.is_complex());
    }

    #[test]
    fn test_is_signed() {
        assert!(DType::F32.is_signed());
        assert!(DType::I32.is_signed());
        assert!(DType::Complex64.is_signed());
        assert!(!DType::U32.is_signed());
        assert!(!DType::Bool.is_signed());
    }

    #[test]
    fn test_is_numeric() {
        assert!(DType::F32.is_numeric());
        assert!(DType::I32.is_numeric());
        assert!(DType::Complex64.is_numeric());
        assert!(!DType::Bool.is_numeric());
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::F16), "F16");
        assert_eq!(format!("{}", DType::BF16), "BF16");
        assert_eq!(format!("{}", DType::F32), "F32");
        assert_eq!(format!("{}", DType::F64), "F64");
        assert_eq!(format!("{}", DType::I8), "I8");
        assert_eq!(format!("{}", DType::I16), "I16");
        assert_eq!(format!("{}", DType::I32), "I32");
        assert_eq!(format!("{}", DType::I64), "I64");
        assert_eq!(format!("{}", DType::U8), "U8");
        assert_eq!(format!("{}", DType::U16), "U16");
        assert_eq!(format!("{}", DType::U32), "U32");
        assert_eq!(format!("{}", DType::U64), "U64");
        assert_eq!(format!("{}", DType::Bool), "Bool");
        assert_eq!(format!("{}", DType::Complex64), "Complex64");
        assert_eq!(format!("{}", DType::Complex128), "Complex128");
    }

    // =========================================================================
    // Default Tests
    // =========================================================================

    #[test]
    fn test_default_is_f32() {
        assert_eq!(DType::default(), DType::F32);
    }

    // =========================================================================
    // Hash Tests
    // =========================================================================

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for dt in DType::ALL.iter() {
            set.insert(*dt);
        }
        assert_eq!(set.len(), 16);
    }

    // =========================================================================
    // FromStr Tests
    // =========================================================================

    #[test]
    fn test_from_str_short_names() {
        assert_eq!(DType::from_str("F16"), Ok(DType::F16));
        assert_eq!(DType::from_str("BF16"), Ok(DType::BF16));
        assert_eq!(DType::from_str("F32"), Ok(DType::F32));
        assert_eq!(DType::from_str("F64"), Ok(DType::F64));
        assert_eq!(DType::from_str("I8"), Ok(DType::I8));
        assert_eq!(DType::from_str("I16"), Ok(DType::I16));
        assert_eq!(DType::from_str("I32"), Ok(DType::I32));
        assert_eq!(DType::from_str("I64"), Ok(DType::I64));
        assert_eq!(DType::from_str("U8"), Ok(DType::U8));
        assert_eq!(DType::from_str("U16"), Ok(DType::U16));
        assert_eq!(DType::from_str("U32"), Ok(DType::U32));
        assert_eq!(DType::from_str("U64"), Ok(DType::U64));
        assert_eq!(DType::from_str("Bool"), Ok(DType::Bool));
        assert_eq!(DType::from_str("Complex64"), Ok(DType::Complex64));
        assert_eq!(DType::from_str("Complex128"), Ok(DType::Complex128));
    }

    #[test]
    fn test_from_str_lowercase() {
        assert_eq!(DType::from_str("f32"), Ok(DType::F32));
        assert_eq!(DType::from_str("f64"), Ok(DType::F64));
        assert_eq!(DType::from_str("i32"), Ok(DType::I32));
        assert_eq!(DType::from_str("u8"), Ok(DType::U8));
    }

    #[test]
    fn test_from_str_numpy_names() {
        assert_eq!(DType::from_str("float32"), Ok(DType::F32));
        assert_eq!(DType::from_str("float64"), Ok(DType::F64));
        assert_eq!(DType::from_str("int32"), Ok(DType::I32));
        assert_eq!(DType::from_str("int64"), Ok(DType::I64));
        assert_eq!(DType::from_str("uint8"), Ok(DType::U8));
        assert_eq!(DType::from_str("float16"), Ok(DType::F16));
        assert_eq!(DType::from_str("complex64"), Ok(DType::Complex64));
    }

    #[test]
    fn test_from_str_aliases() {
        assert_eq!(DType::from_str("half"), Ok(DType::F16));
        assert_eq!(DType::from_str("bfloat16"), Ok(DType::BF16));
        assert_eq!(DType::from_str("single"), Ok(DType::F32));
        assert_eq!(DType::from_str("float"), Ok(DType::F32));
        assert_eq!(DType::from_str("double"), Ok(DType::F64));
        assert_eq!(DType::from_str("int"), Ok(DType::I32));
        assert_eq!(DType::from_str("long"), Ok(DType::I64));
        assert_eq!(DType::from_str("short"), Ok(DType::I16));
        assert_eq!(DType::from_str("boolean"), Ok(DType::Bool));
    }

    #[test]
    fn test_from_str_invalid() {
        assert!(DType::from_str("invalid").is_err());
        assert!(DType::from_str("").is_err());
        assert!(DType::from_str("f128").is_err());
        assert!(DType::from_str("float128").is_err());
    }

    #[test]
    fn test_from_str_error_message() {
        let err = DType::from_str("xyz").unwrap_err();
        assert!(err.contains("Unknown dtype"));
        assert!(err.contains("xyz"));
    }

    #[test]
    fn test_std_from_str_trait() {
        use std::str::FromStr;
        let dt: DType = "f32".parse().unwrap();
        assert_eq!(dt, DType::F32);
    }

    // =========================================================================
    // Can Cast Tests
    // =========================================================================

    #[test]
    fn test_can_cast_same_type() {
        for dt in DType::ALL.iter() {
            assert!(dt.can_cast_from(*dt));
        }
    }

    #[test]
    fn test_can_cast_numeric_to_numeric() {
        let numerics = DType::ALL_NUMERIC;
        for &a in numerics.iter() {
            for &b in numerics.iter() {
                assert!(a.can_cast_from(b), "{} -> {}", b, a);
            }
        }
    }

    #[test]
    fn test_can_cast_bool_to_numeric() {
        for dt in DType::ALL_NUMERIC.iter() {
            assert!(dt.can_cast_from(DType::Bool));
        }
    }

    #[test]
    fn test_can_cast_numeric_to_bool() {
        for dt in DType::ALL_NUMERIC.iter() {
            assert!(DType::Bool.can_cast_from(*dt));
        }
    }

    // =========================================================================
    // Lossless Cast Tests
    // =========================================================================

    #[test]
    fn test_lossless_same_type() {
        for dt in DType::ALL.iter() {
            assert!(dt.is_lossless_cast(*dt));
        }
    }

    #[test]
    fn test_lossless_int_widening() {
        assert!(DType::I16.is_lossless_cast(DType::I8));
        assert!(DType::I32.is_lossless_cast(DType::I8));
        assert!(DType::I64.is_lossless_cast(DType::I8));
        assert!(DType::I32.is_lossless_cast(DType::I16));
        assert!(DType::I64.is_lossless_cast(DType::I16));
        assert!(DType::I64.is_lossless_cast(DType::I32));
    }

    #[test]
    fn test_lossless_uint_widening() {
        assert!(DType::U16.is_lossless_cast(DType::U8));
        assert!(DType::U32.is_lossless_cast(DType::U8));
        assert!(DType::U64.is_lossless_cast(DType::U8));
        assert!(DType::U32.is_lossless_cast(DType::U16));
        assert!(DType::U64.is_lossless_cast(DType::U16));
        assert!(DType::U64.is_lossless_cast(DType::U32));
    }

    #[test]
    fn test_lossless_float_widening() {
        assert!(DType::F32.is_lossless_cast(DType::F16));
        assert!(DType::F64.is_lossless_cast(DType::F16));
        assert!(DType::F64.is_lossless_cast(DType::F32));
    }

    #[test]
    fn test_lossless_int_narrowing_is_not() {
        assert!(!DType::I8.is_lossless_cast(DType::I16));
        assert!(!DType::I8.is_lossless_cast(DType::I32));
        assert!(!DType::I32.is_lossless_cast(DType::I64));
    }

    #[test]
    fn test_lossless_float_narrowing_is_not() {
        assert!(!DType::F32.is_lossless_cast(DType::F64));
        assert!(!DType::F16.is_lossless_cast(DType::F32));
    }

    #[test]
    fn test_lossless_unsigned_to_wider_signed() {
        assert!(DType::I16.is_lossless_cast(DType::U8));
        assert!(DType::I32.is_lossless_cast(DType::U8));
        assert!(DType::I64.is_lossless_cast(DType::U16));
    }

    #[test]
    fn test_lossless_bool_to_numeric() {
        assert!(DType::I8.is_lossless_cast(DType::Bool));
        assert!(DType::F32.is_lossless_cast(DType::Bool));
        assert!(DType::U64.is_lossless_cast(DType::Bool));
    }

    #[test]
    fn test_lossless_int_to_f64() {
        assert!(DType::F64.is_lossless_cast(DType::I32));
        assert!(DType::F64.is_lossless_cast(DType::U32));
    }

    // =========================================================================
    // Promote Tests
    // =========================================================================

    #[test]
    fn test_promote_same_type() {
        for dt in DType::ALL.iter() {
            assert_eq!(DType::promote(*dt, *dt), *dt);
        }
    }

    #[test]
    fn test_promote_floats() {
        assert_eq!(DType::promote(DType::F16, DType::F32), DType::F32);
        assert_eq!(DType::promote(DType::F32, DType::F64), DType::F64);
        assert_eq!(DType::promote(DType::F16, DType::F64), DType::F64);
    }

    #[test]
    fn test_promote_int_and_float() {
        assert_eq!(DType::promote(DType::I32, DType::F32), DType::F32);
        assert_eq!(DType::promote(DType::U8, DType::F64), DType::F64);
    }

    #[test]
    fn test_promote_ints() {
        assert_eq!(DType::promote(DType::I8, DType::I32), DType::I32);
        assert_eq!(DType::promote(DType::U16, DType::U32), DType::U32);
    }

    #[test]
    fn test_promote_complex() {
        assert_eq!(DType::promote(DType::Complex64, DType::F32), DType::Complex64);
        assert_eq!(DType::promote(DType::Complex128, DType::Complex64), DType::Complex128);
        assert_eq!(DType::promote(DType::Complex64, DType::Complex128), DType::Complex128);
    }

    #[test]
    fn test_promote_bool() {
        assert_eq!(DType::promote(DType::Bool, DType::F32), DType::F32);
        assert_eq!(DType::promote(DType::Bool, DType::I32), DType::I8);
        assert_eq!(DType::promote(DType::Bool, DType::Bool), DType::Bool);
    }

    #[test]
    fn test_promote_symmetric() {
        // promote should be symmetric for most types
        assert_eq!(DType::promote(DType::I8, DType::F32), DType::promote(DType::F32, DType::I8));
    }

    // =========================================================================
    // Common Dtype Tests
    // =========================================================================

    #[test]
    fn test_common_dtype_single() {
        assert_eq!(DType::common_dtype(&[DType::F32]).unwrap(), DType::F32);
    }

    #[test]
    fn test_common_dtype_multiple() {
        assert_eq!(
            DType::common_dtype(&[DType::I8, DType::I16, DType::F32]).unwrap(),
            DType::F32
        );
    }

    #[test]
    fn test_common_dtype_empty() {
        assert!(DType::common_dtype(&[]).is_err());
    }

    #[test]
    fn test_common_dtype_all_same() {
        let types = vec![DType::F32; 10];
        assert_eq!(DType::common_dtype(&types).unwrap(), DType::F32);
    }

    // =========================================================================
    // Cast Function Tests
    // =========================================================================

    #[test]
    fn test_cast_f64_to_f32() {
        let input = vec![1.0_f64, 2.5, -3.7, 0.0, 1e30];
        let output = cast_slice_f64_to_f32(&input);
        assert_eq!(output.len(), 5);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_cast_f64_to_f32_preserves_special() {
        let input = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MIN, f64::MAX];
        let output = cast_slice_f64_to_f32(&input);
        assert!(output[0].is_nan());
        assert!(output[1].is_infinite() && output[1] > 0.0);
        assert!(output[2].is_infinite() && output[2] < 0.0);
    }

    #[test]
    fn test_cast_f32_to_f64() {
        let input = vec![1.0_f32, 2.5, -3.7];
        let output = cast_slice_f32_to_f64(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_cast_i32_to_f32() {
        let input = vec![0_i32, 1, -1, 100, -100];
        let output = cast_slice_i32_to_f32(&input);
        assert_eq!(output, vec![0.0, 1.0, -1.0, 100.0, -100.0]);
    }

    #[test]
    fn test_cast_i64_to_f64() {
        let input = vec![0_i64, 1, -1, 1000000];
        let output = cast_slice_i64_to_f64(&input);
        assert_eq!(output, vec![0.0, 1.0, -1.0, 1000000.0]);
    }

    #[test]
    fn test_cast_f32_to_i32() {
        let input = vec![0.0_f32, 1.0, 1.9, -1.0, -1.9];
        let output = cast_slice_f32_to_i32(&input);
        assert_eq!(output, vec![0, 1, 1, -1, -1]);
    }

    #[test]
    fn test_cast_f32_to_i32_clamps() {
        let input = vec![f32::MAX, f32::MIN];
        let output = cast_slice_f32_to_i32(&input);
        assert_eq!(output[0], i32::MAX);
        assert_eq!(output[1], i32::MIN);
    }

    #[test]
    fn test_cast_f32_to_i32_nan() {
        let input = vec![f32::NAN];
        let output = cast_slice_f32_to_i32(&input);
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_cast_bool_to_f32() {
        let input = vec![true, false, true, false];
        let output = cast_slice_bool_to_f32(&input);
        assert_eq!(output, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cast_f32_to_bool() {
        let input = vec![0.0_f32, 1.0, -1.0, 0.5, 0.0];
        let output = cast_slice_f32_to_bool(&input);
        assert_eq!(output, vec![false, true, true, true, false]);
    }

    #[test]
    fn test_cast_u8_to_f32() {
        let input = vec![0_u8, 127, 255];
        let output = cast_slice_u8_to_f32(&input);
        assert_eq!(output, vec![0.0, 127.0, 255.0]);
    }

    #[test]
    fn test_cast_f32_to_u8() {
        let input = vec![0.0_f32, 127.5, 255.0, 256.0, -1.0];
        let output = cast_slice_f32_to_u8(&input);
        assert_eq!(output, vec![0, 127, 255, 255, 0]);
    }

    #[test]
    fn test_cast_i64_to_i32() {
        let input = vec![0_i64, 100, -100, i32::MAX as i64, i32::MIN as i64];
        let output = cast_slice_i64_to_i32(&input);
        assert_eq!(output, vec![0, 100, -100, i32::MAX, i32::MIN]);
    }

    #[test]
    fn test_cast_i64_to_i32_saturates() {
        let input = vec![i64::MAX, i64::MIN];
        let output = cast_slice_i64_to_i32(&input);
        assert_eq!(output[0], i32::MAX);
        assert_eq!(output[1], i32::MIN);
    }

    #[test]
    fn test_cast_empty_slice() {
        let output = cast_slice_f64_to_f32(&[]);
        assert!(output.is_empty());
    }

    // =========================================================================
    // DTypeInfo Tests
    // =========================================================================

    #[test]
    fn test_dtype_info_f32() {
        let info = DTypeInfo::for_dtype(DType::F32);
        assert_eq!(info.name, "F32");
        assert_eq!(info.size_bytes, 4);
        assert_eq!(info.bit_width, 32);
        assert!(info.is_float);
        assert!(!info.is_signed_int);
        assert!(!info.is_unsigned_int);
        assert!(!info.is_bool);
        assert!(!info.is_complex);
        assert!(info.is_numeric);
        assert!(info.is_signed);
        assert_eq!(info.category, "float");
        assert_eq!(info.numpy_name, "float32");
        assert_eq!(info.c_type_name, "float");
        assert_eq!(info.rust_type_name, "f32");
        assert!(info.has_infinity);
        assert!(info.has_subnormal);
        assert!(info.epsilon.is_some());
        assert!(info.mantissa_digits.is_some());
        assert_eq!(info.alignment, 4);
    }

    #[test]
    fn test_dtype_info_i32() {
        let info = DTypeInfo::for_dtype(DType::I32);
        assert_eq!(info.name, "I32");
        assert!(info.is_signed_int);
        assert!(!info.is_float);
        assert!(!info.has_infinity);
        assert!(info.epsilon.is_none());
    }

    #[test]
    fn test_dtype_info_bool() {
        let info = DTypeInfo::for_dtype(DType::Bool);
        assert_eq!(info.name, "Bool");
        assert!(info.is_bool);
        assert!(!info.is_numeric);
    }

    #[test]
    fn test_dtype_info_complex64() {
        let info = DTypeInfo::for_dtype(DType::Complex64);
        assert!(info.is_complex);
        assert!(info.is_numeric);
        assert_eq!(info.size_bytes, 8);
        assert!(info.has_infinity);
    }

    #[test]
    fn test_dtype_info_as_dtype_roundtrip() {
        for dt in DType::ALL.iter() {
            let info = DTypeInfo::for_dtype(*dt);
            assert_eq!(info.as_dtype(), *dt, "roundtrip failed for {:?}", dt);
        }
    }

    #[test]
    fn test_dtype_info_summary() {
        let info = DTypeInfo::for_dtype(DType::F32);
        let summary = info.summary();
        assert!(summary.contains("F32"));
        assert!(summary.contains("4 bytes"));
        assert!(summary.contains("float"));
    }

    #[test]
    fn test_dtype_info_all() {
        let all = DTypeInfo::all();
        assert_eq!(all.len(), 16);
    }

    #[test]
    fn test_dtype_info_display() {
        let info = DTypeInfo::for_dtype(DType::F32);
        let display = format!("{}", info);
        assert!(display.contains("F32"));
    }

    #[test]
    fn test_dtype_info_equality() {
        let a = DTypeInfo::for_dtype(DType::F32);
        let b = DTypeInfo::for_dtype(DType::F32);
        assert_eq!(a, b);
    }

    // =========================================================================
    // Category Tests
    // =========================================================================

    #[test]
    fn test_category() {
        assert_eq!(DType::F32.category(), "float");
        assert_eq!(DType::I32.category(), "signed_int");
        assert_eq!(DType::U8.category(), "unsigned_int");
        assert_eq!(DType::Bool.category(), "bool");
        assert_eq!(DType::Complex64.category(), "complex");
    }

    // =========================================================================
    // Name Tests
    // =========================================================================

    #[test]
    fn test_short_name() {
        for dt in DType::ALL.iter() {
            assert_eq!(dt.short_name(), format!("{}", dt));
        }
    }

    #[test]
    fn test_rust_type_name() {
        assert_eq!(DType::F32.rust_type_name(), "f32");
        assert_eq!(DType::I64.rust_type_name(), "i64");
        assert_eq!(DType::Bool.rust_type_name(), "bool");
    }

    #[test]
    fn test_c_type_name() {
        assert_eq!(DType::F32.c_type_name(), "float");
        assert_eq!(DType::F64.c_type_name(), "double");
        assert_eq!(DType::I32.c_type_name(), "int32_t");
        assert_eq!(DType::Bool.c_type_name(), "bool");
    }

    #[test]
    fn test_numpy_name() {
        assert_eq!(DType::F32.numpy_name(), "float32");
        assert_eq!(DType::F64.numpy_name(), "float64");
        assert_eq!(DType::I32.numpy_name(), "int32");
        assert_eq!(DType::U8.numpy_name(), "uint8");
    }

    // =========================================================================
    // Min/Max Value Tests
    // =========================================================================

    #[test]
    fn test_min_max_values() {
        assert_eq!(DType::I8.min_value(), "-128");
        assert_eq!(DType::I8.max_value(), "127");
        assert_eq!(DType::U8.min_value(), "0");
        assert_eq!(DType::U8.max_value(), "255");
        assert_eq!(DType::U64.max_value(), "18446744073709551615");
        assert_eq!(DType::Bool.min_value(), "false");
        assert_eq!(DType::Bool.max_value(), "true");
    }

    // =========================================================================
    // Float Property Tests
    // =========================================================================

    #[test]
    fn test_epsilon() {
        assert!(DType::F32.epsilon().is_some());
        assert!(DType::F64.epsilon().is_some());
        assert!(DType::F16.epsilon().is_some());
        assert!(DType::I32.epsilon().is_none());
        assert!(DType::Bool.epsilon().is_none());
    }

    #[test]
    fn test_radix() {
        assert_eq!(DType::F32.radix(), Some(2));
        assert_eq!(DType::F64.radix(), Some(2));
        assert_eq!(DType::I32.radix(), None);
    }

    #[test]
    fn test_mantissa_digits() {
        assert_eq!(DType::F16.mantissa_digits(), Some(11));
        assert_eq!(DType::BF16.mantissa_digits(), Some(8));
        assert_eq!(DType::F32.mantissa_digits(), Some(24));
        assert_eq!(DType::F64.mantissa_digits(), Some(53));
        assert!(DType::I32.mantissa_digits().is_none());
    }

    #[test]
    fn test_max_exponent() {
        assert_eq!(DType::F16.max_exponent(), Some(16));
        assert_eq!(DType::BF16.max_exponent(), Some(128));
        assert_eq!(DType::F32.max_exponent(), Some(128));
        assert_eq!(DType::F64.max_exponent(), Some(1024));
    }

    #[test]
    fn test_min_exponent() {
        assert_eq!(DType::F16.min_exponent(), Some(-14));
        assert_eq!(DType::BF16.min_exponent(), Some(-126));
        assert_eq!(DType::F32.min_exponent(), Some(-125));
        assert_eq!(DType::F64.min_exponent(), Some(-1021));
    }

    #[test]
    fn test_decimal_digits() {
        assert_eq!(DType::F16.decimal_digits(), Some(3));
        assert_eq!(DType::BF16.decimal_digits(), Some(2));
        assert_eq!(DType::F32.decimal_digits(), Some(7));
        assert_eq!(DType::F64.decimal_digits(), Some(16));
        assert!(DType::I32.decimal_digits().is_none());
    }

    #[test]
    fn test_has_infinity() {
        assert!(DType::F32.has_infinity());
        assert!(DType::F64.has_infinity());
        assert!(DType::Complex64.has_infinity());
        assert!(!DType::I32.has_infinity());
        assert!(!DType::Bool.has_infinity());
    }

    #[test]
    fn test_has_subnormal() {
        assert!(DType::F32.has_subnormal());
        assert!(!DType::I32.has_subnormal());
    }

    #[test]
    fn test_supports_arithmetic() {
        assert!(DType::F32.supports_arithmetic());
        assert!(!DType::Bool.supports_arithmetic());
    }

    #[test]
    fn test_supports_comparison() {
        assert!(DType::F32.supports_comparison());
        assert!(DType::Bool.supports_comparison());
        assert!(DType::I32.supports_comparison());
    }

    // =========================================================================
    // Quantized Type Tests
    // =========================================================================

    #[test]
    fn test_is_quantized_type() {
        assert!(DType::I8.is_quantized_type());
        assert!(DType::U8.is_quantized_type());
        assert!(DType::I16.is_quantized_type());
        assert!(DType::U16.is_quantized_type());
        assert!(DType::BF16.is_quantized_type());
        assert!(!DType::F32.is_quantized_type());
        assert!(!DType::F64.is_quantized_type());
    }

    #[test]
    fn test_is_gradient_type() {
        assert!(DType::F32.is_gradient_type());
        assert!(DType::F64.is_gradient_type());
        assert!(DType::BF16.is_gradient_type());
        assert!(!DType::I32.is_gradient_type());
        assert!(!DType::Bool.is_gradient_type());
        assert!(!DType::F16.is_gradient_type());
    }

    // =========================================================================
    // Alignment Tests
    // =========================================================================

    #[test]
    fn test_alignment() {
        assert_eq!(DType::F32.alignment(), 4);
        assert_eq!(DType::F64.alignment(), 8);
        assert_eq!(DType::I8.alignment(), 1);
        assert_eq!(DType::I64.alignment(), 8);
    }

    // =========================================================================
    // All Types Iterator Tests
    // =========================================================================

    #[test]
    fn test_all_types_count() {
        assert_eq!(DType::all_types().count(), 16);
    }

    #[test]
    fn test_all_types_contains_all() {
        let all: Vec<DType> = DType::all_types().copied().collect();
        for dt in DType::ALL.iter() {
            assert!(all.contains(dt), "missing {:?}", dt);
        }
    }

    // =========================================================================
    // Constant Array Tests
    // =========================================================================

    #[test]
    fn test_float_types() {
        assert_eq!(DType::FLOAT_TYPES.len(), 4);
        assert_eq!(DType::FLOAT_TYPES[0], DType::F16);
        assert_eq!(DType::FLOAT_TYPES[3], DType::F64);
    }

    #[test]
    fn test_signed_int_types() {
        assert_eq!(DType::SIGNED_INT_TYPES.len(), 4);
        assert_eq!(DType::SIGNED_INT_TYPES[0], DType::I8);
    }

    #[test]
    fn test_unsigned_int_types() {
        assert_eq!(DType::UNSIGNED_INT_TYPES.len(), 4);
        assert_eq!(DType::UNSIGNED_INT_TYPES[0], DType::U8);
    }

    #[test]
    fn test_int_types() {
        assert_eq!(DType::INT_TYPES.len(), 8);
    }

    #[test]
    fn test_complex_types() {
        assert_eq!(DType::COMPLEX_TYPES.len(), 2);
    }

    #[test]
    fn test_all_numeric() {
        assert_eq!(DType::ALL_NUMERIC.len(), 14);
    }

    #[test]
    fn test_all() {
        assert_eq!(DType::ALL.len(), 16);
    }

    #[test]
    fn test_variant_count() {
        assert_eq!(DType::VARIANT_COUNT, 16);
    }

    // =========================================================================
    // DTypeMap Tests
    // =========================================================================

    #[test]
    fn test_dtype_map_from_fn() {
        let map = DTypeMap::from_fn(|dt| dt.size_bytes());
        assert_eq!(map.get(DType::F32), &4);
        assert_eq!(map.get(DType::I8), &1);
        assert_eq!(map.get(DType::Complex128), &16);
    }

    #[test]
    fn test_dtype_map_fill() {
        let map = DTypeMap::fill(42usize);
        assert_eq!(map.get(DType::F32), &42);
        assert_eq!(map.get(DType::I64), &42);
    }

    #[test]
    fn test_dtype_map_get_mut() {
        let mut map = DTypeMap::fill(0usize);
        *map.get_mut(DType::F32) = 100;
        assert_eq!(map.get(DType::F32), &100);
        assert_eq!(map.get(DType::F64), &0);
    }

    // =========================================================================
    // PartialOrd/Ord Tests
    // =========================================================================

    #[test]
    fn test_ordering() {
        assert!(DType::I8 < DType::I16);
        assert!(DType::I16 < DType::I32);
        assert!(DType::I32 < DType::I64);
        assert!(DType::U8 < DType::U16);
        assert!(DType::F32 < DType::F64);
        assert!(DType::Complex64 < DType::Complex128);
    }

    #[test]
    fn test_equality() {
        assert_eq!(DType::F32, DType::F32);
        assert_ne!(DType::F32, DType::F64);
        assert_ne!(DType::I32, DType::U32);
    }

    // =========================================================================
    // Clone and Copy Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let dt = DType::F32;
        let dt2 = dt.clone();
        assert_eq!(dt, dt2);
    }

    #[test]
    fn test_copy() {
        let dt = DType::I64;
        let dt2 = dt;
        assert_eq!(dt, dt2);
    }

    // =========================================================================
    // Debug Tests
    // =========================================================================

    #[test]
    fn test_debug() {
        let dt = DType::F32;
        let debug = format!("{:?}", dt);
        assert!(debug.contains("F32"));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_cast_inf_f64_to_f32() {
        let input = vec![f64::INFINITY, f64::NEG_INFINITY];
        let output = cast_slice_f64_to_f32(&input);
        assert!(output[0].is_infinite());
        assert!(output[1].is_infinite());
    }

    #[test]
    fn test_cast_zero_values() {
        let zeros = vec![0.0_f64; 1000];
        let output = cast_slice_f64_to_f32(&zeros);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_promote_chain() {
        // Test promoting through a chain of types
        let result = DType::promote(
            DType::promote(DType::I8, DType::I16),
            DType::promote(DType::U8, DType::F32),
        );
        assert_eq!(result, DType::F32);
    }

    #[test]
    fn test_dtype_map_all_types_covered() {
        let map = DTypeMap::from_fn(|dt| dt.short_name().to_string());
        for dt in DType::ALL.iter() {
            assert_eq!(map.get(*dt), dt.short_name());
        }
    }

    #[test]
    fn test_common_dtype_chaining() {
        let types = vec![DType::U8, DType::I16, DType::F16, DType::F32, DType::F64];
        assert_eq!(DType::common_dtype(&types).unwrap(), DType::F64);
    }

    // =========================================================================
    // Additional Cast Edge Case Tests
    // =========================================================================

    #[test]
    fn test_cast_f64_to_f32_infinity() {
        let input = vec![f64::INFINITY, f64::NEG_INFINITY];
        let output = cast_slice_f64_to_f32(&input);
        assert!(output[0].is_infinite() && output[0].is_sign_positive());
        assert!(output[1].is_infinite() && output[1].is_sign_negative());
    }

    #[test]
    fn test_cast_f64_to_f32_subnormal() {
        let input = vec![1.0e-40_f64];
        let output = cast_slice_f64_to_f32(&input);
        assert!(output[0].is_subnormal() || output[0] == 0.0);
    }

    #[test]
    fn test_cast_f32_to_u8_clamp_negative() {
        let input = vec![-1.0_f32, -100.0, 0.0, 255.0, 256.0, f32::NAN];
        let output = cast_slice_f32_to_u8(&input);
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 255);
        assert_eq!(output[4], 255);
        assert_eq!(output[5], 0); // NaN -> 0
    }

    #[test]
    fn test_cast_bool_to_f32_large() {
        let input = vec![true; 1000];
        let output = cast_slice_bool_to_f32(&input);
        assert!(output.iter().all(|&x| x == 1.0));
        assert_eq!(output.len(), 1000);
    }

    #[test]
    fn test_cast_empty_all_types() {
        assert!(cast_slice_f64_to_f32(&[]).is_empty());
        assert!(cast_slice_f32_to_f64(&[]).is_empty());
        assert!(cast_slice_i32_to_f32(&[]).is_empty());
        assert!(cast_slice_i64_to_f64(&[]).is_empty());
        assert!(cast_slice_f32_to_i32(&[]).is_empty());
        assert!(cast_slice_bool_to_f32(&[]).is_empty());
        assert!(cast_slice_f32_to_bool(&[]).is_empty());
        assert!(cast_slice_u8_to_f32(&[]).is_empty());
        assert!(cast_slice_f32_to_u8(&[]).is_empty());
        assert!(cast_slice_i64_to_i32(&[]).is_empty());
    }

    #[test]
    fn test_cast_roundtrip_f32_f64() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let f64_vals = cast_slice_f32_to_f64(&input);
        let back = f64_vals.iter().map(|&v| v as f32).collect::<Vec<f32>>();
        for (a, b) in input.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_lossless_symmetry() {
        // If A -> B is lossless, B -> A is not necessarily lossless
        // But if A -> B and B -> C are both lossless, A -> C should be lossless
        assert!(DType::I8.is_lossless_cast(DType::Bool));
        assert!(DType::I32.is_lossless_cast(DType::I8));
        assert!(DType::I32.is_lossless_cast(DType::Bool));
    }

    #[test]
    fn test_lossless_complex() {
        assert!(DType::Complex128.is_lossless_cast(DType::Complex64));
        assert!(!DType::Complex64.is_lossless_cast(DType::Complex128));
    }

    // =========================================================================
    // Additional DTypeInfo Tests
    // =========================================================================

    #[test]
    fn test_dtype_info_for_all_types() {
        for dt in DType::ALL.iter() {
            let info = DTypeInfo::for_dtype(*dt);
            assert!(!info.name.is_empty());
            assert!(info.size_bytes > 0);
            assert!(info.bit_width > 0);
            assert!(!info.category.is_empty());
            assert!(!info.numpy_name.is_empty());
            assert!(!info.c_type_name.is_empty());
            assert!(!info.rust_type_name.is_empty());
            assert!(!info.min_value.is_empty());
            assert!(!info.max_value.is_empty());
            assert!(info.alignment > 0);
        }
    }

    #[test]
    fn test_dtype_info_f16() {
        let info = DTypeInfo::for_dtype(DType::F16);
        assert_eq!(info.name, "F16");
        assert_eq!(info.size_bytes, 2);
        assert!(info.is_float);
        assert!(info.has_infinity);
        assert!(info.has_subnormal);
        assert_eq!(info.decimal_digits, Some(3));
        assert_eq!(info.mantissa_digits, Some(11));
    }

    #[test]
    fn test_dtype_info_bf16() {
        let info = DTypeInfo::for_dtype(DType::BF16);
        assert_eq!(info.name, "BF16");
        assert_eq!(info.size_bytes, 2);
        assert!(info.is_float);
        assert!(info.has_infinity);
        assert_eq!(info.decimal_digits, Some(2));
        assert_eq!(info.mantissa_digits, Some(8));
    }

    #[test]
    fn test_dtype_info_u64() {
        let info = DTypeInfo::for_dtype(DType::U64);
        assert!(info.is_unsigned_int);
        assert!(info.is_numeric);
        assert!(!info.is_signed);
        assert!(!info.has_infinity);
        assert!(info.epsilon.is_none());
    }

    #[test]
    fn test_dtype_info_complex128() {
        let info = DTypeInfo::for_dtype(DType::Complex128);
        assert!(info.is_complex);
        assert_eq!(info.size_bytes, 16);
        assert_eq!(info.bit_width, 128);
        assert!(info.has_infinity);
        assert_eq!(info.decimal_digits, Some(16));
    }

    #[test]
    fn test_dtype_info_categories() {
        let float_info = DTypeInfo::for_dtype(DType::F32);
        let int_info = DTypeInfo::for_dtype(DType::I32);
        let uint_info = DTypeInfo::for_dtype(DType::U8);
        let bool_info = DTypeInfo::for_dtype(DType::Bool);
        let complex_info = DTypeInfo::for_dtype(DType::Complex64);

        assert!(float_info.is_float && !float_info.is_signed_int);
        assert!(int_info.is_signed_int && !int_info.is_unsigned_int);
        assert!(uint_info.is_unsigned_int && !uint_info.is_signed_int);
        assert!(bool_info.is_bool && !bool_info.is_numeric);
        assert!(complex_info.is_complex && !complex_info.is_float);
    }

    // =========================================================================
    // Additional Promotion Tests
    // =========================================================================

    #[test]
    fn test_promote_all_pairs() {
        let types = DType::ALL_NUMERIC;
        for &a in types.iter() {
            for &b in types.iter() {
                let result = DType::promote(a, b);
                assert!(result.can_cast_from(a));
                assert!(result.can_cast_from(b));
            }
        }
    }

    #[test]
    fn test_promote_associative() {
        // promote(a, promote(b, c)) == promote(promote(a, b), c)
        let types = [DType::I8, DType::I32, DType::F32, DType::F64];
        for &a in &types {
            for &b in &types {
                for &c in &types {
                    let r1 = DType::promote(a, DType::promote(b, c));
                    let r2 = DType::promote(DType::promote(a, b), c);
                    assert_eq!(r1, r2, "associativity failed for {:?}, {:?}, {:?}", a, b, c);
                }
            }
        }
    }

    #[test]
    fn test_promote_commutative() {
        let types = DType::ALL_NUMERIC;
        for &a in types.iter() {
            for &b in types.iter() {
                assert_eq!(DType::promote(a, b), DType::promote(b, a));
            }
        }
    }

    #[test]
    fn test_common_dtype_various() {
        assert_eq!(DType::common_dtype(&[DType::U8]).unwrap(), DType::U8);
        assert_eq!(DType::common_dtype(&[DType::F32, DType::F32]).unwrap(), DType::F32);
        assert_eq!(
            DType::common_dtype(&[DType::I8, DType::U8, DType::I16, DType::U16, DType::F16, DType::BF16]).unwrap(),
            DType::F16
        );
    }

    // =========================================================================
    // Additional Feature Tests
    // =========================================================================

    #[test]
    fn test_supports_comparison_all_types() {
        for dt in DType::ALL.iter() {
            assert!(dt.supports_comparison(), "{:?} should support comparison", dt);
        }
    }

    #[test]
    fn test_supports_arithmetic_all() {
        for dt in DType::ALL.iter() {
            if *dt != DType::Bool {
                assert!(dt.supports_arithmetic(), "{:?} should support arithmetic", dt);
            } else {
                assert!(!dt.supports_arithmetic());
            }
        }
    }

    #[test]
    fn test_radix_all_floats() {
        for dt in DType::FLOAT_TYPES.iter() {
            assert_eq!(dt.radix(), Some(2));
        }
    }

    #[test]
    fn test_radix_complex() {
        for dt in DType::COMPLEX_TYPES.iter() {
            assert_eq!(dt.radix(), Some(2));
        }
    }

    #[test]
    fn test_radix_ints() {
        for dt in DType::INT_TYPES.iter() {
            assert_eq!(dt.radix(), None);
        }
    }

    #[test]
    fn test_real_bit_width_all() {
        assert_eq!(DType::F16.real_bit_width(), 16);
        assert_eq!(DType::BF16.real_bit_width(), 16);
        assert_eq!(DType::F32.real_bit_width(), 32);
        assert_eq!(DType::F64.real_bit_width(), 64);
        assert_eq!(DType::I8.real_bit_width(), 8);
        assert_eq!(DType::U64.real_bit_width(), 64);
        assert_eq!(DType::Bool.real_bit_width(), 8);
        assert_eq!(DType::Complex64.real_bit_width(), 32);
        assert_eq!(DType::Complex128.real_bit_width(), 64);
    }

    #[test]
    fn test_size_bytes_consistent_with_bit_width() {
        for dt in DType::ALL.iter() {
            assert_eq!(dt.size_bytes() * 8, dt.bit_width() as usize);
        }
    }

    #[test]
    fn test_all_types_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for dt in DType::ALL.iter() {
            assert!(seen.insert(*dt), "duplicate: {:?}", dt);
        }
    }

    #[test]
    fn test_numpy_names_unique() {
        let mut names = std::collections::HashSet::new();
        for dt in DType::ALL.iter() {
            assert!(names.insert(dt.numpy_name()), "duplicate numpy name: {:?}", dt);
        }
    }

    #[test]
    fn test_c_type_names_unique() {
        let mut names = std::collections::HashSet::new();
        for dt in DType::ALL.iter() {
            assert!(names.insert(dt.c_type_name()), "duplicate C type name: {:?}", dt);
        }
    }

    // =========================================================================
    // DTypeMap Extended Tests
    // =========================================================================

    #[test]
    fn test_dtype_map_debug() {
        let map: DTypeMap<usize> = DTypeMap::fill(42);
        let debug = format!("{:?}", map);
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_dtype_map_clone() {
        let map = DTypeMap::from_fn(|dt| dt.size_bytes());
        let map2 = map.clone();
        assert_eq!(map.get(DType::F32), map2.get(DType::F32));
    }

    #[test]
    fn test_dtype_map_string_values() {
        let map = DTypeMap::from_fn(|dt| dt.numpy_name().to_string());
        assert_eq!(map.get(DType::F32), &"float32");
    }

    #[test]
    fn test_dtype_map_bool_values() {
        let map = DTypeMap::from_fn(|dt| dt.is_float());
        assert_eq!(*map.get(DType::F32), true);
        assert_eq!(*map.get(DType::I32), false);
    }

    #[test]
    fn test_dtype_map_option_values() {
        let map = DTypeMap::from_fn(|dt| dt.epsilon());
        assert!(map.get(DType::F32).is_some());
        assert!(map.get(DType::I32).is_none());
    }

    // =========================================================================
    // Constant Array Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_float_types_correct() {
        assert_eq!(DType::FLOAT_TYPES, [DType::F16, DType::BF16, DType::F32, DType::F64]);
    }

    #[test]
    fn test_signed_int_types_correct() {
        assert_eq!(DType::SIGNED_INT_TYPES, [DType::I8, DType::I16, DType::I32, DType::I64]);
    }

    #[test]
    fn test_unsigned_int_types_correct() {
        assert_eq!(DType::UNSIGNED_INT_TYPES, [DType::U8, DType::U16, DType::U32, DType::U64]);
    }

    #[test]
    fn test_int_types_correct() {
        let expected = [
            DType::I8, DType::I16, DType::I32, DType::I64,
            DType::U8, DType::U16, DType::U32, DType::U64,
        ];
        assert_eq!(DType::INT_TYPES, expected);
    }

    #[test]
    fn test_all_correct() {
        assert_eq!(DType::ALL.len(), 16);
        assert_eq!(DType::ALL[0], DType::F16);
        assert_eq!(DType::ALL[3], DType::F64);
        assert_eq!(DType::ALL[7], DType::I64);
        assert_eq!(DType::ALL[11], DType::U64);
        assert_eq!(DType::ALL[12], DType::Bool);
        assert_eq!(DType::ALL[14], DType::Complex128);
    }

    #[test]
    fn test_all_numeric_correct() {
        assert_eq!(DType::ALL_NUMERIC.len(), 14);
        for dt in DType::ALL_NUMERIC.iter() {
            assert!(dt.is_numeric(), "{:?} should be numeric", dt);
        }
    }

    #[test]
    fn test_no_overlap_between_categories() {
        let float_set: std::collections::HashSet<_> = DType::FLOAT_TYPES.iter().collect();
        let int_set: std::collections::HashSet<_> = DType::INT_TYPES.iter().collect();
        let complex_set: std::collections::HashSet<_> = DType::COMPLEX_TYPES.iter().collect();
        assert!(float_set.is_disjoint(&int_set));
        assert!(float_set.is_disjoint(&complex_set));
        assert!(int_set.is_disjoint(&complex_set));
    }

    // =========================================================================
    // Min/Max Values Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_min_max_values_all_types() {
        assert_eq!(DType::I16.min_value(), "-32768");
        assert_eq!(DType::I16.max_value(), "32767");
        assert_eq!(DType::U16.min_value(), "0");
        assert_eq!(DType::U16.max_value(), "65535");
        assert_eq!(DType::U16.min_value(), "0");
        assert_eq!(DType::U32.max_value(), "4294967295");
        assert_eq!(DType::I32.min_value(), "-2147483648");
        assert_eq!(DType::I32.max_value(), "2147483647");
    }

    // =========================================================================
    // FromStr Alias Tests
    // =========================================================================

    #[test]
    fn test_from_str_bfloat16() {
        assert_eq!(DType::from_str("bfloat16"), Ok(DType::BF16));
    }

    #[test]
    fn test_from_str_boolean_lowercase() {
        assert_eq!(DType::from_str("boolean"), Ok(DType::Bool));
    }

    #[test]
    fn test_from_str_int32() {
        assert_eq!(DType::from_str("int32"), Ok(DType::I32));
    }

    #[test]
    fn test_from_str_uint32() {
        assert_eq!(DType::from_str("uint32"), Ok(DType::U32));
    }

    #[test]
    fn test_from_str_mixed_case() {
        assert_eq!(DType::from_str("Float32"), Ok(DType::F32));
        assert_eq!(DType::from_str("COMPLEX128"), Ok(DType::Complex128));
        assert_eq!(DType::from_str("InT64"), Ok(DType::I64));
    }

    // =========================================================================
    // Large-scale Numerical Property Tests
    // =========================================================================

    #[test]
    fn test_float_types_have_infinity() {
        for dt in DType::FLOAT_TYPES.iter() {
            assert!(dt.has_infinity(), "{:?} should have infinity", dt);
        }
    }

    #[test]
    fn test_int_types_no_infinity() {
        for dt in DType::INT_TYPES.iter() {
            assert!(!dt.has_infinity(), "{:?} should not have infinity", dt);
        }
    }

    #[test]
    fn test_bool_no_infinity() {
        assert!(!DType::Bool.has_infinity());
    }

    #[test]
    fn test_complex_types_have_infinity() {
        for dt in DType::COMPLEX_TYPES.iter() {
            assert!(dt.has_infinity(), "{:?} should have infinity", dt);
        }
    }

    #[test]
    fn test_float_types_have_subnormal() {
        for dt in DType::FLOAT_TYPES.iter() {
            assert!(dt.has_subnormal(), "{:?} should have subnormal", dt);
        }
    }

    #[test]
    fn test_complex_types_have_subnormal() {
        for dt in DType::COMPLEX_TYPES.iter() {
            assert!(dt.has_subnormal(), "{:?} should have subnormal", dt);
        }
    }

    #[test]
    fn test_all_types_numeric_except_bool() {
        for dt in DType::ALL.iter() {
            if *dt == DType::Bool {
                assert!(!dt.is_numeric());
            } else {
                assert!(dt.is_numeric(), "{:?} should be numeric", dt);
            }
        }
    }

    #[test]
    fn test_is_signed_comprehensive() {
        for dt in DType::FLOAT_TYPES.iter() {
            assert!(dt.is_signed(), "{:?} float should be signed", dt);
        }
        for dt in DType::SIGNED_INT_TYPES.iter() {
            assert!(dt.is_signed(), "{:?} signed int should be signed", dt);
        }
        for dt in DType::UNSIGNED_INT_TYPES.iter() {
            assert!(!dt.is_signed(), "{:?} unsigned should not be signed", dt);
        }
        assert!(DType::Complex64.is_signed());
        assert!(!DType::Bool.is_signed());
    }

    // =========================================================================
    // Mantissa/Exponent Detail Tests
    // =========================================================================

    #[test]
    fn test_mantissa_digits_f16() {
        // F16: 1 sign + 5 exp + 10 mantissa = 11 bits total precision
        assert_eq!(DType::F16.mantissa_digits(), Some(11));
    }

    #[test]
    fn test_bf16_fewer_mantissa_bits_than_f16() {
        assert!(DType::BF16.mantissa_digits().unwrap() < DType::F16.mantissa_digits().unwrap());
    }

    #[test]
    fn test_f64_more_precision_than_f32() {
        assert!(DType::F64.mantissa_digits().unwrap() > DType::F32.mantissa_digits().unwrap());
        assert!(DType::F64.decimal_digits().unwrap() > DType::F32.decimal_digits().unwrap());
        assert!(DType::F64.max_exponent().unwrap() > DType::F32.max_exponent().unwrap());
    }

    #[test]
    fn test_exponents_monotonic() {
        assert!(DType::F32.max_exponent().unwrap() > DType::BF16.max_exponent().unwrap());
    }

    // =========================================================================
    // Cast Precision Tests
    // =========================================================================

    #[test]
    fn test_cast_preserves_exact_integers() {
        let input = vec![0_i32, 1, 100, 1000, 10000, 100000];
        let f32_out = cast_slice_i32_to_f32(&input);
        let i32_back = cast_slice_f32_to_i32(&f32_out);
        assert_eq!(input, i32_back);
    }

    #[test]
    fn test_cast_large_i64_to_f64() {
        let input = vec![i64::MAX, i64::MIN, 0, 1, -1];
        let output = cast_slice_i64_to_f64(&input);
        assert_eq!(output.len(), 5);
        assert!((output[0] - i64::MAX as f64).abs() < 1.0);
    }
}
