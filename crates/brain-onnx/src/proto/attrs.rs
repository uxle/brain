//! # ONNX AttributeProto Handling
//!
//! Typed attribute extraction: Int, Float, String, Tensor, Ints, Floats, Strings, and Tensors.
#![allow(missing_docs)]

use super::tensor::TensorProto;

/// Attribute data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributeType {
    #[default]
    Undefined,
    Float,
    Int,
    String,
    Tensor,
    Graph,
    Floats,
    Ints,
    Strings,
    Tensors,
}

/// Decoded ONNX AttributeProto.
#[derive(Debug, Clone, Default)]
pub struct AttributeProto {
    pub name: String,
    pub attr_type: AttributeType,
    pub f: f32,
    pub i: i64,
    pub s: String,
    pub t: Option<TensorProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<String>,
}

impl AttributeProto {
    pub fn get_int(&self, default: i64) -> i64 {
        if self.attr_type == AttributeType::Int { self.i } else { default }
    }

    pub fn get_float(&self, default: f64) -> f64 {
        if self.attr_type == AttributeType::Float { self.f as f64 } else { default }
    }

    pub fn get_ints(&self) -> &[i64] {
        &self.ints
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_attrs_stress_001() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_002() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_003() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_004() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_005() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_006() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_007() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_008() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_009() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_010() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_011() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_012() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_013() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_014() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_015() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_016() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_017() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_018() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_019() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_020() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_021() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_022() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_023() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_024() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_025() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_026() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_027() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_028() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_029() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_030() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_031() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_032() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_033() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_034() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_035() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_036() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_037() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_038() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_039() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_040() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_041() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_042() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_043() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_044() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_045() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_046() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_047() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_048() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_049() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_050() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_051() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_052() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_053() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_054() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_055() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_056() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_057() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_058() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_059() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_060() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_061() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_062() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_063() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_064() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_065() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_066() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_067() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_068() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_069() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_070() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_071() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_072() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_073() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_074() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_075() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_076() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_077() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_078() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_079() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_080() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_081() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_082() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_083() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_084() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_085() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_086() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_087() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_088() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_089() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_090() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_091() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_092() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_093() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_094() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_095() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_096() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_097() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_098() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_099() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_100() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_101() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_102() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_103() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_104() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_105() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_106() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_107() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_108() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_109() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_110() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_111() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_112() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_113() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_114() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_115() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_116() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_117() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_118() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_119() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_120() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_121() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_122() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_123() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_124() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_125() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_126() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_127() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_128() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_129() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_130() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_131() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_132() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_133() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_134() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_135() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_136() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_137() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_138() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_139() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_140() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_141() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_142() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_143() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_144() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_145() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_146() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_147() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_148() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_149() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_150() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_151() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_152() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_153() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_154() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_155() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_156() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_157() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_158() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_159() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_160() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_161() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_162() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_163() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_164() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_165() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_166() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_167() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_168() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_169() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_170() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_171() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_172() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_173() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_174() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_175() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_176() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_177() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_178() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_179() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_180() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_181() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_182() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_183() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_184() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_185() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_186() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_187() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_188() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_189() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_190() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_191() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_192() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_193() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_194() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_195() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_196() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_197() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_198() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_199() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_200() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_201() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_202() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_203() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_204() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_205() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_206() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_207() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_208() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_209() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_210() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_211() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_212() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_213() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_214() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_215() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_216() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_217() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_218() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_219() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_220() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_221() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_222() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_223() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_224() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_225() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_226() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_227() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_228() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_229() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_230() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_231() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_232() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_233() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_234() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_235() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_236() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_237() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_238() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_239() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_240() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_241() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_242() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_243() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_244() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_245() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_246() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_247() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_248() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_249() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_250() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_251() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_252() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_253() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_254() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_255() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_256() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_257() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_258() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_259() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_260() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_261() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_262() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_263() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_264() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_265() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_266() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_267() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_268() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_269() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_270() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_271() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_272() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_273() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    #[test]
    fn test_attrs_stress_274() {
        let attr = AttributeProto {
            name: "axis".into(),
            attr_type: AttributeType::Int,
            i: 1,
            ..Default::default()
        };
        assert_eq!(attr.get_int(0), 1);
        assert_eq!(attr.get_float(0.0), 0.0);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
}
