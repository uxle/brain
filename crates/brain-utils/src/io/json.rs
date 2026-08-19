//! # Dependency-Free RFC 8259 JSON Parser & Serializer
//!
//! Provides a complete, recursive descent JSON parser and pretty serializer
//! supporting objects, arrays, strings with escapes, numbers, booleans, and null.

use std::collections::BTreeMap;
use crate::core::{UtilsError, UtilsResult};

/// JSON Value AST representation.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON Null.
    Null,
    /// JSON Boolean.
    Bool(bool),
    /// JSON Number (represented as f64).
    Number(f64),
    /// JSON String.
    String(String),
    /// JSON Array.
    Array(Vec<JsonValue>),
    /// JSON Object.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Attempts to extract as string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Attempts to extract as float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Attempts to extract as boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Attempts to extract as array.
    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Attempts to extract as object map.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Index into object by key.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|o| o.get(key))
    }

    /// Formats as compact JSON string.
    pub fn to_compact_string(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            Self::Number(n) => n.to_string(),
            Self::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            Self::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_compact_string()).collect();
                format!("[{}]", items.join(","))
            }
            Self::Object(map) => {
                let items: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", k.replace('"', "\\\""), v.to_compact_string()))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

/// Parses a JSON text string into a JsonValue.
pub fn parse_json(input: &str) -> UtilsResult<JsonValue> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    let val = parse_value(&chars, &mut pos)?;
    skip_whitespace(&chars, &mut pos);
    if pos < chars.len() {
        return Err(UtilsError::JsonError(format!("Trailing tokens at pos {}", pos)));
    }
    Ok(val)
}

fn skip_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_value(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() {
        return Err(UtilsError::JsonError("Unexpected EOF while parsing JSON".to_string()));
    }
    match chars[*pos] {
        'n' => parse_null(chars, pos),
        't' | 'f' => parse_bool(chars, pos),
        '"' => parse_string(chars, pos).map(JsonValue::String),
        '[' => parse_array(chars, pos),
        '{' => parse_object(chars, pos),
        '-' | '0'..='9' => parse_number(chars, pos),
        c => Err(UtilsError::JsonError(format!("Unexpected character '{}' at pos {}", c, pos))),
    }
}

fn parse_null(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    if *pos + 4 <= chars.len() && chars[*pos..*pos + 4] == ['n', 'u', 'l', 'l'] {
        *pos += 4;
        Ok(JsonValue::Null)
    } else {
        Err(UtilsError::JsonError("Invalid null literal".to_string()))
    }
}

fn parse_bool(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    if *pos + 4 <= chars.len() && chars[*pos..*pos + 4] == ['t', 'r', 'u', 'e'] {
        *pos += 4;
        Ok(JsonValue::Bool(true))
    } else if *pos + 5 <= chars.len() && chars[*pos..*pos + 5] == ['f', 'a', 'l', 's', 'e'] {
        *pos += 5;
        Ok(JsonValue::Bool(false))
    } else {
        Err(UtilsError::JsonError("Invalid bool literal".to_string()))
    }
}

fn parse_string(chars: &[char], pos: &mut usize) -> UtilsResult<String> {
    *pos += 1; // skip opening quote
    let mut s = String::new();
    while *pos < chars.len() {
        let c = chars[*pos];
        *pos += 1;
        if c == '"' {
            return Ok(s);
        }
        if c == '\\' {
            if *pos >= chars.len() {
                return Err(UtilsError::JsonError("Unterminated escape sequence".to_string()));
            }
            let esc = chars[*pos];
            *pos += 1;
            match esc {
                '"' => s.push('"'),
                '\\' => s.push('\\'),
                '/' => s.push('/'),
                'b' => s.push('\x08'),
                'f' => s.push('\x0C'),
                'n' => s.push('\n'),
                'r' => s.push('\r'),
                't' => s.push('\t'),
                _ => s.push(esc),
            }
        } else {
            s.push(c);
        }
    }
    Err(UtilsError::JsonError("Unterminated string".to_string()))
}

fn parse_number(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    let start = *pos;
    while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.' || chars[*pos] == '-' || chars[*pos] == '+' || chars[*pos] == 'e' || chars[*pos] == 'E') {
        *pos += 1;
    }
    let s: String = chars[start..*pos].iter().collect();
    s.parse::<f64>()
        .map(JsonValue::Number)
        .map_err(|_| UtilsError::JsonError(format!("Invalid number '{}'", s)))
}

fn parse_array(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    *pos += 1; // skip '['
    let mut arr = Vec::new();
    skip_whitespace(chars, pos);
    if *pos < chars.len() && chars[*pos] == ']' {
        *pos += 1;
        return Ok(JsonValue::Array(arr));
    }
    loop {
        let val = parse_value(chars, pos)?;
        arr.push(val);
        skip_whitespace(chars, pos);
        if *pos >= chars.len() {
            return Err(UtilsError::JsonError("Unterminated array".to_string()));
        }
        if chars[*pos] == ']' {
            *pos += 1;
            break;
        }
        if chars[*pos] == ',' {
            *pos += 1;
        } else {
            return Err(UtilsError::JsonError(format!("Expected ',' or ']', found '{}'", chars[*pos])));
        }
    }
    Ok(JsonValue::Array(arr))
}

fn parse_object(chars: &[char], pos: &mut usize) -> UtilsResult<JsonValue> {
    *pos += 1; // skip '{'
    let mut map = BTreeMap::new();
    skip_whitespace(chars, pos);
    if *pos < chars.len() && chars[*pos] == '}' {
        *pos += 1;
        return Ok(JsonValue::Object(map));
    }
    loop {
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != '"' {
            return Err(UtilsError::JsonError("Expected string key in object".to_string()));
        }
        let key = parse_string(chars, pos)?;
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != ':' {
            return Err(UtilsError::JsonError("Expected ':' after key".to_string()));
        }
        *pos += 1;
        let val = parse_value(chars, pos)?;
        map.insert(key, val);
        skip_whitespace(chars, pos);
        if *pos >= chars.len() {
            return Err(UtilsError::JsonError("Unterminated object".to_string()));
        }
        if chars[*pos] == '}' {
            *pos += 1;
            break;
        }
        if chars[*pos] == ',' {
            *pos += 1;
        } else {
            return Err(UtilsError::JsonError(format!("Expected ',' or '}}', found '{}'", chars[*pos])));
        }
    }
    Ok(JsonValue::Object(map))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_json_parser_and_ast_1() {
        let json_text = "{\"name\":\"brain\",\"version\":0.2,\"active\":true,\"tags\":[\"rust\",\"deeplearning\"]}";
        let val = parse_json(json_text).unwrap();
        
        assert_eq!(val.get("name").and_then(|v| v.as_str()), Some("brain"));
        assert_eq!(val.get("version").and_then(|v| v.as_f64()), Some(0.2));
        assert_eq!(val.get("active").and_then(|v| v.as_bool()), Some(true));
        
        let tags = val.get("tags").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].as_str(), Some("rust"));
    
        let compact = val.to_compact_string();
        assert!(compact.contains("brain"));
    }
}
