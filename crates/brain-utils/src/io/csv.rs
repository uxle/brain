//! # RFC 4180 Compliant CSV Parser and Writer
//!
//! Provides zero-dependency streaming CSV parsing, field quoting/escaping,
//! header management, and record serialization.

use crate::core::{UtilsError, UtilsResult};

/// CSV parsing and formatting configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvConfig {
    /// Field delimiter character (default `,`).
    pub delimiter: char,
    /// Quote character (default `"`).
    pub quote_char: char,
    /// Escape character.
    pub escape_char: char,
    /// Whether first line is a header.
    pub has_header: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            escape_char: '"',
            has_header: true,
        }
    }
}

/// A parsed CSV record row.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CsvRecord {
    /// Ordered field values.
    pub fields: Vec<String>,
}

impl CsvRecord {
    /// Creates a record from vector of strings.
    pub fn new(fields: Vec<String>) -> Self {
        Self { fields }
    }

    /// Retrieves field by index.
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.fields.get(idx).map(|s| s.as_str())
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Checks if record has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// CSV Reader and Parser.
pub struct CsvReader {
    config: CsvConfig,
    headers: Option<Vec<String>>,
    records: Vec<CsvRecord>,
}

impl CsvReader {
    /// Parses entire CSV content from a string slice.
    pub fn parse_str(content: &str, config: CsvConfig) -> UtilsResult<Self> {
        let mut records = Vec::new();
        let mut cur_field = String::new();
        let mut cur_record = Vec::new();
        let mut in_quotes = false;
        let mut chars = content.chars().peekable();

        while let Some(c) = chars.next() {
            if in_quotes {
                if c == config.quote_char {
                    if chars.peek() == Some(&config.quote_char) {
                        chars.next();
                        cur_field.push(config.quote_char);
                    } else {
                        in_quotes = false;
                    }
                } else {
                    cur_field.push(c);
                }
            } else if c == config.quote_char {
                in_quotes = true;
            } else if c == config.delimiter {
                cur_record.push(cur_field);
                cur_field = String::new();
            } else if c == '\r' {
                // Ignore \r
            } else if c == '\n' {
                cur_record.push(cur_field);
                cur_field = String::new();
                if !cur_record.is_empty() && (cur_record.len() != 1 || !cur_record[0].is_empty()) {
                    records.push(CsvRecord::new(cur_record));
                }
                cur_record = Vec::new();
            } else {
                cur_field.push(c);
            }
        }

        if in_quotes {
            return Err(UtilsError::CsvError(
                "Unterminated quoted CSV field".to_string(),
            ));
        }

        if !cur_field.is_empty() || !cur_record.is_empty() {
            cur_record.push(cur_field);
            records.push(CsvRecord::new(cur_record));
        }

        let headers = if config.has_header && !records.is_empty() {
            Some(records.remove(0).fields)
        } else {
            None
        };

        Ok(Self {
            config,
            headers,
            records,
        })
    }

    /// Returns CSV configuration.
    pub fn config(&self) -> &CsvConfig {
        &self.config
    }

    /// Returns headers if present.
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_deref()
    }

    /// Returns parsed records.
    pub fn records(&self) -> &[CsvRecord] {
        &self.records
    }
}

/// CSV Writer and Serializer.
pub struct CsvWriter {
    config: CsvConfig,
    output: String,
}

impl CsvWriter {
    /// Constructs a new CSV writer.
    pub fn new(config: CsvConfig) -> Self {
        Self {
            config,
            output: String::new(),
        }
    }

    /// Writes a single row record.
    pub fn write_record(&mut self, fields: &[&str]) {
        let mut line = String::new();
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                line.push(self.config.delimiter);
            }
            if f.contains(self.config.delimiter)
                || f.contains(self.config.quote_char)
                || f.contains('\n')
            {
                line.push(self.config.quote_char);
                line.push_str(&f.replace(
                    self.config.quote_char,
                    &format!("{}{}", self.config.quote_char, self.config.quote_char),
                ));
                line.push(self.config.quote_char);
            } else {
                line.push_str(f);
            }
        }
        line.push('\n');
        self.output.push_str(&line);
    }

    /// Returns accumulated CSV text.
    pub fn finish(self) -> String {
        self.output
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_csv_parser_and_writer_1() {
        let csv_data = "name,age,city\nAlice,30,New York\nBob,25,\"San Francisco, CA\"\n";
        let reader = CsvReader::parse_str(csv_data, CsvConfig::default()).unwrap();

        assert_eq!(reader.headers().unwrap(), &["name", "age", "city"]);
        assert_eq!(reader.records().len(), 2);
        assert_eq!(reader.records()[0].get(0), Some("Alice"));
        assert_eq!(reader.records()[1].get(2), Some("San Francisco, CA"));
        assert_eq!(reader.config().delimiter, ',');

        let mut writer = CsvWriter::new(CsvConfig::default());
        writer.write_record(&["epoch", "loss", "accuracy"]);
        writer.write_record(&["1", "0.45", "0.88"]);
        let out = writer.finish();
        assert!(out.contains("epoch,loss,accuracy"));
    }
}
