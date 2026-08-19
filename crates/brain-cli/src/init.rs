//! # Project Scaffolding & Template Generator
//!
//! Scaffolds a new Brain deep learning project layout with `Cargo.toml`, `.brain.toml`, and model skeleton.

use crate::core::{ExitCode, OutputSink};

/// Scaffolds a new project directory layout.
pub fn scaffold_project(name: &str, sink: &OutputSink) -> ExitCode {
    let project_dir = std::path::Path::new(name);
    sink.println(&format!("Creating new Brain project '{}'...", name));

    let src_dir = project_dir.join("src");
    let workflows_dir = project_dir.join(".github").join("workflows");

    if let Err(err) = std::fs::create_dir_all(&src_dir) {
        sink.println(&format!("error: creating src directory: {}", err));
        return ExitCode::IO_ERROR;
    }
    if let Err(err) = std::fs::create_dir_all(&workflows_dir) {
        sink.println(&format!("error: creating workflows directory: {}", err));
        return ExitCode::IO_ERROR;
    }

    let cargo_toml = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
brain = "1.0.0"
"#,
        name
    );

    let brain_toml = r#"# Brain framework project configuration
[model]
name = "model"
default_arch = "mlp"
"#;

    let main_rs = r#"use brain::core::Tensor;

fn main() {
    println!("Hello from Brain project!");
    let t = Tensor::ones(vec![2, 2]);
    println!("Created tensor: {:?}", t.shape());
}
"#;

    let ci_yml = r#"name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build
      run: cargo build --verbose
    - name: Run tests
      run: cargo test --verbose
"#;

    if let Err(err) = std::fs::write(project_dir.join("Cargo.toml"), cargo_toml) {
        sink.println(&format!("error: writing Cargo.toml: {}", err));
        return ExitCode::IO_ERROR;
    }
    sink.println("  + Created Cargo.toml");

    if let Err(err) = std::fs::write(project_dir.join(".brain.toml"), brain_toml) {
        sink.println(&format!("error: writing .brain.toml: {}", err));
        return ExitCode::IO_ERROR;
    }
    sink.println("  + Created .brain.toml");

    if let Err(err) = std::fs::write(src_dir.join("main.rs"), main_rs) {
        sink.println(&format!("error: writing src/main.rs: {}", err));
        return ExitCode::IO_ERROR;
    }
    sink.println("  + Created src/main.rs");

    if let Err(err) = std::fs::write(workflows_dir.join("ci.yml"), ci_yml) {
        sink.println(&format!("error: writing .github/workflows/ci.yml: {}", err));
        return ExitCode::IO_ERROR;
    }
    sink.println("  + Created .github/workflows/ci.yml");

    sink.println("Project initialized successfully! Run 'cargo run' to start.");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
