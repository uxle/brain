# 🧠 Brain Framework — Complete Commands Reference Manual (`commands.md`)

Welcome to the definitive command reference manual for the **Brain** Deep Learning Framework, Biological Neural Mind Engine, and Portable Autonomous Agent Runtime.

All commands can be invoked either via the installed `brain` binary or directly via Cargo:
```bash
# Using installed binary
brain <command> [flags]

# Or running from the repository source (always with -j 2 for strict concurrency safety)
cargo run -p brain -- <command> [flags]
```

---

## 📋 Master Command Summary

| Command | Category | Purpose | Typical Scenario |
|---|---|---|---|
| [`new`](#1-brain-new) | Biological Mind | Initializes a new 3D interconnected cubic neural mind | Creating a newborn neural mind with custom neuron counts |
| [`chat`](#2-brain-chat--brain-chatbot) | Conversational | Real-time conversational learning & memory retrieval | Teaching facts, asking questions, conversational arithmetic |
| [`make`](#3-brain-make) | Auto-Train | Builds, trains & checkpoints a model in 1 step | Rapid prototyping directly from CSV/TXT data |
| [`run`](#4-brain-run) | Inference | Loads checkpoint and runs sample/batch predictions | Production classification or evaluating single inputs |
| [`agent`](#5-brain-agent) | Autonomous Agent | Runs perception-cognition-action-learning loop | Physical/Virtual computer actuation via USB HID |
| [`train`](#6-brain-train) | Training Engine | Configurable training loop with optimizers & metrics | Training deep MLP/ConvNet architectures |
| [`model`](#7-brain-model) | Architecture | Inspects layer hierarchy, parameters & shapes | Architecture debugging and sanity checking |
| [`check`](#8-brain-check) | Verification | Validates checkpoint integrity & graph invariants | Pre-deployment verification |
| [`init`](#9-brain-init) | Scaffolding | Creates a new Brain deep learning project workspace | Starting a new standalone application |
| [`tensor`](#10-brain-tensor) | Tensor Math | Creates, inspects, slices, and reshapes tensors | Command-line tensor manipulation |
| [`repl`](#11-brain-repl) | Interactive Math | Interactive shell for live matrix math & calculus | Quick calculations and debugging |
| [`script`](#12-brain-script) | Automation | Runs procedural `.brain` pipeline scripts | Automated data preprocessing and pipelines |
| [`bench`](#13-brain-bench) | Benchmarking | High-resolution FLOPS benchmarks (GEMM, FFT, etc.) | Measuring hardware throughput |
| [`dataset`](#14-brain-dataset) | Data Pipeline | Inspects class distributions and creates splits | Dataset EDA and train/val partitioning |
| [`convert`](#15-brain-convert) | Interop | Converts between `.brain`, ONNX, and Safetensors | Model export and migration across frameworks |
| [`doctor`](#16-brain-doctor) | Diagnostics | Audits OS, SIMD support, memory & backend health | Hardware validation before long runs |

---

## 🛠️ Detailed Command Reference

### 1. `brain new`
**Purpose**: Initializes a new 3D interconnected cubic neural mind saved in the tamper-proof `.bn` format (with 32-bit CRC integrity verification).

**When to Use**:
- When creating a newborn biological mind from scratch.
- When initializing a neural mind of a specific dimension ($N 	imes N 	imes N$).
- When pre-teaching a structured knowledge base (e.g., `science.txt`, `mathematics.txt`, `data.txt`) into a saved `.bn` file.

#### Synopsis:
```bash
brain new <brain.bn> [--neurons <N> | --cube <D>] [--teach <file.txt>] [--chat]
```

#### Flags & Options:
- `<brain.bn>` (Positional 1): Target path for the generated neural file.
- `--neurons <N>`, `-n <N>`: Target total neuron count. Cube dimension is calculated as $\lfloor N^{1/3} ceil$.
- `--cube <D>`, `-c <D>`: Explicit cube dimension ($D 	imes D 	imes D$ neurons, default: 10 $	o$ 1,000 neurons).
- `--teach <file.txt>`, `-t <file.txt>`: Path to a knowledge text corpus to learn immediately upon creation.
- `--chat`, `--interactive`: Automatically launches the interactive conversation interface after creation.

#### Concrete Examples:
```bash
# Create a small 5x5x5 brain (125 neurons)
brain new tiny_brain.bn --cube 5

# Create a 1,000-neuron brain and immediately teach it science knowledge
brain new science_mind.bn --neurons 1000 --teach science.txt

# Create, teach, and enter conversation immediately
brain new scholar.bn --teach data.txt --chat
```

---

### 2. `brain chat` / `brain chatbot`
**Purpose**: Starts an interactive conversational session with a dynamic `BrainMind` neural instance. The mind learns in real-time from user statements, answers questions, solves arithmetic, and recalls learned knowledge facts.

**When to Use**:
- To interact with a newborn brain (observing natural imitation and adaptation).
- To teach facts incrementally through conversation (*"A photon is a packet of light"*).
- To query knowledge bases taught to the brain (*"What is photosynthesis?"*).
- To perform arithmetic calculations (*"what is 45 * 12"*).

#### Synopsis:
```bash
brain chat [<brain.bn>]
brain chatbot <brain.bn>
```

#### Concrete Examples:
```bash
# Chat with a new temporary in-memory mind
brain chat

# Chat with an existing persisted mind (updates are saved automatically on exit)
brain chat science_mind.bn
```

#### Interactive Session Dialogue Example:
```text
You: hi
Brain: hi!

You: my name is Lion
Brain: Nice to meet you, Lion! I will remember you.

You: who am I?
Brain: You are Lion.

You: what is 25 * 4
Brain: 25 * 4 = 100

You: A catalyst is a substance that speeds up a chemical reaction without being consumed.
Brain: I learned: 'a catalyst is a substance that speeds up a chemical reaction without being consumed.'

You: what is a catalyst?
Brain: A catalyst is a substance that speeds up a chemical reaction without being consumed.
```

---

### 3. `brain make`
**Purpose**: High-level one-step command that loads a tabular or text dataset, builds an optimal neural network architecture, trains it, and saves a deployable `.brain` checkpoint.

**When to Use**:
- When you have a dataset (e.g. CSV or columnar features with a label column) and want to train an end-to-end model in one command without writing code.

#### Synopsis:
```bash
brain make <output.brain> --data <data.txt> [options]
```

#### Flags & Options:
- `<output.brain>` (Positional 1): Output file path for the trained model checkpoint.
- `--data <path>`: Input dataset path (features in columns $0 \dots N-1$, integer label in last column).
- `--arch <mlp|convnet>`: Architecture type (default: `mlp`).
- `--hidden <N>`: Number of hidden units in the MLP layer (default: `16`).
- `--classes <N>`: Number of target output classes (default: inferred from dataset labels).
- `--optim <sgd|adam>`: Optimizer algorithm (default: `sgd`).
- `--loss <cross_entropy|mse>`: Loss function (default: `cross_entropy`).
- `--lr <F>`: Learning rate (default: `0.1`).
- `--epochs <N>`: Number of training epochs (default: `20`).
- `--batch <N>`: Mini-batch size (default: `8`).

#### Concrete Examples:
```bash
# Train default MLP on iris dataset for 30 epochs
brain make iris_model.brain --data iris.data --epochs 30 --lr 0.05

# Train with Adam optimizer and 64 hidden units
brain make classifier.brain --data dataset.txt --arch mlp --hidden 64 --optim adam --epochs 50
```

---

### 4. `brain run`
**Purpose**: Loads a trained `.brain` checkpoint and executes inference on a single input string or an entire test dataset.

**When to Use**:
- To deploy a trained model and generate predictions on new unseen data.
- To test a single sample vector quickly from the command line.

#### Synopsis:
```bash
brain run <model.brain> [--data <test.data> | --input "<a,b,c,...>"] [--top]
```

#### Flags & Options:
- `<model.brain>` (Positional 1): Path to the saved `.brain` checkpoint.
- `--data <path>`: Path to a test dataset file to evaluate over all samples.
- `--input "<v1,v2,...>"`: Comma-separated list of float features for a single inference sample.
- `--top`: Outputs only the predicted class index (ideal for piping into scripts).

#### Concrete Examples:
```bash
# Predict a single sample vector
brain run iris_model.brain --input "5.1,3.5,1.4,0.2"

# Evaluate accuracy over a test dataset
brain run classifier.brain --data test_features.data

# Pipe predicted class directly to a shell variable
PRED=$(brain run iris_model.brain --input "6.2,3.4,5.4,2.3" --top)
```

---

### 5. `brain agent`
**Purpose**: Runs the portable autonomous learning agent runtime orchestrating visual perception (HDMI/camera), cognition (`WorldModel` + `IntrinsicCuriosityModule`), safety-filtered USB HID actuation (`SafetyGuard`), and continual learning (`EWC`).

**When to Use**:
- When executing autonomous UI tasks, screen perception, and real-time reinforcement learning.

#### Synopsis:
```bash
brain agent <run|info|record|learn> [options]
```

#### Subcommands & Flags:
- `info`: Displays autonomous agent system capabilities and hardware status.
- `run`: Launches the multi-threaded autonomous agent perceive-think-act-learn loop.
  - `--steps <N>`: Maximum number of perception-action iterations to execute (default: `20`).
  - `--dry-run`: Runs cognition and RL updates, but intercepts and blocks physical HID mouse/keyboard signals.
  - `--mock`: Uses simulated screen frames and virtual HID devices.
- `record`: Records human demonstration trajectories.
- `learn`: Trains imitation policies with Elastic Weight Consolidation (EWC).

#### Concrete Examples:
```bash
# Display agent hardware status
brain agent info

# Run 100 autonomous agent steps with live hardware
brain agent run --steps 100

# Run in safe dry-run mode for 50 steps
brain agent run --dry-run --steps 50
```

---

### 6. `brain train`
**Purpose**: Configurable training engine with progress bars, loss curve tracking, learning rate schedulers, and early stopping callbacks.

**When to Use**:
- For multi-epoch training of custom neural architectures with detailed metric reporting.

#### Synopsis:
```bash
brain train [--data <path>] [--epochs <N>] [--batch <N>] [--lr <F>] [--output <out.brain>]
```

#### Flags & Options:
- `--data <path>`: Path to training dataset file.
- `--epochs <N>`: Number of training epochs (default: `10`).
- `--batch <N>`: Mini-batch size (default: `32`).
- `--lr <F>`: Learning rate (default: `0.001`).
- `--output <path>`: Target checkpoint destination.

#### Concrete Examples:
```bash
brain train --data mnist.data --epochs 25 --batch 64 --lr 0.0005 --output mnist_trained.brain
```

---

### 7. `brain model`
**Purpose**: Inspects neural network architectures, layer hierarchies, parameter shapes, and evaluation metrics.

#### Synopsis:
```bash
brain model <summary|eval|export> <model.brain> [options]
```

#### Concrete Examples:
```bash
# Print model summary table
brain model summary model.brain

# Evaluate model metrics on validation data
brain model eval model.brain --data val.data
```

---

### 8. `brain check`
**Purpose**: Audits model checkpoints, verifies shape integrity, checks for NaN/Inf weights, and ensures computational graph validity.

#### Synopsis:
```bash
brain check <model.brain>
```

#### Concrete Examples:
```bash
brain check classifier.brain
```

---

### 9. `brain init`
**Purpose**: Scaffolds a new Brain deep learning project workspace complete with `Cargo.toml`, `.brain.toml`, starter `src/main.rs`, and GitHub Actions CI workflow.

#### Synopsis:
```bash
brain init [project_name]
```

#### Concrete Examples:
```bash
brain init my_vision_pipeline
cd my_vision_pipeline
cargo run -j 2
```

---

### 10. `brain tensor`
**Purpose**: Command-line utility to generate, inspect, and perform arithmetic operations on N-dimensional tensors.

#### Synopsis:
```bash
brain tensor <create|zeros|ones|info|stats|reshape> [dimensions]
```

#### Actions:
- `zeros <d1,d2,...>`: Creates a zero tensor of specified shape.
- `ones <d1,d2,...>`: Creates a ones tensor of specified shape.
- `create <d1,d2,...>`: Allocates a new tensor.
- `info <d1,d2,...>`: Inspects tensor shape, datatype, and memory layout.
- `stats <d1,d2,...>`: Computes sum, mean, min, max, and element count.
- `reshape <d1,d2,...>`: Reshapes tensor dimensions.

#### Concrete Examples:
```bash
# Create a 3x3 matrix of zeros
brain tensor zeros 3,3

# Create a 4-dimensional tensor of ones [2, 3, 32, 32]
brain tensor ones 2,3,32,32

# Get summary statistics of a tensor shape
brain tensor stats 100,50
```

---

### 11. `brain repl`
**Purpose**: Starts an interactive Read-Eval-Print-Loop (REPL) for tensor algebra, autograd calculus, and rapid mathematical experimentation.

#### Synopsis:
```bash
brain repl
```

#### REPL Built-in Meta Commands:
- `:help`: Displays REPL help and available commands.
- `:vars`: Lists all bound variables and their tensor shapes.
- `:clear`: Clears all variables in the current session.
- `:quit` or `:exit`: Exits the REPL session.

#### REPL Session Example:
```text
Brain Interactive REPL v0.2.0. Type ':help' for commands, ':quit' to exit.
>> a = 2 + 3
a = [1]
>> b = [1.0, 2.0, 3.0]
b = [3]
>> :vars
Bound Variables:
  a: [1]
  b: [3]
>> :quit
Goodbye!
```

---

### 12. `brain script`
**Purpose**: Runs declarative `.brain` automation and workflow scripts line by line.

#### Synopsis:
```bash
brain script <script.brain>
```

#### Script File Example (`pipeline.brain`):
```text
# Declarative Brain Script
a = 10
b = 20
c = a + b
```

#### Execution:
```bash
brain script pipeline.brain
```

---

### 13. `brain bench`
**Purpose**: Executes high-resolution micro-benchmarks measuring raw computational throughput (GFLOPS) across GEMM, FFT, Conv2d, and Transformer attention.

#### Synopsis:
```bash
brain bench <gemm|fft|conv2d|transformer|suite> [matrix_size]
```

#### Concrete Examples:
```bash
# Benchmark 512x512 matrix multiplication
brain bench gemm 512

# Benchmark 1024-point FFT
brain bench fft 1024

# Benchmark 2D Convolutions on 256x256 image
brain bench conv2d 256
```

---

### 14. `brain dataset`
**Purpose**: Inspects dataset line counts, sample distributions, partition splits, and cache artifacts.

#### Synopsis:
```bash
brain dataset <inspect|stats|split|cache> [dataset.csv]
```

#### Concrete Examples:
```bash
# Inspect dataset sample and class counts
brain dataset inspect data.txt

# Partition dataset into train and test splits
brain dataset split dataset.csv
```

---

### 15. `brain convert`
**Purpose**: Converts models and tensor archives between Brain `.brain` / `.bn` checkpoints, ONNX protobuf models, and HuggingFace `safetensors`.

#### Synopsis:
```bash
brain convert <source_file> <dest_file> [--format=onnx|bin|json]
```

#### Concrete Examples:
```bash
# Convert ONNX model into Brain format
brain convert resnet18.onnx resnet18.brain

# Convert Safetensors archive into Brain checkpoint
brain convert model.safetensors model.brain
```

---

### 16. `brain doctor`
**Purpose**: Runs an automated diagnostic health check verifying the host operating system, CPU parallelism, AVX2/FMA SIMD vector extensions, system allocators, and tensor execution sanity.

#### Synopsis:
```bash
brain doctor
```

#### Output:
```text
Brain Doctor — System Health Check:
------------------------------------
  [OK] Operating System: Linux x86_64
  [OK] CPU Topology: Available parallelism verified
  [OK] Memory: System allocator operational
  [OK] Tensor Backend: Arithmetic operations verified
All diagnostics passed with zero issues.
```

---

## 🧪 Developer Workflows & Verification

Always enforce the strict `-j 2` bounded concurrency rule:

```bash
# 1. Run all core unit and numerical gradient tests
cargo test -p brain-core --test numerical_check -j 2
cargo test -p brain-autograd --test grad_check -j 2
cargo test -p brain-train --test trainer_regression -j 2
cargo test -p brain-onnx --test onnx_roundtrip -j 2
cargo test -p brain-quantization --test quant_linear -j 2

# 2. Run master 1.0 release audit across all domain subsystems
cargo test -p brain --test master_1_0_release_audit -j 2

# 3. Run cross-crate end-to-end integration pipeline
cargo test -p brain --test cross_crate_pipeline -j 2

# 4. Run full workspace CI across all 33 crates
./scripts/ci.sh
```

---

## 💻 Official Example Programs

```bash
# 1. Autonomous Agent GUI Interaction Demo ("Open Notepad & Type")
cargo run --example agent_notepad_task -j 2

# 2. Vision ConvNet Training & Classification
cargo run --example convnet_train -j 2

# 3. ONNX Model Export & Roundtrip Evaluation
cargo run --example onnx_export --features="export" -j 2

# 4. Dynamic INT8 Linear Quantization & Magnitude Pruning
cargo run --example quantize_linear --features="export" -j 2

# 5. Build Live Bootable Linux USB Image
./scripts/build_bootable_usb.sh
```
