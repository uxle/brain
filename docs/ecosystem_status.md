# Brain Ecosystem Crate Triage

This document categorizes all 33 crates in the repository by readiness and role. Status reflects the current source tree (verified by module/API inventory).

## Tier 1 — Production Core (Fully Tested & Verified)
These crates form the stable foundation and are covered by automated regression and gradient-checking suites:

| # | Crate | Role |
|---|---|---|
| 1 | `brain-core` | Tensor engine: shape algebra, cache-blocked GEMM, BLAS L1-3, linalg, FFT, reductions (incl. cumsum/cumprod/var_mean), adaptive pooling, `BrainMind` |
| 2 | `brain-autograd` | Reverse-mode AD: `Value`/`GradFn`/`Tape`, ~30 ops incl. `abs`/`clamp`/`sin`/`cos`/`where_cond`, Hessian/JVP/Jacobian, checkpointing, `GradScaler` |
| 3 | `brain-nn` | Layers (Linear, Conv1d/2d, ConvTranspose2d, LSTM, GRU, MHA, PixelShuffle, adaptive pools), 30+ activations, normalization (BatchNorm2d, LayerNorm, GroupNorm, RMSNorm, InstanceNorm2d) |
| 4 | `brain-loss` | 30+ losses (CE, focal, KLDiv, Huber, Quantile, InfoNCE, SimCLR, Triplet, ArcFace, CEDice, distillation) with differentiable `forward_value` |
| 5 | `brain-optim` | 16 optimizers (SGD, Adam family, RAdam, Lamb, Lion, NovoGrad, ...), 12 schedulers, clipping, `StateDict` |
| 6 | `brain-train` | Trainer, TrainerBuilder, Batch, ModelState, callbacks (EarlyStopping, MetricHistoryLogger), L2 regularization |
| 7 | `brain-metric` | 60+ metrics: accuracy, ROC/PR AUC, MCC, perplexity, MRR, NDCG, mAP, IoU, calibration, markdown/CSV reports |
| 8 | `brain-onnx` | Pure-Rust ONNX protobuf parser, IR lowering, validator, interpreter (opset 9-21), graph optimizer |
| 9 | `brain-quantization` | Dynamic/static Int8 quantization, calibration, fake quant, QLinear/QConv2d, pruning, CSR sparse ops |
| 10 | `brain-cli` | CLI: `make`, `check`, `run`, `train`, `chat`, `space`, `dataset`, `doctor`, `repl`, `script`, `init`, `bench`, `tensor`, `convert`, `new` |
| 11 | `brain` | Umbrella facade + binary |
| 12 | `brain-data` | DataSource, streaming/mmap loaders, prefetch, backpressure, samplers, collate, caching, RLE/delta compression |
| 13 | `brain-dataset` | DataLoader + WorkerPool, tabular/text/image/audio datasets, transforms, splits, registry, balancing |
| 14 | `brain-graph` | Static graph IR, passes (fold/DCE/CSE/fusion/layout), scheduling, interpreter |
| 15 | `brain-utils` | Hashing, logging, CSV/JSON/INI parsing, fast RNG, rate limiting, system info |

## Tier 2 — Model & Domain Crates (Substantial, Source-Verified)
Full-featured module families with per-crate test suites:

| Crate | Verified Scope |
|---|---|
| `brain-transformer` | Transformer enc/dec, MHA/GQA/MQA/Cross/Relative/Flash attention, RoPE/Alibi, KV cache, Llama/GPT/T5/Bert lites, generation pipelines |
| `brain-rnn` | LSTM/GRU/Vanilla/Peephole cells and sequences, Bidirectional, PackedSequence, BeamSearch, TeacherForcing, online streaming |
| `brain-text` | BPE/WordPiece/SentencePiece/char tokenizers, TF-IDF/BM25, pretrained embeddings, similarity, language modeling |
| `brain-vit` | ViT (PatchEmbed, PosEmbed), detection (BBox/NMS), segmentation heads, training utils, export |
| `brain-cv` | Conv variants, detection (anchors/RoIAlign/NMS), FPN backbones, grid_sample, augmentation |
| `brain-audio` | STFT/Mel/MFCC features, WAV/MP3/FLAC IO, VAD, DTW alignment, denoising, pitch/rhythm/energy features |
| `brain-diffusion` | DDPM/DDIM/PLMS samplers, linear/cosine/scaled schedules, UNet2d, guidance, latent codecs |
| `brain-gan` | DCGAN/ResNet/Conditional generators, PatchGAN discriminators, FID/IS lite, CycleGAN-lite |
| `brain-rl` | DQN/Double/Dueling/Rainbow, PPO, A2C, SAC, GAE, replay buffers (prioritized/N-step), 10+ environments |
| `brain-gnn` | GCN/GAT/SAGE/GIN/GatedConv/EdgeConv, graph sampling, readout pooling, saliency explanation |
| `brain-neuroevolution` | GA, CMA-ES, 1+1-ES, HyperNEAT (CPPN/substrates), fitness benchmarks |
| `brain-regularization` | Dropout family, Mixup, LabelSmoothing, EarlyStopping, WeightNorm, SpectralNorm, L1/L2/ElasticNet, curriculum |

## Tier 3 — Systems & Tooling
| Crate | Verified Scope |
|---|---|
| `brain-distributed` | Ring/tree allreduce, data/model/tensor/pipeline parallelism (1F1B), grad compression, fault tolerance |
| `brain-federated` | FedAvg server, secure aggregation, Gaussian DP noise, top-k/quantization compression |
| `brain-export` | ONNX/TFLite/CoreML/WebNN exporters, export verification, zip archives |
| `brain-compile` | IR lowering, optimization passes, JIT cache, memory/schedule plans, profiler, CUDA/LLVM codegen stubs |
| `brain-benchmark` | Bench runner with warmup, statistics, Welch t-test, energy estimation, hardware probing, 5 report formatters, Prometheus/OTel exporters |

## Known Gaps (Honest List)
- `brain-loss`: `BinaryCrossEntropy` exists only as a `LossKind` name — no dedicated function; no CTC, PoissonNLL, or MarginRanking.
- `brain-optim`: no Adan or L-BFGS; schedulers missing exponential/linear/noam/composed/sequential.
- `brain-train`: no dedicated Checkpointer/Logger/Learner subsystems (metrics live in `brain-metric`, callbacks in `brain-train`).
- `brain-core`: no GPU backend (CPU-only by design); no fp16 arithmetic kernels (f32/f64 only).
- `brain-compile`: CUDA/LLVM codegen are stubs.