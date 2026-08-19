# Brain Public API Surface

This document provides a concise reference to the primary public traits, structs, and functions exposed by the Brain workspace crates. Verify exact signatures in source before relying on them.

## `brain-core`
- **Tensor Types**: `Tensor`, `Shape`, `Dim`, `Strides`, `DType`, `Device`
- **Construction**: `from_vec`, `from_slice`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `rand`, `randn`, `scalar`
- **Arithmetic (`tensor::arithmetic`)**: `matmul`, `bmm`, `addmm`, `outer`, `cross`, `dot`/`vdot`, `matvec`, `cosine_similarity`, `kron`, `min_elem`, `max_elem`, `where_cond`
- **Reductions (`tensor::reduction`)**: `sum`, `mean`, `prod`, `min`, `max`, `ptp`, `var`, `std`, `var_mean`, `sum_along_dim`, `mean_along_dim`, `var_along_dim`, `std_along_dim`, `cumsum`, `cumprod`, `argmax`, `argmin`, `log_sum_exp`, nan-aware variants
- **Pooling (`tensor::pool`)**: `max_pool2d`, `avg_pool2d`, `global_avg_pool2d`, `global_max_pool2d`, `adaptive_avg_pool2d`, `adaptive_max_pool2d`
- **Views**: `reshape`, `view`, `permute`, `transpose`, `t`, `squeeze`, `squeeze_dim`, `unsqueeze`, `flatten`, `expand`, `split`, `chunk`, `narrow`, `slice`, `repeat`, `roll`
- **Linalg (`tensor::linalg`)**: `lu`, `qr`, `cholesky`, `svd_symmetric`, `eigh`, `inv`, `pinv`, `det`, `logdet`, `slogdet`, `trace`, `matrix_power`, `condition_number`, norm family (`norm_l1`..`norm_nuclear`)
- **Special (`tensor::special`)**: `softmax`, `log_softmax`
- **Model (`brain_mind`)**: `BrainMind`, `TeachSummary`

## `brain-autograd`
- **Core Types**: `Value`, `GradFn`, `Tape`
- **Unary ops on `Value`**: `abs`, `clamp`, `sin`, `cos`, `recip`, `square`, `sign`, `exp`, `log`, `log_softmax`, `mean`, `neg`, `relu`, `sigmoid`, `softmax`, `sqrt`, `sum`, `tanh`
- **Binary/ternary ops on `Value`**: `add`, `div`, `matmul`, `mul`, `pow`, `sub`, `min_elem`, `max_elem`, `where_cond`
- **Engine**: `Value::backward`, `Value::grad`, `Value::zero_grad`, `Tape::drain`
- **Advanced**: `graph_closure::{grad, grad_and_hess, hessian, jacobian, jvp, vjp, value_and_grad}`, `parallel_backward`, `GradScaler`, `checkpoint`/`RecomputeGraph`/`CpuOffloader`

## `brain-nn`
- **Modules**: `Linear`, `Conv2d`, `Conv1d`, `ConvTranspose2d`, `Embedding`, `MultiheadAttention`, `LSTM`, `GRU`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `PixelShuffle`, `Identity`, `Bilinear`
- **Normalization**: `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm2d`
- **Activations**: 30+ — `ReLU`, `LeakyReLU`, `GELU`, `FastGELU`, `SiLU`, `Swish`, `Mish`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `ELU`, `CELU`, `SELU`, `GLU`, `SwiGLU`, `Softplus`, `Softsign`, `HardSigmoid`, `HardSwish`, `HardTanh`, `PReLU`, `LogSigmoid`, `TanhShrink`, `HardShrink`, `SoftShrink`, `Shrink`, `ThresholdedReLU`, `Threshold`, `ReLU6`, `Softmin`, `QuietSoftmax`
- **Containers**: `Sequential`, `SequentialNamed`, `ModuleList`, `ModuleDict`
- **Traits/Infra**: `Module`, `Parameter`, `Buffer`, `NamedParameter`, `PruningMask`, hooks (`ForwardPreHook`, `ForwardPostHook`, `HookRegistry`)
- **Init**: kaiming, xavier, orthogonal
- **Dropout**: `Dropout`, `Dropout2d`, `AlphaDropout`, `FusedDropout`

## `brain-loss`
- **Trait**: `Loss` (with differentiable `forward_value`)
- **Classification**: `CrossEntropyLoss` (+ `forward_logits`), `FocalLoss`, `HingeLoss`, `KLDivergenceLoss`, `ArcFaceLoss`
- **Regression**: `MSELoss`, `MAELoss`, `HuberLoss`, `SmoothL1Loss`, `QuantileLoss`, `CauchyLoss`
- **Contrastive**: `InfoNCELoss`, `SimCLRLoss`, `TripletMarginLoss`, `CosineEmbeddingLoss`, `AngularDistanceLoss`
- **Adversarial**: `WassersteinLoss`, `HingeAdversarialLoss`, `LSGANLoss`, `RelativisticLoss`
- **Other**: `CEDiceLoss`, `KnowledgeDistillationLoss`, `CompositeLoss`, `apply_loss_mask`, fused `softmax`/`nll_loss`/`one_hot_target`

## `brain-optim`
- **Optimizers** (trait `Optimizer` with `step`): `Sgd` (+ Nesterov), `Adam`, `AdamW`, `Adamax`, `Nadam`, `RAdam`, `Lamb`, `Lion`, `NovoGrad`, `Rmsprop`, `Adagrad`, `Adadelta`, `Sam`, `Lookahead`, `SwAOptimizer`, `ModelEma`
- **Schedulers**: 12 — step, cosine, cyclic, onecycle, plateau, warmup, ...
- **Utilities**: `clip_grad_norm_`, `clip_grad_value_`, `clip_grad_adaptive_`, `AGC`, `GradScaler`, `LrFinder`, `OptimizerBuilder`, `StateDict`, `OptimizerCheckpoint`

## `brain-train`
- **Pipeline**: `Trainer`, `TrainerBuilder`, `Batch`, `ModelState`, `NamedTensor`, `TrainStep`, `TrainingSummary`, `SyntheticClassification`, `TensorModuleAdapter`, `L2Regularization`
- **Training Methods**: `train_batch`, `fit`, `fit_accumulated`, `evaluate`, `load_state`, `state`
- **Callbacks**: `EarlyStopping`, `MetricHistoryLogger`, `TrainingCallback`, `CallbackAction`

## `brain-transformer`
- **Encoder/Decoder**: `TransformerEncoder`, `TransformerEncoderBlock`, `TransformerDecoder`, `TransformerDecoderLayer`, `TransformerBuilder`
- **Attention**: `MultiHeadAttention`, `GroupedQueryAttention`, `MqaConfig`, `CrossAttention`, `RelativeAttention`, `FlashAttentionLite`, `XformersAttentionLite`
- **Position**: `RotaryEmbedding` (RoPE), `AlibiPositionalBias`, `SinusoidalPositionalEmbedding`, `LearnedPositionalEmbedding`
- **Cache & Generation**: `KvCache`, `LayerKvCache`, `Generator`, `GenerateConfig`, `TextGenerationPipeline`, `SequenceClassificationPipeline`
- **Model Lites**: `LlamaLite`, `GptLite`, `T5Lite`, `BertLite`

## `brain-rnn`
- **Cells**: `LstmCell`, `GruCell`, `VanillaRnnCell`, `PeepholeLstmCell`, `AttentionCell`, `NormLstmCell`
- **Sequences**: `LstmSeq`, `GruSeq`, `VanillaRnnSeq`, `BidirectionalRnn`, `PackedSequence`, `BeamSearch`, `TeacherForcer`
- **Extras**: `RnnBuilder`, `OnlineRnnStreamer`, `VariationalDropout`, `truncate_steps`, `init_lstm_weights`

## `brain-onnx`
- **IR**: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue`
- **Tools**: `parse_model_proto`, `import_model`, `load_onnx`, `evaluate_onnx_model`, `check_model`, `optimize_model` (constant folding, ConvReLU/MatMulAdd fusion), `export_onnx_bytes`, `onnx_summary`, `OpsetTable` (9-21)

## `brain-quantization`
- **Quantization**: `quantize_tensor`, `dequantize_tensor`, `QuantConfig`, `QuantDType`, `QuantTensor`, `DynamicQuantizer`, `StaticQuantizer`, `FakeQuantize`, calibration observers (4)
- **Quantized Ops**: `QLinear`, `QConv2d`, `q8_matmul`
- **Pruning/Sparse**: `apply_magnitude_prune`, `MagnitudePruner`, `StructuredPruner`, `CsrMatrix` (spmm/spmv)
- **Advanced**: `BlockQuantizer`, `ActQuantizer`, `MixedPrecisionQuantizer`, `GraphQuantizer`, `QuantBuilder`, `QuantRuntime`, `analyze_quantization_error`

## `brain-metric`
- **Classification**: `accuracy_score`, `precision_recall_f1`, `confusion_matrix`, `matthews_correlation_coefficient`, `roc_auc_score`, `pr_auc_score`, `hamming_loss`
- **NLP**: `perplexity_score`, `ndcg_at_k`, `mean_reciprocal_rank`, `sentence_bleu`, `meteor_score_lite`, `edit_distance_levenshtein`, `exact_match_ratio`
- **Detection/Seg**: `bbox_iou`, `mean_average_precision`, `miou_and_pixel_accuracy`
- **Regression**: `mse_score`, `mae_score`, `rmse_score`, `r2_score`, `mape_score`, `mase_score`, `median_absolute_error`, `pearson_correlation`
- **Reports**: `MetricTracker`, `aggregate_metric_runs`, markdown/CSV report formatters, `CalibrationReport`, `CompareReport`

## `brain-data` / `brain-dataset`
- **`brain-data`**: `DataSource`, `Sample`, `SampleBatch`, `MemoryLoader`, `StreamDataset`, `MmapChunkReader`, samplers (`Sequential`, `Distributed`), `MapStage`, prefetch/backpressure, `SampleCache`, RLE/delta compression, `DataLoaderConfig`
- **`brain-dataset`**: `DataLoader`, `WorkerPool`, `TabularDataset`, `TextLinesDataset`, `RandomImageDataset`, `RandomSegDataset`, `SyntheticAudioDataset`, `Subset`, transforms (`Compose`, `Normalize`, ...), `random_split_indices`, `DatasetRegistry`, `BalanceConfig`, `DatasetBuilder`

## `brain-graph` / `brain-compile`
- **`brain-graph`**: `GraphIr`, `OpKind`, shape inference, passes (fold/DCE/CSE/fusion/layout), `SchedulePlan`, `GraphInterpreter`, DOT/JSON export, diffing
- **`brain-compile`**: `IrGraph`, `IrOp`, `PassManager`, `Interpreter::evaluate`, `JitCache::get_or_compile`, `MemoryPlan`, `estimate_total_flops`, `export_dot`/`export_text`

## `brain-cli` / `brain`
- **CLI Commands**: `brain make`, `brain check`, `brain run`, `brain train`, `brain chat`, `brain space`, `brain new`, `brain script`, `brain dataset`, `brain convert`, `brain doctor`, `brain repl`, `brain init`, `brain bench`, `brain tensor`
- **Entry**: `brain_cli::run_cli(args, &OutputSink) -> ExitCode`

## Research & Systems
- **`brain-rl`**: `Dqn`/`DoubleDqn`/`DuelingDqn`/`Rainbow`, `Ppo`, `A2c`, `Sac`, `compute_gae`, `ReplayBuffer`/`PrioritizedBuffer`, `Env` (CartPole, GridWorld, ...), `Trainer`
- **`brain-gnn`**: `GCN`, `GAT`, `SAGE`, `GIN`, `GatedConv`, `EdgeConv`, `GnnTrainer`, graph sampling, saliency explanation
- **`brain-diffusion`**: `DiffusionModel`, `Unet2d`, `DdpmSampler`, `DdimSampler`, `PlmsSampler`, linear/cosine/scaled schedules, guidance
- **`brain-gan`**: `Gan`/`GanTrainer`, DCGAN/ResNet/Conditional generators, PatchGAN discriminators, `fid_lite`, `is_lite`
- **`brain-neuroevolution`**: `Ga`, `Cmaes`, `Es1p1`, HyperNEAT (`Cppn`, `SubstrateGrid2D`), fitness benchmarks
- **`brain-distributed`**: ring/tree `allreduce`, `DataParallel`, `ModelParallel`, `TensorParallelLinear`, pipeline 1F1B, `grad_compression`
- **`brain-federated`**: `FederatedServer`, `fed_avg_aggregate`, `SecureAggregator`, Gaussian DP noise, top-k sparsification
- **`brain-export`**: `ModelExporter` trait, ONNX/TFLite/CoreML/WebNN exporters, `ExportBuilder`, `verify_export`
- **`brain-benchmark`**: `Runner`, `BenchmarkSuite`, `Statistics`, `welch_t_test`, `HardwareInfo::probe`, `EnergyEstimator`, report formatters (5)
- **`brain-text`**: BPE/WordPiece/SentencePiece/char tokenizers, `TfIdf`, `Bm25`, similarity, `VocabBuilder`
- **`brain-utils`**: hashing (`fnv1a_64`, `murmur3_32`), `Crc32`/`Adler32`, `StandardLogger`, `ConfigManager`, `FastRng`, `RateLimiter`, `SystemInfo`