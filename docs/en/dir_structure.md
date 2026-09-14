# MindSpeed Project Directory Structure

```plaintext
MindSpeed/
├── README.md                            # Project description document, introducing the features and usage of the MindSpeed Core acceleration library
├── docs/                                # Project documentation directory, containing feature docs, user guides, and API docs in both Chinese and English
├── mindspeed/                           # Core source code directory, containing all acceleration library core implementations
│   ├── args_utils.py                    # Argument utility functions, providing helper functions for argument parsing and validation
│   ├── arguments.py                     # Command-line argument definitions, defining training and configuration-related CLI parameters
│   ├── checkpointing.py                 # Checkpoint management, providing model checkpoint save and load functionality
│   ├── deprecated.py                    # Deprecated functionality module, marking deprecated APIs and features
│   ├── initialize.py                    # Initialization module, handling distributed environment initialization and config loading
│   ├── log_config.py                    # Log configuration, defining log format and output rules
│   ├── megatron_adapter.py              # Megatron-LM adapter, enabling compatibility with the Megatron framework
│   ├── patch_utils.py                   # Patch utilities, providing dynamic code patching and replacement functionality
│   ├── train.py                         # Training module, providing the entry point and main loop control for the training workflow
│   ├── utils.py                         # General utility functions, providing common helper functions for the project
│   ├── yaml_arguments.py                # YAML argument parsing, supporting loading training configurations from YAML files
│   ├── auto_settings/                   # Auto-configuration subsystem, automatically optimizing training configs based on hardware environment
│   │   ├── auto_settings.py             # Auto-configuration main entry, coordinating the auto-configuration workflow of all modules
│   │   ├── search_space.py              # Search space definition, defining the search range of tunable parameters
│   │   ├── config/                      # Configuration definitions, containing model and training configuration templates
│   │   ├── mindspeed_adaptor/           # MindSpeed adaptor, applying auto-configuration to the MindSpeed framework
│   │   ├── model/                       # Model definitions, providing model architecture abstraction and configuration
│   │   ├── module/                      # Core modules, containing modeling modules such as communication, memory, and operators
│   │   ├── profile/                     # Profiling tools, providing performance analysis and diagnostic functions
│   │   └── utils/                       # General utilities, providing configuration and helper functions
│   ├── core/                            # Core functional modules, including parallel strategies, memory management, and other core capabilities
│   │   ├── fp8_utils.py                 # FP8 utility functions, providing FP8 quantization and scaling tools
│   │   ├── mindspeed_parallel_group.py  # Parallel group management, managing distributed training process groups
│   │   ├── parallel_state.py            # Parallel state management, maintaining global parallel state information
│   │   ├── simple_parallel_cfg.py       # Simple parallel configuration, providing simplified parallel config interfaces
│   │   ├── singleton_meta.py            # Singleton metadata, managing singleton pattern metadata
│   │   ├── tensor_parallel_y_union.py   # Tensor parallel Y-union configuration, supporting Y-axis tensor parallel union config
│   │   ├── training.py                  # Training module configuration, providing configuration management for the training workflow
│   │   ├── weight_grad_store.py         # Weight gradient storage management, optimizing weight and gradient storage strategies
│   │   ├── context_parallel/            # Context parallel, implementing sequence-length distributed parallelism
│   │   ├── datasets/                    # Datasets, providing data loading and preprocessing functions
│   │   ├── data_parallel/               # Data parallel, implementing data-parallel distributed training strategies
│   │   ├── distributed/                 # Distributed training, providing distributed training infrastructure
│   │   ├── dist_checkpointing/          # Distributed checkpointing, supporting checkpoint management in distributed environments
│   │   ├── fusions/                     # Operator fusion, providing fused operators for improved computational efficiency
│   │   ├── hccl_buffer/                 # HCCL buffer, optimizing HCCL communication buffer management
│   │   ├── megatron_basic/              # Megatron basic adapter, providing foundational Megatron framework adapters
│   │   ├── memory/                      # Memory management, providing memory optimization and management functions
│   │   ├── models/                      # Model definitions, containing GPT and other model architecture implementations
│   │   ├── multi_modal/                 # Multimodal, supporting multimodal model training and inference
│   │   ├── optimizer/                   # Optimizers, providing AdamW and other optimizer implementations
│   │   ├── performance/                 # Performance optimization, providing performance analysis and optimization tools
│   │   ├── pipeline_parallel/           # Pipeline parallel, implementing model pipeline parallelism strategies
│   │   ├── qat/                         # Quantization-aware training, supporting quantization during training
│   │   ├── qos/                         # QoS management, providing quality of service management and resource control
│   │   ├── tensor_parallel/             # Tensor parallel, implementing tensor-slicing parallelism strategies
│   │   └── transformer/                 # Transformer models, providing Transformer architecture implementations
│   ├── features_manager/                # Feature manager, providing unified registration and configuration of various optimization features
│   │   ├── feature.py                   # Feature management, defining feature base classes and interfaces
│   │   ├── features_manager.py          # Feature manager, providing feature registration, enable, and disable functions
│   │   ├── affinity/                    # Affinity management, optimizing process and device affinity configurations
│   │   ├── ai_framework/                # AI framework adaptation, adapting features for different AI frameworks
│   │   ├── auto_settings/               # Auto-configuration features, providing auto-configuration related features
│   │   ├── ckpt_acceleration/           # Checkpoint acceleration, accelerating checkpoint save and load operations
│   │   ├── compress/                    # Compression features, providing model and gradient compression
│   │   ├── compress_dense/              # Dense compression, providing compression strategies for dense models
│   │   ├── context_parallel/            # Context parallel features, managing context parallel related features
│   │   ├── custom_fsdp/                 # Custom FSDP, providing custom fully sharded data parallel
│   │   ├── data_parallel/               # Data parallel features, managing data parallel related features
│   │   ├── disable_gloo_group/          # Disable Gloo group, disabling Gloo backend communication groups
│   │   ├── distributed/                 # Distributed features, managing distributed training related features
│   │   ├── dist_train/                  # Distributed training, providing feature support for distributed training
│   │   ├── functional/                  # Functional features, providing functional programming related features
│   │   ├── fusions/                     # Fused operator features, managing fused operator features
│   │   ├── hccl_buffer/                 # HCCL buffer features, optimizing HCCL buffer usage
│   │   ├── llava/                       # LLaVA model, providing LLaVA multimodal model support
│   │   ├── megatron_basic/              # Megatron basic features, providing foundational Megatron framework features
│   │   ├── memory/                      # Memory features, managing memory optimization related features
│   │   ├── moe/                         # MoE features, providing Mixture of Experts model support
│   │   ├── optimizer/                   # Optimizer features, managing optimizer related features
│   │   ├── pipeline_parallel/           # Pipeline parallel features, managing pipeline parallel related features
│   │   ├── qat/                         # Quantization-aware training features, managing QAT related features
│   │   ├── qos/                         # QoS features, managing quality of service related features
│   │   ├── recompute/                   # Recompute features, providing activation recomputation functionality
│   │   ├── tensor_parallel/             # Tensor parallel features, managing tensor parallel related features
│   │   ├── tokenizer/                   # Tokenizer features, managing tokenizer related features
│   │   └── transformer/                 # Transformer features, managing Transformer model features
│   ├── fsdp/                            # FSDP fully sharded data parallel, implementing ZeRO-3 and other sharding strategies
│   │   ├── mindspeed_parallel_engine.py # Parallel engine, providing FSDP parallel engine implementation
│   │   ├── parallel_engine_config.py    # Parallel engine configuration, configuring FSDP parallel engine parameters
│   │   ├── distributed/                 # Distributed, providing FSDP distributed communication functionality
│   │   ├── memory/                      # Memory, providing FSDP memory management functionality
│   │   ├── quantization/                # Quantization, providing FSDP quantization support
│   │   └── utils/                       # Utilities, providing FSDP-related helper functions
│   ├── functional/                      # Functional interfaces, providing NPU-related functional APIs
│   │   ├── npu_datadump/                # NPU data dump, providing NPU data export and debugging functionality
│   │   ├── npu_deterministic/           # NPU deterministic computation, ensuring deterministic NPU computation
│   │   ├── profile/                     # Performance profiling, providing performance analysis and diagnostic tools
│   │   ├── profiler/                    # Profiler, providing detailed performance analysis functionality
│   │   └── tflops_calculate/            # TFLOPS calculation, computing training throughput and performance metrics
│   ├── lite/                            # Lite lightweight version, providing lightweight training support
│   │   ├── mindspeed_lite_config.py     # Lite configuration, providing Lite version configuration management
│   │   ├── mindspeed_lite.py            # Lite model, providing Lite version model implementations
│   │   ├── distributed/                 # Distributed, providing Lite version distributed support
│   │   ├── memory/                      # Memory, providing Lite version memory management
│   │   ├── ops/                         # Operators, providing Lite version operator implementations
│   │   └── utils/                       # Utilities, providing Lite version helper functions
│   ├── mindspore/                       # MindSpore framework adaptation, providing MindSpore framework support
│   │   ├── mindspore_adaptor.py         # MindSpore adaptor, implementing MindSpore framework adaptation
│   │   ├── core/                        # Core features, providing MindSpore core feature implementations
│   │   ├── model/                       # Models, providing MindSpore model implementations
│   │   ├── ops/                         # Operators, providing MindSpore operator implementations
│   │   ├── optimizer/                   # Optimizers, providing MindSpore optimizer implementations
│   │   ├── op_builder/                  # Operator builder, providing custom operator build functionality for MindSpore
│   │   ├── third_party/                 # Third-party libraries, providing MindSpore-supported third-party libraries
│   ├── model/                           # Model definitions, providing general model definitions and interfaces
│   ├── moe/                             # MoE Mixture of Experts, providing Mixture of Experts model implementations
│   ├── multi_modal/                     # Multimodal support, providing multimodal model training support
│   │   └── conv3d/                      # 3D convolution, providing 3D convolution operator implementations
│   ├── ops/                             # Operator library, providing high-performance fused operators
│   │   ├── dropout_add_layer_norm.py    # Dropout+LayerNorm fusion, fusing Dropout and LayerNorm operations
│   │   ├── dropout_add_rms_norm.py      # Dropout+RMSNorm fusion, fusing Dropout and RMSNorm operations
│   │   ├── ffn.py                       # FFN operator, feed-forward neural network operator implementation
│   │   ├── fusion_attention_v2.py       # Fusion attention v2, optimized attention mechanism implementation
│   │   ├── gmm.py                       # GMM operator, grouped matrix multiplication operator
│   │   ├── gmm_mxfp8.py                 # MXFP8 GMM operator, GMM operator with MXFP8 support
│   │   ├── grouped_matmul.py            # Grouped matrix multiplication, efficient grouped matrix multiplication implementation
│   │   ├── npu_apply_fused_adamw_v2.py  # Fused AdamW v2, NPU-optimized AdamW optimizer
│   │   ├── npu_bmm_reduce_scatter_all_to_all.py  # BMM+ReduceScatter+AllToAll, fused communication operator
│   │   ├── npu_matmul_add.py            # MatMul+Add fusion, fusing matrix multiplication and addition operations
│   │   ├── npu_rotary_position_embedding.py  # Rotary position embedding, NPU-optimized RoPE implementation
│   │   ├── ...                          # Other operators, including more fusion and optimized operators
│   │   ├── csrc/                        # C++ source code, providing high-performance C++ operator implementations
│   │   └── triton/                      # Triton operators, GPU operator implementations based on Triton
│   ├── optimizer/                       # Optimizers, providing distributed optimizer implementations
│   │   ├── adamw.py                     # AdamW optimizer, NPU-optimized AdamW optimizer implementation
│   │   ├── distrib_optimizer.py         # Distributed optimizer, supporting distributed training optimizers
│   │   └── optimizer.py                 # Optimizer base class, defining optimizer base interfaces
│   ├── op_builder/                      # Operator builder, providing operator compilation and registration functionality
│   ├── run/                             # Runtime module, providing training runtime support
│   │   ├── run.py                       # Run entry, entry file for training scripts
│   │   ├── gpt_dataset.patch            # GPT dataset patch, GPT dataset-related patches
│   │   ├── helpers.patch                # Helper function patches, helper function-related patches
│   │   └── initialize.patch             # Initialization patch, initialization flow-related patches
│   ├── te/                              # Transformer Engine, Transformer model acceleration engine
│   │   └── pytorch/                     # PyTorch adaptation, Transformer Engine implementation for PyTorch
│   │       ├── attention/               # Attention mechanisms, providing optimized attention implementations
│   │       ├── fp8/                     # FP8 support, providing FP8 quantization and computation support
│   │       ├── module/                  # Module definitions, providing Tensor Engine module implementations
│   │       ├── module_typing.py         # Module type hints, providing type definitions for modules
│   │       ├── permutation.py           # Permutation operations, providing tensor permutation operations
│   │       └── utils.py                 # Utility functions, providing Tensor Engine helper functions
│   └── tokenizer/                       # Tokenizer, providing text tokenization functionality
│       ├── tokenizer.py                 # Tokenizer core, core tokenizer implementation
│       └── build_tokenizer/             # Tokenizer building, providing tokenizer build and configuration
├── ci/                                  # CI testing scripts, providing CI/CD automated tests
│   ├── access_control_test.py           # Access control test, testing code access permission control
│   └── docker/                          # Docker build configuration, Docker image build-related configurations
│       └── Dockerfile                   # Docker image definition, defining Docker image build rules
├── docker/                              # Docker-related files, containing Docker configs and scripts
├── tests_extend/                        # Extended test cases, providing additional test cases
└── tools/                               # Auxiliary toolset, providing development and debugging tools
```
