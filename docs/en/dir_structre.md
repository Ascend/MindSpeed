# MindSpeed Project Directory Structure

```plaintext
MindSpeed/
├── README.md                            # Project documentation, introducing the features and usage of the MindSpeed Core acceleration library
├── docs/                                # Project documentation directory, containing feature documentation, user guides, and API documentation in both Chinese and English
├── mindspeed/                           # Core source code directory, containing the core implementation code of all acceleration libraries
│   ├── args_utils.py                    # Argument utility functions, providing auxiliary functions for argument parsing and validation
│   ├── arguments.py                     # Command-line argument definitions, defining training and configuration-related command-line arguments
│   ├── checkpointing.py                 # Checkpoint management, providing model checkpoint save and load functionality
│   ├── deprecated.py                    # Deprecated features module, marking deprecated APIs and features
│   ├── initialize.py                    # Initialization module, handling distributed environment initialization and configuration loading
│   ├── log_config.py                    # Logging configuration, defining log formats and output rules
│   ├── megatron_adapter.py              # Megatron-LM adapter, implements compatibility adaptation with the Megatron framework
│   ├── patch_utils.py                   # Patch utility, provides dynamic code patching and replacement functionality
│   ├── train.py                         # Training module, provides the entry point and main loop control for the training process
│   ├── utils.py                         # General utility functions, provides common helper functions for the project
│   ├── yaml_arguments.py                # YAML argument parsing, supports loading training configurations from YAML files
│   ├── auto_settings/                   # Auto-configuration subsystem, automatically optimizes training configurations based on the hardware environment
│   │   ├── auto_settings.py             # Auto-configuration main entry, coordinates the auto-configuration process across modules
│   │   ├── search_space.py              # Search space definition, defines the search range for tunable parameters
│   │   ├── config/                      # Configuration definitions, contains configuration templates for models and training
│   │   ├── mindspeed_adaptor/           # MindSpeed adapter, applies auto-configuration to the MindSpeed framework
│   │   ├── model/                       # Model definition, providing model architecture abstraction and configuration
│   │   ├── module/                      # Core modules, including modeling modules for communication, memory, and operators
│   │   ├── profile/                     # Profiling tools, providing performance analysis and diagnostic capabilities
│   │   └── utils/                       # General utilities, providing configuration and utility functions
│   ├── core/                            # Core functionality module, containing core capabilities such as parallelism strategies and memory management
│   │   ├── fp8_utils.py                 # FP8 utility functions, providing FP8 quantization and scaling related tools
│   │   ├── mindspeed_parallel_group.py  # Parallel group management, managing process groups for distributed training
│   │   ├── parallel_state.py            # Parallel state management, maintaining global parallel state information
│   │   ├── simple_parallel_cfg.py       # Simple parallel configuration, providing a simplified parallel configuration interface
│   │   ├── singleton_meta.py            # Singleton metadata, managing metadata for the singleton pattern
│   │   ├── tensor_parallel_y_union.py   # tensor parallel Y union configuration, supporting Y-axis tensor parallel union configuration
│   │   ├── training.py                  # training module configuration, providing configuration management for the training process
│   │   ├── weight_grad_store.py         # weight gradient storage management, optimizing storage strategies for weights and gradients
│   │   ├── context_parallel/            # context parallel, implementing distributed parallelism for sequence length
│   │   ├── datasets/                    # datasets, providing data loading and preprocessing functionality
│   │   ├── data_parallel/               # Data parallelism, implementing distributed training strategies for data parallelism
│   │   ├── distributed/                 # Distributed training, providing infrastructure for distributed training
│   │   ├── dist_checkpointing/          # Distributed checkpointing, supporting checkpoint management in distributed environments
│   │   ├── fusions/                     # Operator fusion, providing fused operators to improve computational efficiency
│   │   ├── hccl_buffer/                 # HCCL buffer, optimizing HCCL communication buffer management
│   │   ├── megatron_basic/              # Megatron basic adaptation, providing basic adaptation for the Megatron framework
│   │   ├── memory/                      # Memory management, providing memory optimization and management features
│   │   ├── models/                      # Model definitions, containing model architecture implementations such as GPT
│   │   ├── multi_modal/                 # Multi-modal, supporting training and inference of multi-modal models
│   │   ├── optimizer/                   # Optimizer, providing implementations of optimizers such as AdamW
│   │   ├── performance/                 # Performance optimization, providing performance analysis and optimization tools
│   │   ├── pipeline_parallel/           # Pipeline parallelism, implementing model pipeline parallelism strategy
│   │   ├── qat/                         # Quantization-aware training, supporting quantization during training
│   │   ├── qos/                         # QoS management, providing quality of service management and resource control
│   │   ├── tensor_parallel/             # Tensor parallelism, implementing tensor sharding parallelism strategy
│   │   └── transformer/                 # Transformer model, providing Transformer architecture implementation
│   ├── features_manager/                # Feature manager, centrally managing the registration and configuration of various optimization features
│   │   ├── feature.py                   # Feature management, defining feature base classes and interfaces
│   │   ├── features_manager.py          # Feature manager, providing feature registration, enabling, and disabling functions
│   │   ├── affinity/                    # Affinity management, optimizing process and device affinity configuration
│   │   ├── ai_framework/                # AI framework adaptation, adapting features for different AI frameworks
│   │   ├── auto_settings/               # Auto configuration features, providing auto configuration related features
│   │   ├── ckpt_acceleration/           # Checkpoint acceleration, accelerating checkpoint saving and loading
│   │   ├── compress/                    # Compression features, providing model and gradient compression functionality
│   │   ├── compress_dense/              # Dense compression, providing compression strategies for dense models
│   │   ├── context_parallel/            # Context parallel feature, manages context parallel related features
│   │   ├── custom_fsdp/                 # Custom FSDP, provides custom fully sharded data parallel
│   │   ├── data_parallel/               # Data parallel feature, manages data parallel related features
│   │   ├── disable_gloo_group/          # Disable Gloo group, disables Gloo backend communication group
│   │   ├── distributed/                 # Distributed feature, manages distributed training related features
│   │   ├── dist_train/                  # distributed training, providing feature support for distributed training
│   │   ├── functional/                  # functional features, providing functional programming related features
│   │   ├── fusions/                     # fusion operator features, managing fusion operator features
│   │   ├── hccl_buffer/                 # HCCL buffer features, optimizing HCCL buffer usage
│   │   ├── llava/                       # LLaVA model, providing LLaVA multimodal model support
│   │   ├── megatron_basic/              # Megatron basic features, providing basic features for the Megatron framework
│   │   ├── memory/                      # Memory features, managing memory optimization related features
│   │   ├── moe/                         # MoE features, providing support for Mixture of Experts models
│   │   ├── optimizer/                   # Optimizer features, managing optimizer related features
│   │   ├── pipeline_parallel/           # Pipeline parallel features, managing pipeline parallel related features
│   │   ├── qat/                         # QAT feature, manages quantization-aware training related features
│   │   ├── qos/                         # QoS feature, manages quality of service related features
│   │   ├── recompute/                   # Recomputation feature, provides activation recomputation functionality
│   │   ├── tensor_parallel/             # Tensor parallelism feature, manages tensor parallelism related features
│   │   ├── tokenizer/                   # Tokenizer feature, manages tokenizer related features
│   │   └── transformer/                 # Transformer features, managing Transformer model features
│   ├── fsdp/                            # FSDP fully sharded data parallelism, implementing sharding strategies such as ZeRO-3
│   │   ├── mindspeed_parallel_engine.py # Parallel engine, providing FSDP parallel engine implementation
│   │   ├── parallel_engine_config.py    # Parallel engine configuration, configuring FSDP parallel engine parameters
│   │   ├── distributed/                 # Distributed, providing FSDP distributed communication functionality
│   │   ├── memory/                      # memory, provides FSDP memory management functionality
│   │   ├── quantization/                # quantization, provides FSDP quantization support
│   │   └── utils/                       # utils, provides FSDP-related utility functions
│   ├── functional/                      # functional interface, provides NPU-related functional APIs
│   │   ├── npu_datadump/                # NPU data dump, provides NPU data export and debugging functionality
│   │   ├── npu_deterministic/           # NPU deterministic computation, ensuring deterministic NPU computation
│   │   ├── profile/                     # Performance analysis, providing performance analysis and diagnostic tools
│   │   ├── profiler/                    # Profiler, providing detailed performance analysis functionality
│   │   └── tflops_calculate/            # TFLOPS calculation, computing training throughput and performance metrics
│   ├── lite/                            # Lite lightweight version, providing lightweight training support
│   │   ├── mindspeed_lite_config.py     # Lite configuration, providing configuration management for the Lite version
│   │   ├── mindspeed_lite.py            # Lite model, providing model implementation for the Lite version
│   │   ├── distributed/                 # Distributed, providing distributed support for the Lite version
│   │   ├── memory/                      # Memory, providing memory management for the Lite version
│   │   ├── ops/                         # Operators, providing operator implementation for the Lite version
│   │   └── utils/                       # Utilities, providing utility functions for the Lite version
│   ├── mindspore/                       # MindSpore framework adaptation, providing MindSpore framework support
│   │   ├── mindspore_adapter.py         # MindSpore adapter, implementing adaptation for the MindSpore framework
│   │   ├── core/                        # Core module, providing MindSpore core functionality
│   │   ├── model/                       # Models, providing MindSpore model implementations
│   │   ├── ops/                         # ops, provides MindSpore operator implementations
│   │   ├── optimizer/                   # optimizer, provides MindSpore optimizer implementations
│   │   ├── op_builder/                  # op_builder, provides MindSpore operator building functionality
│   │   └── third_party/                 # third_party, integrates third-party MindSpore libraries
│   ├── model/                           # model, provides general model definitions and interfaces
│   ├── moe/                             # MoE, provides mixture of experts model implementation
│   ├── multi_modal/                     # Multi-modal support, provides multi-modal model training support
│   │   └── conv3d/                      # 3D convolution, provides 3D convolution operator implementation
│   ├── ops/                             # Operator library, provides high-performance fused operators
│   │   ├── dropout_add_layer_norm.py    # Dropout+LayerNorm fusion, fuses Dropout and LayerNorm operations
│   │   ├── dropout_add_rms_norm.py      # Dropout+RMSNorm fusion, fuses Dropout and RMSNorm operations
│   │   ├── ffn.py                       # FFN operator, feedforward neural network operator implementation
│   │   ├── fusion_attention_v2.py       # Fused attention v2, optimized attention mechanism implementation
│   │   ├── gmm.py                       # GMM operator, grouped matrix multiplication operator
│   │   ├── gmm_mxfp8.py                 # MXFP8 GMM operator, GMM operator with MXFP8 support
│   │   ├── grouped_matmul.py            # Grouped matrix multiplication, efficient grouped matrix multiplication implementation
│   │   ├── npu_apply_fused_adamw_v2.py  # Fused AdamW v2, NPU-optimized AdamW optimizer
│   │   ├── npu_bmm_reduce_scatter_all_to_all.py  # BMM+ReduceScatter+AllToAll, fused communication operator
│   │   ├── npu_matmul_add.py            # MatMul+Add fusion, fuses matrix multiplication and addition operations
│   │   ├── npu_rotary_position_embedding.py  # Rotary position embedding, NPU-optimized RoPE implementation
│   │   ├── ...                          # Other operators, including more fused operators and optimized operators
│   │   ├── csrc/                        # C++ source code, providing high-performance operators implemented in C++
│   │   └── triton/                      # Triton operators, GPU operator implementations based on Triton
│   ├── optimizer/                       # Optimizer, providing distributed optimizer implementations
│   │   ├── adamw.py                     # AdamW optimizer, NPU-optimized implementation of the AdamW optimizer
│   │   ├── distrib_optimizer.py         # Distributed optimizer, supporting distributed training optimizers
│   │   └── optimizer.py                 # Optimizer base class, defining the basic interface for optimizers
│   ├── op_builder/                      # Operator builder, providing operator compilation and registration functionality
│   ├── run/                             # Runtime module, providing training runtime support
│   │   ├── run.py                       # Run entry point, the entry file for training scripts
│   │   ├── gpt_dataset.patch            # GPT dataset patch, patches related to GPT datasets
│   │   ├── helpers.patch                # Helper function patch, patches related to helper functions
│   │   └── initialize.patch             # Initialization patch, patches related to the initialization process
│   ├── te/                              # Transformer Engine, acceleration engine for Transformer models
│   │   └── pytorch/                     # PyTorch adaptation, Transformer Engine implementation for the PyTorch framework
│   │       ├── attention/                # Attention mechanism, providing optimized attention implementation
│   │       ├── fp8/                      # FP8 support, providing FP8 quantization and computation support
│   │       ├── module/                   # Module definition, providing Tensor Engine module implementation
│   │       ├── module_typing.py          # Module type hints, providing type definitions for modules
│   │       ├── permutation.py            # Permutation operation, providing tensor permutation operations
│   │       └── utils.py                  # Utility functions, providing Tensor Engine utility functions
│   └── tokenizer/                       # Tokenizer, providing text tokenization functionality
│       ├── tokenizer.py                 # Tokenizer core, the core implementation of the tokenizer
│       └── build_tokenizer/             # Tokenizer builder, providing tokenizer building and configuration
├── ci/                                  # Continuous integration test scripts, providing CI/CD automated testing
│   ├── access_control_test.py           # Access control test, testing code access permission control
│   └── docker/                          # Docker build configuration, Docker image build related configuration
│       └── Dockerfile                   # Docker image definition, defining Docker image build rules
├── docker/                              # Docker related files, containing Docker configuration and scripts
├── tests_extend/                        # Extended test cases, providing additional test cases
└── tools/                               # Auxiliary toolset, providing development and debugging utilities
```
