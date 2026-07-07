# Out-of-the-Box Optimization - Parallel Strategy Auto Settings for Large Models

## Background and Challenges

As the number of configurable parameters for large model parallel training continues to grow—such as DP, TP (and SP), PP, ZERO, VPP, CP, EP, mbs, and recomputation—the impact of various configurations on memory and performance becomes increasingly complex, making manual tuning more and more difficult. Consequently, the industry has begun exploring automatic tuning methods. The main approach involves white-box or gray-box modeling based on the structure of the network model, and then searching for configuration parameters under the guidance of this modeling combined with some profiling.

However, these methods typically have the following two shortcomings:

- White-box or gray-box modeling **makes assumptions about the structure of the model**, but many users modify their models, making it difficult for such modeling to capture these changes. For example, a simple modification like GQA/MQA can cause memory estimation deviations in this type of modeling.
- **The scale of profiling is the same as the actual workload scale**. When conducting large-scale training (e.g., thousands of cards), the overhead of profiling becomes very large.

Therefore, we have designed and developed the feature of auto settings. Compared with existing automatic tuning solutions in the industry, this feature is entirely based on profiling analysis, requires no assumptions about the network structure, and supports using small-scale profiling to estimate optimal training configurations on larger clusters.

## Solution

The auto settings feature relies entirely on data modeling derived from profiling, decoupling it from changes in the network structure, and supports inferring configurations for large-scale clusters from small-scale clusters (for example, a dual-node cluster). Three approaches are provided: white-box search, black-box search, and mixed search.

### White-Box Search

- **Phase 1:** Launch auto settings on a small number of machines. This feature scales down the network size, generates multiple profiling configurations, and automatically launches them multiple times. These profiling runs are primarily used for black-box analysis, such as analyzing which tensors are partitioned, how the shapes of certain operators change, and which operators are added or removed when the configuration changes. After profiling is complete, the result files are parsed to extract the information needed for subsequent black-box modeling.
- **Phase 2:** Perform black-box modeling based on the profiling results. For memory, it automatically analyzes how each tensor is partitioned under different configurations. For performance, it infers how operators are added or removed and how their shapes change with different configurations, and regresses the efficiency of intra-node and inter-node communication. In addition to basic performance and memory modeling, it also analyzes the performance and memory of each candidate recomputation module, enabling the subsequent search to estimate which modules should be selected for recomputation and their impact on performance and memory.
- **Phase 3:** Search for configurations based on the models derived in Phase 2, providing the expected performance and memory for each configuration. This step also relies on an operator performance knowledge base to query the performance of operators with different shapes. Any unseen operators generated during profiling are added to the operator performance knowledge base. If the proportion of operators covered by the knowledge base for a given configuration falls below a threshold, an additional profiling run is launched. This profiling can also simulate large-scale scenarios with smaller setups by simultaneously reducing the network scale and parallelism parameters to obtain operators with the same shapes. If the operator coverage is insufficient to infer performance, the performance of the few uncovered operators is estimated through regression. After the search is complete, the top three configurations with the best performance that meet memory requirements are recommended.

### Black-Box Search

- **Phase 1:** Same as Phase 1 of white-box search, performs modeling and pruning the search space.
- **Phase 2:** Profiles each configuration in the search space to obtain the corresponding memory usage, operator time, and communication time, thereby selecting the most suitable configuration.

### Mixed Search

Balances the efficiency of white-box search with the accuracy of black-box search, providing a comprehensive search solution.

- **Phase 1:** White-box search selects M groups of search configurations.
- **Phase 2:** Applies the black-box search scheme to the N groups of search configurations from Phase 1 to identify the top-k configurations.

Supported models:

- [x] llama2-7b
- [x] mixtral-8*7b
- [x] gpt3-15b

Supported features:

- [x] DP
- [x] TP
- [x] Megatron-SP
- [x] PP
- [x] ZeRO1
- [x] VPP
- [x] CP (ring attention)
- [x] EP (Deepspeed-MOE)
- [x] MicroBatchSize
- [x] Token rearrangement
- [x] Recomputation
- [x] MC2

Planned features:

- [ ] ZeRO2
- [ ] EP (Megatron-MOE)
- [ ] swap-attention
- [ ] Activation recomputation
- [ ] MoE All2All overlap comm

## Usage

Add the following configuration to the parameter list of the training script to enable the Auto settings feature:

```bash
--auto-settings \                                 # Enable the Auto settings featuretings feature
--auto-settings-type mixed \                    # Search scheme, supporting three types: [black, white, mixed]e types: [black, white, mixed]
--auto-settings-work-dir ./auto_settings_dir \    # Working directory where profiling and other files will be savede profiling and other files will be saved
--auto-settings-ranks 16 \                        # Number of cards to search, minimum 16 cardsds to search, minimum 16 cards
--auto-settings-log-level debug \                 # Auto settings log level, options: warning, info, debugnfo, debug
--target-nnodes $NNODES \                         # Number of nodes launched for profiling, consistent with the baseline training scriptr profiling, consistent with the baseline training script
--nproc-per-node $GPUS_PER_NODE \               # Number of processes running on each node, typically equal to the number of devices per node, consistent with the baseline training scripth node, typically equal to the number of devices per node, consistent with the baseline training script
--master-addr $MASTER_ADDR \                    # Master node IP, keep consistent with the baseline training scriptonsistent with the baseline training script
--master-port 6005                           # Master node port, set a port different from the baseline scriptt a port different from the baseline script
--node-rank $NODE_RANK \                        # Keep consistent with the baseline training scripttent with the baseline training script
```

## Environment Variables

The following environment variables are switches used by Auto settings to control staged profiling. They are **for internal use by auto settings only** and **must not** be set during normal training workflows.

**Auto settings will set the following environment variables in an isolated process environment and will not export them to the user environment.**

- "OOTB_OPTIMIZER_MODIFIED_ARGV_PATH=${WORK_DIR}/auto_settings_modified_argv.json": The file location for modifying the profiling launch configuration parameters.
- "OOTB_OPTIMIZER_PARSE_ARGS=TRUE": Obtains hardware-related information and model parameters.
- "OOTB_OPTIMIZER_PARSE_MODEL=TRUE": Obtains the model structure
- "OOTB_OPTIMIZER_PROFILING=TRUE": Obtains complete profiling information and adaptive recomputation profiling information
- "OOTB_OPTIMIZER_PROFILING_BLACK=TRUE": Obtains complete profiling information and adaptive recomputation profiling information, used for black-box search scenarios
