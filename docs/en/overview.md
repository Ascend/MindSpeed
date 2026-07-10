# MindSpeed Overview

MindSpeed is a high-performance acceleration library built for the Ascend platform. It consists of four major components: the MindSpeed Core affinity acceleration module, the MindSpeed LLM suite, the MindSpeed MM suite, and the MindSpeed RL suite.

MindSpeed delivers strong support for foundation model training with outstanding performance and a deeply optimized algorithmic architecture. With MindSpeed, users can fully tap into the high-performance computing capabilities of Ascend devices. Therefore, they can accelerate foundation model training and serve users in the AI field faster and better.

## Overall Architecture

**Figure 1**  MindSpeed overall architecture

![alt text](figures/01_architecture_mindspeed.png)

**Table 1**  Component overview

|Component|Description|
|--|--|
|MindSpeed Core|A foundation model acceleration module built on Ascend devices. It provides optimizations across computation, memory, communication, and parallelism, and supports acceleration features for long sequences, mixture of experts (MoE), and other scenarios.|
|MindSpeed LLM|A large language model (LLM) suite built on the Ascend ecosystem. It aims to provide an end-to-end LLM training solution, including distributed pre-training, distributed instruction tuning, and the corresponding development toolchain, such as multimodal data preprocessing, weight conversion, online inference, and baseline evaluation. It covers mainstream LLMs in the industry.|
|MindSpeed MM|An Ascend multimodal (MM) foundation model suite for large-scale distributed training. It focuses on multimodal generation and multimodal understanding, and provides an end-to-end training process for multimodal foundation models, including multimodal data preprocessing, training and fine-tuning, online inference, and performance evaluation. It covers mainstream multimodal foundation models in the industry.|
|MindSpeed RL|Provides reinforcement learning (RL) core acceleration capabilities for ultra-large Ascend clusters, including training and inference on the same devices, asynchronous pipeline scheduling, and communication for heterogeneous training and inference partitioning.|

## Key Features

- MindSpeed Core:
    - Parallel algorithm optimization: Supports multidimensional parallel strategies such as model parallelism, optimizer parallelism, expert parallelism, and long-sequence parallelism. It provides affinity optimizations for the Ascend hardware and software architecture and significantly improves cluster training performance and efficiency.
    - Memory resource optimization: Provides memory compression, memory reuse, memory swapping, and differentiated recomputation techniques to maximize memory use, alleviate memory bottlenecks, and improve training efficiency.
    - Communication performance optimization: Uses strategies such as fused computation and communication and computation-communication overlap, together with an efficient compute-network collaboration mechanism, to greatly improve compute utilization, reduce communication latency, and optimize overall training performance.
    - Computation performance optimization: Integrates a high-performance fused operator library and combines it with Ascend-aware computation optimizations to fully unleash Ascend computing power and significantly improve computation efficiency.
    - Differentiated capability support: Provides differentiated capabilities for long sequences, weight saving, and automatic search for parallel strategies.

- MindSpeed LLM:
    - Mainstream LLMs: Supports more than 100 mainstream LLMs, such as the Qwen3, DeepSeek, and Mamba2 series. It covers LLM architectures such as Dense, MoE, and structured state space model (SSM), and provides high-performance training scripts tailored to the Ascend architecture, ready to use out of the box.
    - Distributed pre-training: Supports distributed pre-training and provides data preprocessing solutions plus multidimensional parallel strategies including tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), context parallelism (CP), and expert parallelism (EP).
    - Distributed instruction tuning: Supports industry-mainstream full-parameter fine-tuning, low-rank adaptation (LoRA), and quantized LoRA (QLoRA) training algorithms, and provides performance and memory optimization methods for fine-tuning.
    - Model weight conversion: Supports weight conversion between the Megatron and Hugging Face formats and independent or merged conversion of LoRA fine-tuning weights.
    - Online inference and evaluation: Supports distributed online model inference and online evaluation on public benchmark datasets.

- MindSpeed MM:
    - Mainstream multimodal models: Supports mainstream multimodal understanding models such as the InternVL and QwenVL series, mainstream video generation models such as the OpenSoraPlan, CogVideoX, HunyuanVideo, Wan2.1, and Wan2.2 series, and mainstream text-to-image models such as FLUX, SANA, HiDream, and the Qwen Image series. It provides high-performance training scripts tailored to the Ascend architecture, ready to use out of the box.
    - Distributed training: Supports distributed full-parameter fine-tuning and provides data preprocessing solutions plus multidimensional parallel strategies including heterogeneous PP, TP, sequence parallelism (SP), and fully sharded data parallel 2 (FSDP2). Through fine-grained selective recomputation and Async-offload, it makes full use of heterogeneous resources such as memory, host-to-device (H2D), and device-to-host (D2H) to optimize performance for very long sequences. It supports LoRA fine-tuning and direct preference optimization (DPO) training.
    - Model weight conversion: Supports weight conversion between the Megatron and Hugging Face formats and independent or merged conversion of LoRA fine-tuning weights.
    - Online inference and evaluation: Supports distributed online model inference and online evaluation on public benchmark data.

- MindSpeed RL:
    - Memory resource optimization: Supports shared devices for training and inference, switching between the optimal parallel modes for training and inference, and refined memory management techniques for training and inference.
    - Computation flow orchestration optimization: Supports asynchronous Replay Buffer, decouples data dependencies, enables asynchronous training and task pipeline overlap, and greatly improves end-to-end throughput.
    - Load balancing optimization: Supports data load balancing for variable-length sequences and significantly improves compute utilization and end-to-end throughput.
    - Large-scale RL optimization: Supports high-performance training for hundred-billion-parameter MoE long sequences.

## More Information

For more information about MindSpeed, see the online course: [MindSpeed](https://www.hiascend.com/edu/courses?activeTab=MindSpeed).
