# 开箱优化-大模型并行策略自动搜索 Auto settings 特性说明

## 问题背景

随着大模型并行训练可配置的参数越来越多，例如DP、TP（以及SP）、PP、ZERO、VPP、CP、EP、mbs、重计算等，内存和性能受到各种配置的影响变得越来越复杂，人工调优变得越来越困难。于是，业界开始尝试一些自动调优的方法，主要思路是基于网络模型的结构进行白盒或者灰盒建模，在建模的指导下结合一些profiling，进行配置参数的搜索。

但是，这些方法通常存在以下两个不足之处：

- 白盒或灰盒的建模**对网络模型的结构进行了假设**，而很多用户都会对模型做出修改，这类建模难以捕捉到模型的变化。例如，仅仅是GQA/MQA的修改，就会让此类建模的内存出现偏差。
- **profiling的规模和实际的负载规模相同**，当要进行大规模（如千卡）的训练时，profiling的开销会变得很大。

因此，我们设计并开发了一种Auto settings的特性，该特性和业界已有的自动调优方案相比，完全基于profiling的分析，无需对网络的结构做出假设，并且支持”以小仿大“，即以小规模的profiling预估更大集群上的较优训练配置。

## 解决方案

Auto settings特性完全依赖由profiling得出的数据建模，与网络结构的变化解耦，并且支持在小规模集群（如双机）上推测大规模集群的配置。整体提供三个方案：白盒搜索、黑盒搜索、混合搜索

### 白盒搜索
- **阶段1:** 用少量机器拉起auto settings，该特性会裁剪网络大小，并生成多个profiling的配置，自动多次拉起。这些profiling主要是用作黑盒分析，例如分析配置变化时，哪些tensor会被切分，哪些算子的shape会如何变化，会增加或减少哪些算子等。profiling结束后会对结果文件进行解析，提取出后续黑盒建模需要的信息。
- **阶段2:** 依据profiling结果进行黑盒建模。内存方面会自动分析各个tensor在不同配置下的切分情况，性能方面会推断算子随不同配置的增减和shape变化，并回归出机内和机间通信的效率。除了基础的性能和内存建模之外，还会分析各个候选重计算模块的性能和内存，从而可以在后续搜索中预估应该选择哪些模块做重计算，以及其对性能和内存的影响。
- **阶段3:** 根据阶段2得出的建模，进行配置的搜索，给出每个配置下预期的性能和内存。这一步还会依赖一个算子性能知识库，从中查询不同shape的算子的性能。profiling产生的没见过的算子都会被添加到算子性能知识库中。如果某个配置下算子性能知识库覆盖的算子比例小于阈值，则会额外拉起一组profiling，该profiling仍然可以以小仿大，通过同时缩小网络的规模和并行参数，从而得到相同shape的算子。如果算子性能知识库覆盖的算子比例不足以推测算子性能，则未覆盖到的少量算子会通过回归来估计性能。搜索结束后会推荐出内存充足的性能最好的三组配置。

### 黑盒搜索
- **阶段1：**：和白盒搜索阶段1一样，用于建模，对搜索空间剪枝。
- **阶段2：**：对搜索空间每个配置都进行profiling，获取对应的内存占用和算子、通信耗时，从而选择最合适的配置。

### 混合搜索
兼顾白盒搜索的效率和黑盒搜索的精度，提供一个综合搜索方案。
- **阶段1：**：白盒搜索选择M组搜索配置。
- **阶段2：**：对阶段1中N组搜索配置使用黑盒搜索方案，搜索出topk组配置。

已支持的模型:
- [x] llama2-7b
- [x] mixtral-8*7b
- [x] gpt3-15b

已支持的特性:

- [x] DP
- [x] TP
- [x] Megatron-SP
- [x] PP
- [x] ZeRO1
- [x] VPP
- [x] CP (ring attention)
- [x] EP (Deepspeed-MOE)
- [x] MicroBatchSize
- [x] Token重排
- [x] 重计算
- [x] MC2

未来计划支持的特性:

- [ ] ZeRO2
- [ ] EP (Megatron-MOE)
- [ ] swap-attention
- [ ] 激活函数重计算
- [ ] MoE All2All overlap comm

## 使用方法

在训练脚本的参数列表中加入以下配置开启 Auto settings 特性:

```bash
--auto-settings \                                 # 开启 Auto settings 特性
--auto-settings-type mixed \                    # 搜索方案，支持【black, white, mixed】三种
--auto-settings-work-dir ./auto_settings_dir \    # 工作目录，在此会保存profiling等文件
--auto-settings-ranks 16 \                        # 需要搜索的卡数，最低16卡
--auto-settings-log-level debug \                 # Auto settings log记录等级，可选warning, info, debug
--target-nnodes $NNODES \                         # Profiling拉起的节点数，与基线训练脚本保持一致
--nproc-per-node $GPUS_PER_NODE \               # 每个节点上运行的进程数，一般与单节点卡数相同，与基线训练脚本保持一致
--master-addr $MASTER_ADDR \                    # 主节点IP，与基线训练脚本保持一致
--master-port 6005                           # 主节点端口，设置一个与基线脚本不同的端口
--node-rank $NODE_RANK \                        # 与基线训练脚本保持一致
```

## 环境变量
以下环境变量为 Auto settings 控制阶段性 Profiling 所用环境变量开关，**仅为 Auto settings 内部使用**，**禁止**在正常训练流程中设置

**Auto settings会在一个隔离的进程环境中设置以下环境变量，不会export至用户环境中**
- "OOTB_OPTIMIZER_MODIFIED_ARGV_PATH=${WORK_DIR}/auto_settings_modified_argv.json": 修改Profiling拉起配置参数的文件位置
- "OOTB_OPTIMIZER_PARSE_ARGS=TRUE": 获取硬件相关信息及模型参数
- "OOTB_OPTIMIZER_PARSE_MODEL=TRUE": 获取模型结构
- "OOTB_OPTIMIZER_PROFILING=TRUE": 获取完整Profiling信息及自适应重计算Profiling信息
- "OOTB_OPTIMIZER_PROFILING_BLACK=TRUE": 获取完整Profiling信息及自适应重计算Profiling信息，用于黑盒搜索的场景
