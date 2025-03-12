# PP支持多参数传递和动态形状

## 背景与挑战

在深度学习的大规模分布式训练中，流水线并行（Pipeline Parallelism, PP）通过将模型分割为多个阶段并在不同设备上并发执行来提高效率。然而，在处理复杂的多模态数据时，PP面临了新的挑战：

- **对于多参数传递**：传统PP通常只涉及单一张量的传输，但在多参数传递的情况下，需要处理多个变量的传递，这不仅增加了通信复杂度，还要求对每个变量的shape、dtype等属性进行精确管理 。
- **对于动态形状**：当输入数据的序列长度不固定时，传统的方法是将所有序列调整到统一长度，这导致了内存和计算资源的浪费 。

## 解决方案

为了应对这些挑战，开发了一系列优化措施：

- **多参数传递**：开发了一套高效的通信机制，支持多种类型和格式的数据传输，并改进了反向传播算法，使得系统可以自动识别并处理来自多个输出的梯度信息 。
- **动态形状**：引入对动态形状的支持，允许每个微批次中的序列保持其原始长度。这样可以通过在发送张量之前，提前通信张量的形状信息，在各个流水线阶段之间同步即将接收的数据形状，确保内存分配和预处理的准确性 。

## 使用场景

- **多参数传递**：适用于需要处理大量多模态数据的任务，如文本、图像和音频等大型神经网络训练任务，其中流水线并行的各个阶段都需要传递多参数 。
- **动态形状**：非常适合于处理文本长度差异很大的任务，比如文档分类和机器翻译，同时也增强了模型的泛化能力 。

## 使用方法

**注意事项**:
- 用户需代码配置`args.pipeline_tensor_shapes`明确指定各阶段间传递的具体参数及其属性（如shape、dtype）。 
- args.pipeline_tensor_shapes配置参考`tests_extend/system_tests/multi_modal/multi_parameter_pipeline/pretrain_multi_parameter_pipeline_test.py`


**设置训练脚本参数**
- # 支持PP场景
```shell
# 配置举例
# PP >= 2
--pipeline-model-parallel-size ${PP} \
--use-multiparameter-pipeline-model-parallel \
--variable-seq-lengths \
```
- # 支持VPP场景
```shell
# 配置举例
# PP >= 2, num-layers-per-virtual-pipeline-stage不为None
--pipeline-model-parallel-size ${PP} \
--num-layers-per-virtual-pipeline-stage 1 \
--use-multiparameter-pipeline-model-parallel \
--variable-seq-lengths \
```

## 使用效果
同时支持在流水线并行中各阶段间传递多个参数和处理变长输入数据。