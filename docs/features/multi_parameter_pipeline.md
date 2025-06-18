# PP支持多参数传递

## 背景与挑战

在深度学习的大模型分布式训练中，流水线并行（Pipeline Parallelism, PP）是一种将模型分割为多个阶段并在不同设备上并发执行以提高效率的技术。然而，在多模态场景下引入多参数传递支持时，PP面临特定的挑战：

- **通信部分的设计**：传统PP通常只涉及单一张量的传输，但在多参数传递的情况下，需要处理多个变量的传递。这不仅增加了通信复杂度，还要求对每个变量的shape、dtype等属性进行精确管理，这些属性往往与整体模型架构紧密相关，具有高度定制性。
- **前向传播的变量传递**：在前向计算过程中，不仅要根据定义的shape正确传递多个变量，还要确保每个阶段接收到的数据格式符合预期，这对数据流的设计提出了更高的要求。
- **反向传播的运算扩展**：对于反向传播，除了对首个输出进行梯度计算外，还需对其他所有输出进行相应的运算，确保整个训练过程的完整性和准确性。

## 解决方案

针对上述挑战，我们设计了以下解决方案，旨在使PP能够有效支持多参数传递：

- **优化的通信机制**：开发了一套高效的通信机制，支持多种类型和格式的数据传输。针对每个阶段的具体需求定制化配置传输参数。
- **增强的梯度计算逻辑**：改进了反向传播算法，使得系统可以自动识别并处理来自多个输出的梯度信息，保证每个输出都能参与到最终的权重更新中。

## 使用场景

本特性特别适用于以下场景：
- 需要处理大量多模态数据（如文本、图像、音频）的大型神经网络训练任务，并且流水线并行各个阶段传递多参数。

## 使用方法

**注意事项**:
- 用户需代码配置`args.pipeline_tensor_shapes`明确指定各阶段间传递的具体参数及其属性（如shape、dtype）。 
- args.pipeline_tensor_shapes配置参考`tests_extend/system_tests/multi_modal/multi_parameter_pipeline/pretrain_multi_parameter_pipeline_test.py`


**设置训练脚本参数**
- 支持PP场景
```shell
# PP >= 2
--pipeline-model-parallel-size ${PP} \
--use-multiparameter-pipeline-model-parallel \
```
- 支持VPP场景
```shell
# PP >= 2, num-layers-per-virtual-pipeline-stage不为None
--pipeline-model-parallel-size ${PP} \
--num-layers-per-virtual-pipeline-stage 1 \
--use-multiparameter-pipeline-model-parallel \
```
## 使用限制

1. 暂不兼容--moe-fb-overlap和dualpipev特性。

## 使用效果

采用PP支持多参数传递后，用户可以在保持高通信效率的同时，更灵活地处理复杂的多模态数据。