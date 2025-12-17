# SmartSwap

## 背景与挑战

在用户训练过程中产生的OOM问题，现有的内存方案主要为重计算和Swap两个方法。重计算会增加计算开销，而Swap路线需要用户自己编写和控制异步换入换出时机和内存管理，增加较多的使用成本。

## 解决方案

为了在最大限度地利用计算设备显存的同时，提高模型训练的性能，我们支持通过自适应迭代生成Swap策略，这一特性称为SmartSwap。

此功能通过数据采样，策略生成，策略执行等流程的循环迭代，选择有限次数验证下的最优策略。
在迭代中分为3个阶段。

- WarmUp阶段，仅执行数据采样。采集Tensor生命周期信息供后续分析。此时OOM时会通过覆盖底层的内存异常，使得模型能够继续运行。
- SearchPolicy阶段，执行数据采样和策略执行。 在策略生成中，包括候选内存过滤，内存策略生成，内存模拟排布等步骤。
- Stable阶段，仅执行策略执行。在策略执行中，通过多流异步执行内存Swap，掩盖对计算流的耗时影响。

![smart_swap_flowchart](../../sources/images/smart_swap_flowchart.png)

## 使用场景

1. OOM场景：当前训练配置下，出现OOM报错；可开启此功能，将OOM报错拦截，自动生成Swap策略，使训练在可用最大显存内运行。
2. 非OOM场景：当前训练配置下，未出现OOM报错；可开启此功能，根据配置文件中的减少显存值，自动生成Swap策略，使训练在指定显存内运行。
3. 重计算的替代场景：减少模型代码中的重计算生效范围，节省重计算过程。

## 使用方法

1. 在训练脚本中添加此功能的使能参数：`--smart-swap`。
2. （可选）修改此功能的配置文件`mindspeed/core/memory/smart_swap/swap_policy_config.py`进行调试。
3. 新增v2版本，简化用户使用。
```python
self.policy_v2 = True  # True是开启，False是关闭
self.policy_pref = SwapPolicyPref.BETTER_PERFORMANCE  # BETTER_PERFORMANCE是选择activation，BETTER_MEMORY_SAVING是选择activation和optimizer两个部分
self.swap_bucket_size = -1  # 控制swap每层tensor的大小，单位Bytes。默认-1，小于零即视为全选。
self.num_attn_layers_per_stage = 1  # 指定SwapStage划分粒度。默认1，即每个SwapStage包含一个attention layer。
```

## 使用效果

1. 通过减少TP和PP数，获得性能收益；例如在llama2（8p，pp1，seqlen 8K，layer 32），将tp8改为tp1，性能收益25%；
2. 通过关闭或者部分关闭全重计算，获得性能收益；例如在llama2（8p，pp1，seqlen 16K，layer 40），将全重计算关闭，性能收益28%；

## 注意事项

1. SmartSwap适配静态序列场景；暂未适配动态场景，例如MOE类场景。
2. SmartSwap将占用Host内存，例如单机8卡，若每卡均换出`30 GB`到Host，则单机至少需要Host内存`8*30=240 GB`。
3. 针对自定义编译cpp算子，提供手动添加采集Hook的方式，用户需手动修改自定义算子的cpp代码和编译代码，并修改模型训练脚本的`LD_LIBRARY_PATH`环境变量。举例如下。

- 自定义算子的cpp代码

```cpp
// 举例: mindspeed/ops/csrc/cann/gmm.cpp
#include "NPUSwapManager.h"  // NOTE: 添加头文件
// ...
std::vector<at::Tensor> npu_gmm(...)
{
    // NOTE: 添加算子调用前置位的Hook，设置算子名称和算子输入；若输出和输入相同，可在前置位设置。
    c10_npu::swap::NPUSwapManager::GetInstance().BeginHook("gmm_forward");
    c10_npu::swap::NPUSwapManager::GetInstance().TensorHook(x);
    c10_npu::swap::NPUSwapManager::GetInstance().TensorHook(weight);

    // ...

    ACLNN_CMD(aclnnGroupedMatmulV2, x_, weight_, bias_, scale_real, offset_real, antiquant_scale_real,
              antiquant_offset_real, group_list_real, split_item_value, group_type_value, result);

    // NOTE: 添加算子调用后置位的Hook，设置算子输出和结束标志；
    c10_npu::swap::NPUSwapManager::GetInstance().TensorHook(y);
    c10_npu::swap::NPUSwapManager::GetInstance().PostHook();
    c10_npu::swap::NPUSwapManager::GetInstance().EndHook();

    // ...
    return y;
}
```

- 自定义算子的编译代码

```python
# 举例: mindspeed/op_builder/gmm_builder.py

class GMMOpBuilderPublic(MindSpeedOpBuilder):
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def sources(self):
        return ['ops/csrc/cann/gmm.cpp', 'ops/csrc/flop_counter/flop_counter.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        paths.append('ops/csrc/pluggable_allocator/smart_swap')  # NOTE: 添加smart_swap的头文件路径
        return paths

    # ...

    # NOTE: 添加编译链接选项
    def extra_ldflags(self):
        flags = super().extra_ldflags()
        import os
        root_extensions_dir = os.environ.get('TORCH_EXTENSIONS_DIR')
        flags += [
            '-L' + f'{root_extensions_dir}/smart_swap/', '-lsmart_swap',
        ]
        return flags
```

- 模型训练脚本的环境变量修改

```bash
# 举例: pretrain_xxx.sh
export TORCH_EXTENSIONS_DIR="/home/xxx/exts/"
export LD_LIBRARY_PATH=${TORCH_EXTENSIONS_DIR}/smart_swap/:${LD_LIBRARY_PATH}
```
