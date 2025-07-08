# compress-dense

## 背景与挑战

当训练使能不同PP并行策略时，pp_stage=0的情况下，多个warmup阶段所保存的激活值会导致首节点与尾节点内存不均衡。

当存在减小内存的需求时，可以针对首节点dense层激活值进行压缩，以达到节省内存的目的。

## 解决方案

激活值在训练过程中属于高频使用的内存对象，前向产生反向释放，并且不同的并行策略会导致激活值的生命周期产生复杂的变化，因此需要合理并安全的设计可异步的无损压缩方案。通过使能compress-dense的特性，可以利用多流异步的方案实现对大量Dense-层激活值小代价的压缩，具体方法如下图所示：

![Alt text](../../sources/images/compress_dense.png)

## 使用方法

当需要对Dense层MLP产生的激活值进行重计算时候，可通过在训练参数中增加`--compress-dense level1` 或者 `--compress-dense level0` 来使能特性，其中：

- `--compress-dense level0` 只进行Vector压缩，节省空间较少，但是带来较小的性能开销。
- `--compress-dense level1` 在进行Vector压缩的同时，将部分结果swap到CPU上，节省空间较多，但是带来较大的性能开销。

因此，建议在使用该特性时根据matmul的可掩盖时长来适当选择。


### 更一般的情况

事实上，针对任意激活值都可以使用我们的多流异步的方法进行压缩，以达到较小代价便可节省激活值内存的目的。前提是需要找到适合做多流掩盖的计算块，比如大颗粒的矩阵乘（针对tensor的压缩仅用到aicore中的vector计算单元），或者未掩盖的通信部分。并且需要存在重复多层的情况，以便进行层级错位的压缩和解压缩。我们的方法也被抽象为4个步骤，在代码合适的地方加入三个步骤即可：

```python
#1.Create an activation value compression management object and prepare a function for asynchronous compression.
self.ac = ActivationCompress(*args)
self.ac.compress_and_wait_decompress_async_for_previous_layer(x)

#matmul operator

#2.Prepare the asynchronous decompression function.
self.ac.decompress_and_wait_compress_async_for_previous_layer(x)

#some operator

#3.Record the forward and reverse sequence
self.ac.order_record(out)

```

以下是详细的使用示例：

> **注：要使用激活值压缩特性，建议修改范围集中在某个forward内部， 这样可以尽量避免跨层耦合与绑定。**



如下为一个简单模型的使用说明，假设我们希望对模型中第一个matmul产生的激活值进行无损压缩：

定义一个简单三层神经网络 `SimpleModel`，其中每一层由`SimpleLayer`构成，每个Layer为 Linear + relu + Linear 模式。可以通过简单的四个步骤，使能我们的多流异步激活值压缩。

第一步，在第一个Linear层之前，实例化一个`ActivationCompress`类并且调用`compress_and_wait_decompress_async_for_previous_layer` 函数，用于初始化compress tensor相关类以及开始压缩， 其中：

- `train_args`：是一个训练中全局参数对象，如果是在Megatron训练框架下，直接传入 `get_args()` 即可。
- `"simplelayer_ctm"` ：是任意字符串，是对激活值压缩类的命名，推荐使用 "xx_ctm" 的模式。


第二步：在Liner层后，增加 `decompress_and_wait_compress_async_for_previous_layer` 函数，用于异步压缩的等待以及解压缩的触发, 接受的参数即为要压缩的对象(放在下一层压缩，因此不会影响当前层的计算)。

第三步：插入 `order_record` 函数，用于记录压缩顺序。

```python
class SimpleLayer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        #1.Create an activation value compression management object and prepare a function for asynchronous compression.
        if not hasattr(self, "ac"):
            self.ac = ActivationCompress(train_args, "simplelayer_ctm")    
        self.ac.compress_and_wait_decompress_async_for_previous_layer(x)
        
        out = self.linear1(x)
        
        #2.Prepare the asynchronous decompression function.
        self.ac.decompress_and_wait_compress_async_for_previous_layer(out)
        
        out = F.relu(out)
        
        #3.Record the forward and reverse sequence
        self.ac.order_record(out)
        out = self.linear2(out)
        return out

class SimpleModel(nn.Module):

    def __init__(self, input_size=64, hidden_size=128, output_size=64):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dense_layer = SimpleLayer(input_size, hidden_size, output_size)
        self.model = nn.Sequential(self.dense_layer, self.dense_layer, self.dense_layer)

    def forward(self, x):
        out = self.model(x)
        return out
```

可参考文件：`MindSpeed/tests_extend/unit_tests/features/compress_dense/test_compress_tensor.py`


## 注意事项 

- 该特性匹配的 `CANN` 和 `torch_npu` 版本会在2025第二季度版本发布。
- 当前和 `recompute_activation_function` 特性暂不兼容。
