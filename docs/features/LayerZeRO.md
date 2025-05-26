# 分层ZeRO

## 背景与挑战

相比单一类型的LLM，多模态模型具有更复杂的架构，在多模态场景TP开销远大于LLM场景，且与DSP共用有更多额外开销，其中ZeRO3通信域过大带来额外开销较为严重。

## 解决方案

为了解决多模态场景下ZeRO3通信域过大带来额外开销过大的问题，采用分层ZeRO，其将优化器状态进行ZeRO1，micro-batch中使用ZeRO3参数进行参数重建；在运行时参数是重建参数的视图，用后销毁完整参数；每次优化器更新前将梯度同步到ZeRO1，更新后将ZeRO1参数同步到ZeRO3。达到和TP+zero1近似的内存节省效果，同时具有较好的通信隐藏。

### 解决思路

#### 通信组与参数划分

ZeRO1通信组 : 全部dp并行的通信组；
ZeRO3通信组: 部分dp并行的通信组, ZeRO1的通信组子集；
每个设备的LayerZeRO由ZeRO1和ZeRO3部分组成。

#### 参数划分管理

不同的zero3通信组内local rank一致的设备上应保存一致的参数分片；
所有的zero3分片在dp域内进行划分；
zero1的参数划分将所有管理的可训练参数拉平为一维张量, 然后均匀划分到zero1通信组设备；
一个zero1通信组内进行参数与梯度同步。

#### 参数同步与梯度同步

在运行时各个阶段注册hook函数，正确实现运行逻辑：
参数预取的重建和销毁；
梯度内存申请和销毁；
梯度的同步；
优化器梯度的同步。

## 使用场景

在多模态场景长序列激活值远大于模型参数时，开启分层ZeRO后，通过增大并行配置CP并行度，起到与TP+ZeRO1相似的内存节省效果，并加速长序列训练。

## 使用方法

在mindspeed中分层zero的入口是一个配置文件，通过生成配置文件，传入命令行参数即可使用该特性。
```
--layerzero \
--layerzero-config config.yml \
```
config.yml可配置项：
```
    zero3_size: int  # zero3通信组的大小，大于0的整数，一般在机内进行zero3
    transformer_layers:  Optional[Iterable[torch.nn.Module]] = None   # 被包装层的class层级name： module.submodule.class
    param_dtype: Optional[Literal["fp16", "bf16", "fp32"]] = "fp16"   # 混合精度相关：运行时参数精度
    reduce_dtype:  Optional[Literal["fp16", "bf16", "fp32"]] = "fp16" # 混合精度相关：运行时梯度精度
    ignored_modules:  Optional[Iterable[str]] = None                  # 模型不需要被分层ZeRO管理的部分。如果需要训练， 用户需自定义这部分的梯度与参数同步等。 对于模型不需要训练的部分，同时又不想参数分片，需要配置该选项。 非法的情况下默认失效为None
    offload_grads: bool=False    # 在梯度累积过程中是否offload完整梯度
    ckpt_load_path: str=None     # 分层ZeRO相同配置下的ckpt保存绝对路径， 用于断点续训
    autocast_input: bool = True  # 是否自动cast输入到混合精度
    autocast_output: bool = True # 是否cast输出为fp32
```

注意事项：
1. 支持TP，CP，PP(1F1B）并行，梯度累积，梯度Offload；
2. 模型的包装从MegatronModule替换为分层ZeRO，模型的开发中依赖MegatronModule的部分可能会失效；
3. 对于OpenSoraPlan1.3，需要额外设置ignored_modules，忽视一些模块(nn.Module类型)，避免vae部分和text_encoder部分的参数重建，该配置选项是从model提取对应的模块，只需要给出属性名即可；
4. CheckPoint序列化使用到了pickle组件，非授权用户不能拥有UnderFS存储目录及上层目录的写权限，否则可能造成CheckPoint被篡改引起pickle被篡改引起pickle反序列化注入的风险。

## 使用效果
使用MM-OpenSoraPlan1.3模型训练场景下，并行配置(CP=8+分层ZeRO)相较于基线(TP=8+ZeRO1)内存使用情况基本一致，实验组并行配置(CP=8+分层ZeRO)相比于基线(TP=8+ZeRO1)端到端性能提升9.7%。

