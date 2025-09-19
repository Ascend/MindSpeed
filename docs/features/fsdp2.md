# FSDP2

## 背景与挑战

PyTorch的完全分片数据并行（FSDP）旨在提供一个高性能的即时执行模式实现，包含通信分桶和通信/计算掩盖功能。该API通过将一组参数展平拼接成FlatParameter来表示通信桶。然而，这种FlatParameter设计导致难以对桶内单个参数实施差异化操作（如参数冻结、精度转换等），损害了组合灵活性，同时也使内部实现复杂化（例如状态字典逻辑长达数千行代码且需要额外通信）。

## 解决方案

基于上述局限性，FSDP2移除了FlatParameter，采用沿0维分片的DTensor表示分片参数，支持对单个参数的便捷操作、免通信的分片状态字典，以及更简化的初始化流程。

## 使用方法

在mindspeed中FSDP2的入口是一个配置文件，通过生成配置文件，传入命令行参数即可使用该特性。
```
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 设置不能为1
--use-torch-fsdp2 \
--fsdp2-config-path ./fsdp2_config.yaml \
--ckpt-format torch_dist \
--untie-embeddings-and-output-weights \
# 注意不能打开分布式优化器
```
fsdp2_config.yaml的配置项如下：
```
sharding_size: int	# 分片组大小，表示每个参数分片组的NPU数量
sub_modules_to_wrap: Optional[Iterable[torch.nn.Module]] = None	# 需要进行FSDP包装的模块类列表，需通过绝对路径引入：例如：mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
reshard_after_forward: Union[bool, int] = True	# 前向计算后立即重新分片参数
param_dtype: bf16	# 参数存储精度
reduce_dtype: fp32	# 梯度通信精度
cast_forward_inputs: bool = True	# 自动转换前向输入到计算精度
ignored_modules: Optional[Iterable[torch.nn.Module]] = None # 排除FSDP管理的模块类列表, 需要通过绝对路径引入：例如：mindspeed_mm.models.ae.base.AEModel
offload_to_cpu: bool = False # 将权重，梯度以及优化器状态卸载到cpu
pin_memory: bool = True  # 只有当offload_to_cpu为True时才会生效
num_to_forward_prefetch: int  # 指定前向计算预取（forward prefetch）的层数，默认值为0
recompute_modules: Optional[Iterable[torch.nn.Module]] = None # 需要进行重计算的模块类列表，需通过绝对路径引入：例如：mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
```

## 使用效果
针对Llama-7B，FSDP2相比FSDP1实现了更高的MFU，峰值内存降低7%，且保持相同的损失曲线。

## 注意事项
1、当开启fsdp2训练时，需关闭分布式优化器及其相关配置

2、当开启fsdp2训练时，模型权重的保存格式`ckpt-format`仅支持`torch_dist`或`torch_dcp`

- 配置为`torch_dist`，模型需通过继承`MegatronModule`或自定义来实现`sharded_state_dict()`方法；同时需保证模型中所有权重的0维size均大于或等于sharding_size

- 配置为`torch_dcp`，模型需通过继承`MegatronModule`或自定义来实现`state_dict_for_save_checkpoint()`方法，并且其返回的权重字典需要与`model.state_dict()`的返回值一致

3、当开启fsdp2训练时，需关闭重计算的相关配置，包括：`--recompute-granularity`、`--recompute-method`以及`--recompute-num-layers`等