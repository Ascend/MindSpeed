# get_batch 配置缓存

缓存 `get_batch` 内构造的 `TransformerConfig`，减少重复创建配置带来的 Host 开销。

## 使用方法

在训练脚本中添加 `--cache-get-batch-config`，默认关闭。

适配器从传入 `pretrain` 的 `forward_step_func` 定位训练入口。入口需同时提供 `get_batch` 和 `core_transformer_config_from_args`；只缓存由该 `get_batch` 直接发起的配置构造。缓存按 args 对象、配置类、MLA 开关及异构层配置路径区分，不缓存数据 batch。
