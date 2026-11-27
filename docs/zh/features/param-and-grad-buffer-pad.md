# 参数与梯度Buffer对齐

在分布式优化器计算参数布局时，对参数存储起始位置进行字节对齐。

## 使用方法

在训练配置中加入：

```shell
--use-distributed-optimizer
--param-and-grad-buffer-pad 512
```

`--param-and-grad-buffer-pad` 默认不设置，值必须大于0；Ascend推荐设置为512。当前不支持与`--use-layer-wise-distributed-optimizer`组合使用。
