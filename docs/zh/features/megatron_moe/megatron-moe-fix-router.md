# MoE 固定路由

通过确定性的轮转规则分配专家，用于固定路由的 MoE 训练配置。

## 使用方法

添加 `--fix-router`，默认关闭。要求 `--expert-model-parallel-size` 大于 1，不能同时开启 `--moe-router-fusion`。
