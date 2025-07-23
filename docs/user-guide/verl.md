# Verl 使用 MindSpeed 训练后端

## 环境准备

### 1. MindSpeed 安装
按照 MindSpeed 文档，安装对应依赖。
> https://gitee.com/ascend/MindSpeed#%E5%AE%89%E8%A3%85

### 2. Verl 安装
按照 Verl 文档，安装对应依赖。
> https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html

## 使能 MindSpeed 后端

确认模型对应的 `strategy` 配置为 `megatron`，例如 `actor_rollout_ref.actor.strategy=megatron`，可以在 shell 脚本中或者 config 配置文档中设置。

MindSpeed 自定义入参可通过 `override_transformer_config` 参数传入，例如对 `actor` 模型开启 FA 特性可使用 `+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True`

## 特性支持列表
敬请期待...