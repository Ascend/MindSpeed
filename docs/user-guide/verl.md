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
| 特性名称          | 配置参数                                                                                                                                                     | 状态    |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| TP                | actor_rollout_ref.actor.megatron.tensor_model_parallel_size<br>actor_rollout_ref.ref.megatron.tensor_model_parallel_size                                      | Preview |
| PP                | actor_rollout_ref.actor.megatron.pipeline_model_parallel_size<br>actor_rollout_ref.ref.megatron.pipeline_model_parallel_size                                  | Preview |
| EP                | actor_rollout_ref.actor.megatron.expert_model_parallel_size<br>actor_rollout_ref.ref.megatron.expert_model_parallel_size                                      | Preview |
| ETP               | actor_rollout_ref.actor.megatron.expert_tensor_parallel_size<br>actor_rollout_ref.ref.megatron.expert_tensor_parallel_size                                    | Preview |
| SP                | actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel                                                                                | Preview |
| 分布式优化器      | actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer                                                                        | Preview |
| 重计算            | actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method<br>actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity<br>actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers | Preview |
| CP                | actor_rollout_ref.actor.megatron.context_parallel_size<br>actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size                 | Preview |

注："Preview"状态表示预览非正式发布版本，"Released"状态表示正式发布版本，"Dev"状态表示正在开发中。