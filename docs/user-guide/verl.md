# Verl 使用 MindSpeed 训练后端

## 环境准备

### 1. MindSpeed 安装
按照 MindSpeed 文档，安装对应依赖。[MindSpeed安装](https://gitcode.com/Ascend/MindSpeed#%E5%AE%89%E8%A3%85)

### 2. Verl 安装
#### 版本说明
verl固定如下commit id：
```shell
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 3cc7695f4c70620ad871437037856f32182de096
```

若使用的CANN版本高于8.3.RC1，vllm和vllm-ascend安装版本须大于等于0.9.1，0.9.1版本vllm安装可参考：https://vllm-ascend.readthedocs.io/en/v0.9.1/installation.html

#### 安装
参考 Verl 文档，安装对应依赖：[verl_ascend_quick_start](https://github.com/volcengine/verl/blob/3cc7695f4c70620ad871437037856f32182de096/docs/ascend_tutorial/ascend_quick_start.rst)

## 使能 MindSpeed 后端

确认模型对应的 `strategy` 配置为 `megatron`，例如 `actor_rollout_ref.actor.strategy=megatron`，可以在 shell 脚本中或者 config 配置文档中设置。

MindSpeed 自定义入参可通过 `override_transformer_config` 参数传入，例如对 `actor` 模型开启 FA 特性可使用 `+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True`。

## 特性支持列表
| 特性名称     | 配置参数                                                     | 状态    |
| ------------ | ------------------------------------------------------------ | ------- |
| FA（必须开） | +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True | Preview |
| TP           | actor_rollout_ref.actor.megatron.tensor_model_parallel_size  | Preview |
| PP           | actor_rollout_ref.actor.megatron.pipeline_model_parallel_size | Preview |
| EP           | actor_rollout_ref.actor.megatron.expert_model_parallel_size  | Preview |
| ETP          | actor_rollout_ref.actor.megatron.expert_tensor_parallel_size | Preview |
| SP           | actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel | Preview |
| 分布式优化器 | actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer | Preview |
| 重计算       | actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method<br>actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity<br>actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers | Preview |
| CP           | actor_rollout_ref.actor.megatron.context_parallel_size<br>actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size | Preview |

注："Preview"状态表示预览非正式发布版本，"Released"状态表示正式发布版本，"Dev"状态表示正在开发中。