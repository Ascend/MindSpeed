# 问题分析
多模态模型的训练中，由于不同模态模型对算力和内存需求的异构性，会产生以下问题：
- 不同模态模型的最优并行配置不同，全部使用同一种并行配置，造成负载不均衡、资源利用不充分；
- 多模型融合部署，造成静态内存占用偏高，训练内存资源利用率不佳。


# 解决方案
新增dist-train功能，通过对异构模型配置不同的计算资源和并行配置，减少冗余的静态资源和异构模型间的气泡，使能异构模型之间的运行速度达到最优匹配。


# 使用方法
在启动脚本中添加参数`--dist-train`。
需要在MindSpeed-MM仓库中，对应模型目录下的`model.json`中添加`dist_config`字段，具体配置示例如下：
```json
{
  "dist_config": {
    "model_name": "internvl2",  // 多模态模型名称
    "use_multiparam_send_recv": false,  // 模型间是否传递tensor列表
    "model_config": [
      {
        "name": "vit",  // 内部模型名称
        "model_index": 0,  // 模型位于流水线中的序号
        "world_size": 1,  // 模型使用卡数
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "forward_only": false // 是否不做反向计算
      },
      {
        "name": "gpt",
        "model_index": 1,
        "world_size": 3,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 1,
        "forward_only": false,
        "main_dp": true // 配置该项时，代表DP数量以该模型为准，只在需要开启inner_dp时配置该项，且该配置唯一
      }
    ]
  }
}
```


# 使用效果
根据模型不同、参数量不同，效果各有差异，可以针对SPS、MFU等指标进行调优，均有收益。


# 注意事项
- 需要配合多模态模型仓库MindSpeed-MM使用，目前支持模型和对应的子模型名称：internvl2 - [vit, gpt], opensoraplan1.3 - [vae, dit]；
- 需要注意在配置并行策略时，若原生模型不支持某种并行策略，则dist-train配置中也不应该开启此种并行策略；
- 配置并行策略时，需要保证各个模型的DP数量保持一致，若配置有main_dp，则以main_dp的DP数量为准；
- 需要使用dist-train配套的权重转换脚本，和MindSpeed-MM中使用的权重互相转换。