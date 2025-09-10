# Megatron Transformer-engine

## 背景与挑战
Transformer Engine (TE) 是一个专门用于加速基于 Transformer 架构的模型进行训练和推理的库。当前的多个第三方框架依赖该加速库提供的API进行推理及训练，MindSpeed需要对这些需求做出对等支持。

## 解决方法

为了兼容第三方框架对Megatron-TE相关接口的依赖，方便在NPU中进行模型的推理及训练，Mindspeed提供了在Ascend-NPU下等价抽象的TE接口。
目前MindSpeed提供的接口有：
- MindSpeedTELayernorm
- MindSpeedTELayerNormColumnParallelLinear
- MindSpeedTEGroupedLinear

## 使用场景
在模型的训练、推理及第三方框架需要使用相关API时，使用Megatron transformer_engine相关接口。

## 使用方法
脚本中设置`--transformer-impl transformer_engine`，即可使用TE分支。同megatron一致，该参数默认值将设置为`transformer_engine`, 如需回溯早期版本行为，请在脚本中额外设置`--transformer-impl local`.
 
**注意**
- `MindSpeedTELayerNormColumnParallelLinear` 支持与 `ascend-mc2` 同时使能，但不支持与 `ascend-coc` 同时使能。
- `MindSpeedTEGroupedLinear` 在部分重构GMM的特性中，如1f1b-overlap等场景下，可能会失效。


## 使用效果 
执行transformer-impl transformer_engine对应逻辑，进行模型的初始化及训练。
