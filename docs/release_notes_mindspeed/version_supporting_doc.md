# 2.3.0版本配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[分布式训练加速库迁移指南](https://gitcode.com/Ascend/MindSpeed/blob/2.3.0_core_r0.12.1/docs/user-guide/model-migration.md)》|指导具有一定Megatron-LM模型训练基础的用户将原本在其他硬件平台（例如GPU）上训练的模型迁移到昇腾平台（NPU），主要关注点是有效地将Megatron-LM训练模型迁移至昇腾平台上， 并在合理精度误差范围内高性能运行。|-|
|《[MindSpeed LLM基于PyTorch迁移指南](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97%2FMindSpeed-LLM%E8%BF%81%E7%A7%BB%E6%8C%87%E5%8D%97-Pytorch%E6%A1%86%E6%9E%B6.md)》|以Qwen2.5-7B为例，通过比对HuggingFace的模型结构，抓取其模型特性，并将其特性结构适配到MindSpeed LLM中，从而使MindSpeed LLM与开源模型结构对齐，使权重和输入数据做相同运算，确保输出对齐，同时保持或提升该模型的性能。|-|
|《[MindSpeed MM迁移调优指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/2.3.0/docs/user-guide/model-migration.md)》|以Qwen2-VL-7B为例，基于华为昇腾芯片产品（NPU）开发，将原本运行在GPU或其他硬件平台的深度学习模型迁移至NPU， 并保障模型在合理精度误差范围内高性能运行，并对昇腾芯片进行极致优化以发挥芯片最大性能。|-|
|《[MindSpeed RL使用指南](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.3.0/docs/solutions/r1_zero_qwen25_32b.md)》|以Qwen2.5-32B为例，基于GPRO+规则奖励打分进行训练，复现DeepSeek-R1-Zero在Math领域的工作。|-|


