## Ascend MindStudio Training Tools 精度对照
## 背景与挑战

大模型训练中，极小的数据波动就很可能带来最终评分的较大下降，这使得模型在精度对照时往往存在巨大的工作量，尤其是在进行跨平台(GPU→NPU)对照的场景。借助mstt工具，Ascend芯片可以较快的完成整网训练数据的采集，配合确定性计算来达到精度比对的功能。然而mstt使用时需要手动修改代码，并设置调节config，这使得其在mindspeed中的使能存在一定不便。

## 解决方案

为满足上述需求，引入了“精度对照”功能，MindSpeed 通过集成并简化了mstt工具的使用，允许用户通过设置参数，快速进行整网的精度数据dump及比对。

## 使用场景

需要进行精度对比、特定场景复现时。

## 使用方法

要启用此功能，在脚本中加入`--npu-datadump`即可。使用前请先进行下述config.json的修改。默认状态下，采集RANK0、STEP0下的statistics精度。
通过调整`mindspeed\functional\npu_datadump\config.json`，可以进行整网dump时各选项的调整。
通过调整`mindspeed\functional\npu_datadump\compare.json`，可以利用mstt进行各dump数据的精度比对。
关于config设置的更多细节，请参考mstt官方使用教程：[<td><a href="https://gitcode.com/Ascend/mstt/tree/master/debug/accuracy_tools/msprobe">link</a></td>]
- 暂不支持Lite后端。
- dump数据默认保存在Megatron-LM目录下。
## 使用效果

通过精度对照功能，可快速确认整网运行过程中的精度误差。