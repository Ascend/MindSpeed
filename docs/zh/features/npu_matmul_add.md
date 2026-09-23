# matmul_add融合优化

## 背景与挑战

模型训练中开启了梯度累加功能，但累加效率较慢，梯度累加中的 Add 算子占比较高。

## 解决方法

MindSpeed将matmul操作和add操作合并成一个融合算子。算子接口见[npu_matmul_add](../ops/npu_matmul_add.md)。

## 使用场景

LLaMA及GPT大模型均可使用。

## 使用方法

Megatron-LM默认开启`gradient_accumulation_fusion`。去掉`--no-gradient-accumulation-fusion`即可调用MatmulAdd融合路径；该路径的NPU接口由独立的MindSpeed-Ops包提供，不会随MindSpeed自动安装。

默认参数训练需要按[软件安装](../user-guide/install_guide.md)安装MindSpeed-Ops，并确认接口可以导入：

```shell
python -c "from mindspeed_ops.api.atb.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16; print('MindSpeed-Ops MatmulAdd API loaded successfully')"
```

纯推理或评估不执行反向传播，无需安装；训练时也可通过`--no-gradient-accumulation-fusion`关闭融合路径，此时权重梯度由普通路径计算和累加。

Atlas A3训练系列产品和Atlas A2训练系列产品的fp32主梯度场景使用ATB（Ascend Transformer Boost）JIT融合算子，需要安装CANN-NNAL、加载`nnal/atb/set_env.sh`并提供C++/Ninja编译工具链。fp16/bf16主梯度以及Ascend 950PR&950DT系列产品 fp32场景走`addmm_`路径，但开启权重梯度融合时仍需要安装MindSpeed-Ops以提供Python接口。

### 说明

* npu_matmul_add_fp32暂不支持MFU（Model FLOPS Utilization，模型算力利用率）统计。
* 融合算子与小算子之间存在一定的精度差异。
精度差异的根本原因：
小算子matmul操作结束后，会先将得到的结果进行降精度（由fp32到bf16）再升精度（由bf16到fp32）最后进行add操作，这种先降再升的操作会损失一部分精度，而融合算子会跳过这一操作直接进行累加，故精度上存在差异。<br>
具体变化过程如下：
    * 小算子dtype变化过程：`bf16*bf16=fp32->bf16->fp32+fp32=fp32`
    * 融合算子dtype变化过程：`bf16*bf16=fp32+fp32=fp32`

## 使用效果

在内存没有完全使用或占满的情况下，开启Matmul_Add融合算子，模型训练的性能将得到提升，在LLaMA2-7B模型下，性能增益约2%。
