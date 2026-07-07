# matmul_add Fusion

## Background and Challenges

When gradient accumulation is enabled during model training, the accumulation efficiency is slow, and the Add operator accounts for a high proportion of the gradient accumulation process.

## Solution

MindSpeed merges the matmul operation and the add operation into a single fused operator. For the operator interface, see [npu_matmul_add](../ops/npu_matmul_add.md).

## Application Scenario

Applicable to Llama and GPT large models.

## Usage

Enabling the fused operator requires installing ATB (Ascend Transformer Boost). Refer to [Software Installation](../user-guide/install_guide.md) to complete the installation.

Remove `--no-gradient-accumulation-fusion` to call the matmul_add fused operator.

### Notes

* npu_matmul_add_fp32 does not currently support MFU (Model FLOPS Utilization) statistics.
* There is a certain precision difference between the fused operator and the small operators.
Root cause of the precision difference:
After the small operator's matmul operation is completed, the result is first downcast (from fp32 to bf16) and then upcast (from bf16 to fp32) before the final add operation. This downcast-then-upcast process loses some precision, whereas the fused operator skips this step and performs the accumulation directly, resulting in a precision difference.<br>
The specific change process is as follows:
  * Small operator dtype change process: `bf16*bf16=fp32->bf16->fp32+fp32=fp32`
  * Fused operator dtype change process: `bf16*bf16=fp32+fp32=fp32`

## Application Effects

When memory is not fully utilized or saturated, enabling the matmul_add fused operator will improve model training performance. Under the Llama2-7b model, the performance gain is approximately 2%.
