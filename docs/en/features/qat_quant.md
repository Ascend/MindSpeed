# QAT Quantization

## Background and Challenges

As large models continue to grow in scale, the memory required for their training and inference has become a hardware bottleneck. To address this, low-precision training strategies reduce memory usage by compressing weight precision, making them a key technical direction in large model efficiency optimization.

## Solution

For fine-tuning and training scenarios, quantization error is introduced by simulating the quantization-dequantization process of the MXFP4 format, while weights are still stored and updated in high precision (such as BF16), allowing the model to adapt to the precision loss caused by quantization. During forward propagation, weights are quantized to MXFP4 (E2M1 + block scaling) and then dequantized back to high precision for computation. During backward propagation, gradient straight-through estimation (STE) is typically used to enable gradients to effectively update the original weights, thereby adapting to the W4A16 low-precision strategy.

## Usage

The MXFP4 low-precision quantization feature can be enabled in the following two modes. Add the corresponding configuration to the training script:

1. Enable MXFP4 low-precision quantization for all linear layers:
`--qat-scheme w4a16-mxfp4`

2. Enable MXFP4 low-precision quantization only for MOE layers:
`--qat-scheme w4a16-mxfp4-moe-only`

### Notes

Other settings or no settings will use the original strategy.

## Application Effects

Precision loss is controlled within an acceptable range.
