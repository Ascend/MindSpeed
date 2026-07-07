# rms_norm Fusion

## Background and Challenges

In large language models (LLMs) such as Llama, Llama2, and Baichuan, RMSNorm (Root Mean Square Norm) is widely used as a normalization technique. However, since the PyTorch framework does not natively provide an RMSNorm operator interface, RMSNorm is often implemented in a custom manner within models, which impacts execution efficiency to some extent.

## Solution

To address this, MindSpeed performs fusion optimization on the RMSNorm operation, integrating it into a single operator, which effectively reduces the number of data transfers and temporary storage requirements. For the operator interface, see [rms_norm](../ops/rms_norm.md).

## Application Scenario

When the model uses RMSNorm as its normalization method and the training script already includes the following configuration:
`--normalization RMSNorm`.

## Usage

To enable RMSNorm fusion optimization, add the following parameter configuration to the training script:
`--use-fused-rmsnorm`
Under the Mcore branch, this operator only supports enabling the fusion feature.

## Application Effects

Under the Llama2-7b model, after enabling the fused operator RMSNorm, memory savings are approximately 12% and performance improves by about 2.7%, which not only effectively conserves memory resources but also enhances model training efficiency.
