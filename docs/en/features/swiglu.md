# swiglu Fusion

## Background and Challenges

SwiGLU (Swish-Gated Linear Unit) is commonly used in the activation layers of large language models such as LLaMA, LLaMA2, and Baichuan. However, since the PyTorch standard library lacks a direct operator interface for SwiGLU, models typically implement SwiGLU functionality by combining a series of basic operators, which results in suboptimal execution efficiency.

## Solution

To address this challenge, MindSpeed applies fusion optimization to the SwiGLU operation, encapsulating it as a high-performance fused operator that significantly reduces the number of data transfers between memory and the need for temporary data storage. For the operator interface, see [swiglu](../ops/swiglu.md).

## Application Scenario

When the model design uses SwiGLU as the activation function for the MLP layer, and the training script already includes the following configuration option:
`--swiglu`

## Usage

To enable SwiGLU fusion optimization, add the following parameter configuration to the training script:
`--use-fused-swiglu`

Under the mcore branch, only enabling this fusion operator is supported.

## Application Effects

Under the Llama2-7b model, enabling the fusion-optimized SwiGLU operator saves approximately 16.6% memory and improves performance by about 4.7%, which not only effectively reduces memory consumption but also significantly enhances the operational efficiency of model training.
