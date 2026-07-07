# Rotary Position Embedding Fusion

## Background and Challenges

RoPE (Rotary Position Embedding) is a position encoding technique widely used in large language models to effectively encode positional information of text sequences. RoPE combines the stability of absolute position encoding with the flexibility of relative position encoding, while also offering excellent length generalization capability. Although RoPE has been adopted in multiple cutting-edge models such as Llama and GLM, the PyTorch framework currently does not provide a dedicated implementation and optimization for RoPE. As a result, model developers typically need to implement RoPE through custom methods, which often incurs high computational and memory overhead.

## Solution

To address the above issues, we have introduced a fusion optimization solution for Rotary Embedding. By consolidating RoPE operations into a single operator, we significantly reduce the number of data transfers and temporary storage requirements, thereby optimizing model training performance. This optimization is implemented by MindSpeed through calling the torch_npu-side interface, effectively improving the execution efficiency of RoPE in models.

## Application Scenario

Applicable to model architectures that use Rotary Embedding as the position encoding scheme.

## Usage

* Ensure the following parameters are set in the model configuration:
`--position-embedding-type  rope`

* At the same time, enabling the RoPE fusion operator requires setting the following parameter:
`--use-fused-rotary-pos-emb`

## Application Effects

By using the fusion-optimized RoPE operator, model training performance is improved, while memory consumption and computational cost are effectively reduced. On the LLaMA2-7B model, the performance improvement is approximately 1%.
