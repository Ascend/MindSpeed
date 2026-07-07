# Support for the EOD Reset Training Scenario

## EOD Reset Training Scenario

Typically, a text sequence fed into the model within a batch is formed by concatenating multiple documents. By default, these documents are treated as a single sequence, with no masking applied to the self attention between them. In certain cases, the documents are required to be independent, meaning they cannot perform self attention on each other. In this scenario, the attention mask and position ids need to be reset at the end of each document (EOD). When the `--reset-position-ids` parameter is disabled, position encoding is calculated for the entire sequence; when enabled, position encoding is calculated independently within each sequence.

## Solution

The EOD Reset training scenario is supported by invoking the variable-length mode of the underlying flash-attention operator. Additionally, in the EOD Reset training scenario, Ring Attention parallelism for long sequences is supported to accelerate processing for ultra-long sequence scenarios.

## Usage

### 1. Data Preparation

1. First, ensure that an EOD token is added to the end of each document.
2. When `--attention-mask-type` is set to causal, due to internal implementation requirements, the length of each subsequence will be padded online to a multiple of CP × lcm(2, TP), where lcm is the least common multiple. In this scenario, `--variable-seq-lengths` must be enabled because padding will cause the sequence length to change.

### 2. Parameter Settings

1. Enable the `--reset-attention-mask` option.
2. Use the `--reset-position-ids` option to indicate whether position encoding is reset.
3. `--attention-mask-type` can be specified as causal or general; both produce equivalent computation results. causal is an accelerated implementation, while general is the baseline approach.

### 3. Notes

In the Ascend EOD Reset training scenario, when mask-type is set to general, Ring/Hybrid Attention performance drops significantly compared to Ulysses, which is expected behavior. When mask-type is set to causal, the accelerated implementation is used.
