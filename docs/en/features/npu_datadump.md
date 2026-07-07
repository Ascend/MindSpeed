# MindStudio Training Tools Precision Comparison

## Background and Challenges

During large model training, even minimal data fluctuations can lead to a significant drop in the final evaluation score. This often results in a substantial workload for model precision comparison, especially in cross-platform (GPU to NPU) comparison scenarios. With the mstt tool, Ascend chips can complete the collection of full-network training data relatively quickly, and by leveraging deterministic computation, precision comparison can be achieved. However, using mstt requires manual code modification and configuration adjustments, which introduces certain inconveniences when enabling it within MindSpeed.

## Solution

To address the requirements above, the "Precision Comparison" feature has been introduced. MindSpeed integrates and simplifies the use of the mstt tool, allowing users to quickly perform full-network precision data dumping and comparison by setting parameters.

## Application Scenario

When precision comparison or reproduction of specific scenarios is required.

## Usage

To enable this feature, add `--npu-datadump` to the script. Before use, modify the config.json file as described below. By default, statistics precision data for RANK0 and STEP0 is collected.
You can adjust various options for full-network dumping by modifying `mindspeed\functional\npu_datadump\config.json`.
You can use mstt to compare the precision of dumped data by modifying `mindspeed\functional\npu_datadump\compare.json`.
For more details on config settings, refer to the official mstt tutorial: [<td><a href="https://gitcode.com/Ascend/mstt/tree/master/debug/accuracy_tools/msprobe">link</a></td>]

- The Lite backend is not currently supported.
- Dump data is saved in the Megatron-LM directory by default.

## Application Effect

The precision comparison feature allows for quick identification of precision errors during full-network execution.
