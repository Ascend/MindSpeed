# Release Notes

## Version Package Description

### Product Version Information

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.1.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Official version本</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed Core</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Jul, 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance Period</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> For MindSpeed version maintenance, see the [Branch Maintenance Policy](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5).

### Related Product Version Mapping

**Table 1** MindSpeed Core version compatibility table

|MindSpeed Core Code Branch Name|CANN Version|TorchNPU Version|Python Version|PyTorch Version|
|--|--|--|--|--|
|26.1.0_core_r0.12.1|9.1.0|26.1.0|Python3.10|2.7.1|
|26.0.0_core_r0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|

> [!NOTE]
>
> You can choose a MindSpeed code branch as needed to download and install the source code.

## Version Compatibility Information

> [!NOTE]
>
> In the table, "/" indicates not matched, and "Y" indicates matched.

**Table 2**  MindSpeed Core and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="4">TorchNPU</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 3**  MindSpeed Core and CANN version compatibility

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="4">CANN</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## Version Usage Notes

None

## Update Notes

### New Features

- Added support for FP8/MXFP8/HiFloat8 low-precision training formats, as well as w8a16 quantization.
- Added MXFP8-32x32 quantization and FSDP support, releasing bf16 weights after quantization to optimize memory usage.
- Added SwapMuon and mcore muon features, with support for checkpoint saving and loading.
- Added DeepSeek V4 model adaptation and custom Pipeline Parallel (PP) layout support.
- Added Hamilton attention implementation for TE operator layers.

### Deleted Features

- Removed temporary operator adaptations for SFA/SFAG/SLI versions; related functionalities are now supported by formal operators.
- Removed the mindspeed/lite module; Triton operators have been migrated to the `mindspeed/ops/triton` directory.

### Interface Changes

None

### Resolved Issues

 - Fixed the issue in the TE branch where the initialization order of `LayerNormLinear` weights was inconsistent with `NVTE`, and the issue where `LayerNorm` bias was not initialized to zero.
 - Fixed anomalies in the GMM operator and the NPU `sparse_attn_sharedkv` operator, as well as batch consistency issues in l2norm and NaN errors in `recompute_w_u_fwd`.
 - Fixed abnormal memory usage in the `fboverlap` scenario, and time degradation issues in the Triton operator `chunk_bwd_dqkwg`.
 - Fixed HCCL buffer errors in the VeRL scenario.

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

- Impact on service

    The software version upgrade interrupts the service.

- Impact on network communication

    No impact on communication.

### Impact on the System After the Upgrade

None

## Related Documents

|Document Name|Description|Update Notes|
|--|--|--|
|“[MindSpeed Quick Start](../en/user-guide/quickstart.md)”|Introduces how to use MindSpeed to run Megatron-LM efficiently on Ascend devices.|-|
|“[MindSpeed Installation Guide](../en/user-guide/install_guide.md)”|Explains how to install MindSpeed on NPUs with the PyTorch framework. It covers hardware and operating system compatibility, installation of driver firmware and basic CANN software, and the complete installation process in two frameworks, helping users quickly build a distributed training environment for large models.|-|

## Virus Scan and Vulnerability Fix List

### Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Definition Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-07-06|No viruses or malware.|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00|2026-07-06|No viruses or malware.|
|Bitdefender|7.5.1.200224|7.100588|2026-07-06|No viruses or malware.|

### Vulnerability Fix List

None
