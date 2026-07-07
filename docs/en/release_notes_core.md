# Release Notes

## Version Package Description

### Product Version Information

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MindSpeed</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers=><p>26.0.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Release Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>Official release</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>April 2026</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Maintenance</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> For MindSpeed version maintenance, see the [Branch Maintenance Policy](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5).

### Related Product Version Mapping

**Table 1** MindSpeed Core software version compatibility table

|MindSpeed Core Code Branch Name|CANN Version|Ascend Extension for PyTorch Version|Python Version|PyTorch Version|
|--|--|--|--|--|
|26.0.0_core_r0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|
|2.3.0_core_r0.12.1|8.5.0|7.3.0|Python3.10|2.7.1|
|2.2.0_core_r0.12.1|8.3.RC1|7.2.0|Python3.10|2.7.1|

> [!NOTE]
>You can choose a MindSpeed code branch as needed to download and install the source code.

## Version Compatibility Information

|MindSpeed Version|CANN Version|Ascend Extension for PyTorch Version|
|--|--|--|
|26.0.0_core_r0.12.1|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>|26.0.0|
|2.3.0_core_r0.12.1|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|
|2.2.0_core_r0.12.1|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|

## Version Usage Notes

None

## Update Notes

### New Features

 - The Atlas A3 training product family supports generative AI quality of service (QoS) parameter configuration.
 - Supports the KVAllgather context parallel (CP) scheme.
 - Decouples devices in the fully sharded data parallel (FSDP) backend.

### Deleted Features

None

### Interface Changes

None

### Resolved Issues

 - Fixed occasional accuracy anomalies in multi-stream scenarios for the Ascend Transformer Boost operator.
 - Fixed functional exceptions in some scenarios for Norm recomputation in the Transformer Engine branch.

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
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|No viruses or malware.|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00|2026-04-02|No viruses or malware.|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|No viruses or malware.|

### Vulnerability Fix List

None
