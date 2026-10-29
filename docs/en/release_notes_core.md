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
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Release</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed Core</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Sep, 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance Period</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> For MindSpeed version maintenance, see the [Branch Maintenance Policy](https://gitcode.com/Ascend/MindSpeed/tree/26.2.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5).

### Related Product Version Mapping

**Table 1** MindSpeed Core version compatibility table

|MindSpeed Core Code Branch Name|CANN Version|TorchNPU Version|Python Version|PyTorch Version|Triton-Ascend Version|
|--|--|--|--|--|--|
|26.2.0_core_r0.12.1|9.2.0|26.2.0|Python3.12|2.10.0|3.6.0|
|26.1.0_core_r0.12.1|9.1.0|26.1.0|Python3.10|2.7.1|/|
|26.0.0_core_r0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|/|

> [!NOTE]
>
> You can choose a MindSpeed code branch as needed to download and install the source code.

## Version Compatibility Information

> [!NOTE]
>
> In the table, "/" indicates not matched, and "Y" indicates matched.

**Table 2** MindSpeed Core and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="5">TorchNPU</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.X</th>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.X_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.1.X_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
        <tr>
      <td>26.2.X_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 3** MindSpeed Core and CANN version compatibility

<table style="table-layout: fixed; width: 750px ; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">MindSpeed Core</th>
      <th colspan="5">CANN</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.X_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
        <tr>
      <td>26.1.X_core_r0.12.1</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.2.X_core_r0.12.1</td>
      <td>Y</td>
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

None

### Deleted Features

None

### Interface Changes

None

### Resolved Issues

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
|[MindSpeed Quick Start](../en/user-guide/quickstart.md)|Introduces how to use MindSpeed to run Megatron-LM efficiently on Ascend devices.|-|
|[MindSpeed Installation Guide](../en/user-guide/install_guide.md)|Explains how to install MindSpeed on NPUs with the PyTorch framework. It covers hardware and operating system compatibility, installation of driver firmware and basic CANN software, and the complete installation process in two frameworks, helping users quickly build a distributed training environment for large models.|-|

## Virus Scan and Vulnerability Fix List

### Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Definition Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-09-27 08:00:00.0|2026-09-28|No viruses or malware.|
|Kaspersky|12.0.0.6672|2026-09-28 10:09:00|2026-09-28|No viruses or malware.|
|Bitdefender|7.5.1.200224|7.101487|2026-09-28|No viruses or malware.|

### Vulnerability Fix List

None

## Revision History

|Document Version|Release Date|Revision Description|
|--|--|--|
|01|2026-10-15|First official release|
