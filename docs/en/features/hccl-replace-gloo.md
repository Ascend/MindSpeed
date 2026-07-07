# Gloo Snapshot Persistence Optimization

## Background and Challenges

In large-scale clusters, Gloo communication faces scalability limitations and stability issues. On one hand, Gloo communication group creation failures are prone to occur; on the other hand, compared with HCCL communication, Gloo communication is slower.

For the issue of Gloo communication group creation failure with the error `Gloo connectFullMesh failed with ...`, the root cause is that N cards connect to the master node for connection establishment. When the cluster scale is large, the master node's processing capacity is insufficient, which may cause connection establishment failure. This can be mitigated by adjusting parameters related to network connection establishment (verified to be effective in an 8k-card cloud scenario):

```bash
net.ipv4.tcp_max_syn_backlog = 65536
net.core.netdev_max_backlog = 65536
```

Additionally, MindSpeed has designed a Gloo communication optimization solution that uses HCCL communication to replace Gloo.

## Solution

### Approach

1. Replace the Gloo communication group with the HCCL communication group, achieving substitution on top of the existing functionality.

2. Use a slicing method to reduce the amount of data per communication, avoiding GPU memory consumption caused by excessive communication volume.

## Application Scenario

When Gloo communication frequently encounters connection establishment failures, the model startup efficiency is low. In this case, it is necessary to replace the Gloo communication group to improve efficiency.

## Usage

1. Add `--disable-gloo-group` to the training script to enable this feature.

2. Define `--hccl-slice-size N` (optional) in the script to set the communication volume size when the DP group saves and loads distributed optimizer states. The valid range of this parameter is (0, bucket_size/dp], where bucket_size is the size of each bucket in the distributed optimizer. It is recommended to increase this parameter as much as possible, provided that GPU memory allows, to improve communication efficiency.

## Application Effects

### Communication Efficiency Analysis

Theoretically, the communication efficiency of distributed optimizer state saving and loading improves as `hccl-slice-size` increases within a certain range.
<!-- , the following reference data is provided:
- When the slice size is $10 * 1024 * 1024$, the model loading time increases slightly, while the model saving efficiency remains unchanged;
- When the slice size is $30 * 1024 * 1024$, the model saving and loading efficiency is comparable to the original;
- When the slice size is $100 * 1024 * 1024$, the model saving and loading efficiency improves to some extent. -->

### Device Memory Increase Analysis

After enabling this feature, the increase in device memory is `hccl-slice-size * (2 * dp + 1) * 4B`.
