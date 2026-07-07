# HCCL Group Buffer Set

## Background and Challenges

Currently, the communication domain buffer in MindSpeed can only be uniformly set through the environment variable HCCL_BUFFSIZE (default is 200M). However, the buffer sizes required by different communication domains often cannot be generalized. For details, see the [HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/envvar/envref_07_0080.html) section in *CANN Environment Variable Reference*.

## Solution

* Auto-configuration (recommended): Uses an adaptive scheme where MindSpeed adapts the communication domain buffer size based on network parameters.
* Manual configuration: Exposes a switch, allowing users to set the communication domain buffer size according to their own needs.

## Application Scenario

This feature can be enabled in scenarios where video memory is insufficient and memory usage needs to be reduced.

## Usage

* Auto-configuration
`--hccl-group-buffer-adaptive`
**Notes:**
  * When this feature is enabled, the buffer sizes for tp, cp, and pp related communication groups are adaptively set.
  * For ep-related communication groups (such as exp, tp_exp, tp), users can specify the coefficient `--hccl-ep-group-buffer-adaptive-factor` based on the MoE load imbalance degree of the current model to obtain an appropriate communication domain buffer. This coefficient represents the current load imbalance degree. For example, setting `--hccl-ep-group-buffer-adaptive-factor` to 1 indicates the buffer size required under load balancing conditions; setting it to n indicates that the current buffer size is n times that under load balancing conditions. Setting n too large may cause OOM.

  * The communication groups currently supported by auto-configuration are as follows:
    [ "cp", "mp", "mp_exp", "tp", "pp", "tp_cp", "tp_exp", "exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring", "cp_ring_intra","cp_ring_intra_overlap"]

* Manual configuration
`--hccl-group-buffer`
**Notes:**
  * Configure this parameter and specify the desired group and size (for example: dp:200;tp:300;exp:400), in MB.
  * The communication groups currently supported by manual configuration are as follows:
    ["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", "tp_dp", "tp_cp", "tp_exp", "exp", "dp_modulo_exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring","cp_ring_intra", "cp_ring_intra_overlap", "nd1_dim1", "ag_x_sd_rcv_overlap", "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]

## Application Effects

For Llama models, enabling the adaptive scheme can save video memory without performance degradation; for MoE-related models, enabling the adaptive scheme and setting an appropriate load imbalance factor can save video memory without performance degradation.

## Usage Constraints

This feature depends on PTA:FrameworkPTAdapter 7.0.RC1.B020 (including this version) and later versions
