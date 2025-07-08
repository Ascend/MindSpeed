# Hccl Group Buffer Set

## 背景与挑战

当前MindSpeed的通信域Buffer，只能通过环境变量HCCL_BUFFSIZE进行统一设置（默认为200M），但往往不同通信域所需的Buffer大小不能一概而论，具体请参见《CANN 环境变量参考》中的“[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/envvar/envref_07_0080.html)”章节。

## 解决方案

* 自动配置（推荐）：使用自适应方案，MindSpeed根据网络参数自适应通信域缓冲区大小。
* 手动配置：对外呈现开关，使得用户可以根据自己需求自己设置通信域缓冲区大小。

## 使用场景
显存不足，需要降低显存占用的场景可以开启该特性。

## 使用方法
* 自动配置
`--hccl-group-buffer-adaptive`
**注意:**
    * 开启该特性后，会自适应设置tp、cp、pp相关通信组大小。
    * 对于ep相关的通信组（例如exp、tp_exp、tp），用户可自行根据当前模型MoE负载不均衡的程度指定系数--hccl-ep-group-buffer-adaptive-factor，从而得到合适的通信域 Buffer，该系数代表当前负载不均衡的程度。 例如，设置--hccl-ep-group-buffer-adaptive-factor大小为1，代表的是负载均衡情况下需要开启的buffer大小；设置为n，代表当前缓冲区大小是负载均衡情况下的n倍，n配置过大有可能会导致OOM。

    * 自动配置目前支持的通信组如下:
[ "cp", "mp", "mp-exp", "tp", "pp", "tp_cp", "tp_exp", "exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring", "cp_ring_intra","cp_ring_intra_overlap"]

* 手动配置
`--hccl-group-buffer`
**注意:**
    * 配置该参数并指定所需要设定的组以及大小（例如：dp:200;tp:300;exp:400），单位是MB。
    * 手动配置目前支持的通信组如下：
["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", "tp_dp", "tp_cp", "tp_exp", "exp", "dp_modulo_exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring","cp_ring_intra", "cp_ring_intra_overlap", "nd1_dim1", "ag_x_sd_rcv_overlap", "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]

## 使用效果
LLaMA系列模型，开启自适应方案，性能不下降的同时可以节约显存；MoE相关模型，开启自适应方案并设置合适的负载不均衡系数，性能不下降的同时可以节约显存。

## 使用限制
本特性依赖 PTA:FrameworkPTAdapter 7.0.RC1.B020 （包含该版本）之后的版本
