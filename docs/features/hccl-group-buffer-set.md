# Hccl Group Buffer Set

## 问题背景
当前 MindSpeed 的通信域 Buffer，只能通过环境变量 HCCL_BUFFSIZE 进行统一设置（默认为 200M ），但是往往不同的通信域所需的 Buffer 大小不能一概而论

## 解决方案
### 1.手动配置
对外呈现开关，使得用户可以根据自己需求自己设置通信域缓冲区大小
### 2.自动配置(推荐)
使用自适应方案，MindSpeed 根据网络参数自适应通信域缓冲区大小

## 使用方法
### 1.手动配置
打开--hccl-group-buffer，并指定所需要设定的组以及大小（例如：dp:200;tp:300;exp:400），单位是 M 。

手动配置目前支持通信组:

["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", "tp_dp", "tp_cp", "tp_exp", 
 "exp", "dp_modulo_exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring","cp_ring_intra", "cp_ring_intra_overlap",
 "nd1_dim1", "ag_x_sd_rcv_overlap", "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]

### 2.自动配置
打开 --hccl-group-buffer-adaptive ，会自适应设置 tp、cp、pp 相关通信组大小; 需要注意的是，对于 ep 相关的通信组（exp、tp_exp、tp），用户可自行根据当前模型 MOE 负载不均衡的程度指定系数--hccl-ep-group-buffer-adaptive-factor 从而得到合适的通信域 Buffer，该系数代表当前负载不均衡的程度（ 例如，设置--hccl-ep-group-buffer-adaptive-factor 大小为 1， 代表的是负载均衡情况下需要开启的buffer大小；设置为n，代表当前缓冲区大小是负载均衡情况下的 n 倍，n 配置过大有可能会导致OOM ）

自动配置目前支持通信组:

[ "cp", "mp", "mp-exp", "tp", "pp", "tp_cp", "tp_exp", "exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring", "cp_ring_intra","cp_ring_intra_overlap"]

## 使用效果
llama 系列模型，开启自适应方案，性能不下降的同时节约显存；MOE 相关模型，开启自适应方案并设置合适的负载不均衡系数，性能不下降的同时节约显存。

## 使用限制
本特性依赖 PTA:FrameworkPTAdapter 7.0.RC1.B020 （包含该版本）之后的版本
