# Hccl Group Buffer Set

## 问题背景
当前的通信域Buffer，只能通过环境变量HCCL_BUFFSIZE，进行统一设置，但是往往不同的通信域所需的Buffer大小不能一概而论

## 解决方案
1.手动配置:对外呈现开关，使得用户可以根据自己需求自己设置通信域缓冲区大小
2.自动配置:使用自适应方案，MindSpeed，根据网络参数自适应通信域缓冲区大小

## 使用方法
1.手动配置:打开--hccl-group-buffer，并指定所需要设定的组以及大小（例如：dp:200;tp:300;exp:400），单位是M
手动配置目前支持通信组:
["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", "tp_dp", "tp_cp", "tp_exp", 
 "exp", "dp_modulo_exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring","cp_ring_intra", "cp_ring_intra_overlap",
 "nd1_dim1", "ag_x_sd_rcv_overlap", "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]

2.自动配置:打开--hccl-group-buffer-adaptive，会自适应tp、cp、pp相关通信组大小; 需要注意的是，ep相关的通信组（exp、tp_exp、tp）在设计MOE场景的时候默认会配置200M，用户可自行根据负载不均衡的程度指定系数--hccl-ep-group-buffer-adaptive-factor，该系数代表当前负载不均衡的程度（例如，设置--hccl-ep-group-buffer-adaptive-factor 大小为 1， 代表的是负载均衡情况下需要开启的buffer大小；设置为n，代表当前缓冲区大小开启是负载均衡情况下的 n 倍
自动配置目前支持通信组:
[ "cp", "mp", "mp-exp", "tp", "pp", "tp_cp", "tp_exp", "exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring", "cp_ring_intra","cp_ring_intra_overlap"]

## 使用效果
在llama系列模型，开启自适应方案，可节约显存, 提升性能；在MOE相关模型，开启自适应方案并设置合适的负载不均衡系数，可提升性能。
