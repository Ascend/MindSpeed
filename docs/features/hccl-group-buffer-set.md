# Hccl Group Buffer Set

## 问题背景
当前的通信域Buffer，只能通过环境变量HCCL_BUFFSIZE，进行统一设置，但是往往不同的通信域所需的Buffer大小不能一概而论

## 解决方案
对外呈现开关，使得用户可以根据自己需求自己设置通信域缓冲区大小

## 使用方法
打开--hccl-group-buffer，并指定所需要设定的组以及大小（例如：dp:200;tp:300;exp:400），单位是M

目前支持通信组:
["dp", "dp_cp", "cp", "mp", "mp_exp", "tp", "pp", "embd", "tp_dp_cp", "tp_dp", "tp_cp", 
 "tp_exp", "exp", "dp_modulo_exp", "pp_new_stream", "cp2", "cp_ulysses", "cp_ring", "nd1_dim1", 
 "ag_x_sd_rcv_overlap", "nd1_dim2", "ag_y_sd_rcv_overlap", "nd2_dim1", "nd2_dim2"]
