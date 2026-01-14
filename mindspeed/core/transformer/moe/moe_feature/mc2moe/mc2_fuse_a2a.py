# Copyright (c) 2025, Huawei Technologies. All rights reserved.

import torch
import torch_npu
from einops import rearrange


class AlltoallvPermuteGmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                gmm1_inputs, 
                weight1, 
                ep_group, 
                group_list, 
                share_cached_fc1_input, 
                share_expert_weight1, 
                use_shared_experts=False,
                send_counts=None,
                recv_counts=None):
        
        if send_counts is None or recv_counts is None:
            raise AssertionError('Please check your comm input!!!')

        if use_shared_experts:
            mm_x = share_cached_fc1_input
            share_expert_weight1 = share_expert_weight1
        else:
            mm_x = None
            share_expert_weight1 = None
            shared_expert_mm1_out = None

        rank = torch.distributed.get_rank()
        ep_world_size = torch.distributed.get_world_size(ep_group)
        if torch.__version__ > '2.0.1':
            hcom_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = ep_group.get_hccl_comm_name(rank)

        mm1_out, shared_expert_mm1_out, permute2_out = torch_npu.npu_alltoallv_gmm(gmm_x=gmm1_inputs,
                                                            gmm_weight=weight1,
                                                            hcom=hcom_info,
                                                            ep_world_size=ep_world_size,
                                                            send_counts=send_counts,
                                                            recv_counts=recv_counts,
                                                            send_counts_tensor=None,
                                                            recv_counts_tensor=None,
                                                            mm_x=mm_x,
                                                            mm_weight=share_expert_weight1,
                                                            trans_gmm_weight=False,
                                                            trans_mm_weight=False,
                                                            permute_out_flag=True)
        
        
        ctx.save_for_backward(weight1, permute2_out, gmm1_inputs, share_expert_weight1)
        ctx.use_shared_experts = use_shared_experts
        ctx.group_list = group_list
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.ep_world_size = ep_world_size
        ctx.hcom_info = hcom_info
        ctx.mm_x = mm_x
        return mm1_out, shared_expert_mm1_out
        
    @staticmethod
    def backward(ctx, mm1_out_grad, shared_expert_mm1_out_grad):
        use_shared_experts = ctx.use_shared_experts
        group_list = ctx.group_list
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        ep_world_size = ctx.ep_world_size
        hcom_info = ctx.hcom_info
        weight1, permute2_out, gmm1_inputs, share_expert_weight1 = ctx.saved_tensors
        shared_expert_fc1_inputs_grad = None
        shared_expert_weight1_grad = None
        weight1 = rearrange(weight1, 'n h f -> n f h')
        gmm1_inputs_grad, shared_expert_fc1_inputs_grad = torch_npu.npu_gmm_alltoallv(gmm_x=mm1_out_grad,
                                                    gmm_weight=weight1,
                                                    hcom=hcom_info,
                                                    ep_world_size=ep_world_size,
                                                    send_counts=recv_counts,
                                                    recv_counts=send_counts,
                                                    send_counts_tensor=None,
                                                    recv_counts_tensor=None,
                                                    mm_x=shared_expert_mm1_out_grad,
                                                    mm_weight=share_expert_weight1,
                                                    trans_gmm_weight=False,
                                                    trans_mm_weight=False)

        weight1_grad = torch_npu.npu_grouped_matmul([permute2_out.T], [mm1_out_grad], bias=None, group_list=group_list,
                                    split_item=3, group_type=2, group_list_type=1)[0]

        if use_shared_experts:
            shared_expert_weight1_grad = torch_npu.npu_grouped_matmul([gmm1_inputs.T], [shared_expert_mm1_out_grad], bias=None, group_list=group_list,
                            split_item=3, group_type=2, group_list_type=1)[0]
        return gmm1_inputs_grad, weight1_grad, None, None, shared_expert_fc1_inputs_grad, shared_expert_weight1_grad, None, None, None


class GmmUnpermuteAlltoallv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                gmm2_inputs, 
                weight2, 
                ep_group, 
                group_list, 
                share_cached_fc2_input, 
                share_expert_weight2, 
                use_shared_experts=False,
                send_counts=None,
                recv_counts=None):

        if send_counts is None or recv_counts is None:
            raise AssertionError('Please check your comm input!!!')

        if use_shared_experts:
            mm_x = share_cached_fc2_input
            share_expert_weight2 = share_expert_weight2
        else:
            mm_x = None
            share_expert_weight2 = None
            shared_expert_mm2_out = None


        rank = torch.distributed.get_rank()
        ep_world_size = torch.distributed.get_world_size(ep_group)
        if torch.__version__ > '2.0.1':
            hcom_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = ep_group.get_hccl_comm_name(rank)

        alltoall_out, shared_expert_mm2_out = torch_npu.npu_gmm_alltoallv(gmm_x=gmm2_inputs,
                                                    gmm_weight=weight2,
                                                    hcom=hcom_info,
                                                    ep_world_size=ep_world_size,
                                                    send_counts=recv_counts,
                                                    recv_counts=send_counts,
                                                    send_counts_tensor=None,
                                                    recv_counts_tensor=None,
                                                    mm_x=mm_x,
                                                    mm_weight=share_expert_weight2,
                                                    trans_gmm_weight=False,
                                                    trans_mm_weight=False)
        
        
        ctx.save_for_backward(gmm2_inputs, weight2, share_expert_weight2)
        ctx.use_shared_experts = use_shared_experts
        ctx.group_list = group_list
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.ep_world_size = ep_world_size
        ctx.hcom_info = hcom_info
        return alltoall_out, shared_expert_mm2_out
        
    @staticmethod
    def backward(ctx, alltoall_result_grad, shared_expert_mm2_out_grad):
        use_shared_experts = ctx.use_shared_experts
        group_list = ctx.group_list
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        ep_world_size = ctx.ep_world_size
        hcom_info = ctx.hcom_info
        gmm2_inputs, weight2, share_expert_weight2 = ctx.saved_tensors
        shared_expert_fc2_input_grad = None
        shared_expert_weight2_grad = None
        weight2 = rearrange(weight2, 'n h f -> n f h')
        
        gmm2_input_grad, shared_expert_fc2_input_grad, unpermute1_input_grad = torch_npu.npu_alltoallv_gmm(gmm_x=alltoall_result_grad,
                                                            gmm_weight=weight2,
                                                            hcom=hcom_info,
                                                            ep_world_size=ep_world_size,
                                                            send_counts=send_counts,
                                                            recv_counts=recv_counts,
                                                            send_counts_tensor=None,
                                                            recv_counts_tensor=None,
                                                            mm_x=shared_expert_mm2_out_grad,
                                                            mm_weight=share_expert_weight2,
                                                            trans_gmm_weight=False,
                                                            trans_mm_weight=False,
                                                            permute_out_flag=True)

        weight2_grad = torch_npu.npu_grouped_matmul([gmm2_inputs.T], [unpermute1_input_grad], bias=None, group_list=group_list,
                                    split_item=3, group_type=2, group_list_type=1)[0]
        if use_shared_experts:
            shared_expert_weight2_grad = torch_npu.npu_grouped_matmul([gmm2_inputs.T], [shared_expert_mm2_out_grad], bias=None, group_list=group_list,
                            split_item=3, group_type=2, group_list_type=1)[0]
        return gmm2_input_grad, weight2_grad, None, None, shared_expert_fc2_input_grad, shared_expert_weight2_grad, None, None, None
