# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch_npu
from torch import distributed as torch_dist
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
from megatron.training import get_args

from mindspeed.core.tensor_parallel.comm_autograd_function import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_group_api import OverlapCollectiveIntf
from mindspeed.core.tensor_parallel.comm_utils import async_gather_tensors
from mindspeed.core.tensor_parallel.comm_utils import async_reduce_scatter_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim


class Linear2DSplitAlongFirstDim(torch.autograd.Function):
    """2D Linear out axe communication implementation."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        activation_input,
        weight,
        bias,
        ag_comm_intf: CollectiveCommIntf,
        ag_overlap_comm_intf: OverlapCollectiveIntf,
        rs_comm_intf: CollectiveCommIntf,
        rs_overlap_comm_intf: OverlapCollectiveIntf,
        enable_overlap_ag_with_matmul=False,
        enable_overlap_matmul_with_rs=False,
        gradient_accumulation_fusion=False,
    ):
        """
        :param ctx: context to save some tensors or vars for backward use.
        :param activation_input: with shape: [s/(x*cp), b, h/y]
        :param weight: with shape: [h/y, E/x], E means the output size.
        :param bias: bias parameter tensor.
        :param ag_comm_intf: AllGather communication process group interface.
        :param ag_overlap_comm_intf: AllGather communication overlap send and recv comm group
        :param rs_comm_intf: ReduceScatter communication process group interface.
        :param rs_overlap_comm_intf: ReduceScatter communication overlap send and recv comm group
        :param enable_overlap_ag_with_matmul:  enable overlap all-gather with matmul
        :param enable_overlap_matmul_with_rs: enable overlap matmul with reduce-scatter
        :param gradient_accumulation_fusion: enable gradient accumulation fusion
        :return: forward result tensor.
        """
        ctx.save_for_backward(activation_input)
        ctx.weight = weight
        ctx.use_bias = bias is not None
        ctx.rs_comm_intf = rs_comm_intf
        ctx.ag_comm_intf = ag_comm_intf
        ctx.ag_overlap_comm_intf = ag_overlap_comm_intf
        ctx.rs_overlap_comm_intf = rs_overlap_comm_intf
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        if enable_overlap_matmul_with_rs:
            activation_input = activation_input.contiguous()
            return Linear2DSplitAlongFirstDim._do_mm_overlap_reducescatter(
                activation_input, weight.t(), bias, ag_comm_intf, rs_comm_intf
            )

        # first_linear forward: [s/cp, b, H/y] @ [H/y, e/x] -> [s/cp, b, e/x]
        if enable_overlap_ag_with_matmul:
            matmul_res, _ = Linear2DSplitAlongFirstDim._do_allgather_left_tensor_and_matmul_overlap(
                ag_comm_intf,
                ag_overlap_comm_intf,
                part_left_tensor=activation_input,
                full_right_tensor=weight.t(),
            )

            if bias is not None:
                matmul_res += bias
        elif get_args().use_ascend_mc2:  # MC2 适配
            ag_group = ag_comm_intf.get_comm_group()
            rank = ag_comm_intf.get_comm_rank()
            if torch.__version__ > "2.0":
                global_rank = torch.distributed.get_global_rank(ag_group, rank)
                hcomm_info = ag_group._get_backend(torch.device("npu")).get_hccl_comm_name(
                    global_rank
                )
            else:
                hcomm_info = ag_group.get_hccl_comm_name(rank)

            x = activation_input.reshape(-1, activation_input.size(2))

            # [s/(x*cp), b, H/y] --AG(x)--> [s/cp, b, H/y]
            # [s/cp, b, H/y] @ [H/y, e/x] -> [sb/cp, e/x]
            output, _ = torch_npu.npu_all_gather_base_mm(
                x,
                weight.t(),
                hcomm_info,
                ag_comm_intf.get_comm_group_world_size(),
                bias=bias,
                gather_index=0,
                gather_output=False,
            )
            # [sb/cp, e/x]---> [s/cp, b, e/x]
            matmul_res = output.view(
                output.shape[0] // activation_input.shape[1], activation_input.shape[1], output.shape[1]
            )
        else:
            # [s/(x*cp), b, H/y] -> [s/cp, b, H/y]
            activation_input = activation_input.contiguous()
            total_input = sync_gather_along_first_dim(activation_input, ag_comm_intf)
            # [s/cp, b, H/y] @ [H/y, e/x] -> [s/cp, b, e/x]
            matmul_res = torch.matmul(total_input, weight.t())
        # [s/cp, b, E/x] -> [s/(y*cp), b, E/x]
        matmul_res = matmul_res.contiguous()
        matmul_res = sync_reduce_scatter_along_first_dim(matmul_res, rs_comm_intf)
        return matmul_res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward implementation of Linear2DSplitAlongFirstDim, the computation and communication
        overlap:

        ----------------------------------------------------------------------------->time
        | AG(grad_o, Y|X)
        |                AG(activation_input,    X|Y)
        |                part_grad_act = MM(tot_grad_o, weight)
        |                                                      RS(part_grad_act, X|Y)
        |                                                      MM(tot_grad_o^T, tot_act_input)


        :param ctx: context
        :param grad_output: with shape: [s/cp, b, E/(xy)]
        :return:grads of all the input para of forward function as a tuple
        """
        # activation_input shape: [s/(x*cp), b, h/y]
        # weight shape: [h/y, E/x]
        activation_input, = ctx.saved_tensors
        weight = ctx.weight
        use_bias = ctx.use_bias
        # first we prepare the total inputs needed to compute grad_input, grad_weight.
        # [s/(y*cp), b, E/x]---AG(y)---> [s/cp, b, E/x]
        # Use sync AG to avoid communication competition, for the bandwidth is shared for 910C.
        grad_output = grad_output.contiguous()
        total_grad_output = sync_gather_along_first_dim(grad_output, ctx.rs_comm_intf)

        # prepare total activation_input for computing grad weight.
        # [s/(x*cp), b, h/y]---AG(X)--->[s/cp, b, h/y]
        activation_input = activation_input.contiguous()
        gather_input_handle, gathered_tensors = async_gather_tensors(
            local_rank_input=activation_input, ag_comm_intf=ctx.ag_comm_intf
        )

        # [s/cp, b, E/x] @ [E/x, H/y]--> [s/cp, b, H/y] (partial sum)
        partial_grad_input = total_grad_output.matmul(weight).contiguous()
        # [s/cp, b, H/y] (partial sum)---RS(X)--->[s/cp, b, H/(xy)] (full sum)
        rs_grad_input_handle, grad_input = async_reduce_scatter_along_first_dim(
            partial_grad_input, comm_intf=ctx.ag_comm_intf
        )

        # Convert the tensor shapes to 2D for execution compatibility
        sb = total_grad_output.shape[0] * total_grad_output.shape[1]
        # [s/cp, b, E/x]--view--> [sb/cp, E/x]
        total_grad_output = total_grad_output.view(sb, total_grad_output.shape[2])

        if gather_input_handle:
            gather_input_handle.wait()

        # [s/(x*cp), b, h/y]---AG(X)--->[s/cp, b, h/y]
        total_activation_input = gathered_tensors
        # [s/cp, b, h/y]--view--> [sb/cp, h/y]
        total_activation_input = total_activation_input.view(sb, total_activation_input.shape[2])
        if ctx.gradient_accumulation_fusion:
            import fused_weight_gradient_mlp_cuda
            total_grad_output = total_grad_output.contiguous()
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_activation_input, total_grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_activation_input, total_grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=activation_input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            # [E/x, sb/cp] @ [sb/cp, h/y] ---> [E/x, h/y]
            grad_weight = total_grad_output.t().matmul(total_activation_input)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        if rs_grad_input_handle:
            rs_grad_input_handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

    @staticmethod
    def _do_allgather_left_tensor_and_matmul_overlap(
        ag_comm_intf, ag_overlap_comm_intf, part_left_tensor, full_right_tensor, return_ag_res=False
    ):
        cur_ag_rank = ag_comm_intf.get_comm_rank()
        ag_world_sz = ag_comm_intf.get_comm_group_world_size()

        # do tp-x times matmul and reduce the partial res.
        matmul_res = [None] * ag_world_sz
        cur_step_rcv_handle = None
        ring_ag_ranks = ag_overlap_comm_intf.get_ring_global_ranks()
        next_rank = ring_ag_ranks[(cur_ag_rank + ag_world_sz - 1) % ag_world_sz]
        prev_rank = ring_ag_ranks[(cur_ag_rank + 1) % ag_world_sz]
        ag_comm_group = ag_comm_intf.get_comm_group()
        ag_overlap_comm_group = ag_overlap_comm_intf.get_comm_group()
        cur_step_tensor_to_send = part_left_tensor

        # 下一次要计算的数据（本次要从上一个 rank 接收的 tensor。）
        cur_step_rcv_input = torch.empty_like(part_left_tensor)
        all_ag_res = None
        if return_ag_res:
            all_ag_res = [None] * ag_world_sz
            all_ag_res[cur_ag_rank] = part_left_tensor

        # first_linear forward: [H/y, e/x] -> [H/(xy), e/x]
        for step in range(ag_world_sz):
            if step < ag_world_sz - 1 and cur_ag_rank % 2 == 0:  # 偶数 rank 先发再收
                torch_dist.isend(cur_step_tensor_to_send, next_rank, ag_comm_group)
                cur_step_rcv_handle = torch_dist.irecv(
                    cur_step_rcv_input, prev_rank, ag_overlap_comm_group
                )
            elif step < ag_world_sz - 1 and cur_ag_rank % 2 == 1:  # 奇数 rank 先收再发
                cur_step_rcv_handle = torch_dist.irecv(cur_step_rcv_input, prev_rank, ag_comm_group)
                torch_dist.isend(cur_step_tensor_to_send, next_rank, ag_overlap_comm_group)

            # compute: part_left_tensor @ split_right(split by inner dim)
            # [e/x, h/(xy)]
            cur_tensor_idx = (step + cur_ag_rank) % ag_world_sz
            if return_ag_res and step > 0:
                all_ag_res[cur_tensor_idx] = cur_step_tensor_to_send.clone()

            # first linear forward: [s/(x*cp), b, H/y] @ [H/y, e/x] -> [s/(x*cp), b, e/x]
            cur_step_matmul_res = torch.matmul(cur_step_tensor_to_send, full_right_tensor)
            matmul_res[cur_tensor_idx] = cur_step_matmul_res

            if step < ag_world_sz - 1:
                cur_step_rcv_handle.wait()
                cur_step_tensor_to_send = cur_step_rcv_input.clone()

        final_matmul_res = torch.cat(matmul_res)

        return final_matmul_res, all_ag_res

    @staticmethod
    def _do_mm_overlap_reducescatter(activation_input, weight, bias, ag_comm_intf, rs_comm_intf):
        # [s/(x*cp), b, H/y] -> [s/cp, b, H/y]
        activation_input = activation_input.contiguous()
        total_input = sync_gather_along_first_dim(activation_input, ag_comm_intf)
        # [s/cp, b, H/y] @ [H/y, e/x] -> [s/cp, b, e/x]
        chunk_num = rs_comm_intf.get_comm_group_world_size()
        rs_chunks = []
        rs_handle_and_tmp_tensors = []
        # convert tuple to list to free used tensors ahead.
        seq_len, b, h = total_input.size()
        chunk_size = seq_len // chunk_num
        input_chunks = torch.reshape(total_input.view(chunk_size, -1, h).transpose(0, 1), (chunk_num, -1, h))
        rs_res = torch.empty((chunk_size, b, weight.size(1)), dtype=weight.dtype, device=weight.device)
        for idx in range(chunk_num):
            input_chunk = input_chunks[idx].reshape(chunk_size, -1, h)
            # [s/(cp*y), b, H/y] @  [H/y, e/x]  -> [s/(cp*y), b, e/x]
            chunk_matmul_res = torch.matmul(input_chunk, weight).contiguous()
            if bias is not None:
                chunk_matmul_res += bias

            # [s/(cp*y), b, e/x]--rs--> [s/(cp*y*y), b, e/x]
            rs_handle, rs_chunk = async_reduce_scatter_along_first_dim(
                chunk_matmul_res, rs_comm_intf
            )
            rs_chunks.append(rs_chunk)
            rs_handle_and_tmp_tensors.append((idx, rs_handle, chunk_matmul_res))

        offset = 0
        sub_chunk_size = chunk_size // chunk_num
        for idx, rs_handle, chunk_matmul_res_tensor in rs_handle_and_tmp_tensors:
            if rs_handle:
                rs_handle.wait()
                chunk_matmul_res_tensor.untyped_storage().resize_(0)
                rs_res[offset:offset + sub_chunk_size] = rs_chunks[idx]
                offset += sub_chunk_size

        # [s / (cp * y * y), b, e / x] ->  [s/(cp*y), b, e/x]
        final_res = torch.reshape(rs_res.view(chunk_num, -1, weight.size(1)).transpose(0, 1), (chunk_size, -1, weight.size(1)))
        return final_res
