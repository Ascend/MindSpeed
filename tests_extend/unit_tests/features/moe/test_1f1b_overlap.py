# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from unittest.mock import patch
import os
import types
import sys

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
sys.argv = [
    sys.argv[0],
    "--transformer-impl",
    "local",
    "--moe-grouped-gemm"
]
from functools import partial, wraps
import pytest
import torch
import torch.nn.functional as F

from mindspeed import megatron_adaptor

from megatron.training.arguments import parse_args
from megatron.core.parallel_state import get_expert_model_parallel_rank, get_tensor_model_parallel_rank
from megatron.training.global_vars import set_args
from megatron.core.parallel_state import destroy_model_parallel

from megatron.core.models.gpt import GPTModel
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from megatron.core.transformer import TransformerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

from mindspeed.functional.npu_deterministic.npu_deterministic import extend_seed_all

from mindspeed.core.transformer.moe.moe_feature.fb_overlap.transformer_block import (
    transformer_block_backward
)
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.gpt_model import gpt_model_backward

from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed, initialize_model_parallel




def initialize_gpt_model(seed, etp_size, pre_process=False, post_process=False, **config_kwargs):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    
    def model_provider_func(seed, pre_process_true, post_process_true, pre_process=False, post_process=False, **config_kwargs):
        default_config_kwargs = dict(
            num_layers=4, hidden_size=4096, num_attention_heads=64, num_query_groups=8, params_dtype=torch.bfloat16,
            hidden_dropout=0, attention_dropout=0, add_bias_linear=False, add_qkv_bias=False,
            activation_func=F.silu, num_moe_experts=16, ffn_hidden_size=1024,
            moe_router_topk=4, moe_grouped_gemm=True, moe_token_dispatcher_type='alltoall_seq', gated_linear_unit=True,
            use_cpu_initialization=True, perform_initialization=False, expert_model_parallel_size=8
        )
        default_config_kwargs.update(**config_kwargs)
        transformer_config = TransformerConfig(**default_config_kwargs)
        transformer_config.expert_tensor_parallel_size = etp_size
        transformer_spec = get_gpt_layer_local_spec(num_experts=transformer_config.num_moe_experts, moe_grouped_gemm=True)
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_spec,
            vocab_size=16384,
            max_sequence_length=2048,
            pre_process=pre_process_true,
            post_process=post_process_true,
        )
        for param in model.parameters():
            torch.nn.init.normal_(param.data, mean=0.0, std=1.0)

        return model

    model = get_model(partial(model_provider_func, seed, pre_process, post_process, pre_process=pre_process, post_process=post_process, **config_kwargs))

    return unwrap_model(model)[0]


def revert_patches():
    from mindspeed.patch_utils import MindSpeedPatchesManager
    import megatron


    megatron.core.transformer.transformer_block.TransformerBlock.__init__ = \
        MindSpeedPatchesManager.patches_info['megatron.core.transformer.transformer_block.TransformerBlock.__init__'].orig_func
    MindSpeedPatchesManager.patches_info.pop('megatron.core.transformer.transformer_block.TransformerBlock.__init__')
    megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward = \
        MindSpeedPatchesManager.patches_info['megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward'].orig_func
    MindSpeedPatchesManager.patches_info.pop('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward')
    megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook = \
        MindSpeedPatchesManager.patches_info['megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook'].orig_func
    MindSpeedPatchesManager.patches_info.pop('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook')



def get_input_split_by_sp_ep(tp_size, ep_size, sequence_length, hidden_size):
    total_input = torch.randn((sequence_length * ep_size, 1, hidden_size), device='npu', dtype=torch.bfloat16)
    input_cur_ep = total_input.chunk(ep_size, dim=0)[get_expert_model_parallel_rank()]
    input_cur_tp_ep = input_cur_ep.chunk(tp_size, dim=0)[get_tensor_model_parallel_rank()]
    input_cur_tp_ep.requires_grad = True
    return input_cur_tp_ep


class TestMoeFbOverlapFeature(DistributedTest):
    world_size = 8
    sequence_length = 2048
    hidden_size = 4096

    @pytest.mark.parametrize("tp_ep_etp", [(2, 8, 1)])
    @pytest.mark.parametrize('n_shared_experts', [1])
    @pytest.mark.parametrize('dispatcher_type', ['alltoall'])
    @pytest.mark.parametrize('memory_option', ['disable', 'level0'])
    def test_transformer_layer_forward_backward_overlap(self, tp_ep_etp, dispatcher_type, memory_option, n_shared_experts):
        # mock the validate in transformer_config post init
        with patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.pre_validate_features_args') as mock_pre_validate, \
             patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.post_validate_features_args') as mock_post_validate, \
             patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.validate_features_args') as mock_validate:

            mock_pre_validate.return_value = None
            mock_post_validate.return_value = None
            mock_validate.return_value = None

            from mindspeed.patch_utils import MindSpeedPatchesManager
            tp_size, ep_size, etp_size = tp_ep_etp
            args = parse_args(None, True)
            args.npu_deterministic = True
            args.use_flash_attn = True
            args.moe_token_dispatcher_type = dispatcher_type
            args.gradient_accumulation_fusion = False
            args.moe_zero_memory = 'disable'
            recompute_options = {'recompute_granularity' : None, 'recompute_method' : None, 'recompute_num_layers' : None}
            if memory_option == 'full':
                recompute_options = {
                    'recompute_granularity' : 'full', 'recompute_method' : 'uniform', 'recompute_num_layers' : 1
                }
            args.sequence_parallel = tp_size > 1
            moe_shared_expert_intermediate_size = None
            if n_shared_experts > 0:
                args.n_shared_experts = n_shared_experts
                moe_shared_expert_intermediate_size = n_shared_experts * 1024

            set_args(args)
            seed = 1234
            set_random_seed(seed)
            extend_seed_all(seed)
            initialize_model_parallel(tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size, expert_tensor_parallel_size=etp_size)
            args.moe_fb_overlap = False
            ref_model = initialize_gpt_model(
                seed, etp_size, pre_process=False, post_process=False, sequence_parallel=tp_size > 1,
                expert_model_parallel_size=ep_size, num_layers=2, tensor_model_parallel_size=tp_size,
                moe_token_dispatcher_type=dispatcher_type, moe_shared_expert_overlap=True,
                moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size, **recompute_options
            )
            ref_block = ref_model.decoder

            input1 = get_input_split_by_sp_ep(tp_size, ep_size, self.sequence_length, self.hidden_size)
            input2 = get_input_split_by_sp_ep(tp_size, ep_size, self.sequence_length, self.hidden_size)
            out1_grad = torch.ones_like(input1) * 1e-4
            out2_grad = torch.ones_like(input2) * 1e-4

            # ref_model run 2 ref_model forward & backward microbatch
            ref_block.set_input_tensor(input1)
            out1_ref = ref_block(input1, None) # fwd
            out1_ref.backward(out1_grad) # bwd
            input1_ref_grad = input1.grad
            input1.grad = None

            ref_block.set_input_tensor(input2)
            out2_ref = ref_block(input2, None)  # fwd
            out2_ref.backward(out2_grad)  # bwd
            input2_ref_grad = input2.grad
            input2.grad = None

            args.moe_fb_overlap = True
            if memory_option == 'level0':
                args.moe_zero_memory = memory_option
            from mindspeed.features_manager.moe.fb_overlap import MoEFwdBwdOverlapFeature
            MoEFwdBwdOverlapFeature().register_patches(MindSpeedPatchesManager, args)
            MindSpeedPatchesManager.apply_patches()


            model = initialize_gpt_model(
                seed, etp_size, pre_process=False, post_process=False, sequence_parallel=tp_size > 1,
                expert_model_parallel_size=ep_size, num_layers=2, tensor_model_parallel_size=tp_size,
                moe_token_dispatcher_type=dispatcher_type, moe_shared_expert_overlap=True,
                moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size, **recompute_options
            )
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                param.data = ref_param.data.clone()
            block = model.decoder

            # overlapped model run 1 forward, 1 forward and backward overlapping, 1 backward
            input1, input2 = input1.clone(), input2.clone()

            # 1 fwd
            block.set_input_tensor(input1)
            out1 = block(input1, None)
            out1_graphs = block.get_fwd_layer_graphs()
            out1 = out1.clone() # out1 will be resized in backward

            # 1 forward and backward overlapping
            block.set_input_tensor(input2)
            out2 = block(
                input2,
                None,
                bwd_block_output_grad=torch.ones_like(input1) * 1e-4,
                bwd_block_graphs=out1_graphs
            )
            out2_graphs = block.get_fwd_layer_graphs()
            input1_grad = block.get_pp_comm_output().input_tensor_grad
            out2 = out2.clone()


            # 1 bwd
            input2_grad = transformer_block_backward(torch.ones_like(out2) * 1e-4, out2_graphs)

            # Check the fwd output, bwd grad(dx) are all equal.
            assert torch.equal(out1, out1_ref)
            assert torch.equal(out2, out2_ref)
            assert torch.equal(input1_grad, input1_ref_grad)
            assert torch.equal(input2_grad, input2_ref_grad)

            revert_patches()
            destroy_model_parallel()



    @pytest.mark.parametrize('pre_post', [(False, True)])
    def test_gpt_model_forward_backward_overlap(self, pre_post):
        def embedding_forward_wrapper(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                out = fn(*args, **kwargs)
                return out.bfloat16()

            return wrapper

        # mock the validate in transformer_config post init
        with patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.pre_validate_features_args') as mock_pre_validate, \
             patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.post_validate_features_args') as mock_post_validate, \
             patch('mindspeed.features_manager.features_manager.MindSpeedFeaturesManager.validate_features_args') as mock_validate:

            mock_pre_validate.return_value = None
            mock_post_validate.return_value = None
            mock_validate.return_value = None

            preprocess, postprocess = pre_post
            from mindspeed.patch_utils import MindSpeedPatchesManager
            tp_size, ep_size, etp_size = 2, 8, 1

            args = parse_args(None, True)
            args.npu_deterministic = True
            args.use_flash_attn = True
            args.moe_token_dispatcher_type = 'alltoall'
            args.gradient_accumulation_fusion = False
            args.moe_zero_memory = 'disable'
            args.sequence_parallel = tp_size > 1
            args.n_shared_experts = 1
            moe_shared_expert_intermediate_size = args.n_shared_experts * 1024

            set_args(args)
            seed = 1234
            set_random_seed(seed)
            extend_seed_all(seed)
            initialize_model_parallel(tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size, expert_tensor_parallel_size=etp_size)

            args.moe_fb_overlap = False
            ref_model = initialize_gpt_model(
                seed, etp_size, pre_process=preprocess, post_process=postprocess, sequence_parallel=tp_size > 1,
                expert_model_parallel_size=ep_size, num_layers=2, tensor_model_parallel_size=tp_size,
                moe_token_dispatcher_type='alltoall', moe_shared_expert_overlap=True,
                moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size
            )
            if preprocess:
                ref_model.embedding.forward = types.MethodType(embedding_forward_wrapper(ref_model.embedding.forward), ref_model.embedding)

            if preprocess:
                position_ids = torch.arange(0, self.sequence_length, device='npu').view(1, -1)
                input1 = torch.arange(0, self.sequence_length, device='npu').view(1, -1)
                input2 = torch.arange(1, self.sequence_length + 1, device='npu').view(1, -1)
            else:
                input1 = get_input_split_by_sp_ep(tp_size, ep_size, self.sequence_length, self.hidden_size)
                input2 = get_input_split_by_sp_ep(tp_size, ep_size, self.sequence_length, self.hidden_size)

            if preprocess:
                out1_ref = ref_model(input1, position_ids, None)
            else:
                ref_model.set_input_tensor(input1)
                out1_ref = ref_model(None, None, None)

            out1_ref.backward(torch.ones_like(out1_ref) * 1e-4)

            if preprocess:
                out2_ref = ref_model(input2, position_ids, None)
            else:
                ref_model.set_input_tensor(input2)
                out2_ref = ref_model(None, None, None)

            out2_ref.backward(torch.ones_like(out2_ref) * 1e-4)


            args.moe_fb_overlap = True
            from mindspeed.features_manager.moe.fb_overlap import MoEFwdBwdOverlapFeature
            MoEFwdBwdOverlapFeature().register_patches(MindSpeedPatchesManager, args)
            MindSpeedPatchesManager.apply_patches()

            model = initialize_gpt_model(
                seed, etp_size, pre_process=preprocess, post_process=postprocess, sequence_parallel=tp_size > 1,
                expert_model_parallel_size=ep_size, num_layers=2, tensor_model_parallel_size=tp_size,
                moe_token_dispatcher_type='alltoall', moe_shared_expert_overlap=True,
                moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size
            )
            if preprocess:
                model.embedding.forward = types.MethodType(embedding_forward_wrapper(model.embedding.forward), model.embedding)

            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                param.data = ref_param.data.clone()

            input1, input2 = input1.clone(), input2.clone()

            if preprocess:
                position_ids = position_ids.clone()

            if preprocess:
                out1 = model(input1, position_ids, None)
            else:
                model.set_input_tensor(input1)
                out1 = model(None, None, None)

            out1_result = out1.clone()

            out1_graphs = model.decoder.get_fwd_layer_graphs()
            if postprocess:
                out1.backward(torch.ones_like(out1) * 1e-4)
                bwd_block_output_grad = out1_graphs[-1].unperm2_graph[1].grad
            else:
                bwd_block_output_grad = torch.ones_like(out1) * 1e-4

            extra_block_kwargs = {
                'bwd_block_output_grad': bwd_block_output_grad, 'bwd_block_graphs': out1_graphs,
                'pp_comm_params': None, 'bwd_pp_comm_params': None
            }

            if preprocess:
                out2 = model(input2, position_ids, None, extra_block_kwargs=extra_block_kwargs)
            else:
                model.set_input_tensor(input2)
                out2 = model(None, None, None, extra_block_kwargs=extra_block_kwargs)

            out2_result = out2.clone() # out2 will be resized on backward, so clone the result for compare.
            out2_graphs = model.decoder.get_fwd_layer_graphs()
            if postprocess:
                out2.backward(torch.ones_like(out2) * 1e-4)
                model_grad = None
            else:
                model_grad = torch.ones_like(out2) * 1e-4

            gpt_model_backward(model_grad, out2_graphs)

            assert torch.equal(out1_result, out1_ref)
            assert torch.equal(out2_result, out2_ref)

            revert_patches()
            destroy_model_parallel()





















