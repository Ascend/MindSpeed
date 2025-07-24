# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Union
import logging

import torch

from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.common.language_module.language_module import LanguageModule

from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)


def setup_embeddings_and_output_layer_with_mtp(self) -> None:
    """Sets up embedding layer in first stage and output layer in last stage.

    This function initalizes word embeddings in the final stage when we are
    using pipeline parallelism and sharing word embeddings, and sets up param
    attributes on the embedding and output layers.
    """

    # Set `is_embedding_or_output_parameter` attribute.
    if self.pre_process:
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    if self.post_process and self.output_layer.weight is not None:
        self.output_layer.weight.is_embedding_or_output_parameter = True

    # If share_embeddings_and_output_weights is True, we need to maintain duplicated
    # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
    # we also need to maintain duplicated embedding weights in mtp process stage.
    # So we need to copy embedding weights from pre processing stage as initial parameters
    # in these cases.

    # DualpipeV.
    if not self.share_embeddings_and_output_weights and \
        not getattr(
        self.config, 'mtp_num_layers', 0) or \
        self.config.schedules_method == 'dualpipev':
        return

    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        self.shared_embedding_or_output_weight().zero_out_wgrad = True
        return

    if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
        self.shared_embedding_or_output_weight().shared_embedding = True

    if (self.post_process or getattr(self, 'mtp_process', False)) and not self.pre_process:
        assert not parallel_state.is_pipeline_first_stage()
        # set weights of the duplicated embedding to 0 here,
        # then copy weights from pre processing stage using all_reduce below.
        weight = self.shared_embedding_or_output_weight()
        weight.data.fill_(0)
        weight.shared = True
        weight.shared_embedding = True

    # Parameters are shared between the word embeddings layers, and the
    # heads at the end of the model. In a pipelined setup with more than
    # one stage, the initial embedding layer and the head are on different
    # workers, so we do the following:
    # 1. Create a second copy of word_embeddings on the last stage, with
    #    initial parameters of 0.0.
    # 2. Do an all-reduce between the first and last stage to ensure that
    #    the two copies of word_embeddings start off with the same
    #    parameter values.
    # 3. In the training loop, before an all-reduce between the grads of
    #    the two word_embeddings layers to ensure that every applied weight
    #    update is the same on both stages.

    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if parallel_state.is_rank_in_embedding_group():
            weight = self.shared_embedding_or_output_weight()
            weight.data = weight.data.cuda()
            torch.distributed.all_reduce(
                weight.data, group=parallel_state.get_embedding_group()
            )

    elif not getattr(LanguageModule, "embedding_warning_printed", False):
        logging.getLogger(__name__).warning(
            "Distributed processes aren't initialized, so the output layer "
            "is not initialized with weights from the word embeddings. "
            "If you are just manipulating a model this is fine, but "
            "this needs to be handled manually. If you are training "
            "something is definitely wrong."
        )
        LanguageModule.embedding_warning_printed = True

 
 
def model_provider_mtp(pre_process=True, post_process=True, use_dualpipe_mtp=False) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None and use_dualpipe_mtp:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)
            post_process = True

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

    return model


def dualpipev_get_mtp_num_layers_to_build(config: TransformerConfig) -> int:
    """Get the number of MTP layers to build in dualpipeV."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    if mpu.is_pipeline_first_stage() and config.schedules_method == 'dualpipev' and not config.dualpipev_first_chunk: 
        return config.mtp_num_layers if config.mtp_num_layers else 0
    if mpu.is_pipeline_last_stage() and not config.schedules_method == 'dualpipev':
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0
