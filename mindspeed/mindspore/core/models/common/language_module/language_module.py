# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import mindspore
import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule


def setup_embeddings_and_output_layer(language_model) -> None:
    """Sets up embedding layer in first stage and output layer in last stage.

    This function initalizes word embeddings in the final stage when we are
    using pipeline parallelism and sharing word embeddings, and sets up param
    attributes on the embedding and output layers.
    """

    # Set `is_embedding_or_output_parameter` attribute.
    if language_model.pre_process:
        language_model.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    if language_model.post_process and language_model.output_layer.weight is not None:
        language_model.output_layer.weight.is_embedding_or_output_parameter = True

    if not language_model.share_embeddings_and_output_weights and not getattr(
        language_model.config, 'mtp_num_layers', 0
    ):
        return

    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        language_model.shared_embedding_or_output_weight().zero_out_wgrad = True
        return

    if parallel_state.is_pipeline_first_stage() and language_model.pre_process and not language_model.post_process:
        language_model.shared_embedding_or_output_weight().shared_embedding = True

    if (language_model.post_process or getattr(language_model, 'mtp_process', False)) and not language_model.pre_process:
        assert not parallel_state.is_pipeline_first_stage()
        # set weights of the duplicated embedding to 0 here,
        # then copy weights from pre processing stage using all_reduce below.
        weight = language_model.shared_embedding_or_output_weight()
        device_target = mindspore.get_context('device_target')
        if device_target == 'CPU':
            np_data = np.zeros(weight.data.shape, dtype=np.float32)
            weight.data = mindspore.Tensor(np_data, dtype=weight.data.dtype)
        else:
            weight.data.fill_(0)
        weight.shared = True
        weight.shared_embedding = True


    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if parallel_state.is_rank_in_embedding_group():
            weight = language_model.shared_embedding_or_output_weight()
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