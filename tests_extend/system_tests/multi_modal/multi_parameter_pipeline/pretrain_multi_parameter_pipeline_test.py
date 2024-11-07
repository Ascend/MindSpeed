# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.init as init

import mindspeed.megatron_adaptor

from functools import partial

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.training import pretrain
from megatron.training.arguments import core_transformer_config_from_args


class MultiParamSimpleModel(nn.Module):
    def __init__(self, input_size):
        super(MultiParamSimpleModel, self).__init__()
        args = get_args()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        self.input_tensor = []

        self.fc1 = nn.Linear(input_size, input_size, bias=False)
        self.fc2 = nn.Linear(input_size, input_size, bias=False)
        self.fc3 = nn.Linear(input_size, input_size, bias=False)
        self.fc4 = nn.Linear(input_size, input_size, bias=False)
        self.fc5 = nn.Linear(input_size, input_size, bias=False)
        self.fc6 = nn.Linear(input_size, input_size, bias=False)
        self.fc7 = nn.Linear(input_size, input_size, bias=False)
        self.fc8 = nn.Linear(input_size, input_size, bias=False)

        init.constant_(self.fc1.weight, 0.001)
        init.constant_(self.fc2.weight, 0.001)
        init.constant_(self.fc3.weight, 0.001)
        init.constant_(self.fc4.weight, 0.001)
        init.constant_(self.fc5.weight, 0.001)
        init.constant_(self.fc6.weight, 0.001)
        init.constant_(self.fc7.weight, 0.001)
        init.constant_(self.fc8.weight, 0.001)

        if args.virtual_pipeline_model_parallel_size is not None:
            vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            if vpp_rank == 0:
                self.layers = [self.fc1, self.fc2, self.fc3, self.fc4][pp_rank:pp_rank + 1]
            else:
                self.layers = [self.fc5, self.fc6, self.fc7, self.fc8][pp_rank:pp_rank + 1]
        else:
            if pp_size > 1:
                self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8][
                              pp_rank * 2:(pp_rank + 1) * 2]
            else:
                self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]

    def forward(self, x, y):
        if len(self.input_tensor) > 0 and self.input_tensor[0] is not None:
            x = self.input_tensor[0]
            y = self.input_tensor[1]
            self.input_tensor = []

        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            x = x * y + y
        return [x, y]

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = MultiParamSimpleModel(args.hidden_size)
    model.config = config

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    model.pre_process = False
    model.post_process = False
    model.share_embeddings_and_output_weights = False

    if args.virtual_pipeline_model_parallel_size is not None:
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        if vpp_rank == 0 and pp_rank == 0:
            model.pre_process = True
        if vpp_rank == 1 and pp_rank == pp_size - 1:
            model.post_process = True
    else:
        if pp_rank == 0:
            model.pre_process = True
        if pp_rank == pp_size - 1:
            model.post_process = True

    args.pipeline_tensor_shapes = [
        {'shape': (args.micro_batch_size, args.hidden_size), 'dtype': torch.float32},
        {'shape': (args.micro_batch_size, args.hidden_size), 'dtype': torch.float32}
    ]
    setattr(forward_step, 'pipeline_tensor_shapes', args.pipeline_tensor_shapes)

    return model


def loss_func(label, output_tensor):
    criterion = nn.MSELoss()
    output_tensor = output_tensor[0]
    loss = criterion(label, output_tensor)
    reporting_loss = loss.clone().detach()
    return loss, {'lm loss': reporting_loss}


def forward_step(data_iterator, model):
    args = get_args()
    if args.virtual_pipeline_model_parallel_size is not None:
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
    else:
        vpp_rank = 0
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if vpp_rank == 0 and pp_rank == 0:
        x = torch.randn(args.micro_batch_size, args.hidden_size, device=torch.cuda.current_device()).npu()
        y = torch.randn(args.micro_batch_size, args.hidden_size, device=torch.cuda.current_device()).npu()
    else:
        x = None
        y = None

    label = torch.zeros(args.micro_batch_size, args.hidden_size, device=torch.cuda.current_device())

    output_tensor = model(x, y)

    return output_tensor, partial(loss_func, label)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)
    config.mock = True

    dataset_type = MockGPTDataset

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
