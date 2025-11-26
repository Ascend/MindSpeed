import os
import pytest
import torch
import torch_npu
import mindspeed.megatron_adaptor

from tests_extend.unit_tests.common import DistributedTest
import megatron.core.parallel_state as Utils
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core import ModelParallelConfig


class TestPPSchedules(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)

    def test_get_forward_backward_func(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
        assert(schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
        assert(schedule.get_forward_backward_func() == schedule.forward_backward_pipelining_without_interleaving)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=2)
        assert(schedule.get_forward_backward_func() == schedule.forward_backward_pipelining_with_interleaving)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=4)
        assert(schedule.get_forward_backward_func() == schedule.forward_backward_pipelining_with_interleaving)
        Utils.destroy_model_parallel()

    def test_deallocate_output_tensor(self):
        out = torch.tensor([[1, 2, 3], [4, 5, 6]])
        schedule.deallocate_output_tensor(out)
        assert(out.nelement() == 6) 

    def test_forward_backward_func_with_pipeline_parallel(self):
        from megatron.core.pipeline_parallel import get_forward_backward_func
        rank = int(os.environ['LOCAL_RANK'])
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)

        def forward_step_func(data_iterator, model):
            rank = int(os.environ['LOCAL_RANK'])

            def loss_func(output_tensor):
                return rank, {'loss_reduced': rank}
            return torch.rand(512, 8, 256).cuda(), loss_func

        model = torch.nn.Linear(4, 1)
        model.model_type = 'unit-test'

        def set_input_tensor(input_tensor):
            return None
        model.set_input_tensor = set_input_tensor

        forward_backward_func = get_forward_backward_func()
        assert(schedule.get_forward_backward_func() == schedule.forward_backward_pipelining_without_interleaving)

        sequence_length = 512
        micro_batch_size = 8
        hidden_size = 256

        config = ModelParallelConfig(
            pipeline_model_parallel_size=4,
            sequence_parallel=False,
            pipeline_dtype=torch.float,
        )
        config.hidden_size = hidden_size
        model.config = config
        
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=None,
            model=[model],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            forward_only=True) 
        
        loss_reduced_expected = [{'loss_reduced': rank}, {'loss_reduced': rank}, {'loss_reduced': rank}, {'loss_reduced': rank}]
        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(losses_reduced)
            assert(i['loss_reduced'] == j['loss_reduced'])
        Utils.destroy_model_parallel()  

    def test_forward_backward_func_with_interleaving(self):
        from megatron.core.enums import ModelType
        from megatron.core.pipeline_parallel import get_forward_backward_func
        rank = int(os.environ['LOCAL_RANK'])

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )

        def forward_step_func(data_iterator, model):
            rank = int(os.environ['LOCAL_RANK'])

            def loss_func(output_tensor):
                return rank, {'loss_reduced': rank}

            return torch.rand(512, 8, 256).cuda(), loss_func

        model = torch.nn.Linear(4, 1)

        def set_input_tensor(input_tensor):
            return None

        model.set_input_tensor = set_input_tensor

        forward_backward_func = get_forward_backward_func()
        assert (
            schedule.get_forward_backward_func()
            == schedule.forward_backward_pipelining_with_interleaving
        )

        sequence_length = 512
        micro_batch_size = 8
        hidden_size = 256

        config = ModelParallelConfig(
            pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
        )
        config.hidden_size = hidden_size
        model.config = config

        loss_reduced_expected = [
            {'loss_reduced': rank},
            {'loss_reduced': rank},
            {'loss_reduced': rank},
            {'loss_reduced': rank},
        ]

        model.model_type = ModelType.encoder_and_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        model.model_type = ModelType.encoder_or_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=256,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        with pytest.raises(RuntimeError):
            model.model_type = ModelType.encoder_or_decoder
            forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=[range(0, 100), range(0, 100)],
                model=[model, model],
                num_microbatches=7,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=512,
                forward_only=True,
            )

        model.model_type = ModelType.encoder_or_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        Utils.destroy_model_parallel()

    def test_forward_backward_func_with_uneven_interleaving(self):
        from megatron.core.enums import ModelType
        from megatron.core.pipeline_parallel import get_forward_backward_func
        rank = int(os.environ['LOCAL_RANK'])

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )

        def forward_step_func(data_iterator, model):
            rank = int(os.environ['LOCAL_RANK'])

            def loss_func(output_tensor):
                return rank, {'loss_reduced': rank}

            return torch.rand(512, 8, 256).cuda(), loss_func

        model_a = torch.nn.Linear(4, 1)
        model_b = torch.nn.Linear(8, 1)

        def set_input_tensor(input_tensor):
            return None

        model_a.set_input_tensor = set_input_tensor
        model_b.set_input_tensor = set_input_tensor

        forward_backward_func = get_forward_backward_func()
        assert (
            schedule.get_forward_backward_func()
            == schedule.forward_backward_pipelining_with_interleaving
        )

        sequence_length = 512
        micro_batch_size = 8
        hidden_size = 256

        config = ModelParallelConfig(
            pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
        )
        config.hidden_size = hidden_size
        model_a.config = config
        model_b.config = config

        loss_reduced_expected = [
            {'loss_reduced': rank},
            {'loss_reduced': rank},
            {'loss_reduced': rank},
            {'loss_reduced': rank},
        ]

        model_a.model_type = ModelType.encoder_and_decoder
        model_b.model_type = ModelType.encoder_and_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=256,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        with pytest.raises(RuntimeError):
            model_a.model_type = ModelType.encoder_or_decoder
            model_b.model_type = ModelType.encoder_or_decoder
            forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=[range(0, 100)],
                model=[model_a, model_b],
                num_microbatches=7,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=512,
                forward_only=True,
            )

        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=micro_batch_size,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=True,
        )

        for i, j in zip(losses_reduced, loss_reduced_expected):
            print(f"losses_reduced: {i} loss_reduced_expected: {j}")
            assert i['loss_reduced'] == j['loss_reduced']

        Utils.destroy_model_parallel()

