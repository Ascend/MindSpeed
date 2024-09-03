import queue
import pytest
import torch
import torch_npu
from unit_tests.common import DistributedTest
from mindspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from mindspeed.core.weight_grad_store import WeightGradStore


class TestDistributedOptimizer(DistributedTest):
    world_size = 4

    @staticmethod
    def setup_weight_grad_store():
        WeightGradStore.cache = []
        WeightGradStore.weight_grad_queue = queue.Queue()
        WeightGradStore.stored_grads = None
        WeightGradStore.store_grad_cache = None
        WeightGradStore.swap_event = []
        WeightGradStore.prefetch_stream = None
        WeightGradStore.gather_stream = None
        WeightGradStore.host_tensors_gradoutput = []
        WeightGradStore.host_pipe_experts_grad = []
        WeightGradStore.host_tensors_input = []
        WeightGradStore.ori_storage = []

        WeightGradStore.is_decoupleBlock = False

    @pytest.mark.parametrize("grad_output", [torch.tensor([[0.01, 0.02, 0.01], [0.02, 0.01, 0.03]]), None])
    def test_flush(self, grad_output):
        self.setup_weight_grad_store()

        args = parse_args(None, True)
        args.use_nanopipe_swap = True
        set_args(args)

        total_input = torch.tensor([[0.1, 0.2], [0.4, 0.5]])
        WeightGradStore.put(total_input, grad_output, 'weight1', True, False, False)
        if grad_output is not None:
            assert len(WeightGradStore.host_tensors_gradoutput) == 1
            grad_output_expected_size, grad_output_expected_tensor = (6, grad_output.cpu())
            grad_output_actual_size, grad_output_actual_tensor = WeightGradStore.host_tensors_gradoutput[0]
            assert grad_output_expected_size == grad_output_actual_size
            assert torch.equal(grad_output_actual_tensor, grad_output_expected_tensor), "grad_output张量内容不匹配"

        assert len(WeightGradStore.host_tensors_input) == 1
        total_input_expected_size, total_input_expected_tensor = (4, total_input.cpu())
        total_input_actual_size, total_input_actual_tensor = WeightGradStore.host_tensors_input[0]
        assert total_input_actual_size == total_input_expected_size
        assert torch.equal(total_input_actual_tensor, total_input_expected_tensor), "total_input张量内容不匹配"

        assert len(WeightGradStore.cache) == 1
        assert torch.equal(WeightGradStore.cache[0][0], total_input)
        if grad_output is not None:
            assert torch.equal(WeightGradStore.cache[0][1], grad_output)
        else:
            assert WeightGradStore.cache[0][1] is None
        assert WeightGradStore.cache[0][2:] == ('weight1', True, False, False)

        WeightGradStore.flush()
        assert WeightGradStore.cache == []
        assert not WeightGradStore.weight_grad_queue.empty()
        assert WeightGradStore.weight_grad_queue.get() == [(total_input, grad_output, 'weight1', True, False, False)]

    @pytest.mark.parametrize("tp_pp", [(2, 2)])
    @pytest.mark.parametrize("row", [True, False])
    @pytest.mark.parametrize("sequence_parallel", [True, False])
    def test_overlap_all_gather(self, tp_pp, row, sequence_parallel):
        self.setup_weight_grad_store()
        input_data = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).npu()
        grad_output = torch.tensor([[0.01, 0.02, 0.01], [0.02, 0.01, 0.03]]).npu()
        weight = torch.tensor([[0.3, 0.2, 0.1], [0.5, 0.4, 0.3]]).npu()
        (tp, pp) = tp_pp
        in_row = row
        sequence_parallel = sequence_parallel
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None)
        WeightGradStore.cache = [(input_data, grad_output, weight, sequence_parallel, in_row, False)]
        if WeightGradStore.gather_stream is None:
            WeightGradStore.gather_stream = torch_npu.npu.Stream(device=torch.npu.current_device)
        result, handle = WeightGradStore.overlap_all_gather()
        diff_value = 0.00001
        if handle is not None:
            handle.wait()
        if not sequence_parallel:
            assert result == (input_data, grad_output, weight, sequence_parallel, in_row, False)
        elif in_row:
            assert torch.allclose(result[0], input_data, rtol=diff_value, atol=diff_value)
            assert torch.allclose(result[1], torch.cat((grad_output, grad_output), dim=0), rtol=diff_value, atol=diff_value)
        else:
            assert torch.allclose(result[0], torch.cat((input_data, input_data), dim=0), rtol=diff_value, atol=diff_value)
            assert torch.allclose(result[1], grad_output, rtol=diff_value, atol=diff_value)

    @pytest.mark.parametrize("tp_pp", [(2, 2)])
    def test_pop(self, tp_pp):
        self.setup_weight_grad_store()
        input_data = torch.randn(1, 2, 3).npu()
        grad_output = torch.randn(1, 2, 3).npu()
        weight = torch.randn(3, 3).npu()
        weight.grad = torch.randn(3, 3).npu()
        setattr(weight, 'main_grad', weight.clone())
        (tp, pp) = tp_pp

        args = parse_args(None, True)
        args.use_nanopipe_swap = True
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None)

        if WeightGradStore.gather_stream is None:
            WeightGradStore.gather_stream = torch_npu.npu.Stream(device=torch.npu.current_device)

        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)
        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)

        WeightGradStore.pop()
        assert WeightGradStore.store_grad_cache is None
        assert WeightGradStore.stored_grads is None
        assert WeightGradStore.swap_event == []
        assert WeightGradStore.grad_store == []
        assert WeightGradStore.host_pipe_experts_grad == []


    @pytest.mark.parametrize("tp_pp", [(2, 2)])
    def test_swap_tensors(self, tp_pp):
        self.setup_weight_grad_store()
        input_data = torch.randn(1, 2, 3).npu()
        grad_output = torch.randn(1, 2, 3).npu()
        weight = torch.randn(3, 3).npu()
        weight.grad = torch.randn(3, 3).npu()
        setattr(weight, 'main_grad', weight.clone())
        (tp, pp) = tp_pp

        args = parse_args(None, True)
        args.use_nanopipe_swap = True
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
                                  pipeline_model_parallel_size=pp,
                                  virtual_pipeline_model_parallel_size=None,
                                  pipeline_model_parallel_split_rank=None)

        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)
        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)

        WeightGradStore.swap_tensors()
        assert WeightGradStore.swap_event is not None
