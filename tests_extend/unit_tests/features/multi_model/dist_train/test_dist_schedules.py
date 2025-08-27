# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import unittest
from unittest.mock import patch, MagicMock, call, ANY
import torch
from torch_npu.contrib import transfer_to_npu
from mindspeed.core.multi_modal.dist_train import dist_schedules


class TestDistSchedules(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.dist_train = True
        self.mock_args.world_size = 8
        self.mock_mpu = MagicMock()
        self.mock_config = MagicMock(
            hidden_size=256,
            params_dtype=torch.float32,
            sequence_parallel=False,
            deallocate_pipeline_outputs=False,
            overlap_p2p_comm=False,
            timers=None,
            no_sync_func=None,
            barrier_with_L1_time=False,
            grad_sync_func=None,
            finalize_model_grads_func=None,
            calculate_per_token_loss=False,
            num_microbatches_with_partial_activation_checkpoints=None
        )

        self.dist_patcher = patch("torch.distributed.is_initialized", return_value=True)
        self.dist_patcher.start()
        self.rank_patcher = patch("torch.distributed.get_rank", return_value=0)
        self.rank_patcher.start()

    def tearDown(self):
        self.dist_patcher.stop()
        self.rank_patcher.stop()

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_args")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_all_config")
    def test_initialize_distributed_wrapper(self, mock_get_all_config, mock_get_args):
        mock_get_args.return_value = self.mock_args
        mock_get_all_config.return_value = {
            "model1": MagicMock(world_size=4),
            "model2": MagicMock(world_size=4)
        }

        mock_original = MagicMock()
        wrapped = dist_schedules.initialize_distributed_wrapper(mock_original)

        wrapped("arg1", kwarg="value")

        self.assertEqual(self.mock_args.world_size, 8)
        mock_original.assert_called_once_with("arg1", kwarg="value")
        self.assertEqual(self.mock_args.world_size, 8)

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.mpu")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_dist_model_name")
    def test_get_checkpoint_name(self, mock_get_dist_model_name, mock_mpu):
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 2
        mock_mpu.get_tensor_model_parallel_rank.return_value = 1
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        mock_get_dist_model_name.return_value = "model"

        path = dist_schedules._get_checkpoint_name("/checkpoints", 100)
        self.assertIn("mp_model_rank_01_000", path)

        release_path = dist_schedules._get_checkpoint_name("/checkpoints", 100, release=True)
        self.assertIn("release", release_path)

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_pipeline_model_parallel_rank")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_pipeline_model_parallel_prev_rank")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_pipeline_model_parallel_next_rank")
    def test_p2p_ops_even_rank(self, mock_next_rank, mock_prev_rank, mock_rank):
        mock_rank.return_value = 0
        mock_prev_rank.return_value = 3
        mock_next_rank.return_value = 1

        send_prev = torch.ones(2)
        recv_prev = torch.zeros(2)
        send_next = torch.ones(3)
        recv_next = torch.zeros(3)

        with patch("torch.distributed.isend") as mock_isend, \
                patch("torch.distributed.irecv") as mock_irecv:
            dist_schedules._p2p_ops(
                tensor_send_prev=send_prev,
                tensor_recv_prev=recv_prev,
                tensor_send_next=send_next,
                tensor_recv_next=recv_next,
                group=None,
                prev_pipeline_rank=mock_prev_rank.return_value,
                next_pipeline_rank=mock_next_rank.return_value
            )

            mock_isend.assert_has_calls([
                call(tensor=send_next, dst=1, group=None),
                call(tensor=send_prev, dst=3, group=None)
            ], any_order=False)
            mock_irecv.assert_has_calls([
                call(tensor=recv_prev, src=3, group=None),
                call(tensor=recv_next, src=1, group=None)
            ])

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.is_use_multiparam_send_recv")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_tensor_model_parallel_world_size")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.is_pipeline_stage_before_split")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_context_parallel_world_size")
    def test_get_tensor_shapes(self, mock_cp_size, mock_before_split, mock_tp_size, mock_use_multiparam):
        mock_cp_size.return_value = 2
        mock_tp_size.return_value = 4
        mock_before_split.return_value = True
        mock_use_multiparam.return_value = False

        shapes = dist_schedules.get_tensor_shapes(
            rank=0,
            model_type=dist_schedules.ModelType.encoder_and_decoder,
            seq_length=512,
            micro_batch_size=2,
            decoder_seq_length=256,
            config=self.mock_config,
            encoder_decoder_xattn=False
        )

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], (256, 2, 256))

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_pipeline_model_parallel_rank")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_model_config")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_dist_model_config")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.is_forward_only_model")
    def test_forward_backward_pipeline_forward_only(
            self, mock_is_forward_only_model, mock_dist_config, mock_get_config, mock_rank):
        mock_rank.return_value = 1
        mock_get_config.return_value = self.mock_config
        mock_dist_config.return_value = MagicMock(
            pipeline_model_parallel_size=4,
            model_index=0,
            forward_only=True
        )
        mock_is_forward_only_model.return_value = True
        mock_model = MagicMock()
        mock_data_iter = MagicMock()

        with patch('mindspeed.core.multi_modal.dist_train.dist_schedules.get_tensor_shapes') as mock_get_shapes, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.schedules.forward_step") as mock_forward_step, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.recv_forward") as mock_recv_forward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.send_forward") as mock_send_forward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.generate_send_recv_mask") as mock_send_recv_mask:
            mock_get_shapes.return_value = [{'shape': ((10, 4, 8)), 'dtype': torch.float16}]
            mock_forward_step.return_value = ([torch.tensor(1.0)], torch.tensor(16))
            mock_recv_forward.return_value = [torch.randn(2, 4, 256)]
            mock_send_forward.return_value = [torch.randn(2, 4, 256)]
            mock_send_recv_mask.return_value = {
                'send_forward': True,
                'send_backward': False,
                'recv_forward': True,
                'recv_backward': False
            }
            dist_schedules.forward_backward_pipelining_without_interleaving(
                forward_step_func=MagicMock(),
                data_iterator=mock_data_iter,
                model=mock_model,
                num_microbatches=4,
                seq_length=512,
                micro_batch_size=2,
                forward_only=False
            )
        mock_forward_step.assert_called()

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_pipeline_model_parallel_rank")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_model_config")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_dist_model_config")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.is_forward_only_model")
    def test_forward_backward_pipeline(
            self, mock_is_forward_only_model, mock_dist_config, mock_get_config, mock_rank):
        mock_rank.return_value = 1
        mock_get_config.return_value = self.mock_config
        mock_dist_config.return_value = MagicMock(
            pipeline_model_parallel_size=4,
            model_index=0,
            forward_only=False
        )

        mock_is_forward_only_model.return_value = True

        mock_model = MagicMock()
        mock_data_iter = MagicMock()

        with patch('mindspeed.core.multi_modal.dist_train.dist_schedules.get_tensor_shapes') as mock_get_shapes, \
                patch(
                    "mindspeed.core.multi_modal.dist_train.dist_schedules.schedules.forward_step") as mock_forward_step, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.recv_forward") as mock_recv_forward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.send_forward") as mock_send_forward, \
                patch(
                    "mindspeed.core.multi_modal.dist_train.dist_schedules.generate_send_recv_mask") as mock_send_recv_mask, \
                patch(
                    "mindspeed.core.multi_modal.dist_train.dist_schedules.send_forward_recv_backward") as mock_send_forward_recv_backward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.recv_backward") as mock_recv_backward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules._backward_step") as mock_backward_step, \
                patch(
                    "mindspeed.core.multi_modal.dist_train.dist_schedules.send_backward_recv_forward") as mock_send_backward_recv_forward, \
                patch("mindspeed.core.multi_modal.dist_train.dist_schedules.send_backward") as mock_send_backward:
            mock_get_shapes.return_value = [{'shape': ((10, 4, 8)), 'dtype': torch.float16}]
            mock_forward_step.return_value = ([torch.tensor(1.0)], torch.tensor(16))
            mock_recv_forward.return_value = [torch.randn(2, 4, 256)]
            mock_send_forward.return_value = [torch.randn(2, 4, 256)]
            mock_send_recv_mask.return_value = {
                'send_forward': False,
                'send_backward': False,
                'recv_forward': False,
                'recv_backward': False
            }
            mock_send_forward_recv_backward.return_value = torch.tensor(10.0)
            mock_backward_step.return_value = torch.tensor(5.0)
            mock_send_backward_recv_forward.return_value = torch.tensor(10.0)
            mock_send_backward.return_value = torch.tensor(5.0)
            mock_recv_backward.return_value = torch.tensor(8.0)
            dist_schedules.forward_backward_pipelining_without_interleaving(
                forward_step_func=MagicMock(),
                data_iterator=mock_data_iter,
                model=mock_model,
                num_microbatches=4,
                seq_length=512,
                micro_batch_size=2,
                forward_only=False
            )
        mock_forward_step.assert_called()

    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_send_recv_fun")
    @patch("mindspeed.core.multi_modal.dist_train.dist_schedules.get_dst_ranks")
    def test_send_forward(self, mock_dst_ranks, mock_send_func):
        send_recv_ops = {"send_forward": True}
        mock_dst_ranks.return_value = [2, 3]
        mock_send_func.return_value = MagicMock()
        dist_schedules.send_forward(
            ["tensor_data"],
            "tensor_shapes",
            self.mock_config,
            send_recv_ops
        )
        mock_send_func.return_value.assert_called_once_with(["tensor_data"], False, [2, 3])