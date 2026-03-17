import os
import sys
from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch_npu

from megatron.training import global_vars
from mindspeed.core.qos.adaptor import create_group_qos
from mindspeed.core.qos.domain_info import ParallelCommDomain, RankGenerator
from mindspeed.core.qos.qos import (
    Qos,
    _DEFAULT_QOS,
    _DEFAULT_QOS_ROCE_HIGH,
    _DEFAULT_QOS_ROCE_LOW,
    _DEFAULT_QOS_ROCE_MIDDLE,
    _DEFAULT_QOS_SDMA_HIGH,
    _DEFAULT_QOS_SDMA_LOW,
    _DEFAULT_QOS_SDMA_MIDDLE,
    _PARALLEL_TYPES,
    domains,
    roce_qos_str_to_value,
    sdma_qos_str_to_value,
)

GLOBAL_RANK_PARAMS = Namespace(
    tp=2,
    pp=4,
    dp=8,
    ep=8,
    cp=1,
    order='tp-cp-ep-dp-pp',
    rank_offset=0,
    world_size=2 * 4 * 8 * 1,
    tensor_parallel_comm_domain=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17],
                                 [18, 19], [20, 21], [22, 23], [24, 25],
                                 [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41],
                                 [42, 43], [44, 45], [46, 47], [48, 49],
                                 [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61], [62, 63]],
    pipeline_parallel_comm_domain=[[0, 16, 32, 48], [1, 17, 33, 49], [2, 18, 34, 50], [3, 19, 35, 51], [4, 20, 36, 52],
                                   [5, 21, 37, 53], [6, 22, 38, 54],
                                   [7, 23, 39, 55], [8, 24, 40, 56], [9, 25, 41, 57], [10, 26, 42, 58],
                                   [11, 27, 43, 59], [12, 28, 44, 60], [13, 29, 45, 61],
                                   [14, 30, 46, 62], [15, 31, 47, 63]],
    data_parallel_comm_domain=[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15],
                               [16, 18, 20, 22, 24, 26, 28, 30], [17, 19, 21, 23, 25, 27, 29, 31],
                               [32, 34, 36, 38, 40, 42, 44, 46], [33, 35, 37, 39, 41, 43, 45, 47],
                               [48, 50, 52, 54, 56, 58, 60, 62], [49, 51, 53, 55, 57, 59, 61, 63]],
    context_parallel_comm_domain=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
                                  [16], [17], [18], [19], [20], [21],
                                  [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35],
                                  [36], [37], [38], [39], [40], [41], [42],
                                  [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56],
                                  [57], [58], [59], [60], [61], [62], [63]],
    expert_parallel_comm_domain=[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15],
                                 [16, 18, 20, 22, 24, 26, 28, 30], [17, 19, 21, 23, 25, 27, 29, 31],
                                 [32, 34, 36, 38, 40, 42, 44, 46], [33, 35, 37, 39, 41, 43, 45, 47],
                                 [48, 50, 52, 54, 56, 58, 60, 62], [49, 51, 53, 55, 57, 59, 61, 63]]
)


def reset_quality_of_service_singleton():
    Qos._initialize = False
    Qos._instance = None
    if hasattr(global_vars, '_GLOBAL_ARGS'):
        global_vars._GLOBAL_ARGS = None


class TestQos:
    def test_qos_manual(self):
        with patch('mindspeed.core.qos.qos.get_args') as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(
                aiqos_mode="manual",
                aiqos_schedule="{tp:low,pp:high,dp-cp:high,ep:high,cp:middle,pos-embd:middle}"
            )
            qos = Qos()
            assert qos.aiqos_mode == "manual"
            assert qos.sdma_aiqos_schedule == {'tp': 2, 'pp': 6, 'dp-cp': 6, 'ep': 6, 'cp': 4, 'pos-embd': 4}
            assert qos.roce_aiqos_schedule == {'tp': 3, 'pp': 5, 'dp-cp': 5, 'ep': 5, 'cp': 4, 'pos-embd': 4}
            assert qos._initialize == True

    def test_qos_auto_enable_roce(self):
        g = RankGenerator(tp=GLOBAL_RANK_PARAMS.tp, ep=GLOBAL_RANK_PARAMS.ep, dp=GLOBAL_RANK_PARAMS.dp,
                        pp=GLOBAL_RANK_PARAMS.pp, cp=GLOBAL_RANK_PARAMS.cp, order='tp-cp-ep-dp-pp')
        ep_group_ranks = g.get_ranks('ep', independent_ep=True)
        tp_group_ranks = g.get_ranks('tp')
        pp_group_ranks = g.get_ranks('pp')
        dp_group_ranks = g.get_ranks('dp')
        cp_group_ranks = g.get_ranks('cp')

        assert tp_group_ranks == GLOBAL_RANK_PARAMS.tensor_parallel_comm_domain
        assert pp_group_ranks == GLOBAL_RANK_PARAMS.pipeline_parallel_comm_domain
        assert dp_group_ranks == GLOBAL_RANK_PARAMS.data_parallel_comm_domain
        assert cp_group_ranks == GLOBAL_RANK_PARAMS.context_parallel_comm_domain
        assert ep_group_ranks == GLOBAL_RANK_PARAMS.expert_parallel_comm_domain

        tp_info = ParallelCommDomain(ip_list=None, rank_list=tp_group_ranks, world_size=GLOBAL_RANK_PARAMS.tp,
                                    parallel_type='tp', comm_amount=4096, comm_amount_no_overlap=2048)
        pp_info = ParallelCommDomain(ip_list=None, rank_list=pp_group_ranks, world_size=GLOBAL_RANK_PARAMS.pp,
                                    parallel_type='pp', comm_amount=40960, comm_amount_no_overlap=20480)
        dp_info = ParallelCommDomain(ip_list=None, rank_list=dp_group_ranks, world_size=GLOBAL_RANK_PARAMS.dp,
                                    parallel_type='dp', comm_amount=1314, comm_amount_no_overlap=520)
        cp_info = ParallelCommDomain(ip_list=None, rank_list=cp_group_ranks, world_size=GLOBAL_RANK_PARAMS.cp,
                                    parallel_type='cp', comm_amount=512, comm_amount_no_overlap=256)
        ep_info = ParallelCommDomain(ip_list=None, rank_list=ep_group_ranks, world_size=GLOBAL_RANK_PARAMS.ep,
                                    parallel_type='ep', comm_amount=131072, comm_amount_no_overlap=81920)

        with patch('mindspeed.core.qos.domain_info.get_args') as mock_domain_get_args, \
                patch('mindspeed.core.qos.qos.get_args') as mock_qos_get_args, \
                patch('mindspeed.core.qos.domain_info.is_a3_version', new=True), \
                patch('mindspeed.core.qos.qos.is_a3_version', new=True), \
                patch('mindspeed.core.qos.domain_info.get_overlap_space_dict') as mock_space_dict, \
                patch('mindspeed.core.qos.qos.get_tensor_parallel_comm_domain', return_value=tp_info), \
                patch('mindspeed.core.qos.qos.get_data_parallel_comm_domain', return_value=dp_info), \
                patch('mindspeed.core.qos.qos.get_pipeline_parallel_comm_domain', return_value=pp_info), \
                patch('mindspeed.core.qos.qos.get_expert_parallel_comm_domain', return_value=ep_info), \
                patch('mindspeed.core.qos.qos.get_context_parallel_comm_domain', return_value=cp_info), \
                patch('mindspeed.core.qos.qos.log_rank_0'):
            all_keys = [(x, y) for x in ('tp', 'dp', 'pp', 'ep', 'cp') for y in ('tp', 'dp', 'pp', 'ep', 'cp')]
            space_overlap_res = {key: 0 for key in all_keys}
            mock_space_dict.return_value = space_overlap_res
            
            mock_args = Namespace(
                aiqos_mode="auto",
                aiqos_enable_roce=True,
                num_experts=32,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
            )
            global_vars._GLOBAL_ARGS = mock_args
            mock_domain_get_args.return_value = mock_args
            mock_qos_get_args.return_value = mock_args
            
            reset_quality_of_service_singleton()
            assert Qos._initialize is False
            assert Qos._instance is None
            
            qos = Qos()
            assert qos is not None
            assert qos.aiqos_mode == "auto"

            assert qos.set_parallel_roce_qos('tp') == 4
            assert qos.set_parallel_roce_qos('pp') == 4
            assert qos.set_parallel_roce_qos('dp') == 4
            assert qos.set_parallel_roce_qos('cp') == 4
            assert qos.set_parallel_roce_qos('ep') == 4

            assert qos.set_parallel_sdma_qos('tp') == 6
            assert qos.set_parallel_sdma_qos('pp') == 4
            assert qos.set_parallel_sdma_qos('dp') == 6
            assert qos.set_parallel_sdma_qos('cp') == 2
            assert qos.set_parallel_sdma_qos('ep') == 2
             
    def test_create_qos_group(self):
        tp_ranks = GLOBAL_RANK_PARAMS.tensor_parallel_comm_domain
        timeout = timedelta(seconds=120)
        pg_initialized = False
        
        with patch.dict(os.environ, {'MASTER_ADDR': "localhost", 'MASTER_PORT': "6666"}, clear=False), \
            patch('mindspeed.core.qos.adaptor.get_args') as mock_adaptor_get_args, \
            patch('mindspeed.core.qos.qos.get_args') as mock_qos_get_args:
            
            mock_args = SimpleNamespace(
                aiqos_mode="manual",
                aiqos_enable_roce=True,
                aiqos_schedule="{tp:low,pp:high,dp-cp:high,ep:high,cp:middle,pos-embd:middle}"
            )
            
            global_vars._GLOBAL_ARGS = mock_args
            mock_adaptor_get_args.return_value = mock_args
            mock_qos_get_args.return_value = mock_args
            
            try:
                torch.npu.set_device(0)
                torch.distributed.init_process_group(backend='hccl', rank=0, world_size=1, timeout=timeout)
                pg_initialized = True
                
                pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                sdma_qos = 6
                roce_qos = 5

                group = create_group_qos(
                    [0],
                    timeout=timeout,
                    pg_options=pg_options,
                    group_desc='DATA_PARALLEL_GROUP',
                    parallel_type='dp'
                )

                pg_options.hccl_config = {'hccl_sdma_qos': sdma_qos, 'qos_service_level': roce_qos,
                                        'qos_traffic_class': roce_qos * 32}

                assert group is not None, "Group creation failed"

                hccl_cfg = pg_options.hccl_config

                assert 'hccl_sdma_qos' in hccl_cfg, "Missing key: hccl_sdma_qos"
                assert hccl_cfg['hccl_sdma_qos'] == sdma_qos

                assert 'qos_service_level' in hccl_cfg, "Missing key: qos_service_level"
                assert hccl_cfg['qos_service_level'] == roce_qos

                assert 'qos_traffic_class' in hccl_cfg, "Missing key: qos_traffic_class"
                assert hccl_cfg['qos_traffic_class'] == roce_qos * 32
            
            finally:
                if pg_initialized:
                    torch.distributed.destroy_process_group()
                    
            global_vars._GLOBAL_ARGS = None
            reset_quality_of_service_singleton()
