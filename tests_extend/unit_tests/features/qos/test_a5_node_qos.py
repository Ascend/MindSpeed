from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from megatron.training import global_vars
from mindspeed.core.qos.a5_node_qos import (
    cal_a5_auto_qos,
    cal_node_conflict_degree,
    majority_vote_qos,
    map_rank_queue_weights_to_levels,
    select_min_conflict_groups,
    simulate_node_traffic,
)
from mindspeed.core.qos.adaptor import get_nccl_options_qos_wrapper
from mindspeed.core.qos.domain_info import ParallelCommDomain
from mindspeed.core.qos.qos import Qos, domains, generate_distributions
from mindspeed.ops.npu_matmul_add import NPUVersion

from tests_extend.unit_tests.features.qos.test_qos import reset_quality_of_service_singleton

DOMAINS = domains
TIME_PP_VS_ALL = {key: 0 for key in [(x, y) for x in DOMAINS for y in DOMAINS]}
for other in ('tp', 'dp', 'ep', 'cp'):
    TIME_PP_VS_ALL[('pp', other)] = 1
    TIME_PP_VS_ALL[(other, 'pp')] = 1


def _domain(name, rank_list, comm_amount, world_size=None):
    if world_size is None:
        world_size = max((rank for group in rank_list for rank in group), default=0) + 1
    return ParallelCommDomain(
        ip_list=None,
        rank_list=rank_list,
        world_size=world_size,
        parallel_type=name,
        comm_amount=comm_amount,
        comm_amount_no_overlap=comm_amount,
    )


class TestSimulateNodeTraffic:
    def test_size_one_group_has_zero_traffic(self):
        domains_info = [_domain('tp', [[0], [1]], comm_amount=100)]

        traffic = simulate_node_traffic(domains_info, world_size=2)

        assert traffic[0]['tp'] == 0
        assert traffic[1]['tp'] == 0

    def test_mesh_splits_group_volume_across_all_pairs(self):
        domains_info = [_domain('tp', [[0, 1, 2, 3]], comm_amount=12)]

        traffic = simulate_node_traffic(domains_info, world_size=4)

        assert traffic[0]['tp'] == 6
        assert traffic[1]['tp'] == 6
        assert traffic[2]['tp'] == 6
        assert traffic[3]['tp'] == 6

    def test_pp_head_and_tail_have_less_traffic_than_middle(self):
        domains_info = [_domain('pp', [[0, 1, 2, 3]], comm_amount=6)]

        traffic = simulate_node_traffic(domains_info, world_size=4)

        assert traffic[0]['pp'] == 2
        assert traffic[3]['pp'] == 2
        assert traffic[1]['pp'] == 4
        assert traffic[2]['pp'] == 4

    def test_cp_ring_wraps_and_is_uniform(self):
        domains_info = [_domain('cp', [[0, 1, 2, 3]], comm_amount=8)]

        traffic = simulate_node_traffic(domains_info, world_size=4)

        assert traffic[0]['cp'] == 4
        assert traffic[1]['cp'] == 4
        assert traffic[2]['cp'] == 4
        assert traffic[3]['cp'] == 4


class TestNodeConflictAndPriority:
    def test_conflict_skips_pairs_without_time_overlap(self):
        node_traffic = [
            {'tp': 10, 'dp': 4, 'pp': 8, 'ep': 0, 'cp': 0},
            {'tp': 10, 'dp': 4, 'pp': 1, 'ep': 0, 'cp': 0},
        ]
        comb = [[0, 1]]

        degree = cal_node_conflict_degree(comb, node_traffic, TIME_PP_VS_ALL)

        assert degree == 0

    def test_conflict_sums_min_traffic_on_each_rank(self):
        node_traffic = [
            {'tp': 10, 'dp': 4, 'pp': 8, 'ep': 0, 'cp': 0},
            {'tp': 10, 'dp': 4, 'pp': 1, 'ep': 0, 'cp': 0},
        ]
        comb = [[0, 2]]

        degree = cal_node_conflict_degree(comb, node_traffic, TIME_PP_VS_ALL)

        assert degree == 9

    def test_min_conflict_puts_time_conflict_domains_in_different_queues(self):
        node_traffic = [
            {'tp': 10, 'dp': 0, 'pp': 10, 'ep': 0, 'cp': 0},
        ]
        combs = generate_distributions(3, 2)

        min_comb = select_min_conflict_groups(combs, node_traffic, TIME_PP_VS_ALL)

        queued_together = any(0 in queue and 2 in queue for queue in min_comb)
        assert queued_together is False

    def test_single_nonempty_queue_maps_to_middle(self):
        node_traffic = {'tp': 9, 'dp': 4, 'pp': 1, 'ep': 0, 'cp': 0}
        groups = [[0, 1, 2, 3, 4]]

        mapping = map_rank_queue_weights_to_levels(groups, node_traffic, queue_values=(2, 4, 6))

        assert mapping == {'tp': 4, 'dp': 4, 'pp': 4, 'ep': 4, 'cp': 4}

    def test_two_nonempty_queues_map_to_high_and_low(self):
        node_traffic = {'tp': 9, 'dp': 1, 'pp': 0, 'ep': 0, 'cp': 0}
        groups = [[0], [1]]

        mapping = map_rank_queue_weights_to_levels(groups, node_traffic, queue_values=(2, 4, 6))

        assert mapping['tp'] == 6
        assert mapping['dp'] == 2
        assert mapping['pp'] == 4
        assert mapping['ep'] == 4
        assert mapping['cp'] == 4

    def test_three_queues_sort_by_sqrt_weight(self):
        node_traffic = {'tp': 9, 'dp': 1, 'pp': 4, 'ep': 16, 'cp': 0}
        groups = [[0, 3], [2], [1, 4]]

        mapping = map_rank_queue_weights_to_levels(groups, node_traffic, queue_values=(2, 4, 6))

        assert mapping['tp'] == 6
        assert mapping['ep'] == 6
        assert mapping['pp'] == 4
        assert mapping['dp'] == 2
        assert mapping['cp'] == 2

    def test_majority_vote_picks_the_most_common_mapping(self):
        mapping_a = {'tp': 6, 'dp': 2, 'pp': 4, 'ep': 6, 'cp': 2}
        mapping_b = {'tp': 2, 'dp': 6, 'pp': 4, 'ep': 2, 'cp': 6}
        rank_mappings = [mapping_a, mapping_a, mapping_b]
        node_traffic = [
            {'tp': 1, 'dp': 1, 'pp': 1, 'ep': 1, 'cp': 1},
            {'tp': 1, 'dp': 1, 'pp': 1, 'ep': 1, 'cp': 1},
            {'tp': 100, 'dp': 100, 'pp': 100, 'ep': 100, 'cp': 100},
        ]

        winner = majority_vote_qos(rank_mappings, node_traffic)

        assert winner == mapping_a

    def test_majority_tie_prefers_larger_total_traffic(self):
        mapping_a = {'tp': 6, 'dp': 2, 'pp': 4, 'ep': 6, 'cp': 2}
        mapping_b = {'tp': 2, 'dp': 6, 'pp': 4, 'ep': 2, 'cp': 6}
        rank_mappings = [mapping_a, mapping_b]
        node_traffic = [
            {'tp': 1, 'dp': 0, 'pp': 0, 'ep': 0, 'cp': 0},
            {'tp': 10, 'dp': 0, 'pp': 0, 'ep': 0, 'cp': 0},
        ]

        winner = majority_vote_qos(rank_mappings, node_traffic)

        assert winner == mapping_b

    def test_majority_double_tie_uses_mapping_order(self):
        mapping_a = {'tp': 6, 'dp': 2, 'pp': 4, 'ep': 6, 'cp': 2}
        mapping_b = {'tp': 2, 'dp': 6, 'pp': 4, 'ep': 2, 'cp': 6}
        rank_mappings = [mapping_a, mapping_b]
        node_traffic = [
            {'tp': 5, 'dp': 0, 'pp': 0, 'ep': 0, 'cp': 0},
            {'tp': 5, 'dp': 0, 'pp': 0, 'ep': 0, 'cp': 0},
        ]

        winner = majority_vote_qos(rank_mappings, node_traffic)

        assert winner == mapping_a


class TestA5AutoQosIntegration:
    def test_cal_a5_auto_qos_returns_only_hccl_domain_map(self):
        domain_list = [
            _domain('tp', [[0, 1], [2, 3]], 8),
            _domain('dp', [[0, 2], [1, 3]], 2),
            _domain('pp', [[0, 1, 2, 3]], 6),
            _domain('ep', [[0], [1], [2], [3]], 100),
            _domain('cp', [[0, 1, 2, 3]], 8),
        ]

        result = cal_a5_auto_qos(domain_list, TIME_PP_VS_ALL, queue_values=(2, 4, 6))

        assert set(result) == set(DOMAINS)
        assert all(value in (2, 4, 6) for value in result.values())

    def test_qos_auto_on_a5_does_not_fill_roce_schedule(self):
        tp_info = _domain('tp', [[0, 1]], 8, world_size=2)
        dp_info = _domain('dp', [[0], [1]], 0, world_size=2)
        pp_info = _domain('pp', [[0, 1]], 6, world_size=2)
        ep_info = _domain('ep', [[0], [1]], 0, world_size=2)
        cp_info = _domain('cp', [[0], [1]], 0, world_size=2)

        with (
            patch('mindspeed.core.qos.domain_info.get_args') as mock_domain_get_args,
            patch('mindspeed.core.qos.qos.get_args') as mock_qos_get_args,
            patch('mindspeed.core.qos.qos.get_npu_version', return_value=NPUVersion.A5),
            patch('mindspeed.core.qos.qos.get_tensor_parallel_comm_domain', return_value=tp_info),
            patch('mindspeed.core.qos.qos.get_data_parallel_comm_domain', return_value=dp_info),
            patch('mindspeed.core.qos.qos.get_pipeline_parallel_comm_domain', return_value=pp_info),
            patch('mindspeed.core.qos.qos.get_expert_parallel_comm_domain', return_value=ep_info),
            patch('mindspeed.core.qos.qos.get_context_parallel_comm_domain', return_value=cp_info),
            patch('mindspeed.core.qos.qos.log_rank_0'),
        ):
            mock_args = Namespace(
                aiqos_mode="auto",
                aiqos_enable_roce=True,
                num_experts=None,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            )
            global_vars._GLOBAL_ARGS = mock_args
            mock_domain_get_args.return_value = mock_args
            mock_qos_get_args.return_value = mock_args
            reset_quality_of_service_singleton()

            qos = Qos()

            assert qos.roce_aiqos_schedule == {}
            assert qos.set_parallel_hccl_qos('tp') in (2, 4, 6)
            assert qos.set_parallel_hccl_qos('pp') in (2, 4, 6)
            assert qos.set_parallel_roce_qos('tp') == 4

            global_vars._GLOBAL_ARGS = None
            reset_quality_of_service_singleton()

    def test_a5_adaptor_writes_only_hccl_sdma_qos(self):
        mock_get_nccl_options = MagicMock(return_value=SimpleNamespace(hccl_config={'hccl_buffer_size': 200}))

        with (
            patch('mindspeed.core.qos.adaptor.get_args') as mock_adaptor_get_args,
            patch('mindspeed.core.qos.qos.get_args') as mock_qos_get_args,
            patch('mindspeed.core.qos.adaptor.get_npu_version', return_value=NPUVersion.A5),
            patch('mindspeed.core.qos.qos.get_npu_version', return_value=NPUVersion.A5),
            patch('mindspeed.core.qos.adaptor.log_rank_0'),
        ):
            mock_args = SimpleNamespace(
                aiqos_mode="manual",
                aiqos_enable_roce=True,
                aiqos_schedule="{tp:low,pp:high,dp-cp:high,ep:high,cp:middle,pos-embd:middle}",
            )
            global_vars._GLOBAL_ARGS = mock_args
            mock_adaptor_get_args.return_value = mock_args
            mock_qos_get_args.return_value = mock_args
            reset_quality_of_service_singleton()

            get_nccl_options_qos = get_nccl_options_qos_wrapper(mock_get_nccl_options)
            options = get_nccl_options_qos('pp', {})

            hccl_cfg = options.hccl_config
            assert hccl_cfg['hccl_sdma_qos'] == 6
            assert 'qos_service_level' not in hccl_cfg
            assert 'qos_traffic_class' not in hccl_cfg
            assert hccl_cfg['hccl_buffer_size'] == 200

            global_vars._GLOBAL_ARGS = None
            reset_quality_of_service_singleton()

    def test_qos_auto_on_a5_fills_composite_hccl_domains(self):
        tp_info = _domain('tp', [[0, 1]], 8, world_size=2)
        dp_info = _domain('dp', [[0], [1]], 0, world_size=2)
        pp_info = _domain('pp', [[0, 1]], 6, world_size=2)
        ep_info = _domain('ep', [[0], [1]], 0, world_size=2)
        cp_info = _domain('cp', [[0], [1]], 0, world_size=2)

        base_schedule = {
            'tp': 6,
            'dp': 2,
            'pp': 6,
            'ep': 2,
            'cp': 6,
        }

        with (
            patch('mindspeed.core.qos.domain_info.get_args') as mock_domain_get_args,
            patch('mindspeed.core.qos.qos.get_args') as mock_qos_get_args,
            patch(
                'mindspeed.core.qos.qos.get_npu_version',
                return_value=NPUVersion.A5,
            ),
            patch(
                'mindspeed.core.qos.qos.get_tensor_parallel_comm_domain',
                return_value=tp_info,
            ),
            patch(
                'mindspeed.core.qos.qos.get_data_parallel_comm_domain',
                return_value=dp_info,
            ),
            patch(
                'mindspeed.core.qos.qos.get_pipeline_parallel_comm_domain',
                return_value=pp_info,
            ),
            patch(
                'mindspeed.core.qos.qos.get_expert_parallel_comm_domain',
                return_value=ep_info,
            ),
            patch(
                'mindspeed.core.qos.qos.get_context_parallel_comm_domain',
                return_value=cp_info,
            ),
            patch(
                'mindspeed.core.qos.a5_node_qos.cal_a5_auto_qos',
                return_value=base_schedule,
            ),
            patch('mindspeed.core.qos.qos.log_rank_0'),
        ):
            mock_args = Namespace(
                aiqos_mode='auto',
                aiqos_enable_roce=False,
                num_experts=8,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            )
            global_vars._GLOBAL_ARGS = mock_args
            mock_domain_get_args.return_value = mock_args
            mock_qos_get_args.return_value = mock_args
            reset_quality_of_service_singleton()

            try:
                qos = Qos()

                expected = {
                    'dp-cp': 2,
                    'ep-dp': 2,
                    'mp': 6,
                    'tp-ep-pp': 6,
                    'tp-ep-mp': 6,
                }

                for parallel_type, expected_qos in expected.items():
                    assert parallel_type in qos.hccl_aiqos_schedule
                    assert qos.set_parallel_hccl_qos(parallel_type) == expected_qos
            finally:
                global_vars._GLOBAL_ARGS = None
                reset_quality_of_service_singleton()
