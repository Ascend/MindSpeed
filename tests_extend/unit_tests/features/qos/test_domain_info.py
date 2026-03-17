import os
import re
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from mindspeed.core.qos.domain_info import (
    is_a3,
    get_tensor_parallel_comm_domain,
    get_pipeline_parallel_comm_domain,
    get_data_parallel_comm_domain,
    get_context_parallel_comm_domain,
    get_expert_parallel_comm_domain,
    get_overlap_time_dict,
    get_overlap_space_dict,
    is_cross_boundary
)
from mindspeed.core.qos.domain_info import ParallelCommDomain, RankGenerator, domains


class TestIsCrossBoundary:
    def test_is_cross_boundary_single_machine(self):
        ranks = [[0, 1, 2], [3, 4, 5]]
        assert is_cross_boundary(ranks, boundary=8) is False

    def test_is_cross_boundary_multi_machine(self):
        ranks = [[0, 8], [1, 9]]
        assert is_cross_boundary(ranks, boundary=8) is True

    def test_is_cross_boundary_empty_domain(self):
        ranks = [[]]
        assert is_cross_boundary(ranks, boundary=8) is False

    def test_is_cross_boundary_invalid_boundary(self):
        ranks = [[0, 1]]
        with pytest.raises(ValueError) as exc_info:
            is_cross_boundary(ranks, boundary=0)
        assert "Boundary value must be a positive integer" in str(exc_info.value)


class TestIsA3:
    @patch('os.popen')
    def test_is_a3_910B(self, mock_popen):
        mock_popen.return_value.read.return_value = "Chip Name: Ascend910B"
        assert is_a3() is True

    @patch('os.popen')
    def test_is_a3_ascend910(self, mock_popen):
        mock_popen.return_value.read.return_value = "Chip Name: Ascend910"
        assert is_a3() is True

    @patch('os.popen')
    def test_is_a3_other_chip(self, mock_popen):
        mock_popen.return_value.read.return_value = "Chip Name: Ascend310"
        assert is_a3() is False

    @patch('os.popen')
    def test_is_a3_command_fail(self, mock_popen):
        mock_popen.side_effect = Exception("Command not found")
        with pytest.raises(RuntimeError) as exc_info:
            is_a3()
        assert "Fail to get chip name" in str(exc_info.value)


class TestGetOverlapTimeDict:
    @patch('mindspeed.core.qos.domain_info.get_args')
    def test_default_overlap_time_dict(self, mock_get_args):
        mock_args = MagicMock()
        mock_args.overlap_grad_reduce = False
        mock_args.overlap_param_gather = False
        mock_get_args.return_value = mock_args

        time_overlap = get_overlap_time_dict()
        all_keys = [(x, y) for x in domains for y in domains]
        assert set(time_overlap.keys()) == set(all_keys)

        pp_related_keys = [
            ('pp', 'tp'), ('pp', 'dp'), ('pp', 'cp'), ('pp', 'ep'),
            ('tp', 'pp'), ('dp', 'pp'), ('cp', 'pp'), ('ep', 'pp')
        ]
        for key in pp_related_keys:
            assert time_overlap[key] == 1

        for key in time_overlap:
            if key not in pp_related_keys:
                assert time_overlap[key] == 0

    @patch('mindspeed.core.qos.domain_info.get_args')
    def test_overlap_time_dict_with_grad_reduce(self, mock_get_args):
        mock_args = MagicMock()
        mock_args.overlap_grad_reduce = True
        mock_args.overlap_param_gather = False
        mock_get_args.return_value = mock_args

        time_overlap = get_overlap_time_dict()

        dp_related_keys = [
            ('dp', 'tp'), ('dp', 'pp'), ('dp', 'cp'), ('dp', 'ep'),
            ('tp', 'dp'), ('pp', 'dp'), ('cp', 'dp'), ('ep', 'dp')
        ]
        for key in dp_related_keys:
            assert time_overlap[key] == 1

    @patch('mindspeed.core.qos.domain_info.get_args')
    def test_overlap_time_dict_with_param_gather(self, mock_get_args):
        mock_args = MagicMock()
        mock_args.overlap_grad_reduce = False
        mock_args.overlap_param_gather = True
        mock_get_args.return_value = mock_args

        time_overlap = get_overlap_time_dict()

        dp_related_keys = [
            ('dp', 'tp'), ('dp', 'pp'), ('dp', 'cp'), ('dp', 'ep'),
            ('tp', 'dp'), ('pp', 'dp'), ('cp', 'dp'), ('ep', 'dp')
        ]
        for key in dp_related_keys:
            assert time_overlap[key] == 1


class TestGetOverlapSpaceDict:
    @patch('mindspeed.core.qos.domain_info.is_a3_version', new=True)
    @patch('mindspeed.core.qos.domain_info.is_adjacent_two_node_group')
    @patch('mindspeed.core.qos.domain_info.overlap_space_padding')
    def test_a3_sdma_link(
            self,
            mock_padding: MagicMock,
            mock_adjacent: MagicMock,
    ):
        def _is_not_in_forbidden_list(x):
            """Check if x is NOT in the forbidden adjacent list."""
            forbidden = [[[0, 2], [4, 5]], [[1, 3], [5, 7]]]
            return x not in forbidden

        mock_adjacent.side_effect = _is_not_in_forbidden_list
        domain_part_info = {
            'pp': [[0, 1], [2, 3]],  # 返回True
            'tp': [[0, 2], [4, 5]],  # 返回False
            'dp': [[6, 7], [8, 9]],  # 返回True
            'cp': [[1, 3], [5, 7]],  # 返回False
            'ep': [[10, 11], [12, 13]]  # 返回True
        }
        domain_part_info = {d: domain_part_info.get(d, [[0, 1]]) for d in domains}
        get_overlap_space_dict(domain_part_info, link_type="SDMA")
        assert mock_adjacent.call_count == len(domains)
        forbidden_groups = [[[0, 2], [4, 5]], [[1, 3], [5, 7]]]
        expected_cross = [
            d for d in domains
            if (value := domain_part_info.get(d)) and value in forbidden_groups
        ]
        mock_padding.assert_called_once_with(expected_cross)

    @patch('mindspeed.core.qos.domain_info.is_a3_version', new=True)
    @patch('mindspeed.core.qos.domain_info.is_cross_boundary')
    @patch('mindspeed.core.qos.domain_info.overlap_space_padding')
    def test_a3_roce_default_boundary(
            self,
            mock_padding: MagicMock,
            mock_cross: MagicMock,
    ):
        mock_cross.return_value = False
        domain_part_info = {d: [[0, 1]] for d in domains}
        get_overlap_space_dict(domain_part_info, link_type="ROCE")
        boundary_values = []
        for call in mock_cross.call_args_list:
            boundary = call[0][1]
            boundary_values.append(int(boundary) if isinstance(boundary, str) else boundary)
        assert 384 in boundary_values

    @patch('mindspeed.core.qos.domain_info.is_a3_version', new=True)
    def test_a3_unsupported_link_type(self):
        domain_part_info = {d: [[0, 1]] for d in domains}
        with pytest.raises(ValueError) as exc_info:
            get_overlap_space_dict(domain_part_info, link_type="ETH")
        assert "Unsupported link type: ETH" in str(exc_info.value)

    @patch('mindspeed.core.qos.domain_info.is_a3_version', new=False)
    @patch('mindspeed.core.qos.domain_info.is_cross_boundary')
    @patch('mindspeed.core.qos.domain_info.overlap_space_padding')
    def test_a2_roce_only(
            self,
            mock_padding: MagicMock,
            mock_cross: MagicMock,
    ):
        def _is_comm_in_forbidden_ranges(comm, boundary):
            forbidden_ranges = [
                [[0, 8], [1, 9]],
                [[7, 8], [15, 16]],
                [[8, 9], [10, 11]]
            ]
            return comm in forbidden_ranges

        mock_cross.side_effect = _is_comm_in_forbidden_ranges
        domain_part_info = {
            'pp': [[0, 8], [1, 9]],
            'tp': [[0, 1], [2, 3]],
            'dp': [[7, 8], [15, 16]],
            'cp': [[4, 5], [6, 7]],
            'ep': [[8, 9], [10, 11]]
        }
        domain_part_info = {d: domain_part_info.get(d, [[0, 1]]) for d in domains}
        get_overlap_space_dict(domain_part_info, link_type="SDMA")
        boundary_values = []
        for call in mock_cross.call_args_list:
            boundary = call[0][1]
            boundary_values.append(int(boundary) if isinstance(boundary, str) else boundary)
        assert 8 in boundary_values
        target_patterns = [
            [[0, 8], [1, 9]],
            [[7, 8], [15, 16]],
            [[8, 9], [10, 11]]
        ]

        expected_cross = [
            d for d in domains
            if domain_part_info.get(d) in target_patterns
        ]
        mock_padding.assert_called_once_with(expected_cross)

    @patch('mindspeed.core.qos.domain_info.is_a3_version', new=False)
    @patch('mindspeed.core.qos.domain_info.is_cross_boundary')
    @patch('mindspeed.core.qos.domain_info.overlap_space_padding')
    def test_empty_domain_part_info(
            self,
            mock_padding: MagicMock,
            mock_cross: MagicMock,
    ):
        mock_cross.return_value = False
        empty_domain_part_info = {d: [] for d in domains}
        get_overlap_space_dict(empty_domain_part_info)
        assert mock_cross.call_count == len(domains)
        mock_padding.assert_called_once_with([])
