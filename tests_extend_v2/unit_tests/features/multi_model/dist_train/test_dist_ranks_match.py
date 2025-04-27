# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
from mindspeed.core.multi_modal.dist_train import dist_ranks_match as match
from tests_extend.unit_tests.common import DistributedTest


class TestDistRanksMatchWithDistributed(DistributedTest):
    world_size = 1

    @staticmethod
    def _get_ranks_matrix(start_rank, row_size, col_size, dp_size=1):
        all_row_size = row_size * dp_size
        matrix = [[rank_ for rank_ in range(rank, rank + all_row_size * col_size, all_row_size)]
                  for rank in range(start_rank, start_rank + all_row_size)]
        pp_ranks = matrix
        tp_ranks = [[row[i] for row in matrix] for i in range(col_size)]
        tp_ranks = [
            ranks[i * row_size: (i + 1) * row_size]
            for ranks in tp_ranks
            for i in range(dp_size)
        ]
        return pp_ranks, tp_ranks

    @staticmethod
    def _flip_ranks(pp_ranks, tp_ranks):
        pp_ranks_ = [list(reversed(ranks)) for ranks in pp_ranks]
        tp_ranks_ = list(reversed(tp_ranks))
        return pp_ranks_, tp_ranks_

    @pytest.mark.parametrize('shape_pair', (['0x0x1', '2x2x2'], ['1x1x1', '0x0x1']))
    # shape -- row x col x stack -- tp_size x pp_size x dp_size
    def test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series_should_get_value_error(self, shape_pair):
        shape_1, shape_2 = shape_pair
        shape_1 = [int(s) for s in shape_1.split('x')]
        shape_2 = [int(s) for s in shape_2.split('x')]
        pp_ranks_1, tp_ranks_1 = self._get_ranks_matrix(0, *shape_1)
        pp_ranks_2, tp_ranks_2 = self._get_ranks_matrix(shape_1[0] * shape_1[1] * shape_1[2], *shape_2)
        match.clear_model_comm_ranks()
        with pytest.raises(ValueError) as excinfo:
            match.generate_model_comm_ranks(pp_ranks_1, tp_ranks_1, pp_ranks_2, tp_ranks_2)
        assert 'tp ranks must not empty' in str(excinfo.value)

    def _test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(self, shape_1, shape_2, answer):
        shape_1 = [int(s) for s in shape_1.split('x')]
        shape_2 = [int(s) for s in shape_2.split('x')]
        pp_ranks_1, tp_ranks_1 = self._get_ranks_matrix(0, *shape_1)
        pp_ranks_2, tp_ranks_2 = self._get_ranks_matrix(shape_1[0] * shape_1[1] * shape_1[2], *shape_2)

        def check_get_dst_ranks():
            for rank in range(shape_1[0] * shape_1[1] * shape_1[2] + shape_2[0] * shape_2[1] * shape_2[2]):
                if rank not in answer.keys():
                    assert match.get_dst_ranks(rank) is None, f'{match._MODEL_COMM_RANKS=} should equals to {answer}'
                else:
                    assert match.get_dst_ranks(rank) == answer.get(rank, None), \
                        f'{match._MODEL_COMM_RANKS=} should equals to {answer}'

        def check_model_comm_ranks_and_get_dst_ranks(pp_ranks_1_, tp_ranks_1_, pp_ranks_2_, tp_ranks_2_):
            nonlocal answer
            match.clear_model_comm_ranks()
            match.generate_model_comm_ranks(pp_ranks_1_, tp_ranks_1_, pp_ranks_2_, tp_ranks_2_)
            check_get_dst_ranks()
            assert match._MODEL_COMM_RANKS == answer, f'{match._MODEL_COMM_RANKS=} should equals to {answer}'

        check_model_comm_ranks_and_get_dst_ranks(pp_ranks_1, tp_ranks_1, pp_ranks_2, tp_ranks_2)
        # expend ut by flipping pp_ranks and tp_ranks
        if shape_1 == shape_2:
            return
        pp_ranks_1, tp_ranks_1 = self._flip_ranks(pp_ranks_1, tp_ranks_1)
        pp_ranks_2, tp_ranks_2 = self._flip_ranks(pp_ranks_2, tp_ranks_2)
        check_model_comm_ranks_and_get_dst_ranks(pp_ranks_2, tp_ranks_2, pp_ranks_1, tp_ranks_1)

    def test_generate_model_comm_ranks_one_dp_to_one_dp_and_more_tp_to_less_tp_and_rank_num_is_divisible(self):
        shape_1, shape_2, answer = '4x3x1', '2x3x1', {8: [12], 9: [12], 10: [13], 11: [13], 12: [8, 9], 13: [10, 11]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_one_dp_to_one_dp_and_less_tp_to_more_tp_and_rank_num_is_divisible(self):
        shape_1, shape_2, answer = '2x4x1', '4x2x1', {6: [8, 9], 7: [10, 11], 8: [6], 9: [6], 10: [7], 11: [7]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_one_dp_to_one_dp_and_tp_is_equal_and_rank_num_is_divisible(self):
        shape_1, shape_2, answer = '2x4x1', '2x3x1', {6: [8], 7: [9], 8: [6], 9: [7]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_one_dp_to_one_dp_and_rank_num_is_not_divisible(self):
        shape_1, shape_2, answer = '3x3x1', '2x3x1', {6: [9], 7: [9], 8: [10], 9: [6, 7], 10: [8]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_one_dp_to_one_dp_and_rank_num_is_equal(self):
        shape_1, shape_2, answer = '3x3x1', '3x3x1', {6: [9], 7: [10], 8: [11], 9: [6], 10: [7], 11: [8]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_two_dp_to_one_dp_and_rank_num_is_divisible(self):
        shape_1, shape_2, answer = '3x1x2', '3x1x1', {0: [6], 1: [6], 2: [7], 3: [7], 4: [8], 5: [8], 6: [0, 1],
                                                      7: [2, 3], 8: [4, 5]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_two_dp_to_one_dp_and_rank_num_is_not_divisible(self):
        shape_1, shape_2, answer = '2x1x2', '3x1x1', {0: [4], 1: [4], 2: [5], 3: [6], 4: [0, 1], 5: [2], 6: [3]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_two_dp_to_two_dp_and_rank_num_is_not_divisible(self):
        shape_1, shape_2, answer = '2x1x2', '2x1x2', {0: [4], 1: [5], 2: [6], 3: [7], 4: [0], 5: [1], 6: [2], 7: [3]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_generate_model_comm_ranks_four_dp_to_two_dp_and_rank_num_is_divisible(self):
        shape_1, shape_2, answer = '2x1x4', '2x1x2', {0: [8], 1: [8], 2: [9], 3: [9], 4: [10], 5: [10], 6: [11],
                                                      7: [11], 8: [0, 1], 9: [2, 3], 10: [4, 5], 11: [6, 7]}
        self._test_generate_model_comm_ranks_and_get_dst_ranks_with_two_model_series(shape_1, shape_2, answer)

    def test_get_dst_ranks_with_none_input(self):
        pp_ranks_1, tp_ranks_1 = self._get_ranks_matrix(0, 4, 1, 1)
        pp_ranks_2, tp_ranks_2 = self._get_ranks_matrix(4 * 1 * 1, 8, 2, 1)
        match.clear_model_comm_ranks()
        match.generate_model_comm_ranks(pp_ranks_1, tp_ranks_1, pp_ranks_2, tp_ranks_2)
        assert match.get_dst_ranks() == [4, 5], f'Getting incorrect ranks ({match.get_dst_ranks()}).'
