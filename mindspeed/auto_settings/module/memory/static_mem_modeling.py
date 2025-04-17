from typing import no_type_check, Any, DefaultDict, Iterable, List, Set, Tuple
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import math
import os

from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.search_space import SearchSpace
from mindspeed.auto_settings.utils.dtype import DTYPE
from mindspeed.auto_settings.utils.file_utils import restricted_read
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.mem_utils import mem_b_to_mb
from mindspeed.auto_settings.utils.utils import get_tp_for_profiling

_Param = DefaultDict[str, List[int]]


class StaticMemModeling:

    @no_type_check
    def __init__(self, model_cfg: ModelConfig) -> None:
        self.model_cfg = model_cfg
        self._logger = get_logger("static_mem")
        self._baseline_tp = get_tp_for_profiling()
        self.params_inputs: _Param = None
        self.params_per_layer_wo_experts: _Param = None
        self.params_per_experts: _Param = None
        self.params_outputs: _Param = None
        self.params_tp_unaffected: Set[str] = set()

    @staticmethod
    def _cal_embedding_size(
        vocab_size: int,
        make_vocab_size_divisible_by: int,
        tp: int,
        hidden_size: int
    ) -> int:
        division = make_vocab_size_divisible_by * tp
        padded_vocab_size = int(math.ceil(vocab_size / division) * division)
        return padded_vocab_size * hidden_size

    @staticmethod
    def _diff_params(left: _Param, right: _Param) -> _Param:
        result = deepcopy(left)
        for k, v in right.items():
            result[k].extend([-params for params in v])

        zeros_out_keys = list()
        for k, v in result.items():
            if sum(v) == 0:
                zeros_out_keys.append(k)
        for k in zeros_out_keys:
            result.pop(k)

        return result

    @staticmethod
    def _merge_params(params_list: Iterable[_Param]) -> _Param:
        result = defaultdict(list)
        for params in params_list:
            for k, v in params.items():
                result[k].extend(v)
        return result

    def model_static_mem(self, working_dir: str) -> None:
        def _decode(filename: str) -> Any:
            filepath = os.path.join(working_dir, filename)
            return restricted_read(filepath)

        def _get_pp4_params_list(filename: str) -> List[_Param]:
            result: List[_Param] = [None] * 4  # type: ignore
            for pp_rank, model_params in _decode(filename):
                if not result[pp_rank]:
                    result[pp_rank] = defaultdict(list)
                    for p_name, p_size in model_params:
                        result[pp_rank][p_name].append(p_size)
            return result

        total_pp4_params = _get_pp4_params_list(SearchSpace.PP4_FILENAME)
        per_layer_w_experts_params = total_pp4_params[1]
        self.params_inputs = self._diff_params(
            total_pp4_params[0],
            per_layer_w_experts_params
        )
        self.params_outputs = self._diff_params(
            total_pp4_params[-1],
            per_layer_w_experts_params
        )

        if self.model_cfg.is_moe():
            self.params_per_experts = self._diff_params(
                _get_pp4_params_list(SearchSpace.EXPERT2_FILENAME)[0],
                self._merge_params((
                    self.params_inputs,
                    per_layer_w_experts_params,
                    self.params_outputs
                ))
            )
        else:
            self.params_per_experts = defaultdict(list)
        self.params_per_layer_wo_experts = self._diff_params(
            per_layer_w_experts_params,
            self.params_per_experts
        )

        tp1_params = self._merge_params((
            self.params_inputs,
            self.params_per_layer_wo_experts,
            self.params_per_experts,
            self.params_outputs
        ))
        tp2_params = _get_pp4_params_list(SearchSpace.TP2_FILENAME)[0]
        last_embedding_name = set(tp1_params.keys()).difference(tp2_params.keys())
        last_embedding_params: Set[int] = set()
        for k in last_embedding_name:
            last_embedding_params.add(sum(self.params_outputs.pop(k)))
        first_embedding_name: Set[str] = set()
        for k, v in self.params_inputs.items():
            if sum(v) in last_embedding_params:
                first_embedding_name.add(k)
                last_embedding_params.remove(sum(v))
        for k in first_embedding_name:
            self.params_inputs.pop(k)

        for k in tp2_params.keys():
            if sum(tp1_params.get(k, list())) == sum(tp2_params.get(k, list())):
                self.params_tp_unaffected.add(k)

        for k, v in self.params_inputs.items():
            if k not in self.params_tp_unaffected:
                self.params_inputs[k] = [p * self._baseline_tp for p in v]
        for k, v in self.params_per_layer_wo_experts.items():
            if k not in self.params_tp_unaffected:
                self.params_per_layer_wo_experts[k] = [p * self._baseline_tp for p in v]
        for k, v in self.params_per_experts.items():
            if k not in self.params_tp_unaffected:
                self.params_per_experts[k] = [p * self._baseline_tp for p in v]
        for k, v in self.params_outputs.items():
            if k not in self.params_tp_unaffected:
                self.params_outputs[k] = [p * self._baseline_tp for p in v]

        self._logger.debug("== first embedding name:")
        for k in first_embedding_name:
            self._logger.debug(k)
        self._logger.debug("== headings params:")
        for k, v in self.params_inputs.items():
            self._logger.debug(f"{k}: {sum(v)} | {v}")
        self._logger.debug("== layer_wo_experts params:")
        for k, v in self.params_per_layer_wo_experts.items():
            self._logger.debug(f"{k}: {sum(v)} | {v}")
        self._logger.debug("== experts params:")
        for k, v in self.params_per_experts.items():
            self._logger.debug(f"{k}: {sum(v)} | {v}")
        self._logger.debug("== outputs params:")
        for k, v in self.params_outputs.items():
            self._logger.debug(f"{k}: {sum(v)} | {v}")
        self._logger.debug(f"== last embedding name:")
        for k in last_embedding_name:
            self._logger.debug(k)
        self._logger.debug("== not tp affected params:")
        for name in self.params_tp_unaffected:
            self._logger.debug(name)

    def cal_static_mem(self, cfg: SearchConfig) -> List[float]:
        dtype = self.model_cfg.dtype
        non_expert_zero1 = cfg.dp * cfg.cp
        expert_zero1 = cfg.dp * cfg.cp / (cfg.ep if cfg.ep else 1)

        def _cal_static_mem_per_stage(
            non_expert_params: int,
            expert_params: int,
            not_zero1_div_bytes: int,
            zero1_div_bytes: int
        ) -> float:
            result = float(0)
            if cfg.zero1:
                result += non_expert_params * \
                    (not_zero1_div_bytes + zero1_div_bytes / non_expert_zero1)
                result += expert_params * \
                    (not_zero1_div_bytes + zero1_div_bytes / expert_zero1)
            else:
                result += (non_expert_params + expert_params) * \
                    (not_zero1_div_bytes + zero1_div_bytes)
            result = mem_b_to_mb(result * dtype.value[1])
            result += 5000  # roughly estimated cann+hccl+driver+os memory
            return result

        static_mem_stages: List[float] = list()
        for stage_id in range(cfg.pp):
            non_expert_params_per_stage, expert_params_per_stage = \
                self._cal_num_params_per_stage(stage_id, cfg)
            if dtype == DTYPE.fp16:
                static_mem_per_stage = _cal_static_mem_per_stage(
                    non_expert_params_per_stage,
                    expert_params_per_stage,
                    1 + 1,
                    8
                )
            elif dtype == DTYPE.bf16:
                static_mem_per_stage = _cal_static_mem_per_stage(
                    non_expert_params_per_stage,
                    expert_params_per_stage,
                    1 + 2,
                    6
                )
            else:
                static_mem_per_stage = _cal_static_mem_per_stage(
                    non_expert_params_per_stage,
                    expert_params_per_stage,
                    1 + 1,
                    2
                )
            static_mem_stages.append(static_mem_per_stage)
        return static_mem_stages

    def _cal_num_params_per_stage(
        self,
        stage_id: int,
        cfg: SearchConfig
    ) -> Tuple[int, int]:
        def _cal_num_params(p_name: str, p_size: List[int], ep: int = 1):
            if p_name in self.params_tp_unaffected:
                return sum(p_size)
            else:
                return sum(p_size) // ep // cfg.tp

        num_layers = self.model_cfg.num_layers

        non_expert_params = 0
        for p_name, p_size in self.params_per_layer_wo_experts.items():
            non_expert_params += _cal_num_params(p_name, p_size)
        non_expert_params *= num_layers // cfg.pp

        expert_params = 0
        if cfg.num_experts and cfg.ep:
            for p_name, p_size in self.params_per_experts.items():
                expert_params += _cal_num_params(p_name, p_size, ep=cfg.ep)
            expert_params *= (num_layers * cfg.num_experts) // cfg.pp

        if stage_id == 0:
            for p_name, p_size in self.params_inputs.items():
                non_expert_params += _cal_num_params(p_name, p_size)
            non_expert_params += self._cal_embedding_size(
                self.model_cfg.vocab_size,
                self.model_cfg.make_vocab_size_divisible_by,
                cfg.tp,
                self.model_cfg.hidden_size
            ) // cfg.tp

        if stage_id == cfg.pp - 1:
            for p_name, p_size in self.params_outputs.items():
                non_expert_params += _cal_num_params(p_name, p_size)
            if cfg.pp != 1 or self.model_cfg.untie_embeddings_and_output_weights:
                non_expert_params += self._cal_embedding_size(
                    self.model_cfg.vocab_size,
                    self.model_cfg.make_vocab_size_divisible_by,
                    cfg.tp,
                    self.model_cfg.hidden_size
                ) // cfg.tp

        return non_expert_params, expert_params
