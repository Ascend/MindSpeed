from typing import no_type_check, Any, List, Set, Tuple
from dataclasses import replace
from itertools import chain
import os.path


from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.memory.model_param import ModelParam
from mindspeed.auto_tuning.utils.dtype import DTYPE
from mindspeed.auto_tuning.utils.mem_utils import mem_b_to_mb
from mindspeed.auto_tuning.utils.restricted_unpickler import restricted_loads


class StaticMemModeling:
    LAYER1_FILENAME = "auto_tuning_static_model_layer1.json"
    PP4_FILENAME = "auto_tuning_static_model_pp4.json"
    EXPERT2_FILENAME = "auto_tuning_static_model_expert2.json"
    TP2_FILENAME = "auto_tuning_static_model_tp2.json"

    @no_type_check
    def __init__(self, model_cfg: ModelConfig) -> None:
        self.model_cfg = model_cfg
        self._logger = get_logger("static_mem")
        self.params_first_embedding: List[ModelParam] = None
        self.params_per_layer_wo_experts: List[ModelParam] = None
        self.params_per_experts: List[ModelParam] = None
        self.params_last_layernorm_and_embedding: List[ModelParam] = None
        self.params_pp_affected: List[ModelParam] = None
        self.params_tp_unaffected: Set[str] = set()

    @staticmethod
    def _diff_params(left: List[ModelParam],
                     right: List[ModelParam]
                     ) -> List[ModelParam]:
        """
        Finds the difference between two lists of parameters.
        The result follows these conditions:
            1. If a param exists in right but not in left,
            it gets appended directly into the result

            2. If a param exists in both lists and sharing a same name,
            however the shape is different, the shape difference is appended

            3. If a param (say A) exists only in left,
            we assume there's another param B with the same name
            but shape of 0 in the right list,
            thus 0 (B's shape) subtracted by A's shape gets appended
        """
        diff: List[ModelParam] = list()

        left_iter = iter(left)
        left_p = next(left_iter, None)
        for right_p in right:
            cmp_result = ModelParam.cmp(left_p, right_p)
            if cmp_result == 1:
                left_p = next(left_iter, None)
            elif cmp_result == -1 and left_p:
                diff.append(ModelParam(left_p.name,
                                       right_p.num_parameters -
                                       left_p.num_parameters
                                       ))
                left_p = next(left_iter, None)
            else:
                diff.append(right_p)

        while left_p:
            diff.append(ModelParam(left_p.name, -left_p.num_parameters))
            left_p = next(left_iter, None)

        return diff

    def generate_static_mem_profiling_list(self) -> List[Tuple[SearchConfig, str]]:
        result: List[Tuple[SearchConfig, str]] = list()

        layer1_cfg = SearchConfig()
        layer1_cfg.copy_from_config(self.model_cfg)
        layer1_cfg.tensor_model_parallel_size = 1
        layer1_cfg.context_parallel_size = 1
        layer1_cfg.pipeline_model_parallel_size = 1
        layer1_cfg.num_layers = 1
        if self.model_cfg.is_moe():
            layer1_cfg.num_experts = 1
            layer1_cfg.expert_model_parallel_size = 1
        result.append((layer1_cfg, self.LAYER1_FILENAME))

        pp4_cfg = replace(layer1_cfg,
                          pipeline_model_parallel_size=4,
                          num_layers=4)
        result.append((pp4_cfg, self.PP4_FILENAME))

        if self.model_cfg.is_moe():
            expert2_cfg = replace(pp4_cfg, num_experts=2)
            result.append((expert2_cfg, self.EXPERT2_FILENAME))

        tp2_cfg = replace(pp4_cfg, tensor_model_parallel_size=2)
        result.append((tp2_cfg, self.TP2_FILENAME))

        for cfg, _ in result:
            cfg.prepare_for_profiling()

        return result

    def model_static_mem(self, working_dir: str) -> None:
        def _decode(filename: str) -> Any:
            filepath = os.path.join(working_dir, filename)
            with open(filepath, mode="rb") as file:
                decode = restricted_loads(file)
                return decode

        def _get_pp_params(filename: str) -> List[List[ModelParam]]:
            params = [None] * 4
            for pp_rank, model_params in _decode(filename):
                if not params[pp_rank]:
                    params[pp_rank] = model_params
            return params  # type: ignore

        total_pp4_params = _get_pp_params(self.PP4_FILENAME)
        per_layer_w_experts_params = total_pp4_params[1]
        self.params_first_embedding = \
            self._diff_params(per_layer_w_experts_params,
                              total_pp4_params[0])
        self.params_last_layernorm_and_embedding = \
            self._diff_params(per_layer_w_experts_params,
                              total_pp4_params[-1])

        if self.model_cfg.is_moe():
            total_expert2_params = _get_pp_params(self.EXPERT2_FILENAME)
            self.params_per_experts = \
                self._diff_params(per_layer_w_experts_params,
                                  total_expert2_params[1])
        else:
            self.params_per_experts = list()
        self.params_per_layer_wo_experts = \
            self._diff_params(self.params_per_experts,
                              per_layer_w_experts_params)

        total_layer1_params: List[List[ModelParam]] = \
            [p for _, p in _decode(self.LAYER1_FILENAME)]
        layer1_params = total_layer1_params[0]
        self.params_pp_affected = \
            self._diff_params(self.params_first_embedding +
                              self.params_per_layer_wo_experts +
                              self.params_per_experts +
                              self.params_last_layernorm_and_embedding,
                              layer1_params)

        total_tp2_params = _get_pp_params(self.TP2_FILENAME)
        total_pp4_params_concat = list(chain.from_iterable(total_pp4_params))
        total_tp2_params_concat = list(chain.from_iterable(total_tp2_params))
        for i, param in enumerate(total_pp4_params_concat):
            if param == total_tp2_params_concat[i]:
                self.params_tp_unaffected.add(param.name)

        self._logger.debug("\n== first embedding params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_first_embedding]) +
                           "\n== layer_wo_experts params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_per_layer_wo_experts]) +
                           "\n== experts params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_per_experts]) +
                           "\n== last layer norm and embedding params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_last_layernorm_and_embedding]) +
                           "\n== pp affected params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_pp_affected]) +
                           "\n== not tp affected params:\n" +
                           "\n".join(
                               [str(p) for p in self.params_tp_unaffected]))

    def cal_static_mem(self, cfg: SearchConfig) -> List[float]:
        dtype = self.model_cfg.dtype
        non_expert_zero1 = cfg.dp * cfg.cp
        expert_zero1 = cfg.dp * cfg.cp / (cfg.ep if cfg.ep else 1)

        def _cal_static_mem_per_stage(non_expert_params: int,
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
                static_mem_per_stage = \
                    _cal_static_mem_per_stage(non_expert_params_per_stage,
                                              expert_params_per_stage,
                                              1 + 1,
                                              8)
            elif dtype == DTYPE.bf16:
                static_mem_per_stage = \
                    _cal_static_mem_per_stage(non_expert_params_per_stage,
                                              expert_params_per_stage,
                                              1 + 2,
                                              6)
            else:
                static_mem_per_stage = \
                    _cal_static_mem_per_stage(non_expert_params_per_stage,
                                              expert_params_per_stage,
                                              1 + 1,
                                              2)
            static_mem_stages.append(static_mem_per_stage)
        return static_mem_stages

    def _cal_num_params_per_stage(self,
                                  stage_id: int,
                                  cfg: SearchConfig
                                  ) -> Tuple[int, int]:
        def _cal_num_params(param: ModelParam, ep: int = 1):
            if param.name in self.params_tp_unaffected:
                return param.num_parameters
            else:
                return param.num_parameters // ep // cfg.tp

        num_layers = self.model_cfg.num_layers

        non_expert_params = 0
        for param in self.params_per_layer_wo_experts:
            non_expert_params += _cal_num_params(param)
        non_expert_params *= num_layers // cfg.pp

        expert_params = 0
        if cfg.num_experts and cfg.ep:
            for param in self.params_per_experts:
                expert_params += _cal_num_params(param, ep=cfg.ep)
            expert_params *= (num_layers * cfg.num_experts) // cfg.pp

        if stage_id == 0:
            for param in self.params_first_embedding:
                non_expert_params += _cal_num_params(param)
        if stage_id == cfg.pp - 1:
            for param in self.params_last_layernorm_and_embedding:
                non_expert_params += _cal_num_params(param)

        if cfg.pp == 1:
            for param in self.params_pp_affected:
                non_expert_params += _cal_num_params(param)

        return non_expert_params, expert_params
