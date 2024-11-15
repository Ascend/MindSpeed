from typing import no_type_check, List, Tuple
from collections import namedtuple
from dataclasses import replace
import os.path

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_config import ProfilingModelInfo
from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_tuning.utils.utils import get_prof_dir

ProfileResult = namedtuple("ProfileResult", ["cfg", "prof"])
MemModule = namedtuple(
    "MemModule",
    [
        "checkpoint_activation_layer",
        "checkpoint_activation_embedding",
        "checkpoint_activation_loss",
        "forward_peak",
        "loss_peak",
        "backward_peak",
        "optimizer_peak"
    ]
)


class DynamicMemModeling:
    BASELINE_SEQLEN = 4096

    @no_type_check
    def __init__(self, model_cfg: ModelConfig) -> None:
        self.model_cfg = model_cfg
        self._logger = get_logger("dynamic_mem")
        self.ckpt_act_layer: float = None
        self.ckpt_act_embedding: float = None
        self.ckpt_act_tp_b_embedding: float = None
        self.ckpt_act_loss: float = None
        self.forward_peak: float = None
        self.tp_b_forward_peak: float = None
        self.backward_peak: float = None
        self.tp_b_backward_peak: float = None
        self.loss_peak: float = None
        self.tp_b_loss_peak: float = None
        self.optimizer_peak: float = None
        self.tp_b_optimizer_peak: float = None
        self.seq_b_optimizer_peak: float = None

    @staticmethod
    def _cal_peak_mem_per_stage(mem_module,
                                cfg: SearchConfig,
                                schedule: str,
                                nlayer: int,
                                stage_id: int
                                ) -> float:
        checkpoint_activation_layer, \
            checkpoint_activation_embedding, \
            checkpoint_activation_loss, \
            forward_peak, \
            loss_peak, \
            backward_peak, \
            _ = mem_module

        if schedule == "1f1b":
            if not cfg.vpp:
                num_warmup = cfg.pp - stage_id
                num_embd = cfg.pp
            else:
                num_warmup = cfg.pp * (cfg.vpp + 1) - 1 - 2 * stage_id
                num_embd = cfg.pp * 2 - 1

            estimated_forward_peak = checkpoint_activation_layer * nlayer * (num_warmup - 1) + \
                checkpoint_activation_layer * (nlayer - 1 + 1) + \
                forward_peak

            estimated_backward_peak = checkpoint_activation_layer * nlayer * num_warmup + \
                backward_peak

            if stage_id == 0:
                estimated_forward_peak += checkpoint_activation_embedding * num_embd
                estimated_backward_peak += checkpoint_activation_embedding * num_embd

            if stage_id == cfg.pp - 1:
                estimated_forward_peak += checkpoint_activation_loss
                estimated_backward_peak += checkpoint_activation_loss

                estimated_loss_peak = checkpoint_activation_layer * nlayer * num_warmup + \
                    checkpoint_activation_loss * (num_warmup - 1) + \
                    loss_peak
            else:
                estimated_loss_peak = 0

            peak_mem = max(estimated_forward_peak,
                           estimated_backward_peak,
                           estimated_loss_peak)
        else:
            peak_mem = 0

        return peak_mem

    def generate_dynamic_mem_profiling_list(self) -> List[SearchConfig]:
        result: List[SearchConfig] = list()

        baseline_cfg = SearchConfig()
        baseline_cfg.copy_from_config(self.model_cfg)
        baseline_cfg.tensor_model_parallel_size = 4
        baseline_cfg.context_parallel_size = 1
        baseline_cfg.pipeline_model_parallel_size = 1
        baseline_cfg.num_layers = 1
        baseline_cfg.seq_length = self.BASELINE_SEQLEN
        if self.model_cfg.is_moe():
            baseline_cfg.num_experts = 4
            baseline_cfg.expert_model_parallel_size = 1
        result.append(baseline_cfg)

        tp8_cfg = replace(baseline_cfg,
                          tensor_model_parallel_size=8)
        result.append(tp8_cfg)

        seq8k_cfg = replace(baseline_cfg,
                            seq_length=2 * self.BASELINE_SEQLEN)
        result.append(seq8k_cfg)

        for cfg in result:
            cfg.prepare_for_profiling()

        return result

    def model_dynamic_mem(self, working_dir: str) -> None:
        def _get_profiling(cfg: SearchConfig) -> ProfilingModelInfo:
            profiling_path = os.path.join(working_dir, get_prof_dir(cfg))
            profiling_node_parse = GatherNodeProfiling(profiling_path)
            return profiling_node_parse.fuse_node_pkl()

        baseline_cfg, tp8_cfg, seq8k_cfg = \
            self.generate_dynamic_mem_profiling_list()

        tp4seq4k_prof = _get_profiling(baseline_cfg)
        tp8seq4k_prof = _get_profiling(tp8_cfg)
        tp4seq8k_prof = _get_profiling(seq8k_cfg)

        self._get_ckpt_act_layer_modeling(baseline_cfg, tp4seq4k_prof)
        self._get_ckpt_act_embedding_modeling(baseline_cfg,
                                              tp8_cfg,
                                              tp4seq4k_prof,
                                              tp8seq4k_prof)
        self._get_ckpt_act_loss_modeling(baseline_cfg, tp4seq4k_prof)
        self._get_forward_peak_modeling(baseline_cfg,
                                        tp8_cfg,
                                        tp4seq4k_prof,
                                        tp8seq4k_prof)
        self._get_backward_peak_modeling(baseline_cfg,
                                         tp8_cfg,
                                         tp4seq4k_prof,
                                         tp8seq4k_prof)
        self._get_loss_peak_modeling(baseline_cfg,
                                     tp8_cfg,
                                     tp4seq4k_prof,
                                     tp8seq4k_prof)
        self._get_optimizer_peak_modeling(
            ProfileResult(cfg=baseline_cfg, prof=tp4seq4k_prof),
            ProfileResult(cfg=seq8k_cfg, prof=tp4seq8k_prof),
            ProfileResult(cfg=tp8_cfg, prof=tp8seq4k_prof)
        )

        self._logger.debug("== ckpt_act_layer:")
        self._logger.debug(f"{self.ckpt_act_layer}")
        self._logger.debug("== ckpt_act_embedding:")
        self._logger.debug(f"{self.ckpt_act_embedding}, {self.ckpt_act_tp_b_embedding}")
        self._logger.debug("== ckpt_act_loss:")
        self._logger.debug(f"{self.ckpt_act_loss}")
        self._logger.debug("== forward_peak:")
        self._logger.debug(f"{self.forward_peak}, {self.tp_b_forward_peak}")
        self._logger.debug("== backward_peak:")
        self._logger.debug(f"{self.backward_peak}, {self.tp_b_backward_peak}")
        self._logger.debug("== loss_peak:")
        self._logger.debug(f"{self.loss_peak}, {self.tp_b_loss_peak}")
        self._logger.debug("== optimizer_peak:")
        self._logger.debug(f"{self.optimizer_peak}, {self.tp_b_optimizer_peak}, {self.seq_b_optimizer_peak}")

    def cal_dynamic_mem(self,
                        cfg: SearchConfig
                        ) -> Tuple[List[float], float]:
        mem_module = self._cal_mem_module(cfg)
        optimizer_peak = mem_module[-1]

        nlayer = self.model_cfg.num_layers // cfg.pp
        if cfg.layers_per_vpp:
            nlayer = cfg.layers_per_vpp

        schedule = "1f1b"
        dynamic_mem_stages: List[float] = list()
        for stage_id in range(cfg.pp):
            peak_mem = self._cal_peak_mem_per_stage(mem_module,
                                                    cfg,
                                                    schedule,
                                                    nlayer,
                                                    stage_id)
            peak_mem *= (cfg.mbs / 1)  # mbs in profiling cfg equals 1
            dynamic_mem_stages.append(peak_mem)
        return dynamic_mem_stages, optimizer_peak

    def _get_ckpt_act_layer_modeling(self,
                                     base_cfg: SearchConfig,
                                     base_prof: ProfilingModelInfo
                                     ) -> None:
        self.ckpt_act_layer = base_cfg.tp * \
            (base_prof.loss.start_memory[0][0] -
             base_prof.forward.start_memory[0][0])

    def _get_ckpt_act_embedding_modeling(self,
                                         base_cfg: SearchConfig,
                                         bi_tp_cfg: SearchConfig,
                                         base_prof: ProfilingModelInfo,
                                         bi_tp_prof: ProfilingModelInfo) -> None:
        base_embd = base_prof.forward.start_memory[0][0] - \
            base_prof.embedding.start_memory[0][0]
        bi_tp_embd = bi_tp_prof.forward.start_memory[0][0] - \
            bi_tp_prof.embedding.start_memory[0][0]
        self.ckpt_act_tp_b_embedding = bi_tp_embd * \
            (bi_tp_cfg.tp // base_cfg.tp) - \
            base_embd
        self.ckpt_act_embedding = base_embd * base_cfg.tp - \
            self.ckpt_act_tp_b_embedding * (base_cfg.tp - 1)

    def _get_ckpt_act_loss_modeling(self,
                                    base_cfg: SearchConfig,
                                    base_prof: ProfilingModelInfo) -> None:
        self.ckpt_act_loss = base_cfg.tp * \
            (base_prof.backward.start_memory[0][0] -
             base_prof.loss.start_memory[0][0])

    def _get_forward_peak_modeling(self,
                                   base_cfg: SearchConfig,
                                   bi_tp_cfg: SearchConfig,
                                   base_prof: ProfilingModelInfo,
                                   bi_tp_prof: ProfilingModelInfo) -> None:
        base_forward_peak = base_prof.forward.peak_memory[0][0] - \
            base_prof.loss.start_memory[0][0]
        bi_tp_forward_peak = bi_tp_prof.forward.peak_memory[0][0] - \
            bi_tp_prof.loss.start_memory[0][0]
        self.tp_b_forward_peak = bi_tp_forward_peak * \
            (bi_tp_cfg.tp // base_cfg.tp) - \
            base_forward_peak
        self.forward_peak = base_forward_peak * base_cfg.tp - \
            self.tp_b_forward_peak * (base_cfg.tp - 1)

    def _get_backward_peak_modeling(self,
                                    base_cfg: SearchConfig,
                                    bi_tp_cfg: SearchConfig,
                                    base_prof: ProfilingModelInfo,
                                    bi_tp_prof: ProfilingModelInfo) -> None:
        base_backward_peak = base_prof.backward.peak_memory[0][0] - \
            base_prof.backward.start_memory[0][0]
        bi_tp_backward_peak = bi_tp_prof.backward.peak_memory[0][0] - \
            bi_tp_prof.backward.start_memory[0][0]
        self.tp_b_backward_peak = bi_tp_backward_peak * \
            (bi_tp_cfg.tp // base_cfg.tp) - \
            base_backward_peak
        self.backward_peak = base_backward_peak * base_cfg.tp - \
            self.tp_b_backward_peak * (base_cfg.tp - 1)

    def _get_loss_peak_modeling(self,
                                base_cfg: SearchConfig,
                                bi_tp_cfg: SearchConfig,
                                base_prof: ProfilingModelInfo,
                                bi_tp_prof: ProfilingModelInfo) -> None:
        base_loss_peak = base_prof.loss.peak_memory[0][0] - \
            base_prof.loss.start_memory[0][0]
        bi_tp_loss_peak = bi_tp_prof.loss.peak_memory[0][0] - \
            bi_tp_prof.loss.start_memory[0][0]
        self.tp_b_loss_peak = bi_tp_loss_peak * \
            (bi_tp_cfg.tp // base_cfg.tp) - \
            base_loss_peak
        self.loss_peak = base_loss_peak * base_cfg.tp - \
            self.tp_b_loss_peak * (base_cfg.tp - 1)

    def _get_optimizer_peak_modeling(
        self,
        base_res: ProfileResult,
        bi_seq_res: ProfileResult,
        bi_tp_res: ProfileResult
    ) -> None:
        base_cfg, base_prof = base_res
        bi_seq_cfg, bi_seq_prof = bi_seq_res
        bi_tp_cfg, bi_tp_prof = bi_tp_res
        base_optimizer_peak = base_prof.optimizer.peak_memory[0][0] - \
            base_prof.optimizer.start_memory[0][0]
        bi_seq_optimizer_peak = bi_seq_prof.optimizer.peak_memory[0][0] - \
            bi_seq_prof.optimizer.start_memory[0][0]
        bi_tp_optimizer_peak = bi_tp_prof.optimizer.peak_memory[0][0] - \
            bi_tp_prof.optimizer.start_memory[0][0]
        self.seq_b_optimizer_peak = (base_optimizer_peak *
                                     (bi_seq_cfg.seq_length // base_cfg.seq_length) -
                                     bi_seq_optimizer_peak) * base_cfg.tp
        self.tp_b_optimizer_peak = bi_tp_optimizer_peak * \
            (bi_tp_cfg.tp // base_cfg.tp) - \
            base_optimizer_peak
        self.optimizer_peak = base_optimizer_peak * base_cfg.tp - \
            self.tp_b_optimizer_peak * (base_cfg.tp - 1)

    def _cal_mem_module(self, cfg: SearchConfig) -> MemModule:
        seq_length = self.model_cfg.seq_length
        nseq = seq_length // cfg.cp // self.BASELINE_SEQLEN
        tp = cfg.tp
        tp_w = cfg.tp - 1

        checkpoint_activation_layer = self.ckpt_act_layer * nseq / tp

        checkpoint_activation_embedding = \
            (self.ckpt_act_embedding +
             tp_w * self.ckpt_act_tp_b_embedding) * nseq / tp

        checkpoint_activation_loss = self.ckpt_act_loss * nseq / tp

        forward_peak = \
            (self.forward_peak +
             tp_w * self.tp_b_forward_peak) * nseq / tp

        loss_peak = \
            (self.loss_peak +
             tp_w * self.tp_b_loss_peak) * nseq / tp

        backward_peak = \
            (self.backward_peak +
             tp_w * self.tp_b_backward_peak) * nseq / tp

        optimizer_peak = \
            ((self.optimizer_peak +
              tp_w * self.tp_b_optimizer_peak) * nseq -
             self.seq_b_optimizer_peak * (nseq - 1)) / tp

        self._logger.debug(f"== checkpoint_activation_layer: {checkpoint_activation_layer}")
        self._logger.debug(f"== checkpoint_activation_embedding: {checkpoint_activation_embedding}")
        self._logger.debug(f"== checkpoint_activation_loss: {checkpoint_activation_loss}")
        self._logger.debug(f"== forward_peak: {forward_peak}")
        self._logger.debug(f"== loss_peak: {loss_peak}")
        self._logger.debug(f"== backward_peak: {backward_peak}")
        self._logger.debug(f"== optimizer_peak: {optimizer_peak}")

        return MemModule(
            checkpoint_activation_layer,
            checkpoint_activation_embedding,
            checkpoint_activation_loss,
            forward_peak,
            loss_peak,
            backward_peak,
            optimizer_peak
        )
