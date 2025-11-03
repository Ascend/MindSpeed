import os
import time
import copy
import torch
import torch.distributed as dist
import shutil

from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.mindspeed_adaptor.mindspeed_executor import ExecutorFlag
from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_settings.mindspeed_adaptor.mindspeed_mm_adaptor import rewrite_json_file
from mindspeed.auto_settings.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_settings.module.communication.comm_perf_predictor_manager import CommPerfPredictorManager
from mindspeed.auto_settings.module.model_performance import ModelPerformance
from mindspeed.auto_settings.module.operator.operator import OperatorPerformance
from mindspeed.auto_settings.utils.utils import get_prof_dir, TimeRecorder, check_file_exists
from mindspeed.auto_settings.config.generate_profiling_configs import generate_profiling_configs
from mindspeed.auto_settings.utils.file_utils import restricted_read

_AUTO_SETTING_FILENAME = "/auto_settings.json"


class ModelSpaces:
    def __init__(self):
        self.sources_setting = get_system_config()
        self.logger = get_logger("main")
        self.time_recorder = TimeRecorder()
        self.model_list: list[SingleModel(TimeRecorder)] = []
        self.space_search_model = None
        self.model_type = ""

    def dist_train_config(self):
        self.logger.info("<==========dist_train_config==========>")
        self.model_list = []
        single_model_name = set()
        temp_world_size = 0
        from mindspeed.core.multi_modal.dist_train.dist_train_config import get_dist_model_name
        devices = torch.cuda.device_count()
        single_model_name = ["gpt", "vit"]
        self.copy_source_file(self.sources_setting.mm_model, (self.sources_setting.work_dir + _AUTO_SETTING_FILENAME))
        self.sources_setting.mm_model = self.sources_setting.work_dir + _AUTO_SETTING_FILENAME
        for model_name in single_model_name:
            single_model = copy.deepcopy(SingleModel)(self.time_recorder)
            single_model.model_config.dist_train = True
            single_model.model_settings.dist_train = True
            single_model.model_config.sub_work_dir = os.path.join(self.sources_setting.work_dir, model_name)
            single_model.model_config.mm_model = (self.sources_setting.work_dir + '/' + model_name + _AUTO_SETTING_FILENAME)
            self.copy_source_file(self.sources_setting.mm_model, single_model.model_config.mm_model)
            single_model.model_config.world_size = torch.cuda.device_count()
            single_model.model_config.mm_model_name = model_name
            single_model.model_config.parallel_switch = ["dp", "pp"]
            self.model_list.append(single_model)


    def copy_source_file(self, src_file, dst_file):
        # 将路径转换为绝对路径
        src_file_abs = os.path.abspath(src_file)
        dst_file_abs = os.path.abspath(dst_file)
         # 确保目标目录存在
        dst_dir = os.path.dirname(dst_file_abs)
        if not os.path.exists(dst_dir):
            try:
                os.makedirs(dst_dir, exist_ok=True)
                self.logger.error(f"创建目标目录: {dst_dir}")
            except Exception as e:
                self.logger.error(f"创建目标目录失败: {str(e)}")
                return False
        try:
            shutil.copy2(src_file_abs, dst_file_abs)
            self.logger.info(f"文件 {src_file_abs} 已成功拷贝到 {dst_file_abs}")
        except FileNotFoundError:
            self.logger.info(f"源文件 {src_file_abs} 未找到。")
        except PermissionError:
            self.logger.info(f"没有足够的权限进行文件拷贝操作。")
        except Exception as e:
            self.logger.info(f"发生未知错误: {e}")


class SingleModel:
    def __init__(self, time_recorder):
        # setting继续使用全局，改变model config
        self.model_settings = get_system_config()
        self.model_config = ModelConfig()
        self.gather_node_profiling = None
        self.profiling_result = []
        self.pkl = PostInfo
        self.model_performance: ModelPerformance = None
        self.memory_model: MemoryModeling = None

        self.logger = get_logger("single_model")
        self.time_recorder: TimeRecorder = time_recorder

    def parse_args(self, model_type):
        # Force refresh model args just in case model has been modified after previous run.
        self.logger.info("<==========Begin to parse args==========>")
        self.logger.info(f"self.model_config.sub_work_dir:{self.model_config.sub_work_dir}")
        self.model_settings.executor.execute(
            self.pkl.FILENAME,
            model_config=self.model_config,
            flag=ExecutorFlag.PARSE_ARGS
        )
        self.logger.info(f"self.model_config.sub_work_dir:  {self.model_config.sub_work_dir},  self.pkl.FILENAME:  {self.pkl.FILENAME}")
        self.pkl: PostInfo = restricted_read(
            os.path.join(self.model_config.sub_work_dir, self.pkl.FILENAME)
        )
        self.model_settings.load_settings(self.pkl)
        for key, value in vars(self.model_settings.model_config).items():
            if value is not None:
                if "mm_model" in key:
                    continue
                setattr(self.model_config, key, value)
        self.model_settings.model_config.mm_model = self.model_settings.mm_model
        if not dist.is_initialized():
            dist.init_process_group(
                backend=dist.Backend.GLOO,
                world_size=self.model_settings.nnodes,
                rank=self.model_settings.node_rank
            )
        #为了传参保留的通信组
        gloo_group = dist.new_group(ranks=list(range(self.model_settings.nnodes)), backend=dist.Backend.GLOO)
        self.model_settings.gloo_group = gloo_group
        self.logger.info("<==========Finished parsing args==========>")
        #todu wass
    
    def _memory_model(self):
        self.logger = get_logger("main")
        self.time_recorder.start_time = time.time()

        # Memory modeling
        self.memory_model = MemoryModeling(self.model_config)

        #mm目前不支持tp
        if self.model_settings.dist_train:
            self.memory_model._static_modeling._baseline_tp = 1
            self.memory_model._dynamic_modeling._baseline_tp = 1
            self.model_settings.use_multiparameter_pipeline_model_parallel = True
        static_list, dynamic_list = self.memory_model.generate_mem_modeling_profiling_list()
        self.logger.info("<==========Begin to profile static memory==========>")
        for cfg, filename in static_list:
            if cfg.dist_train:
                rewrite_json_file(self.model_config.mm_model, cfg)
            if not self._check_file_exists(filename):
                self.model_settings.executor.execute(
                    filename,
                    model_config=self.model_config,
                    cfg=cfg,
                    flag=ExecutorFlag.PARSE_MODEL,
                    gloo_group=self.model_settings.gloo_group
                )
        self.logger.info("<==========Finished profiling static memory==========>")
        self.logger.info("<==========Begin to profile dynamic memory==========>")
        for cfg in dynamic_list:
            if cfg.dist_train:
                rewrite_json_file(self.model_config.mm_model, cfg)
            if not self._check_file_exists(get_prof_dir(cfg)):
                self.model_settings.executor.execute(
                    get_prof_dir(cfg),
                    model_config=self.model_config,
                    cfg=cfg,
                    gloo_group=self.model_settings.gloo_group
                )
        self.logger.info("<==========Finished profiling dynamic memory==========>")
        self.memory_model.modeling(self.model_config.sub_work_dir)
        self.time_recorder.model_parser_end_time = time.time()
        self.logger.info("Model parser cost time: %s ms",
                         str((self.time_recorder.model_parser_end_time - self.time_recorder.start_time) * 1000))


    def _performance_model(self):
        profiling_cfg_list = generate_profiling_configs(self.model_settings, self.model_config)
        self.logger.info("profile_cfgs (tp, pp, dp, cp, ep, #layers, seq_len):")
        self.logger.info(",".join(
            str((cfg.tp,
                 cfg.pp,
                 cfg.dp,
                 cfg.cp,
                 cfg.ep,
                 cfg.num_layers,
                 cfg.seq_length))
            for cfg, _ in profiling_cfg_list))

        self.time_recorder.generate_profiling_config_end_time = time.time()

        profiling_results = []
        self.logger.info("<==========Begin profiling==========>")
        self.logger.info("This process will run the script and get some profiling results.")
        self.logger.info("Please wait for a while.")

        for index, (profiling_cfg, _) in enumerate(profiling_cfg_list):
            # tracking the order of profiling all over the list
            self.logger.info('<==========the %s/%s loop==========>', str(index + 1), str(len(profiling_cfg_list)))
            self.logger.info("profile_db_configs (tp, pp, dp, cp, ep, #layers, seq_len):")
            self.logger.info(str([profiling_cfg.tp,
                                  profiling_cfg.pp,
                                  profiling_cfg.dp,
                                  profiling_cfg.cp,
                                  profiling_cfg.ep,
                                  profiling_cfg.num_layers,
                                  profiling_cfg.seq_length]))
            if self.model_config.dist_train:
                rewrite_json_file(self.model_config.mm_model, profiling_cfg)
            if not self._check_file_exists(get_prof_dir(profiling_cfg)):
                self.model_settings.executor.execute(
                    get_prof_dir(profiling_cfg),
                    model_config=self.model_config,
                    cfg=profiling_cfg,
                    gloo_group=self.model_settings.gloo_group
                )

            self.gather_node_profiling = GatherNodeProfiling(
                os.path.join(self.model_config.sub_work_dir, get_prof_dir(profiling_cfg)))
            # when self.model_settings.DEBUG_MODE
            # self.gather_node_profiling.parse_nodel_pkl_debug(profiling_cfg, self.model_settings)
            profiling_res = self.gather_node_profiling.fuse_node_pkl()

            profiling_results.append([profiling_cfg, profiling_res])

        self.time_recorder.profiling_and_parser_end_time = time.time()

        # Performance Modeling

        self.model_performance = ModelPerformance(
            working_dir=self.model_config.sub_work_dir,
            predictor_mgr=CommPerfPredictorManager(self.model_settings, self.model_config),
            operator=OperatorPerformance(model_config=self.model_config, 
                                        working_dir=self.model_config.sub_work_dir, 
                                        model_settings=self.model_settings)
    )
        self.model_performance.get_profiling_info(profiling_results)
    
    def _check_file_exists(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.model_config.sub_work_dir, filename))
