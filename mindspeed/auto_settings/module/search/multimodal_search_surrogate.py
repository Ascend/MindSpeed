from typing import Deque, List, Optional, Tuple
from collections import deque
from copy import deepcopy
import os
import sys
import traceback as tb
import numpy as np
from typing import Deque, List, Optional, Tuple
from multiprocessing import Pool, JoinableQueue, Process, Event, queues, Manager
from mindspeed.auto_settings.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.config.model_config import ModelConfig
from io import StringIO
from functools import partial
from mindspeed.auto_settings.module.model_performance import ModelPerformance
from mindspeed.auto_settings.auto_settings import SingleModel

from mindspeed.auto_settings.module.search.multimodal_search import MultimodalSpaceSearch
from mindspeed.auto_settings.auto_settings import SingleModel
from mindspeed.auto_settings.utils.logger import get_logger, change_stream_handler
from mindspeed.auto_settings.module.search.stage_1_prune_multimodal import stage_1_discrete_search_space_prune
import copy


class SurrogateElement:
    def __init__(self):
        self.performance = 0 # 当performance为None，表示被“要求某维度必须相等”剪枝
        self.cfg = None     # 存储该位置的最优搜索结果
        self.active_cfg = None # 存储该位置被激活的cfg
        self.all_cfg = None # 存储该位置可能的全部搜索维度
        self.refreshed = False # 该位置是否已被访问


class MultimodalSpaceSearchSurrogate(MultimodalSpaceSearch):
    def __init__(self):
        super(MultimodalSpaceSearch, self).__init__()
        self._logger = get_logger("multimodal_search_surrogate_logger")
        self.surrogate_model: SurrogateModel = None

    def _ones_multimodal_search(self, start_search_num: list, single_model_max_num: int, models: list[SingleModel], cpu_num: int):
        self._logger.info(f"multimodal_search_surrogate.py _ones_multimodal_search in")
        # 寻找到对应数据
        start_cfgs_all = []
        start_cfgs_worldspace = []
        each_modal_perf = []
        position_index = start_search_num

        for position_index_eachmodal, world_space, model in zip(position_index, self.surrogate_model.search_space, models):

            while position_index_eachmodal < single_model_max_num:
                for item in world_space:
                    if item:
                        self._logger.debug(item.refreshed)
                    else:
                        self._logger.debug("item == none")
                
                position_index_eachmodal += self.find_position(world_space[position_index_eachmodal:], None)
                single_model_search_cfg: list[SearchConfig] = self.search_demo_parallel(model,
                                                                world_size=position_index_eachmodal + 1, num_workers=cpu_num,
                                                                search_space_cfg=world_space[position_index_eachmodal].active_cfg)
                if single_model_search_cfg[0] is not None:
                    start_cfgs_worldspace.append(position_index_eachmodal) # 从0开始计数
                    start_cfgs_all.append(single_model_search_cfg[0])
                    each_modal_perf.append(single_model_search_cfg[0].performance)

                    # all_cfg中应该需要删除search_demo_parallel中被判定为OOM的类
                    world_space[position_index_eachmodal].cfg = single_model_search_cfg[0]
                    break
                else:
                    world_space[position_index_eachmodal] = None
                    position_index_eachmodal += 1 # 全被剪枝，搜索域跳过该值

            if position_index_eachmodal > single_model_max_num:
                self._logger.error("No enough workspace for one modal, exit")
                return
        return start_cfgs_all, start_cfgs_worldspace, each_modal_perf

    def _space_search(self, models: list[SingleModel], cpu_num: int, max_repeat_time: int = 100):
        self._logger.debug(f"multimodal_search_surrogate.py _space_search in")

        total_world_dev_num = models[0].model_settings.search_world_size
        modal_num = len(models)
        single_model_max_num = total_world_dev_num - modal_num + 1
        self._logger.debug(f"single_model_max_num = {single_model_max_num}\
                            total_world_dev_num = {total_world_dev_num}\
                            modal_num = {modal_num}")
        start_search_num = []
        start_search_num_each_modal = 0 # index从0开始计算
        for i in range(modal_num):
            start_search_num.append(start_search_num_each_modal)
            i = i + 1
        self.surrogate_model = SurrogateModel(model_list=models, single_model_max_num=single_model_max_num)

        start_cfgs_all, start_cfgs_worldspace, _ = self._ones_multimodal_search(start_search_num, single_model_max_num, models, cpu_num)
        
        self.surrogate_model.data_load(model_cfg=start_cfgs_all, world_size=start_cfgs_worldspace)
        
        perf_each_dp = [None] * (single_model_max_num + 1)
        positions_each_dp = [None] * (single_model_max_num + 1)
        world_space = []
        init_search_space = copy.deepcopy(self.surrogate_model.search_space)
        for dp in range(single_model_max_num):
            each_dp_search_space = copy.deepcopy(init_search_space)
            self.surrogate_model.search_space = each_dp_search_space            
            self.surrogate_model.generate_search_space(dp + 1)
            world_space.append(self.surrogate_model.search_space)
            
            for i in range(max_repeat_time): 
                perf, positions = self.search_best_cfg_multimodal(self.surrogate_model.search_space, total_world_dev_num)
                if len(positions[-1]) != 0:
                    new_cfgs_all, new_cfgs_worldspace, each_modal_perf = self._ones_multimodal_search(positions[-1], total_world_dev_num, models, cpu_num)
                    self.surrogate_model.add_refresh(new_cfgs_all, new_cfgs_worldspace)
                    
                else:
                    positions_each_dp[dp] = None
                    break

                if new_cfgs_worldspace == positions[-1]:
                    positions_each_dp[dp] = positions[-1]
                    perf_each_dp[dp] = max(each_modal_perf)
                    break
                
            
            self._logger.info("The model is not converged, and the result may be inaccurate.")
            positions_each_dp[dp] = new_cfgs_worldspace


            # 根据性能选择最优的dp与world_size 
        perf_each_dp = np.array(perf_each_dp)
        best_dp = np.argmin(np.where(perf_each_dp == None, np.inf, perf_each_dp))
        positions = positions_each_dp[best_dp]

        self.surrogate_model.search_space = world_space[best_dp]


        self._logger.info(f"performance: {perf}")
        for k in range(modal_num):
            self._logger.info(f"modal {models[k].model_config.mm_model_name} refer: {positions[k] + 1} cards\n")
            self._logger.info(f"\n{self.surrogate_model.search_space[k][positions[k]].cfg}")
                
    def find_position(self, a: list[SurrogateElement], val):
        return super().find_position(a, val)

        
    def search_best_cfg_multimodal(self, each_modal_best_cfgs_each_worldspace: list[list[SurrogateElement]], total_world_dev_num: int):

        return super().search_best_cfg_multimodal(each_modal_best_cfgs_each_worldspace, total_world_dev_num - self.surrogate_model.modal_num)

    def search_demo_parallel(
        self, 
        model: SingleModel,
        re_profiling_flag=False,# 兼容性测试完成
        recomp_cfg_list=None,
        num_workers=None,
        world_size=0,
        working_dir=None,
        search_space_cfg=None
    ) -> [List[Optional[SearchConfig]]]:
        self._logger.debug(f"search_engine.py-search_demo_parallel  in")
        mem_model = model.memory_model
        perfmodel = model.model_performance
        setting = model.model_settings
        model_config = model.model_config
        if not working_dir:
            working_dir = model.model_settings.work_dir,
        if world_size == 0:
            world_size = model.model_settings.search_world_size

        if search_space_cfg is None:
            stage_1_valid_ptd_configs: list[SearchConfig] = stage_1_discrete_search_space_prune(model_config)
        else:
            stage_1_valid_ptd_configs: list[SearchConfig] = search_space_cfg


        device_mem_cap = setting.memory_cap
        self._logger.info(f"Search: total_device_num: {world_size}")
        

        self._logger.info(f"Stage [1] pruned result: number of valid PTD configurations [{len(stage_1_valid_ptd_configs)}]")
        
        if len(stage_1_valid_ptd_configs) <= 1:
            self.perf_cfg_map = deque([(float("inf"), None)] * 3, 3)
        else:
            self.perf_cfg_map = deque([(float("inf"), None)] * len(stage_1_valid_ptd_configs), len(stage_1_valid_ptd_configs))
      

        for cfg in stage_1_valid_ptd_configs:
            self._logger.info(f"Stage [1] pruned config: TP=[{cfg.tp}] PP=[{cfg.pp}] LAYERS_PER_VPP=[{cfg.layers_per_vpp}] DP=[{cfg.dp}] CP=[{cfg.cp}] EP=[{cfg.ep}] ZeRO=[{cfg.zero1}]")
        
        queue = JoinableQueue()
        manager = Manager()
        lock = manager.Lock()
        share_list = manager.list([0])

        terminate_event = Event()
        perfmodel.operator.del_db_connection()
        best_cfg_handling = Process(target=self.best_cfg, args=(queue, terminate_event))
        best_cfg_handling.start()
        partial_compute_cfg = partial(
            self.compute_cfg, 
            lock=lock,
            profile_count=share_list,
            mem_model=mem_model,
            perf_model=perfmodel,
            model_config=model_config, 
            working_dir=working_dir, 
            re_profiling_flag=re_profiling_flag, 
            recomp_cfg_list=recomp_cfg_list,
            device_mem_cap=device_mem_cap
            )
        pool = Pool(processes=num_workers)
        for cfg in stage_1_valid_ptd_configs:
            pool.apply_async(
                partial_compute_cfg, 
                args=(cfg,), 
                callback=lambda res: self.put_in_queue(res, queue), 
                error_callback=self.err_callback
                )
        
        pool.close()
        pool.join()
        terminate_event.set()
        queue.join()
        best_cfg_handling.join()
        self.perf_cfg_map = queue.get()
        queue.join()
        return [cfg for _, cfg in self.perf_cfg_map]

    def put_in_queue(self, result, queue):
        self._logger.info("Stage 1 complete!")
        if result is not None:
            queue.put(result)

    def err_callback(self, err):
        self._logger.error(f'error: {str(err)}')

    def compute_cfg(
        self,
        cfg,
        lock, 
        profile_count: Manager().list(),
        mem_model: MemoryModeling,
        perf_model: ModelPerformance,
        working_dir: str,
        model_config: ModelConfig,
        re_profiling_flag=False,
        recomp_cfg_list=None,
        device_mem_cap=65535
        ):
        output1 = StringIO()
        rearch_logger = get_logger("multimodal_search_surrogate_logger")
        # change_stream_handler(rearch_logger, output1)
        
        rearch_logger.info(f"Search: device_mem_cap: {device_mem_cap}")
        uncovered_prof = []
        fw_performance = 0
    
        rearch_logger.info("====================")
        rearch_logger.info(f"Looking at:\n\n{cfg}")
        recompute_mem, peak_stage_mem, optimizer_peak = mem_model.estimate(cfg, parallel=True, output=output1)
        if max(peak_stage_mem, optimizer_peak) <= device_mem_cap:
            try:
                perf, uncovered_prof, use_mc2, fw_performance = perf_model.performance(
                    cfg, working_dir, profile_count, re_profiling_flag, output1, lock
                )
                
            except Exception as err:
                rearch_logger.warning(f"Search: ERROR during perf_modeling_calculation: {type(err).__name__}")
                tb.print_exc()
    
            rearch_logger.info(f"before recompute, perf = {perf} and memory = {peak_stage_mem}"
                          f"success enter recompute_solver and tp = {cfg.tensor_model_parallel_size} "
                          f"pp = {cfg.pipeline_model_parallel_size} "
                          f"layers_per_vpp={cfg.num_layers_per_virtual_pipeline_stage} "
                          f"dp = {cfg.data_parallel_size} cp = {cfg.context_parallel_size} "
                          f"ep = {cfg.expert_model_parallel_size} zero = {cfg.use_distributed_optimizer}")
            need_recompute, new_perf, add_mem, recompute_layer = self.full_recompute_solver(
                device_mem_cap - peak_stage_mem, model_config, perf, cfg, recompute_mem, fw_performance
            )
            new_memory = add_mem + peak_stage_mem   # 优化器内存峰值如何与重计算solver后的内存峰值进行比较
            rearch_logger.info(f"after recompute, perf = {new_perf} and need_recompute = {need_recompute}")
            # if not need_recompute:
            rearch_logger.info(f"cur mem_estimated = {new_memory}, recompute_layer = {recompute_layer}")
    
            rearch_logger.info(f"{output1.getvalue()}\n")
    
            return new_perf, need_recompute, new_memory, recompute_layer, use_mc2, uncovered_prof, cfg
        else:
            rearch_logger.info(f"OOM found, next!")
            rearch_logger.info(f"{output1.getvalue()}\n")
            return None
    
    def best_cfg(self, Queue: JoinableQueue, terminate_event):
    
        while not terminate_event.is_set() or not Queue.empty():
            better_found = False
            try:
                new_perf, need_recompute, new_memory, recompute_layer, use_mc2, uncovered_prof, cfg = Queue.get(timeout=0.1)
                
                for i, perf_cfg in enumerate(self.perf_cfg_map):
                    if new_perf < perf_cfg[0]:
                        better_found = True
                        cfg.performance = new_perf
                        cfg.memory = new_memory
                        cfg.recompute_num_layers = recompute_layer
                        cfg.use_ascend_mc2 = use_mc2 if cfg.tensor_model_parallel_size > 1 else False
                        self._logger.info(f"Search: SUCCESSFUL Better #{i} Config Found.")
                        self._logger.info(f"Performance Estimation: {new_perf}.")
                        self.perf_cfg_map.pop()
                        self.perf_cfg_map.insert(i, (new_perf, cfg))
                        break
                if not better_found:
                    self._logger.info(f"Sub-optimal performance, next!")
                Queue.task_done()
            except queues.Empty:
                continue
        Queue.put(self.perf_cfg_map)
        Queue.task_done()


#代理模型：代替搜索空间
class SurrogateModel:
    def __init__(self, model_list: list[SingleModel], single_model_max_num: int, parameters=2):
        #type: list[list]   代表已被探索的配置
        #type: list[list]   配置对应的时间表现，即因变量y
        #type: list[list]   配置对应的卡数，即自变量x  (存储时从0开始计数)
        #type: list[list]   代理模型的参数值
        #type:int           代理模型最大参数量，默认值为2
        # 总卡数，在这里指每个模态组件限定的搜索维度
        # 搜索空间
        
        self.model_config = None               
        self.model_config_performance = None   
        self.world_size = None             
        self.val = None                     
        self.paramenters = parameters       
        self.single_model_max_num = single_model_max_num          
        self.search_space = []
        self._logger = get_logger("SurrogateModel")

        # 搜索空间提前剪枝
        for each_modal in model_list:
            each_modal_worldspace = []
            for each_world_size in range(1, single_model_max_num + 1):
                self._logger.info(f"SurrogateModel each_world_size {each_world_size}")
                all_cfg = stage_1_discrete_search_space_prune(each_modal.model_config, each_world_size)
                if len(all_cfg) != 0:
                    element = SurrogateElement()
                    element.all_cfg = all_cfg
                    # 默认全部激活，后续调整时调整部分激活
                    element.active_cfg = all_cfg
                else:
                    element = None
                each_modal_worldspace.append(element)
            self.search_space.append(each_modal_worldspace)

        self.modal_num = len(model_list)


    def data_load(self, model_cfg: list[SearchConfig], world_size):         # 分模态存储数据，初始化
        self.model_config = []
        self.model_config_performance = []
        self.world_size = []       
        self.val = [] 

        for i, (each_modal, each_modal_worldsize) in enumerate(zip(model_cfg, world_size)):
            
            each_modal_list = []
            each_modal_list.append(each_modal)
            self.model_config.append(each_modal_list)
            perf = []
            perf.append(each_modal.performance)
            self.model_config_performance.append(perf)

            self.search_space[i][each_modal_worldsize].cfg = each_modal

            each_modal_worldsize_list = []
            each_modal_worldsize_list.append(each_modal_worldsize)
            self.world_size.append(each_modal_worldsize_list)



    def add_refresh(self, model_cfg: list[SearchConfig], world_size: list):       # 增量更新数据
        for i, (each_modal_cfg, each_modal_worldsize, each_modal_performance) in \
            enumerate(zip(self.model_config, self.world_size, self.model_config_performance)):
            each_modal_cfg.append(model_cfg[i])
            each_modal_worldsize.append(world_size[i])
            each_modal_performance.append(model_cfg[i].performance)
            self.search_space[i][world_size[i]].cfg = model_cfg[i]

    # 拟合求解方法
    def solve_value(self):
        self.val = []
        X_all = self.generate_matrix()
        y_all = self.model_config_performance
        for x, y in zip(X_all, y_all):
            val = np.linalg.pinv(x) @ y
            self.val.append(val)

    # 按条件生成、更新搜索空间
    def generate_search_space(self, dp=1):
        self.solve_value()
        for each_modal_each_worldsize, each_modal_val in zip(self.search_space, self.val):
            for i in range(self.single_model_max_num):
                if each_modal_each_worldsize[i] is None:
                    continue # 被剪枝的值直接跳过
                each_modal_each_worldsize[i].performance = 0
                for val_num, val in enumerate(each_modal_val):
                    #参数更新顺序设定：y = a0/x + a1 + a2/x^2 + ... + an/x^n
                    if val_num == 0:
                        # i从0开始的偏移量
                        each_modal_each_worldsize[i].performance += val * 1 / (i + 1)
                    elif val_num == 1:
                        each_modal_each_worldsize[i].performance += val * 1
                    else:
                        each_modal_each_worldsize[i].performance += val * 1 / ((i + 1) ** val_num)
                each_modal_each_worldsize[i].active_cfg = []
                for cfg in each_modal_each_worldsize[i].all_cfg:
                    if cfg is None:
                        del cfg
                        continue
                    if cfg.dp == dp:
                        each_modal_each_worldsize[i].active_cfg.append(cfg)
                if len(each_modal_each_worldsize[i].active_cfg) == 0:
                    #不存在合适的DP值，置零
                    each_modal_each_worldsize[i].performance = None
        
    #参数更新顺序设定：y = a0/x + a1 + a2/x^2 + ... + an/x^n， 根据该参数生成每个模态的X矩阵, 目前暂定参数量为2
    def generate_matrix(self):
        #每次全量更新参数，获取每个模态X值
        X_all = []
        val_len = len(self.world_size[0])
        if val_len >= 1 and self.paramenters >= 1:
            for each_modal_world_size in self.world_size:
                val_each_model = []
                for each_modal_each_world_size in each_modal_world_size:
                    # index从0开始的偏移量
                    val_each_model_each_world_size = [1 / (each_modal_each_world_size + 1)]
                    val_each_model.append(val_each_model_each_world_size)
                X_all.append(val_each_model)
        if val_len >= 2 and self.paramenters >= 2:
            for X_one_modal in X_all:
                for each_worldspace in X_one_modal:
                    # index从0开始的偏移量
                    each_worldspace.append(1)

        while val_len >= 3 and self.paramenters >= 3:
            for i in range(2, min(val_len, self.paramenters)):
                for each_worldspace in X_one_modal:
                    for modal_index, X_one_modal in enumerate(X_all):
                        each_worldspace.append(1 / ((self.world_size[modal_index][i] + 1) ** i))
        return X_all
