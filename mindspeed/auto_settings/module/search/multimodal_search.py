import time

from mindspeed.auto_settings.module.search.search_engine import SpaceSearch
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.auto_settings import SingleModel
import numpy as np


class MultimodalSpaceSearch(SpaceSearch):
    def __init__(self):
        super(MultimodalSpaceSearch, self).__init__()
        self._logger = get_logger("multimodal_search_logger")
    
    def _space_search(self, models: list[SingleModel], cpu_num: int):
        timelist = []
        total_world_space = models[0].model_settings.search_world_size
        each_modal_best_cfgs_each_worldspace = []

        for model in models:
            best_cfgs_each_worldspace = [[None for _ in range(total_world_space)] for _ in range(total_world_space)]
            # type: list[list[SearchConfig]]
            modal_num = len(models)
            max_search_num = total_world_space - modal_num + 1
            search_cfg_start_time = time.time()
            for world_size in range(1, max_search_num + 1):
                
                profiling_and_parser_end_time = time.time()
                single_model_search_cfg: list[SearchConfig] = self.search_demo_parallel(model,
                                                                world_size, num_workers=cpu_num)
                search_cfg_end_time = time.time()
                self._logger.info(">>>>>> Search_cfg cost time: %s ms",
                    str((search_cfg_end_time - profiling_and_parser_end_time) * 1000))

                for each_cfg in single_model_search_cfg:
                    if each_cfg is not None and best_cfgs_each_worldspace[world_size][each_cfg.dp] is not None:
                        if best_cfgs_each_worldspace[world_size][each_cfg.dp].performance > each_cfg.performance:
                            best_cfgs_each_worldspace[world_size][each_cfg.dp] = each_cfg
                    else:
                        if each_cfg is not None:
                            best_cfgs_each_worldspace[world_size][each_cfg.dp] = each_cfg
                        continue

            each_modal_best_cfgs_each_worldspace.append(best_cfgs_each_worldspace)
            search_cfg_end_time_end = time.time()
            timelist.append(str((search_cfg_end_time_end - search_cfg_start_time) * 1000)) 
        
        
        each_modal_best_cfgs_each_worldspace = np.array(each_modal_best_cfgs_each_worldspace).transpose(2, 0, 1)


        each_dp_best_perf, best_position_list = [], []
        for each_dp_best_cfgs_each_wordspace in each_modal_best_cfgs_each_worldspace:
            perf, positions = self.search_best_cfg_multimodal(each_dp_best_cfgs_each_wordspace, total_world_space)
            each_dp_best_perf.append(perf)
            best_position_list.append(positions[-1])
        search_cfg_end_time_end = time.time()
        start_index = 0
        for i in range(len(each_dp_best_perf)):
            if each_dp_best_perf[i] is not None:
                start_index = i
                break

        best_perf = each_dp_best_perf[start_index]
        best_positions = best_position_list[start_index]
        best_dp = 1
        #在each_dp_best_perf中寻找到最大的perf，返回该perf以及对应的position
        for i, (perf, positions) in enumerate(zip(each_dp_best_perf, best_position_list)):
            if perf is not None and perf < best_perf:
                best_perf = perf
                best_positions = positions
                best_dp = i
        
        self._logger.info(f">>>>>> Search_cfg each modal cost time: {timelist}ms")
        self._logger.info(f"performance: {best_perf}")        
        for k in range(modal_num):
            self._logger.info(f"modal {models[k].model_config.mm_model_name} refer: {best_positions[k]} cards\n")
            self._logger.info(each_modal_best_cfgs_each_worldspace[best_dp][k][best_positions[k]])

    #搜索算法：
    def search_best_cfg_multimodal(self, each_modal_best_cfgs_each_worldspace: list[list[SearchConfig]], total_world_space: int):
        modal_num = len(each_modal_best_cfgs_each_worldspace)
        if modal_num > total_world_space:
            self._logger.info("no enough card! exit")
        for each_modal in each_modal_best_cfgs_each_worldspace:
            #当某一模态全是None的时候直接退出
            if all(num is None or num.performance is None for num in each_modal):
                self._logger.info("no valid cfg! exit")
                return None, [[]]

        # Find the lower and upper bounds for binary search
        low = min(num.performance for row in each_modal_best_cfgs_each_worldspace for num in row if num is not None and num.performance is not None)
        high = max(num.performance for row in each_modal_best_cfgs_each_worldspace for num in row if num is not None and num.performance is not None)
        best_position: list[list[int]] = [[]]
        while abs(low - high) > 1e-5:
            mid = (low + high) / 2
            # Find the positions for each list
            positions = []
            for k in range(modal_num):
                pos = self.find_position(each_modal_best_cfgs_each_worldspace[k], mid)
                positions.append(pos)
                if pos is None:
                    break

            # 搜索空间中不存在满足该配置限制的值
            if positions[-1] is None:
                low = mid # 二分法提升限制、跳过搜索
                continue
                

            # Check if the sum of positions is <= total_world_space
            if sum(positions[k] for k in range(modal_num)) <= total_world_space:
                # positions[k]对应的index从1开始
                high = mid
                best_position.append(positions)
            else:
                low = mid
        if len(best_position[-1]) == 0:
            return None, [[]]

        return max(each_modal_best_cfgs_each_worldspace[k][best_position[-1][k]].performance for k in range(modal_num)), best_position

    # def best_assign(position, world_size_limit, top_k = 3):
        
    def find_position(self, a: list[SearchConfig], val):
        left = 0
        for i in a:
            if i is None or i.performance is None:
                left = left + 1
            else:
                if val is None or i.performance <= val:
                    return left
                else:
                    left = left + 1
        return None