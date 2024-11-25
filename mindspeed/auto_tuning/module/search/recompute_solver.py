from copy import deepcopy
from typing import List, Dict
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.parse.recompute_module_info import ModuleRecomputeInfo


class RecomputeSolver:

    def __init__(self, first_layer_context, perf, static_memory, memory_limit, search_cfg: SearchConfig, model_config):
        self.num_layers_per_pp = model_config.num_layers // search_cfg.pipeline_model_parallel_size
        self.layer_num_per_chunk = 0
        self.virtual_pipeline_model_parallel_size = 1 if not search_cfg.num_layers_per_virtual_pipeline_stage \
            else (search_cfg.num_layers // search_cfg.num_layers_per_virtual_pipeline_stage //
                  search_cfg.pipeline_model_parallel_size)
        self.search_config = search_cfg
        self.model_config = model_config
        self.module_layers: List[ModuleRecomputeInfo] = []
        self.parent_layers: List[ModuleRecomputeInfo] = []
        self.parent_children_dict: Dict[str, List[ModuleRecomputeInfo]] = {}

        self.first_layer_context = first_layer_context
        self.first_layer_recompute_info = ModuleRecomputeInfo(self.first_layer_context)
        self.full_recompute_performance = perf
        self.static_memory = static_memory
        self.memory_limit = memory_limit

        self.recompute_module: Dict[str, ModuleRecomputeInfo] = {}

        self.layers_combination: List[LayerCombination] = []
        self.layer_full_recompute_combination: LayerCombination = None
        self.layer_without_recompute_combination: LayerCombination = None
        self.layer_recompute_one_combination: LayerCombination = None

        self.node_split_flag = ','

        self.num_warmup_micro_batches_per_chunk = []
        self.num_micro_batches = 0

        if search_cfg.num_layers_per_virtual_pipeline_stage:
            self.num_model_chunks = (search_cfg.num_layers // search_cfg.num_layers_per_virtual_pipeline_stage //
                                     search_cfg.pipeline_model_parallel_size)
        else:
            self.num_model_chunks = 1

    def get_num_warmup_micro_batches(self):
        pipeline_parallel_size = self.search_config.pipeline_model_parallel_size
        data_parallel_size = self.search_config.data_parallel_size
        self.num_micro_batches = self.model_config.global_batch_size // self.model_config.micro_batch_size // data_parallel_size
        if pipeline_parallel_size <= 1:
            self.num_warmup_micro_batches_per_chunk.append(1)
            return
        pipeline_parallel_rank = 0
        total_num_micro_batches = self.num_micro_batches * self.num_model_chunks
        if self.num_model_chunks == 1:
            num_warmup_micro_batches = pipeline_parallel_size - pipeline_parallel_rank - 1
            num_warmup_micro_batches += 1
            self.num_warmup_micro_batches_per_chunk.append(num_warmup_micro_batches)
        else:
            num_warmup_micro_batches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_micro_batches += (self.num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_micro_batches += 1
            num_warmup_micro_batches = min(num_warmup_micro_batches, total_num_micro_batches)
            remain_batch_num = (num_warmup_micro_batches - pipeline_parallel_size * self.num_model_chunks)
            for i in range(self.num_model_chunks):
                if i == 0:
                    self.num_warmup_micro_batches_per_chunk.append(pipeline_parallel_size + max(0, remain_batch_num))
                elif i == self.num_model_chunks - 1:
                    self.num_warmup_micro_batches_per_chunk.append(pipeline_parallel_size + min(0, remain_batch_num))
                else:
                    self.num_warmup_micro_batches_per_chunk.append(pipeline_parallel_size)

    def build_solver_info(self):
        self.prune_no_recompute_layer()
        self.layers_combination_init(0)
        self.get_num_warmup_micro_batches()
        return self.knapsack_best()

    def get_recompute_op(self):
        recompute_nodes = []
        parent_node_list = []
        for module_recompute_info in self.module_layers:
            if not module_recompute_info.recompute:
                continue
            name = module_recompute_info.full_name
            recompute_nodes.append(name)
            separate_node_name_list = name.split(".")
            for i in range(1, len(separate_node_name_list)):
                parent_node_name = ".".join(separate_node_name_list[:-i])
                if parent_node_name not in parent_node_list:
                    parent_node_list.append(parent_node_name)

        for n in parent_node_list:
            if n in recompute_nodes:
                recompute_nodes.clear()
                return recompute_nodes
        return self.remove_full_selective_node(recompute_nodes)

    def prune_no_recompute_layer(self):
        module_layers = []
        parent_layers = [self.first_layer_recompute_info]
        children_module_list = []
        self.recursive_prune_modules(self.first_layer_context, module_layers, parent_layers, children_module_list)
        cur_layer_name = self.first_layer_recompute_info.full_name
        self.parent_children_dict.update({cur_layer_name: children_module_list})
        self.parent_layers = parent_layers
        self.module_layers = module_layers

    def recursive_prune_modules(self, parent_module, module_layers: List, parent_layers: List,
                                children_module_list: List):
        if "layers" not in parent_module:
            return
        parent_modules = parent_module['layers']
        parent_module_recompute_info = ModuleRecomputeInfo(parent_module)
        if len(parent_modules) == 0:
            return
        parent_module_memory_time_rate = get_module_memory_time_rate(parent_module_recompute_info)
        cur_sub_module_list = []
        for sub_layer in parent_modules:
            sub_layer_recompute_info = ModuleRecomputeInfo(sub_layer)
            cur_layer_name = sub_layer_recompute_info.full_name
            cur_sub_module_list.append(sub_layer_recompute_info)
            children_layer_name = []
            self.recursive_prune_modules(sub_layer, module_layers, parent_layers, children_layer_name)
            if children_layer_name:
                self.parent_children_dict.update({cur_layer_name: children_layer_name})
                parent_layers.append(sub_layer_recompute_info)
            sub_layer_memory_time_rate = get_module_memory_time_rate(sub_layer_recompute_info)
            if sub_layer_memory_time_rate < parent_module_memory_time_rate:
                continue
            if not sub_layer_recompute_info.memory or len(children_layer_name) == 1 and children_layer_name[0].memory == sub_layer.get("memory"):
                continue
            module_layers.append(sub_layer_recompute_info)
            self.recompute_module.update({cur_layer_name: sub_layer_recompute_info})

        children_module_list.extend(cur_sub_module_list)

    def remove_full_selective_node(self, recompute_nodes):
        if len(recompute_nodes) == 0:
            return recompute_nodes
        try:
            for parent_module in self.parent_layers:
                parent_module_name = parent_module.full_name
                if parent_module_name not in self.parent_children_dict.keys():
                    continue
                sub_layers_recompute_count = 0
                for sub_layer in self.parent_children_dict[parent_module_name]:
                    if sub_layer.full_name in recompute_nodes:
                        sub_layers_recompute_count += 1
                    if sub_layers_recompute_count == len(self.parent_children_dict[parent_module_name]):
                        recompute_nodes.clear()
                        break
        except KeyError:
            print("[ERROR] Some of these keys don't exist.")
        return recompute_nodes

    def layers_combination_init(self, idx):
        if idx == 0:
            self.layer_full_recompute_combination = LayerCombination({
                "name": "full_recompute",
                "memory": self.first_layer_recompute_info.input_size,
                "cost": self.first_layer_recompute_info.time,
                "policy_name": "n_full"
            })
            self.layers_combination.append(self.layer_full_recompute_combination)
            self.layer_without_recompute_combination = LayerCombination({
                "name": "without_recompute",
                "memory": self.first_layer_recompute_info.memory,
                "cost": 0,
                "policy_name": "n_without"
            })
            self.layers_combination.append(self.layer_without_recompute_combination)
        try:
            if idx >= len(self.module_layers):
                recompute_nodes = self.get_recompute_op()
                if len(recompute_nodes) == 0:
                    return

                stash_mem_per_layer = (self.first_layer_recompute_info.memory -
                                       self.first_layer_recompute_info.input_size)
                recompute_cost = 0
                for recompute_module in recompute_nodes:
                    stash_mem_per_layer -= (self.recompute_module.get(recompute_module).memory -
                                            self.recompute_module.get(recompute_module).input_size)
                    recompute_cost += self.recompute_module.get(recompute_module).time
                self.layer_recompute_one_combination = LayerCombination({
                    "name": self.node_split_flag.join(recompute_nodes),
                    "memory": stash_mem_per_layer,
                    "cost": recompute_cost,
                    "policy_name": "n_selective"
                })
                self.layers_combination.append(self.layer_recompute_one_combination)
                return
        except KeyError:
            print("[ERROR] The key \"module_layers\" doesn't exist.")
        if self.module_layers[idx].memory > self.module_layers[idx].input_size:
            self.module_layers[idx].recompute = True
            self.layers_combination_init(idx + 1)
        self.module_layers[idx].recompute = False
        self.layers_combination_init(idx + 1)

    def get_max_goods_value(self, idx, ans):
        i, j, k = idx[0], idx[1], idx[2]
        pre_step_ans = ans[i - 1][j - k]
        if k == 0:
            return deepcopy(pre_step_ans)

        goods_value = ans[i][j]
        memory = pre_step_ans.memory
        pre_layer_num = j - k
        for index in range(k):
            cur_layer_index = pre_layer_num + index
            cur_layer_chunk_rank = cur_layer_index // self.layer_num_per_chunk
            memory += self.num_warmup_micro_batches_per_chunk[cur_layer_chunk_rank] * self.layers_combination[i].memory
        cost = pre_step_ans.cost + k * self.layers_combination[i].cost * self.num_micro_batches
        if pre_step_ans.cost == float('inf'):
            cost = k * self.layers_combination[i].cost * self.num_micro_batches

        device_memory = self.memory_limit

        if device_memory >= memory and cost <= goods_value.cost and (len(pre_step_ans.layer_names) + k) == j:
            goods_value.memory = memory
            goods_value.cost = cost
            goods_value.layer_names.clear()
            if len(pre_step_ans.layer_names) > 0:
                goods_value.layer_names.extend(pre_step_ans.layer_names)
            goods_value.layer_names.extend(self.layers_combination[i].name for _ in range(k))

        return goods_value

    def knapsack_best(self):
        combination_num = len(self.layers_combination)
        base_memory = (self.static_memory - self.num_layers_per_pp / self.num_model_chunks * sum(self.num_warmup_micro_batches_per_chunk) *
                       self.first_layer_recompute_info.input_size)
        base_cost = (self.full_recompute_performance - self.num_layers_per_pp * self.num_micro_batches *
                     self.first_layer_recompute_info.time)
        ans = [[GoodsValue(base_memory, base_cost) for _ in range(self.num_layers_per_pp + 1)] for _ in range(combination_num)]
        self.layer_num_per_chunk = self.num_layers_per_pp // self.num_model_chunks
        for i in range(1, self.num_layers_per_pp + 1):
            ans[0][i].cost += self.first_layer_recompute_info.time * self.num_micro_batches * i
            for j in range(i):
                cur_layer_chunk_rank = j // self.layer_num_per_chunk
                ans[0][i].memory += (self.first_layer_recompute_info.input_size *
                                     self.num_warmup_micro_batches_per_chunk[cur_layer_chunk_rank])
            ans[0][i].layer_names.extend([self.layer_full_recompute_combination.name for _ in range(i)])

        for i in range(1, combination_num):
            for j in range(1, self.num_layers_per_pp + 1):
                k = 0
                while k <= j:
                    ans[i][j] = self.get_max_goods_value([i, j, k], ans)
                    k += 1

        best_goods_value = ans[combination_num - 1][self.num_layers_per_pp]
        print(f"after solve, current memory is {best_goods_value.memory} and current perf = {best_goods_value.cost} "
              f"and cur_recompute_combination is {best_goods_value.layer_names}")
        need_recompute = False
        for combination_name in best_goods_value.layer_names:
            if combination_name != self.layer_without_recompute_combination.name:
                need_recompute = True
                break
        return need_recompute, best_goods_value.memory, best_goods_value.cost


class LayerCombination:
    def __init__(self, config):
        self.name = config["name"]
        self.memory = config["memory"]
        self.cost = config["cost"]
        self.policy_name = config["policy_name"]


class GoodsValue:
    def __init__(self, memory, cost=float('inf')):
        self.layer_names = []
        self.memory = memory
        self.cost = cost


def get_module_memory_time_rate(module_recompute_info: ModuleRecomputeInfo):
    if module_recompute_info.memory:
        return (module_recompute_info.memory - module_recompute_info.input_size) / module_recompute_info.time
    return -1


def module_name_split_num(module):
    module_name_split_list = module['prefix_name'].split(".")
    return len(module_name_split_list)


def get_parent_module_list(sub_layer_full_name):
    parent_module_list = []
    separate_node_name_list = sub_layer_full_name.split(".")
    for i in range(1, len(separate_node_name_list)):
        parent_node_name = ".".join(separate_node_name_list[:-i])
        parent_module_list.append(parent_node_name)

    return parent_module_list


def is_cur_module_tag_recompute(recompute_module_name_list, cur_module_name):
    if cur_module_name in recompute_module_name_list:
        return True
    else:
        return False
