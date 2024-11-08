import sys
from copy import deepcopy

import networkx as nx
import torch
import numpy as np

from megatron.training import print_rank_0
from megatron.core import parallel_state


class GraphSolver:
    graph_solver = None

    def __init__(self):
        self.total_recompute_cost = 0
        self.total_forward_cost = 0
        self.num_layers_module = []
        self.layers_num = 0
        self.transformer_module_memory = 0
        self.recompute_policy = {}
        self.layers_combination = []
        self.layer_full_recompute_combination = None
        self.layer_without_recompute_combination = None
        self.layer_recompute_one_combination = None
        self.module_layers = {}
        self.node_split_flag = ", "
        self.without_recompute_info = ""
        self.all_recompute_info = ""
        self.selective_recompute_info = ""
        self.final_policy_info = ""
        self.static_memory = 0
        self.pp = 1
        self.module_chunk = 1
        self.chp_input = 0
        self.chp_time = 0
        self.full_activation = 0
        self.first_layer_module = None
        self.mp = 1
        self.dp = 1

    @staticmethod
    def get_dg(module_layers):
        dg = nx.DiGraph()
        dg.add_nodes_from([
            (i, {"name": module_layers[i]['name'],
                 "mem": module_layers[i]['memory'],
                 "input": module_layers[i]['input'],
                 "compute": module_layers[i]['time'],
                 "recompute": False,
                 "status": "no_status"})
            for i in range(len(module_layers))
        ])
        dg.add_edges_from([
            (i, i + 1) for i in range(len(module_layers) - 1)
        ])
        return dg

    def broadcast_in_mp_dp(self, tensor, src):
        if self.mp > 1 and parallel_state.get_tensor_model_parallel_src_rank() == src:
            torch.distributed.broadcast(tensor,
                                        src=parallel_state.get_tensor_model_parallel_src_rank(),
                                        group=parallel_state.get_tensor_model_parallel_group())
        if self.dp > 1:
            torch.distributed.broadcast(tensor, src=parallel_state.get_data_parallel_src_rank(),
                                        group=parallel_state.get_data_parallel_group())

    def get_no_recompute_layer(self):
        self.first_layer_module = self.num_layers_module[0]['layers'][0]
        layer_module = self.first_layer_module['layers']
        module_layers = []
        if len(layer_module) == 0:
            return module_layers
        parent_layers = []
        for layer in layer_module:
            if "memory" not in layer:
                continue
            module_layers.append(layer)
            parent_layers.append(layer)
            if "layers" not in layer:
                continue
            parent_name = layer['name']
            sub_layer_name = []
            for sub_layer in layer['layers']:
                if "memory" not in sub_layer:
                    continue
                # rename sub_layer name, like 'self_attention.core_attention'
                sub_layer['name'] = "{}.{}".format(parent_name, sub_layer['name'])
                module_layers.append(sub_layer)
                sub_layer_name.append(sub_layer)
            self.module_layers.update({parent_name: sub_layer_name})
        self.module_layers.update({"parent_layers": parent_layers})
        self.module_layers.update({"module_layers": module_layers})
        return

    # remove full select node, like 'input_layernorm', 'self_attention', 'post_attention_layernorm' and 'mlp' in list
    def remove_full_selective_node(self, recompute_nodes):
        if len(recompute_nodes) == 0:
            return recompute_nodes

        layers_recompute_count = 0
        try:
            for layer in self.module_layers["parent_layers"]:
                name = layer['name']
                if name in recompute_nodes:
                    layers_recompute_count += 1
                if layers_recompute_count == len(self.module_layers["parent_layers"]):
                    recompute_nodes.clear()
                    break
                if name not in self.module_layers.keys():
                    continue
                sub_layers_recompute_count = 0
                for sub_layer in self.module_layers[name]:
                    if sub_layer['name'] in recompute_nodes:
                        sub_layers_recompute_count += 1
                    if sub_layers_recompute_count == len(self.module_layers[name]):
                        recompute_nodes.clear()
                        break
        except KeyError:
            print_rank_0("[ERROR] Some of these keys don't exist.")
        return recompute_nodes

    def get_recompute_op(self, graph):
        recompute_nodes = []
        p_node = []
        for node in graph.nodes:
            if not graph.nodes[node]['recompute']:
                continue
            name = graph.nodes[node]['name']
            recompute_nodes.append(name)
            spd = name.split(".")
            if len(spd) == 2 and spd[0] not in p_node:
                p_node.append(spd[0])
        # remove parent and sub in list together, like 'self_attention' and 'self_attention.core_attention' in list
        for n in p_node:
            if n in recompute_nodes:
                recompute_nodes.clear()
                break
        return self.remove_full_selective_node(recompute_nodes)

    def broadcast_recompute_policy(self, recompute_policy_list):
        try:
            self.mp = parallel_state.get_tensor_model_parallel_world_size()
            self.dp = parallel_state.get_data_parallel_world_size()
        except:
            print_rank_0("WARNING: mp, dp is not defined")
        global_rank = torch.distributed.get_rank()
        src = (global_rank // (self.mp * self.dp)) * self.dp * self.mp

        policy_shape = np.array(recompute_policy_list).shape
        policy_len_tensor = torch.tensor(policy_shape, device=torch.npu.current_device())
        self.broadcast_in_mp_dp(policy_len_tensor, src)
        policy_len = tuple(policy_len_tensor.cpu().numpy().tolist())
        if global_rank == src:
            recompute_policy_tensor = torch.tensor(recompute_policy_list, dtype=torch.int8,
                                                   device=torch.npu.current_device())
        else:
            recompute_policy_tensor = torch.empty(policy_len, dtype=torch.int8,
                                                  device=torch.npu.current_device())

        self.broadcast_in_mp_dp(recompute_policy_tensor, src)
        result = recompute_policy_tensor.cpu().numpy().tolist()
        del recompute_policy_tensor
        return result

    def set_recompute_info_to_module(self, module, recompute_nodes_info):
        for sub_module in module:
            name = sub_module["name"]
            if name not in recompute_nodes_info.keys():
                continue
            info = recompute_nodes_info[name]
            if isinstance(info, bool):
                sub_module["recompute"] = info
                continue
            if "child_module" in info.keys():
                self.set_recompute_info_to_module(sub_module["layers"], info["child_module"])
                continue
            if name in info.keys():
                sub_module["recompute"] = info[name]

    def covert_recompute_node_idx_to_name(self, recompute_nodes):
        result = {}
        try:
            module_layers = self.module_layers["module_layers"]
        except KeyError as e:
            print_rank_0("[ERROR] The key \"module_layers\" doesn't exist.")
            raise e
        for i, node in enumerate(recompute_nodes):
            if node != self.layer_recompute_one_combination.broadcast_value:
                continue
            name = module_layers[i]["name"]
            parent_name = name
            sub_name = ""
            if "." in name:
                parent_name, sub_name = name.split(".")
            if parent_name not in result.keys():
                result[parent_name] = {}
            if sub_name == "":
                result[parent_name].update({name: True})
                continue
            if "child_module" not in result[parent_name].keys():
                result[parent_name]["child_module"] = {}
            result[parent_name]["child_module"].update({name: True, sub_name: True})
        return result

    def set_to_module(self, module, recompute_nodes, idx):
        if len(recompute_nodes) == 0:
            module["recompute"] = True
            return
        recompute_nodes_info = self.covert_recompute_node_idx_to_name(recompute_nodes)
        if len(recompute_nodes_info) == 0:
            return
        self.set_recompute_info_to_module(module["layers"], recompute_nodes_info)

    def apply_policy_to_model(self, recompute_policy_list):
        full_layers = []
        for layer in self.num_layers_module:
            if 'is_module_list' in layer:
                full_layers.extend(layer["layers"])
            else:
                full_layers.append(layer)
        if len(recompute_policy_list) == 0:
            return
        idx = 0
        if (recompute_policy_list[-1][2] == self.layer_full_recompute_combination.broadcast_value
                or recompute_policy_list[0][2] == self.layer_without_recompute_combination.broadcast_value):
            recompute_policy_list = list(reversed(recompute_policy_list))
        for policy in recompute_policy_list:
            n = policy[0]
            combination_idx = policy[1]
            recompute_nodes = []
            if policy[2] == self.layer_without_recompute_combination.broadcast_value:
                status = self.layer_without_recompute_combination.broadcast_value
                try:
                    recompute_nodes = [status for _ in range(len(self.module_layers["module_layers"]))]
                except KeyError:
                    print_rank_0("[ERROR] The key \"module_layers\" doesn't exist.")
            if policy[2] == self.layer_recompute_one_combination.broadcast_value:
                recompute_nodes = policy[3:]
            for i in range(idx, idx + n):
                self.set_to_module(full_layers[i], recompute_nodes, combination_idx)
            idx += n

    # minimize the number of memory, results in all recompute
    def calculate_cost_mem(self, g: nx.DiGraph, idx):
        subtotal_cost = 0
        subtotal_compute_cost = 0
        memory_cost = (g.nodes[idx]['mem'] if not g.nodes[idx]['recompute'] else g.nodes[idx]['input'])
        compute_cost = (g.nodes[idx]['compute'] if g.nodes[idx]['recompute'] else 0)

        successors = g.successors(idx)
        for successor in successors:
            a, b = self.calculate_cost_mem(g, successor)
            subtotal_cost += a
            subtotal_compute_cost += b

        return subtotal_cost + memory_cost, subtotal_compute_cost + compute_cost

    def cal_non_transformer_memory(self, model_context, num_model_chunks):
        # total memory used
        model_memory = 0
        for layer in model_context['layers']:
            model_memory += layer['memory']
            break
        non_size = (model_memory - self.transformer_module_memory) * num_model_chunks
        return non_size

    def reset_cost(self, g: nx.DiGraph, idx, reset_node_name):
        node_name = g.nodes[idx]['name']
        if node_name in reset_node_name:
            g.nodes[idx]['mem'] = 0
            g.nodes[idx]['input'] = 0
            g.nodes[idx]['compute'] = 0
        successors = g.successors(idx)
        for successor in successors:
            self.reset_cost(g, successor, reset_node_name)
        return

    # remove dg redundant nodes, like: self_attention and self_attention.core_attention, remove one
    def reset_redundant_nodes(self, dg, recompute_nodes):
        dg_copy = deepcopy(dg)
        reset_node_name = []
        try:
            for parent_layer in self.module_layers["parent_layers"]:
                parent_name = parent_layer['name']
                if parent_name not in self.module_layers.keys():
                    continue
                sub_reset_node_name = []
                for sub_layer in self.module_layers[parent_name]:
                    sub_reset_node_name.append(sub_layer['name'])
                    if sub_layer['name'] in recompute_nodes:
                        reset_node_name.append(parent_name)
                        sub_reset_node_name.clear()
                        break
                if len(sub_reset_node_name) != 0:
                    reset_node_name.extend(sub_reset_node_name)
        except KeyError:
            print_rank_0("[ERROR] The key \"parent_layers\" doesn't exist.")
        self.reset_cost(dg_copy, 0, reset_node_name)
        return dg_copy

    def layers_combination_init(self, g, idx):
        if idx == 0:
            self.layer_full_recompute_combination = LayerCombination({
                "name": "full_recompute",
                "num": self.layers_num,
                "memory": self.chp_input,
                "cost": self.chp_time,
                "broadcast_value": 0,
                "policy_name": "n_full"
            })
            self.layers_combination.append(self.layer_full_recompute_combination)
            self.layer_without_recompute_combination = LayerCombination({
                "name": "without_recompute",
                "num": self.layers_num,
                "memory": self.full_activation,
                "cost": 0,
                "broadcast_value": 2,
                "policy_name": "n_without"
            })
            self.layers_combination.append(self.layer_without_recompute_combination)
        try:
            if idx >= len(self.module_layers["module_layers"]):
                recompute_nodes = self.get_recompute_op(g)
                if len(recompute_nodes) == 0:
                    return
                dg = self.reset_redundant_nodes(g, recompute_nodes)
                stash_mem_per_layer, recompute_cost = self.calculate_cost_mem(dg, 0)
                self.layer_recompute_one_combination = LayerCombination({
                    "name": self.node_split_flag.join(recompute_nodes),
                    "num": self.layers_num,
                    "memory": stash_mem_per_layer,
                    "cost": recompute_cost,
                    "broadcast_value": 1,
                    "policy_name": "n_selective"
                })
                self.layers_combination.append(self.layer_recompute_one_combination)
                return
        except KeyError:
            print_rank_0("[ERROR] The key \"module_layers\" doesn't exist.")
        if g.nodes[idx]['mem'] >= g.nodes[idx]['input']:
            g.nodes[idx]['recompute'] = True
            self.layers_combination_init(g, idx + 1)            
        g.nodes[idx]['recompute'] = False
        self.layers_combination_init(g, idx + 1)

    def get_max_goods_value(self, idx, ans, device_memory):
        i, j, k = idx[0], idx[1], idx[2]
        pre_step_ans = ans[i - 1][j - k]
        if k == 0:
            return pre_step_ans

        goods_value = ans[i][j]
        memory = pre_step_ans.memory + k * self.layers_combination[i].memory
        cost = pre_step_ans.cost + k * self.layers_combination[i].cost
        if pre_step_ans.cost == float('inf'):
            cost = k * self.layers_combination[i].cost
        try:
            device_memory = max(device_memory - self.static_memory, 0) / self.pp
        except ZeroDivisionError:
            device_memory = max(device_memory - self.static_memory, 0)
            print_rank_0("[ERROR] pipeline model parallel world size is 0. ")

        if device_memory >= memory and cost <= goods_value.cost:
            goods_value.memory = memory
            goods_value.cost = cost
            goods_value.layer_names.clear()
            if len(pre_step_ans.layer_names) > 0:
                goods_value.layer_names.extend(pre_step_ans.layer_names)
            goods_value.layer_names.extend(self.layers_combination[i].name for _ in range(k))

        return goods_value

    def print_recompute_policy(self, memory, cost):
        fmt_str = "With selective recompute:\n"
        for k, v in self.recompute_policy.items():
            if k == self.layer_full_recompute_combination.name:
                policy_name = self.layer_full_recompute_combination.policy_name
            elif k == self.layer_without_recompute_combination.name:
                policy_name = self.layer_without_recompute_combination.policy_name
            else:
                policy_name = self.layer_recompute_one_combination.policy_name
                fmt_str += "recomputeNodes=[{}], ".format(k)
            fmt_str += "{} {}; ".format(v, policy_name)
        all_recompute_cost = self.layers_num * self.layer_full_recompute_combination.cost
        try:
            performance = (all_recompute_cost - cost) / (all_recompute_cost * 4)
        except ZeroDivisionError:
            performance = 0
            print_rank_0("[ERROR] all recompute cost is 0. ")
        fmt_str = fmt_str.strip().rstrip(";")
        fmt_str += "\ntotal mem cost: {:.1f} GiB + {:.1f} GiB, speed up compared with all recompute {:.2%}".format(
            self.static_memory / 1024, memory * self.pp / 1024, performance)
        self.selective_recompute_info = fmt_str

    def get_all_layer_policy(self, combination_num, layer_num, ans):
        layer_nodes = [self.layer_full_recompute_combination.name for _ in range(layer_num)]
        memory = layer_num * self.layer_full_recompute_combination.memory
        cost = layer_num * self.layer_full_recompute_combination.cost
        for i in range(layer_num, 0, -1):
            size = layer_num - len(ans[combination_num][i].layer_names)
            if size != layer_num:
                l_nodes = []
                l_nodes.extend(ans[combination_num][i].layer_names)
                # if the policies of all layers are not found, the remaining layers ues all recompute policy.
                l_nodes.extend(self.layer_full_recompute_combination.name for _ in range(size))
                l_memory = ans[combination_num][i].memory + size * self.layer_full_recompute_combination.memory
                l_cost = ans[combination_num][i].cost + size * self.layer_full_recompute_combination.cost
                if l_cost < cost:
                    cost = l_cost
                    memory = l_memory
                    layer_nodes.clear()
                    layer_nodes.extend(l_nodes)

        for nodes in layer_nodes:
            if nodes not in self.recompute_policy.keys():
                self.recompute_policy.update({nodes: 1})
                continue
            self.recompute_policy.update({nodes: self.recompute_policy[nodes] + 1})

        self.print_recompute_policy(memory, cost)

    def knapsack_best(self, device_memory):
        combination_num = len(self.layers_combination) - 1
        if self.layers_combination[0] is not None:
            combination_num = len(self.layers_combination)
            # make combination index id begin for 1.
            self.layers_combination.insert(0, None)
        # init ans
        ans = [[GoodsValue() for _ in range(self.layers_num + 1)] for _ in range(combination_num + 1)]
        # find max goods value
        for i in range(1, combination_num + 1):
            for j in range(self.layers_num + 1):
                k = 0
                while k <= self.layers_combination[i].num and k <= j:
                    ans[i][j] = self.get_max_goods_value([i, j, k], ans, device_memory)
                    k += 1
        self.get_all_layer_policy(combination_num, self.layers_num, ans)

    def get_combination_idx(self, nodes_name):
        for i in range(len(self.layers_combination)):
            if self.layers_combination[i] is None:
                continue
            if nodes_name == self.layers_combination[i].name:
                return i
        return -1

    def analyse_policy_to_list(self):
        recompute_policy_list = []
        module_layers = []
        try:
            module_layers = self.module_layers["module_layers"]
        except KeyError:
            print_rank_0("[ERROR] The key \"module_layers\" doesn't exist.")
        module_layers_num = len(module_layers)
        for nodes_name, v in self.recompute_policy.items():
            idx = self.get_combination_idx(nodes_name)
            nodes_count = [v, idx]
            if nodes_name == self.layer_without_recompute_combination.name:
                broadcast_value = self.layer_without_recompute_combination.broadcast_value
                nodes_count.extend(broadcast_value for _ in range(module_layers_num + 1))
            elif nodes_name == self.layer_full_recompute_combination.name:
                broadcast_value = self.layer_full_recompute_combination.broadcast_value
                nodes_count.extend(broadcast_value for _ in range(module_layers_num + 1))
            else:
                nodes_count.append(self.layer_recompute_one_combination.broadcast_value)
                recompute_nodes = nodes_name.split(self.node_split_flag)
                for layer in module_layers:
                    if layer["name"] in recompute_nodes:
                        nodes_count.append(self.layer_recompute_one_combination.broadcast_value)
                        continue
                    nodes_count.append(self.layer_without_recompute_combination.broadcast_value)
            recompute_policy_list.append(nodes_count)
        return recompute_policy_list

    def print_list_to_policy(self, recompute_policy_list):
        try:
            module_layers = self.module_layers["module_layers"]
        except KeyError as e:
            print_rank_0("[ERROR] The key \"module_layers\" doesn't exist.")
            raise e
        module_layers_num = len(module_layers)
        if len(recompute_policy_list) == 0:
            return
        fmt_str = ">> final selective strategy <<\n"
        for policy in recompute_policy_list:
            n = policy[0]
            if policy[2] == self.layer_without_recompute_combination.broadcast_value:
                policy_name = self.layer_without_recompute_combination.policy_name
            elif policy[2] == self.layer_full_recompute_combination.broadcast_value:
                policy_name = self.layer_full_recompute_combination.policy_name
            else:
                policy_name = self.layer_recompute_one_combination.policy_name
                policy = policy[3:]
                nodes = []
                for i in range(module_layers_num):
                    if policy[i] == self.layer_recompute_one_combination.broadcast_value:
                        nodes.append(module_layers[i]["name"])
                fmt_str += "recomputeNodes=[{}], ".format(self.node_split_flag.join(nodes))
            fmt_str += "{} {}\n".format(n, policy_name)
        self.final_policy_info = fmt_str.rstrip("\n")

    def get_layers_module(self, model, parent_ctx):
        if 'is_recomputing_layer' in model:
            if 'is_module_list' in model and 'memory' in parent_ctx:
                self.transformer_module_memory += parent_ctx['memory']
            elif 'is_module_list' not in model and 'memory' in model:
                self.transformer_module_memory += model['memory']
            self.num_layers_module.append(model)
            if "layers" in model:
                self.layers_num += len(model["layers"])
            return
        if "layers" not in model:
            return
        for sub_model in model["layers"]:
            self.get_layers_module(sub_model, model)

    def build_solver_info(self, model_context, pp, num_model_chunks):
        self.pp = max(self.pp, pp)
        self.get_layers_module(model_context, "")
        self.total_recompute_cost = sys.maxsize
        # first layer is not recompute
        self.get_no_recompute_layer()
        self.chp_input = self.first_layer_module['input']
        self.chp_time = self.first_layer_module['time']
        self.full_activation = self.first_layer_module['memory']
        self.module_chunk = len(model_context['layers'])
        self.total_forward_cost = self.chp_time * self.layers_num
        self.static_memory = model_context['used_mem'] + self.cal_non_transformer_memory(model_context,
                                                                                         num_model_chunks)

        parent_dg = self.get_dg(self.module_layers["parent_layers"])
        stash_mem_per_layer, _ = self.calculate_cost_mem(parent_dg, 0)
        stash_mem_total = stash_mem_per_layer * self.layers_num
        self.without_recompute_info = (
            f"Without recompute: total mem cost: {self.static_memory / 1024:.1f} GiB + "
            f"{stash_mem_total / 1024:.1f} GiB, total recompute 0, speed up over all recompute 25%")
        stash_mem_total = self.chp_input * self.layers_num
        self.all_recompute_info = (f"With all recompute: total mem cost: {self.static_memory / 1024:.1f} GiB + "
                                   f"{stash_mem_total / 1024:.1f} GiB, total recompute all")

        dg = self.get_dg(self.module_layers["module_layers"])
        self.layers_combination_init(dg, 0)

    def get_policy(self, device_memory):
        self.recompute_policy.clear()
        self.knapsack_best(device_memory)
        recompute_policy_list = self.analyse_policy_to_list()
        if torch.distributed.is_initialized():
            recompute_policy_list = self.broadcast_recompute_policy(recompute_policy_list)
        return recompute_policy_list


def get_graph_solver():
    if GraphSolver.graph_solver is None:
        GraphSolver.graph_solver = GraphSolver()
    return GraphSolver.graph_solver


class LayerCombination:
    def __init__(self, config):
        self.name = config["name"]
        self.num = config["num"]
        self.memory = config["memory"]
        self.cost = config["cost"]
        self.broadcast_value = config["broadcast_value"]
        self.policy_name = config["policy_name"]


class GoodsValue:
    def __init__(self):
        self.layer_names = []
        self.memory = 0
        self.cost = float('inf')
