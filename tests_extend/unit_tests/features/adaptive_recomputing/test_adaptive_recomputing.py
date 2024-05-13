# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from ascendspeed import megatron_adaptor
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_solver import GraphSolver

from unit_tests.common import DistributedTest


class TestAdaptiveRecomputing(DistributedTest):
    world_size = 1
    transformer_layer_info_fmt = """
     {
        "prefix_name": "xxx",
        "name": "%d",
        "allowed_recomputing": true,
        "layers": [{
            "prefix_name": "xxx",
            "name": "input_layernorm",
            "memory_bytes": 0,
            "memory": 384,
            "time": 1.9710063934326172,
            "input": 64.0,
            "peak_memory": 402705408,
            "forward_cnt": 2,
            "pre_total_time": 3.9420127868652344
        }, {
            "prefix_name": "xxx",
            "name": "attention",
            "layers": [{
                "prefix_name": "xxx",
                "name": "query_key_value",
                "memory_bytes": 0,
                "memory": 192,
                "time": 9.331226348876953,
                "input": 64.0,
                "peak_memory": 402654208,
                "forward_cnt": 2,
                "pre_total_time": 18.662452697753906
            }, {
                "prefix_name": "xxx",
                "name": "rotary_emb",
                "memory_bytes": 0,
                "memory": 0,
                "time": 1.7354488372802734,
                "input": 64.0,
                "peak_memory": 0,
                "forward_cnt": 2,
                "pre_total_time": 3.470897674560547
            }, {
                "prefix_name": "xxx",
                "name": "triangle_attn",
                "layers": [{
                    "prefix_name": "xxx",
                    "name": "scaled_masked_softmax",
                    "memory_bytes": 0,
                    "memory": 512,
                    "time": 465.08251536976206,
                    "input": 516.0,
                    "peak_memory": 542107136,
                    "forward_cnt": 11,
                    "pre_total_time": 5115.907669067383
                }],
                "memory_bytes": 0,
                "memory": 1664,
                "time": 22.87912368774414,
                "input": 208.0,
                "peak_memory": 2818581504,
                "forward_cnt": 2,
                "pre_total_time": 45.75824737548828
            }, {
                "prefix_name": "xxx",
                "name": "dense",
                "memory_bytes": 0,
                "memory": 64,
                "time": 8.333802223205566,
                "input": 64.0,
                "peak_memory": 536871936,
                "forward_cnt": 2,
                "pre_total_time": 16.667604446411133
            }],
            "memory_bytes": 0,
            "memory": 1792,
            "time": 50.97508430480957,
            "input": 80.0,
            "peak_memory": 2684364288,
            "forward_cnt": 2,
            "pre_total_time": 101.95016860961914
        }, {
            "prefix_name": "xxx",
            "name": "post_attention_layernorm",
            "memory_bytes": 0,
            "memory": 384,
            "time": 1.8906593322753906,
            "input": 64.0,
            "peak_memory": 402705408,
            "forward_cnt": 2,
            "pre_total_time": 3.7813186645507812
        }, {
            "prefix_name": "xxx",
            "name": "mlp",
            "layers": [{
                "prefix_name": "xxx",
                "name": "gate_proj",
                "memory_bytes": 0,
                "memory": 172,
                "time": 9.36591625213623,
                "input": 64.0,
                "peak_memory": 360711168,
                "forward_cnt": 2,
                "pre_total_time": 18.73183250427246
            }, {
                "prefix_name": "xxx",
                "name": "up_proj",
                "memory_bytes": 0,
                "memory": 172,
                "time": 8.879423141479492,
                "input": 64.0,
                "peak_memory": 360711168,
                "forward_cnt": 2,
                "pre_total_time": 17.758846282958984
            }, {
                "prefix_name": "xxx",
                "name": "down_proj",
                "memory_bytes": 0,
                "memory": 64,
                "time": 13.797521591186523,
                "input": 172.0,
                "peak_memory": 536871936,
                "forward_cnt": 2,
                "pre_total_time": 27.595043182373047
            }],
            "memory_bytes": 0,
            "memory": 752,
            "time": 38.39600086212158,
            "input": 64.0,
            "peak_memory": 1258294272,
            "forward_cnt": 2,
            "pre_total_time": 76.79200172424316
        }],
        "memory_bytes": 0,
        "memory": 3312,
        "time": 100.17907619476318,
        "input": 64.0,
        "peak_memory": 3942760960,
        "forward_cnt": 2,
        "pre_total_time": 200.35815238952637
    }
    """
    module_all_fmt = """
    {
        "module": [],
        "layers": [{
            "prefix_name": "xxx",
            "name": "module",
            "layers": [{
                "prefix_name": "xxx",
                "name": "module",
                "layers": [{
                    "prefix_name": "xxx",
                    "name": "embedding",
                    "layers": [{
                        "prefix_name": "xxx",
                        "name": "word_embeddings",
                        "memory_bytes": 0,
                        "memory": 256,
                        "time": 13.043999671936035,
                        "input": 0.25,
                        "peak_memory": 268797952,
                        "forward_cnt": 2,
                        "pre_total_time": 26.08799934387207
                    }],
                    "memory_bytes": 0,
                    "memory": 64,
                    "time": 16.85166358947754,
                    "input": 0.25,
                    "peak_memory": 604310016,
                    "forward_cnt": 2,
                    "pre_total_time": 33.70332717895508
                }, {
                    "prefix_name": "xxx",
                    "name": "language_model",
                    "layers": [{
                        "prefix_name": "xxx",
                        "name": "layers",
                        "is_recomputing_layer": true,
                        "is_module_list": true,
                        "layers": [%s]
                    }],
                    "memory_bytes": 0,
                    "memory": 4336,
                    "time": 1621.1401224136353,
                    "input": 80.0,
                    "peak_memory": 5331085312,
                    "forward_cnt": 2,
                    "pre_total_time": 3242.2802448272705
                }],
                "memory_bytes": 0,
                "memory": 4336,
                "time": 1642.3271894454956,
                "input": 16.25,
                "peak_memory": 5398523392,
                "forward_cnt": 2,
                "pre_total_time": 3284.654378890991
            }],
            "memory_bytes": 0,
            "memory": 4336,
            "time": 1645.2174186706543,
            "input": 16.25,
            "peak_memory": 5398523392,
            "forward_cnt": 2,
            "pre_total_time": 3290.4348373413086
        }],
        "used_mem": 16600,
        "max_device_memory": 58960
    }
    """

    def get_module(self, size):
        module_layers = [self.transformer_layer_info_fmt % i for i in range(size)]
        module_layers_context = self.module_all_fmt % (",".join(module_layers))
        module = json.loads(module_layers_context)
        return module

    def get_transformer_layers(self, module):
        transformer_layers = None
        for sub_module in module["layers"]:
            if sub_module["name"] == "layers":
                transformer_layers = sub_module["layers"]
                break
            if "layers" not in sub_module:
                continue
            transformer_layers = self.get_transformer_layers(sub_module)
        return transformer_layers

    @staticmethod
    def is_recompute_module(module):
        if "recompute" in module and module["recompute"]:
            return True
        return False

    def get_module_recompute_layer(self, module):
        recompute_module_layer = []
        for sub_module in module:
            if self.is_recompute_module(sub_module):
                recompute_module_layer.append(sub_module["name"])
                continue
            if "layers" not in sub_module:
                continue
            for child_module in sub_module["layers"]:
                name = child_module["name"]
                if "." not in child_module["name"]:
                    name = "{}.{}".format(sub_module["name"], child_module["name"])
                if self.is_recompute_module(child_module):
                    recompute_module_layer.append(name)
        return recompute_module_layer

    @staticmethod
    def check_full_or_without(type_name, policy):
        if type_name not in policy:
            return False
        if policy[type_name] <= 0:
            return False
        policy[type_name] -= 1
        return True

    @staticmethod
    def check_selective(recompute_layer, policy):
        if "n_selective" not in policy:
            return False
        n_selective_list = policy["n_selective"]
        for n_selective in n_selective_list:
            n = n_selective["n"]
            recompute_node = n_selective["recompute_nodes"]
            if n <= 0 or len(recompute_layer) != len(recompute_node):
                continue
            if len(set(recompute_layer) | set(recompute_node)) == len(recompute_node) and n > 0:
                n_selective["n"] -= 1
                return True
        return False

    def assert_policy(self, module, policy):
        transformer_layers = self.get_transformer_layers(module)
        for module in transformer_layers:
            # n_full
            if self.is_recompute_module(module):
                if self.check_full_or_without("n_full", policy):
                    continue
                return False
            sub_module_recompute_layer = self.get_module_recompute_layer(module["layers"])
            # n_without
            if len(sub_module_recompute_layer) == 0:
                if self.check_full_or_without("n_without", policy):
                    continue
                return False
            # n_selective
            if self.check_selective(sub_module_recompute_layer, policy):
                continue
            return False
        return True

    def do_solve_graph(self, layer_num, pp, device_memory):
        module = self.get_module(layer_num)
        solver = GraphSolver()
        solver.build_solver_info(module, pp)
        recompute_policy_list = solver.get_policy(device_memory)
        solver.apply_policy_to_model(recompute_policy_list)
        solver.print_list_to_policy(recompute_policy_list)
        print(solver.without_recompute_info)
        print(solver.all_recompute_info)
        print(solver.selective_recompute_info)
        print(solver.final_policy_info)
        return module

    def test_solve_graph_by_module_10_layer_pp_2_52G(self):
        print("=== start to test solve graph: module 10 layer, pp 2, memory 52GB ===")
        module = self.do_solve_graph(10, 2, 52 * 1024)
        policy = {
            "n_without": 2,
            "n_selective": [
                {
                    "n": 8,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_2_54G(self):
        print("=== start to test solve graph: module 10 layer, pp 2, memory 54GB ===")
        module = self.do_solve_graph(10, 2, 54 * 1024)
        policy = {
            "n_without": 3,
            "n_selective": [
                {
                    "n": 7,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_1_52G(self):
        print("=== start to test solve graph: module 10 layer, pp 1, memory 52GB ===")
        module = self.do_solve_graph(10, 1, 52 * 1024)
        policy = {
            "n_without": 10
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_1_54G(self):
        print("=== start to test solve graph: module 10 layer, pp 1, memory 54GB ===")
        module = self.do_solve_graph(10, 1, 54 * 1024)
        policy = {
            "n_without": 10
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_2_52G(self):
        print("=== start to test solve graph: module 32 layer, pp 2, memory 52GB ===")
        module = self.do_solve_graph(32, 2, 52 * 1024)
        policy = {
            "n_full": 11,
            "n_selective": [
                {
                    "n": 19,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm",
                                        "mlp.up_proj"]
                },
                {
                    "n": 2,
                    "recompute_nodes": ["input_layernorm", "attention.query_key_value", "attention.triangle_attn",
                                        "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_2_54G(self):
        print("=== start to test solve graph: module 32 layer, pp 2, memory 54GB ===")
        module = self.do_solve_graph(32, 2, 54 * 1024)
        policy = {
            "n_full": 10,
            "n_selective": [
                {
                    "n": 20,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm",
                                        "mlp.up_proj"]
                },
                {
                    "n": 2,
                    "recompute_nodes": ["input_layernorm", "attention.query_key_value", "attention.triangle_attn",
                                        "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_1_52G(self):
        print("=== start to test solve graph: module 32 layer, pp 1, memory 52GB ===")
        module = self.do_solve_graph(32, 1, 52 * 1024)
        policy = {
            "n_without": 3,
            "n_selective": [
                {
                    "n": 28,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm",
                                        "mlp.up_proj"]
                },
                {
                    "n": 1,
                    "recompute_nodes": ["input_layernorm", "attention.rotary_emb", "attention.triangle_attn"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_1_54G(self):
        print("=== start to test solve graph: module 32 layer, pp 1, memory 54GB ===")
        module = self.do_solve_graph(32, 1, 54 * 1024)
        policy = {
            "n_without": 4,
            "n_selective": [
                {
                    "n": 28,
                    "recompute_nodes": ["input_layernorm", "attention.triangle_attn", "post_attention_layernorm",
                                        "mlp.up_proj"]
                },
            ]
        }
        assert (self.assert_policy(module, policy))
