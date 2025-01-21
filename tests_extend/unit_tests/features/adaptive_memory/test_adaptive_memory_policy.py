import json
from mindspeed import megatron_adaptor
from mindspeed.core.memory.adaptive_memory.adaptive_memory_policy import AdaptMemPolicyManager
from megatron.training.global_vars import set_args, get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches, init_num_microbatches_calculator
from megatron.training.arguments import parse_args
from megatron.core import parallel_state as ps


default_context = """{
    "name": "layers",
    "deep": 5,
    "prefix_name": "module0.module.module.language_model.encoder",
    "submodules": [
         {
              "name": "0",
              "deep": 6,
              "prefix_name": "module0.module.module.language_model.encoder.layers",
              "allowed_adapt": true,
              "submodules": [
                   {
                        "name": "input_norm",
                        "deep": 7,
                        "prefix_name": "module0.module.module.language_model.encoder.layers.0",
                        "is_modlue_of_layer0": true,
                        "input": 16,
                        "memory": 32.0087890625,
                        "forward_cnt": 8,
                        "pre_total_time": 0.6377400048077106,
                        "avg_time": 0.07971750060096383,
                        "output": 16,
                        "module_forward_cnt": 4,
                        "module_forward_total_time": 1.285620030015707,
                        "module_forward_avg_time": 0.32140500750392675,
                        "module_swap_cnt": 4,
                        "module_swap_total_time": 3.5634199380874634,
                        "module_swap_avg_time": 0.8908549845218658,
                        "module_swap_total_memory": 64,
                        "module_swap_avg_memory": 16,
                        "is_swap": true
                   },
                   {
                        "name": "self_attention",
                        "deep": 7,
                        "prefix_name": "module0.module.module.language_model.encoder.layers.0",
                        "submodules": [
                             {
                                  "name": "query_key_value",
                                  "deep": 8,
                                  "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention",
                                  "is_modlue_of_layer0": true,
                                  "input": 16,
                                  "memory": 40.00048828125,
                                  "forward_cnt": 8,
                                  "pre_total_time": 12.090399861335754,
                                  "avg_time": 1.5112999826669693,
                                  "output": 24,
                                  "module_forward_cnt": 4,
                                  "module_forward_total_time": 5.709099888801575,
                                  "module_forward_avg_time": 1.4272749722003937,
                                  "module_swap_cnt": 4,
                                  "module_swap_total_time": 2.4392800331115723,
                                  "module_swap_avg_time": 0.6098200082778931,
                                  "module_swap_total_memory": 64,
                                  "module_swap_avg_memory": 16,
                                  "is_swap": true
                             },
                             {
                                  "name": "core_attention",
                                  "deep": 8,
                                  "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention",
                                  "submodules": [
                                       {
                                            "name": "scale_mask_softmax",
                                            "deep": 9,
                                            "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention.core_attention",
                                            "is_modlue_of_layer0": true,
                                            "is_swap": true
                                       },
                                       {
                                            "name": "attention_dropout",
                                            "deep": 9,
                                            "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention.core_attention",
                                            "is_modlue_of_layer0": true,
                                            "is_swap": true
                                       }
                                  ],
                                  "is_modlue_of_layer0": true,
                                  "is_swap": true
                             },
                             {
                                  "name": "core_attention_flash",
                                  "deep": 8,
                                  "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention",
                                  "is_modlue_of_layer0": true,
                                  "input": 52,
                                  "memory": 72.00146484375,
                                  "forward_cnt": 8,
                                  "pre_total_time": 12.260059833526611,
                                  "avg_time": 1.5325074791908264,
                                  "output": 16,
                                  "module_forward_cnt": 4,
                                  "module_forward_total_time": 6.384199857711792,
                                  "module_forward_avg_time": 1.596049964427948,
                                  "module_swap_cnt": 4,
                                  "module_swap_total_time": 10.591600179672241,
                                  "module_swap_avg_time": 2.6479000449180603,
                                  "module_swap_total_memory": 272,
                                  "module_swap_avg_memory": 68,
                                  "is_swap": true
                             },
                             {
                                  "name": "dense",
                                  "deep": 8,
                                  "prefix_name": "module0.module.module.language_model.encoder.layers.0.self_attention",
                                  "is_modlue_of_layer0": true,
                                  "input": 16,
                                  "memory": 32.00048828125,
                                  "forward_cnt": 8,
                                  "pre_total_time": 10.596959948539734,
                                  "avg_time": 1.3246199935674667,
                                  "output": 16,
                                  "module_forward_cnt": 4,
                                  "module_forward_total_time": 6.443620085716248,
                                  "module_forward_avg_time": 1.610905021429062,
                                  "is_swap": true
                             }
                        ],
                        "is_modlue_of_layer0": true,
                        "input": 20,
                        "memory": 112.00537109375,
                        "forward_cnt": 8,
                        "pre_total_time": 42.454280376434326,
                        "avg_time": 5.306785047054291,
                        "output": 16,
                        "module_forward_cnt": 4,
                        "module_forward_total_time": 34.08845901489258,
                        "module_forward_avg_time": 8.522114753723145,
                        "module_swap_cnt": 4,
                        "module_swap_total_time": 28.252699851989746,
                        "module_swap_avg_time": 7.0631749629974365,
                        "module_swap_total_memory": 368,
                        "module_swap_avg_memory": 92,
                        "is_swap": true
                   },
                   {
                        "name": "post_attention_norm",
                        "deep": 7,
                        "prefix_name": "module0.module.module.language_model.encoder.layers.0",
                        "is_modlue_of_layer0": true,
                        "input": 16,
                        "memory": 32.0087890625,
                        "forward_cnt": 8,
                        "pre_total_time": 0.41586000099778175,
                        "avg_time": 0.05198250012472272,
                        "output": 16,
                        "module_forward_cnt": 4,
                        "module_forward_total_time": 0.20914000272750854,
                        "module_forward_avg_time": 0.052285000681877136,
                        "module_swap_cnt": 4,
                        "module_swap_total_time": 2.3505600094795227,
                        "module_swap_avg_time": 0.5876400023698807,
                        "module_swap_total_memory": 64,
                        "module_swap_avg_memory": 16,
                        "is_swap": true
                   },
                   {
                        "name": "mlp",
                        "deep": 7,
                        "prefix_name": "module0.module.module.language_model.encoder.layers.0",
                        "submodules": [
                             {
                                  "name": "block",
                                  "deep": 8,
                                  "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp",
                                  "submodules": [
                                       {
                                            "name": "moe_layer",
                                            "deep": 9,
                                            "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block",
                                            "submodules": [
                                                 {
                                                      "name": "gate",
                                                      "deep": 10,
                                                      "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer",
                                                      "submodules": [
                                                           {
                                                                "name": "weight",
                                                                "deep": 11,
                                                                "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.gate",
                                                                "is_modlue_of_layer0": true,
                                                                "is_swap": true
                                                           }
                                                      ],
                                                      "is_modlue_of_layer0": true,
                                                      "input": 16,
                                                      "memory": 101.2587890625,
                                                      "forward_cnt": 8,
                                                      "pre_total_time": 22.908900022506714,
                                                      "avg_time": 2.8636125028133392,
                                                      "output": 44.062503814697266,
                                                      "module_forward_cnt": 4,
                                                      "module_forward_total_time": 5.89467990398407,
                                                      "module_forward_avg_time": 1.4736699759960175,
                                                      "module_swap_cnt": 4,
                                                      "module_swap_total_time": 7.2082600593566895,
                                                      "module_swap_avg_time": 1.8020650148391724,
                                                      "module_swap_total_memory": 163.25,
                                                      "module_swap_avg_memory": 40.8125,
                                                      "is_swap": true
                                                 },
                                                 {
                                                      "name": "experts",
                                                      "deep": 10,
                                                      "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer",
                                                      "submodules": [
                                                           {
                                                                "name": "experts",
                                                                "deep": 11,
                                                                "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts",
                                                                "submodules": [
                                                                     {
                                                                          "name": "0",
                                                                          "deep": 12,
                                                                          "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts",
                                                                          "submodules": [
                                                                               {
                                                                                    "name": "w1",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 16.77929997444153,
                                                                                    "avg_time": 2.097412496805191,
                                                                                    "output": 30.84375,
                                                                                    "module_forward_cnt": 4,
                                                                                    "module_forward_total_time": 4.455719947814941,
                                                                                    "module_forward_avg_time": 1.1139299869537354,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 1.2993800342082977,
                                                                                    "module_swap_avg_time": 0.32484500855207443,
                                                                                    "module_swap_total_memory": 35.25,
                                                                                    "module_swap_avg_memory": 8.8125,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w2",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 30.84375,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.789600014686584,
                                                                                    "avg_time": 1.098700001835823,
                                                                                    "output": 8.8125,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 4.568459987640381,
                                                                                    "module_swap_avg_time": 1.1421149969100952,
                                                                                    "module_swap_total_memory": 123.375,
                                                                                    "module_swap_avg_memory": 30.84375,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w3",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 9.187660098075867,
                                                                                    "avg_time": 1.1484575122594833,
                                                                                    "output": 30.84375,
                                                                                    "module_forward_cnt": 4,
                                                                                    "module_forward_total_time": 4.457119941711426,
                                                                                    "module_forward_avg_time": 1.1142799854278564,
                                                                                    "is_swap": true
                                                                               }
                                                                          ],
                                                                          "is_modlue_of_layer0": true,
                                                                          "input": 8.8125,
                                                                          "memory": 141.00244140625,
                                                                          "forward_cnt": 8,
                                                                          "pre_total_time": 35.54563927650452,
                                                                          "avg_time": 4.443204909563065,
                                                                          "output": 8.8125,
                                                                          "module_forward_cnt": 4,
                                                                          "module_forward_total_time": 14.720059871673584,
                                                                          "module_forward_avg_time": 3.680014967918396,
                                                                          "module_swap_cnt": 4,
                                                                          "module_swap_total_time": 19.869799613952637,
                                                                          "module_swap_avg_time": 4.967449903488159,
                                                                          "module_swap_total_memory": 528.75,
                                                                          "module_swap_avg_memory": 132.1875,
                                                                          "is_swap": true
                                                                     },
                                                                     {
                                                                          "name": "1",
                                                                          "deep": 12,
                                                                          "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts",
                                                                          "submodules": [
                                                                               {
                                                                                    "name": "w1",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.389880061149597,
                                                                                    "avg_time": 1.0487350076436996,
                                                                                    "output": 30.84375,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 1.2995599806308746,
                                                                                    "module_swap_avg_time": 0.32488999515771866,
                                                                                    "module_swap_total_memory": 35.25,
                                                                                    "module_swap_avg_memory": 8.8125,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w2",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 30.84375,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.446460008621216,
                                                                                    "avg_time": 1.055807501077652,
                                                                                    "output": 8.8125,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 4.565199971199036,
                                                                                    "module_swap_avg_time": 1.141299992799759,
                                                                                    "module_swap_total_memory": 123.375,
                                                                                    "module_swap_avg_memory": 30.84375,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w3",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 9.265139937400818,
                                                                                    "avg_time": 1.1581424921751022,
                                                                                    "output": 30.84375,
                                                                                    "is_swap": true
                                                                               }
                                                                          ],
                                                                          "is_modlue_of_layer0": true,
                                                                          "input": 8.8125,
                                                                          "memory": 141.00244140625,
                                                                          "forward_cnt": 8,
                                                                          "pre_total_time": 26.79997992515564,
                                                                          "avg_time": 3.349997490644455,
                                                                          "output": 8.8125,
                                                                          "module_forward_cnt": 4,
                                                                          "module_forward_total_time": 13.817359924316406,
                                                                          "module_forward_avg_time": 3.4543399810791016,
                                                                          "module_swap_cnt": 4,
                                                                          "module_swap_total_time": 19.430299758911133,
                                                                          "module_swap_avg_time": 4.857574939727783,
                                                                          "module_swap_total_memory": 528.75,
                                                                          "module_swap_avg_memory": 132.1875,
                                                                          "is_swap": true
                                                                     },
                                                                     {
                                                                          "name": "2",
                                                                          "deep": 12,
                                                                          "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts",
                                                                          "submodules": [
                                                                               {
                                                                                    "name": "w1",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.2",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.872340083122253,
                                                                                    "avg_time": 1.1090425103902817,
                                                                                    "output": 30.84375,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 1.2996800243854523,
                                                                                    "module_swap_avg_time": 0.32492000609636307,
                                                                                    "module_swap_total_memory": 35.25,
                                                                                    "module_swap_avg_memory": 8.8125,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w2",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.2",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 30.84375,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.863640069961548,
                                                                                    "avg_time": 1.1079550087451935,
                                                                                    "output": 8.8125,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 4.567259907722473,
                                                                                    "module_swap_avg_time": 1.1418149769306183,
                                                                                    "module_swap_total_memory": 123.375,
                                                                                    "module_swap_avg_memory": 30.84375,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w3",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.2",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 9.201199889183044,
                                                                                    "avg_time": 1.1501499861478806,
                                                                                    "output": 30.84375,
                                                                                    "is_swap": true
                                                                               }
                                                                          ],
                                                                          "is_modlue_of_layer0": true,
                                                                          "input": 8.8125,
                                                                          "memory": 141.00244140625,
                                                                          "forward_cnt": 8,
                                                                          "pre_total_time": 27.632519960403442,
                                                                          "avg_time": 3.4540649950504303,
                                                                          "output": 8.8125,
                                                                          "module_forward_cnt": 4,
                                                                          "module_forward_total_time": 13.569799900054932,
                                                                          "module_forward_avg_time": 3.392449975013733,
                                                                          "module_swap_cnt": 4,
                                                                          "module_swap_total_time": 19.436500072479248,
                                                                          "module_swap_avg_time": 4.859125018119812,
                                                                          "module_swap_total_memory": 528.75,
                                                                          "module_swap_avg_memory": 132.1875,
                                                                          "is_swap": true
                                                                     },
                                                                     {
                                                                          "name": "3",
                                                                          "deep": 12,
                                                                          "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts",
                                                                          "submodules": [
                                                                               {
                                                                                    "name": "w1",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.3",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.63152003288269,
                                                                                    "avg_time": 1.0789400041103363,
                                                                                    "output": 30.84375,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 1.2999800145626068,
                                                                                    "module_swap_avg_time": 0.3249950036406517,
                                                                                    "module_swap_total_memory": 35.25,
                                                                                    "module_swap_avg_memory": 8.8125,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w2",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.3",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 30.84375,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 8.314799904823303,
                                                                                    "avg_time": 1.039349988102913,
                                                                                    "output": 8.8125,
                                                                                    "module_swap_cnt": 4,
                                                                                    "module_swap_total_time": 4.569100022315979,
                                                                                    "module_swap_avg_time": 1.1422750055789948,
                                                                                    "module_swap_total_memory": 123.375,
                                                                                    "module_swap_avg_memory": 30.84375,
                                                                                    "is_swap": true
                                                                               },
                                                                               {
                                                                                    "name": "w3",
                                                                                    "deep": 13,
                                                                                    "prefix_name": "module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.3",
                                                                                    "is_modlue_of_layer0": true,
                                                                                    "input": 8.8125,
                                                                                    "memory": 39.65673828125,
                                                                                    "forward_cnt": 8,
                                                                                    "pre_total_time": 9.212700009346008,
                                                                                    "avg_time": 1.151587501168251,
                                                                                    "output": 30.84375,
                                                                                    "is_swap": false
                                                                               }
                                                                          ],
                                                                          "is_modlue_of_layer0": true,
                                                                          "input": 8.8125,
                                                                          "memory": 141.00244140625,
                                                                          "forward_cnt": 8,
                                                                          "pre_total_time": 26.85856056213379,
                                                                          "avg_time": 3.3573200702667236,
                                                                          "output": 8.8125,
                                                                          "module_forward_cnt": 4,
                                                                          "module_forward_total_time": 13.761080026626587,
                                                                          "module_forward_avg_time": 3.4402700066566467,
                                                                          "module_swap_cnt": 4,
                                                                          "module_swap_total_time": 19.43123960494995,
                                                                          "module_swap_avg_time": 4.857809901237488,
                                                                          "module_swap_total_memory": 528.75,
                                                                          "module_swap_avg_memory": 132.1875,
                                                                          "is_swap": false
                                                                     }
                                                                ],
                                                                "is_modlue_of_layer0": true,
                                                                "is_swap": false
                                                           }
                                                      ],
                                                      "is_modlue_of_layer0": true,
                                                      "input": 35.25,
                                                      "memory": 599.26025390625,
                                                      "forward_cnt": 8,
                                                      "pre_total_time": 119.19857883453369,
                                                      "avg_time": 14.899822354316711,
                                                      "output": 35.25,
                                                      "module_forward_cnt": 4,
                                                      "module_forward_total_time": 56.19960021972656,
                                                      "module_forward_avg_time": 14.04990005493164,
                                                      "module_swap_cnt": 4,
                                                      "module_swap_total_time": 77.52988052368164,
                                                      "module_swap_avg_time": 19.38247013092041,
                                                      "module_swap_total_memory": 2115,
                                                      "module_swap_avg_memory": 528.75,
                                                      "is_swap": false
                                                 }
                                            ],
                                            "is_modlue_of_layer0": true,
                                            "input": 16,
                                            "memory": 672.45654296875,
                                            "forward_cnt": 8,
                                            "pre_total_time": 178.45436096191406,
                                            "avg_time": 22.306795120239258,
                                            "output": 16,
                                            "module_forward_cnt": 4,
                                            "module_forward_total_time": 101.57614326477051,
                                            "module_forward_avg_time": 25.394035816192627,
                                            "module_swap_cnt": 4,
                                            "module_swap_total_time": 120.47393989562988,
                                            "module_swap_avg_time": 30.11848497390747,
                                            "module_swap_total_memory": 2553.75,
                                            "module_swap_avg_memory": 638.4375,
                                            "is_swap": false
                                       }
                                  ],
                                  "is_modlue_of_layer0": true,
                                  "input": 16,
                                  "memory": 672.45654296875,
                                  "forward_cnt": 8,
                                  "pre_total_time": 179.55749893188477,
                                  "avg_time": 22.444687366485596,
                                  "output": 16.000003814697266,
                                  "module_forward_cnt": 4,
                                  "module_forward_total_time": 88.85853958129883,
                                  "module_forward_avg_time": 22.214634895324707,
                                  "module_swap_cnt": 4,
                                  "module_swap_total_time": 115.26704025268555,
                                  "module_swap_avg_time": 28.816760063171387,
                                  "module_swap_total_memory": 2553.75,
                                  "module_swap_avg_memory": 638.4375,
                                  "is_swap": false
                             }
                        ],
                        "is_modlue_of_layer0": true,
                        "input": 16,
                        "memory": 672.45654296875,
                        "forward_cnt": 8,
                        "pre_total_time": 180.64299964904785,
                        "avg_time": 22.58037495613098,
                        "output": 16,
                        "module_forward_cnt": 4,
                        "module_forward_total_time": 87.97980117797852,
                        "module_forward_avg_time": 21.99495029449463,
                        "module_swap_cnt": 4,
                        "module_swap_total_time": 113.94688034057617,
                        "module_swap_avg_time": 28.486720085144043,
                        "module_swap_total_memory": 2553.75,
                        "module_swap_avg_memory": 638.4375,
                        "is_swap": false
                   }
              ],
              "is_layer0_of_module0": true,
              "is_modlue_of_layer0": true,
              "input": 20,
              "memory": 800.4794921875,
              "forward_cnt": 8,
              "pre_total_time": 227.34654235839844,
              "avg_time": 28.418317794799805,
              "output": 16,
              "module_forward_cnt": 4,
              "module_forward_total_time": 203.36761856079102,
              "module_forward_avg_time": 50.841904640197754,
              "module_swap_cnt": 4,
              "module_swap_total_time": 209.38258361816406,
              "module_swap_avg_time": 52.345645904541016,
              "module_swap_total_memory": 3049.75,
              "module_swap_avg_memory": 762.4375,
              "is_swap": false
         }
    ],
    "is_module_list": true,
    "is_adapt_layer": true
}"""


def init_args():
    args = parse_args(None, True)
    set_args(args)
    args.num_layers = 24
    args.global_batch_size = 16
    args.micro_batch_size = 2
    args.data_parallel_size = 1
    try:
        get_num_microbatches()
    except Exception as e:
        init_num_microbatches_calculator(0, None, args.global_batch_size, args.micro_batch_size,
                                         args.data_parallel_size)


parallels = []


def mock_pp_size():
    return parallels[0]


def mock_vpp_size():
    return parallels[1]


def mock_pp_rank():
    return parallels[2]


def mock_function(pp, vpp, rank):
    parallels.clear()
    parallels.extend([pp, vpp, rank])

    global_args = get_args()
    global_args.pipeline_model_parallel_size = parallels[0]
    global_args.virtual_pipeline_model_parallel_size = parallels[1]
    global_args.num_layers_per_virtual_pipeline_stage = global_args.num_layers // pp // vpp

    old_pp_func = ps.get_pipeline_model_parallel_world_size
    old_vpp_func = ps.get_virtual_pipeline_model_parallel_world_size
    old_pp_rank = ps.get_pipeline_model_parallel_rank
    ps.get_pipeline_model_parallel_world_size = mock_pp_size
    ps.get_virtual_pipeline_model_parallel_world_size = mock_vpp_size
    ps.get_pipeline_model_parallel_rank = mock_pp_rank
    return old_pp_func, old_vpp_func, old_pp_rank


def reset_function(old_pp_func, old_vpp_func, old_pp_rank):
    ps.get_pipeline_model_parallel_world_size = old_pp_func
    ps.get_virtual_pipeline_model_parallel_world_size = old_vpp_func
    ps.get_pipeline_model_parallel_rank = old_pp_rank


class TestPolicyGeneration:
    def test_policy_combination_generate(self):
        init_args()
        f1, f2, f3 = mock_function(4, 3, 0)

        manager = AdaptMemPolicyManager()
        unittest_default_context = json.loads(default_context)
        manager.traversal_model_context(unittest_default_context)
        for comb in manager.policy_combinations:
            comb.memory = comb.memory + manager.without_adapt_mem
        assert manager.full_recompute_comb is not None, "no full recompute comb"
        assert manager.full_swap_comb is not None, "no full swap comb"
        assert manager.without_adaptive_comb is not None, "no without adaptive comb"
        assert len(manager.policy_combinations) != 0

        policy_list = set()
        for comb in manager.policy_combinations:
            # Join and sort the recompute and swap lists into strings
            recompute_str = "recompute:" + ','.join(sorted(comb.recompute or []))
            swap_str = "swap:" + ','.join(sorted(comb.swap or []))

            # Generate the policy string
            policy = recompute_str + ';' + swap_str

            # Check if the policy already exists in the set
            assert policy not in policy_list, "Duplicate policy found"

            # Add the policy string to the set
            policy_list.add(policy)
