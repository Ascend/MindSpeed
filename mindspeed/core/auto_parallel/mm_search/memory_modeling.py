# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import json

import torch

from mindspeed.core.auto_parallel.mm_search.help import get_json


def get_model_parameters(model_config):
    transformer_params_count = 12 * model_config["hidden_size"] ** 2
    total_params_count = transformer_params_count * model_config["num_layers"]
    return total_params_count


def get_model_total_static_memory(args, parallel_config):
    model_config = get_json(args.mm_model)
    DP = parallel_config[2]

    if model_config.get("image_encoder"):
        vit_model_cfg = {"hidden_size": model_config["image_encoder"]["vision_encoder"]["hidden_size"], 
                         "num_layers": model_config["image_encoder"]["vision_encoder"]["num_layers"]}
        vit_model_params_count = get_model_parameters(vit_model_cfg)
    if model_config.get("text_decoder"):
        llm_model_cfg = {"hidden_size": model_config["text_decoder"]["hidden_size"], 
                         "num_layers": model_config["text_decoder"]["num_layers"]}
        llm_model_params_count = get_model_parameters(llm_model_cfg)

    mem_para, mem_grad, mem_optimizer = 0, 0, 0
    if model_config["image_encoder"]["vision_encoder"]["params_dtype"] == "bf16":
        if not model_config["image_encoder"]["vision_encoder"].get("freeze", False):
            mem_para += 2 * vit_model_params_count
            mem_grad += 4 * vit_model_params_count
            mem_optimizer += 4 * vit_model_params_count + 8 * vit_model_params_count / DP
        else:
            mem_para += 2 * vit_model_params_count
    if model_config["text_decoder"]["params_dtype"] == "bf16":
        if not model_config["text_decoder"].get("freeze", False):
            mem_para += 2 * llm_model_params_count
            mem_grad += 4 * llm_model_params_count
            mem_optimizer += 4 * llm_model_params_count + 8 * llm_model_params_count / DP
        else:
            mem_para += 2 * llm_model_params_count

    model_total_static_memory = mem_para + mem_grad + mem_optimizer
    return model_total_static_memory / (1024 ** 2)


def parallel_cluster_is_oom(args, parallel_config, static_mem):
    PP, TP = parallel_config[0], parallel_config[1]

    max_available_memory = torch.npu.get_device_properties(0).total_memory * 0.95 / 1024**2

    if PP * TP * max_available_memory < static_mem:
        return True
    else:
        return False


def count_module_param(model):
    for mod in model:
        precision_placeholder = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
        module_param_property = {name: [param.numel(), precision_placeholder.get(param.dtype, 0), param.requires_grad] for name, param in mod.named_parameters()}
        # model_para, optimizer, grad
        module_param_dict = [0, 0, 0]
        for module_param in module_param_property:
            module_param_dict[0] += module_param_property[module_param][0] * \
                module_param_property[module_param][1] / 1024 ** 2
            if module_param_property[module_param][2]:
                module_param_dict[1] += (module_param_property[module_param][0] * 4 + \
                    module_param_property[module_param][0] * 8) / 1024 ** 2
                module_param_dict[2] += module_param_property[module_param][0] * 4 / 1024 ** 2
    module_param_property_json = json.dumps(module_param_property)
    with open(f'raw_profile_{torch.distributed.get_rank()}.json', 'w') as f:
        f.write(module_param_property_json)
    return module_param_dict


