from typing import Dict, List, Optional, Tuple, Union
import json
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.core.multi_modal.dist_train.dist_train_config import merge_dist_train_args


def read_json_file(path: str) -> Dict:
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def rewrite_json_file(path: str, cfg):
    total_config = read_json_file(path)
    config = total_config['dist_config']['model_config']
    mm_model_name = cfg.sub_work_dir.split('/')[-1]
    fix_config = []
    for item in config:
        item['name'] = mm_model_name
        item['model_index'] = 0
        item['tensor_model_parallel_size'] = cfg.tensor_model_parallel_size
        item['pipeline_model_parallel_size'] = cfg.pipeline_model_parallel_size
        item['context_parallel_size'] = cfg.context_parallel_size
        item['world_size'] = cfg.world_size
        item['auto_tuning_flag'] = True
        # 'gpt' in mm_model_name:
        if 'text_decoder' in total_config and isinstance(total_config['text_decoder'], dict) and 'num_layers' in total_config['text_decoder']:
            model = total_config['text_decoder']
            total_config['text_decoder']['num_layers'] = cfg.pipeline_model_parallel_size
            total_config['text_decoder']['pipeline_num_layers'] = \
                [1 for _ in range(cfg.pipeline_model_parallel_size)]
            total_config['text_decoder'] = add_gpt_recompute(cfg, model)
            if total_config['text_decoder']['max_position_embeddings'] <= cfg.seq_length:
                total_config['text_decoder']['max_position_embeddings'] = cfg.seq_length
        elif 'predictor' in total_config and isinstance(total_config['predictor'], dict) and 'num_layers' in total_config['predictor']:
            model = total_config['predictor']
            total_config['predictor']['num_layers'] = cfg.pipeline_model_parallel_size
            total_config['predictor']['pipeline_num_layers'] = \
                [1 for _ in range(cfg.pipeline_model_parallel_size)]
            total_config['predictor'] = add_gpt_recompute(cfg, model)
        if 'image_encoder' in total_config and isinstance(total_config['image_encoder'], dict) and 'num_layers' in total_config['image_encoder']['vision_encoder']:
            total_config['image_encoder']['vision_encoder']["num_layers"] = cfg.pipeline_model_parallel_size
            total_config['image_encoder']['vision_encoder']['pipeline_num_layers'] = \
                [1 for _ in range(cfg.pipeline_model_parallel_size)]
            total_config = add_vit_recompute(cfg, total_config)
        fix_config = [item]
    total_config['dist_config']['model_config'] = fix_config
    
    with open(path, 'w') as file:
        json.dump(total_config, file, indent=4)
    return cfg


def add_gpt_recompute(config_list, model):
    if "recompute_method" not in model.keys() or \
            "recompute_granularity" in model or \
            "recompute_num_layers" in model:
        model['recompute_method'] = "block"
        model['recompute_granularity'] = "full"
        model['recompute_num_layers'] = config_list.pipeline_model_parallel_size
    return model


def add_vit_recompute(config_list, total_config):
    if "recompute_method" not in total_config['image_encoder']['vision_encoder'].keys() or \
            "recompute_granularity" in total_config['image_encoder']['vision_encoder'] or \
            "recompute_num_layers" in total_config['image_encoder']['vision_encoder']:
        total_config['image_encoder']['vision_encoder']["recompute_method"] = "block"
        total_config['image_encoder']['vision_encoder']["recompute_granularity"] = "full"
        total_config['image_encoder']['vision_encoder']["recompute_num_layers"] = config_list.pipeline_model_parallel_size
    return total_config


def wrapper(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return config
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error processing file: {e}")
            return None
    return inner
