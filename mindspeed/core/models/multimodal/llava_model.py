# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps


def llava_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        
        # set llava flag for language_transformer_config
        kwargs['language_transformer_config'].is_llava = True

        # set llava flag for vision_transformer_config
        kwargs['vision_transformer_config'].is_llava = True

        fn(self, *args, **kwargs)

    return wrapper