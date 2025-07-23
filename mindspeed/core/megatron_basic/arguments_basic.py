# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import inspect
from logging import getLogger
from functools import wraps
from dataclasses import make_dataclass, field

from mindspeed.args_utils import get_full_args
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

LOG = getLogger(__name__)


def extra_args_provider_decorator(extra_args_provider):
    """Make a extra args parser  for megatron."""

    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        MindSpeedFeaturesManager.register_features_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    """Decorate parse_args function of megatron."""

    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def validate_args_wrapper(validate_args):
    """A decorator for megatron arguments validation function."""

    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        # make prev validation and copy some args.
        MindSpeedFeaturesManager.pre_validate_features_args(args)

        # make megatron args validation then restore args thar are copied.
        args = validate_args(args, defaults)

        # make post validation after megatron validation.
        MindSpeedFeaturesManager.post_validate_features_args(args=args)

        args.create_attention_mask_in_dataloader = False
        args.reduce_recompute_for_last_chunk = False

        MindSpeedFeaturesManager.validate_features_args(args=args)

        # _print_args is patched, so it has three arguments.
        from megatron.training.arguments import _print_args
        _print_args("arguments", args, True)

        return args

    return wrapper


def print_args_wrapper(fn):
    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        if self.num_moe_experts is None:
            _ori_var_seq = getattr(self, 'variable_seq_lengths', False)
            self.variable_seq_lengths = False
        fn(self)
        if self.num_moe_experts is None:
            self.variable_seq_lengths = _ori_var_seq

    return wrapper


def transformer_config_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        known_config = {}
        unknown_config = {}
        full_args = vars(get_full_args()).copy()
        full_args.update(dict(kwargs))

        config_key = inspect.signature(self.__class__).parameters
        for key, value in full_args.items():
            if key in config_key:
                known_config[key] = value
            else:
                unknown_config[key] = value

        fields = []
        for key, value in unknown_config.items():
            if not hasattr(self, key):
                fields.append((str(key), type(value), field(init=False)))
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))
        for key, value in unknown_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        fn(self, *args, **known_config)

    return wrapper
