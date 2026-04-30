import torch
import torch_npu
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


def add_layer_norm_sp_support(config, instance):
    setattr(instance, 'config', config)
    sequence_parallel = False if not hasattr(config, 'sequence_parallel') else config.sequence_parallel
    persist_layer_norm = False if not hasattr(config, 'persist_layer_norm') else config.persist_layer_norm
    setattr(instance, 'sequence_parallel', sequence_parallel)
    setattr(instance.weight, 'sequence_parallel', sequence_parallel)
    setattr(instance.bias, 'sequence_parallel', sequence_parallel)
    setattr(instance, 'persist_layer_norm', persist_layer_norm)


class TENorm:
    """
    Conditional Initialization of Transformer-Engine’s LayerNorm or RMSNorm Instance
    """

    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            if getattr(config, "tp_2d", False):
                instance = LayerNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
            else:
                try:
                    # using apex implementation
                    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
                    instance = FusedLayerNorm(config=config, hidden_size=hidden_size, eps=eps)
                except ImportError:
                    # using torch implementation
                    instance = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
                    add_layer_norm_sp_support(config, instance)
        elif config.normalization == "RMSNorm":
            if getattr(config, "tp_2d", False):
                instance = RMSNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
                instance.use_fused_rmsnorm = False
            else:
                from mindspeed.core.fusions.fused_rms_norm import RMSNorm
                instance = RMSNorm(dim=hidden_size, eps=eps, sequence_parallel=config.sequence_parallel, config=config)
                instance.config.use_fused_rmsnorm = True
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
