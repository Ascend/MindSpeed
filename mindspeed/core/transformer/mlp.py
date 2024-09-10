from functools import wraps
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule


def mlp_init_wrapper(init_func):
    @wraps(init_func)
    def mlp_init(
            self,
            config: TransformerConfig,
            submodules,
            is_expert=False,
            input_size=None,
            shared_expert=False
    ):
        init_func(self, config,
                  submodules,
                  is_expert,
                  input_size)

        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
        del self.linear_fc1
        del self.linear_fc2
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            shared_expert=shared_expert
        )

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            shared_expert=shared_expert
        )

    return mlp_init
