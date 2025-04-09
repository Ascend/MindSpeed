import re
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


def parse_string_to_array(input_string):
    pattern = re.compile(r"(\d+|\*|\[|\])")
    tokens = pattern.findall(input_string)

    def parse(tokens):
        stack = []
        while tokens:
            token = tokens.pop(0)
            if token == "[":
                stack.append(parse(tokens))
            elif token == "]":
                return stack
            elif token == "*":
                multiplier = int(tokens.pop(0))
                last_element = stack.pop()
                stack.extend([last_element] * multiplier)
            else:
                stack.append(int(token))
        return stack

    result = parse(tokens)
    return result[0] if len(result) > 0 else []


class UnalignedPipelineFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__("pipeline-num-transformer-layers")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--pipeline-num-transformer-layers", type=str,
                           help="Set pipeline nums for transformer layers.")

    def validate_args(self, args):
        self.incompatible_check(args, "noop_layers")
        self.incompatible_check(args, "num_layer_list")
        if args.pipeline_num_transformer_layers is None:
            return

        pipe_layers = parse_string_to_array(args.pipeline_num_transformer_layers)
        if len(pipe_layers) == 0 or len(pipe_layers) != args.pipeline_model_parallel_size:
            raise AssertionError(f"pipeline_num_transformer_layers' length{len(pipe_layers)} "
                                 f"should equal to pipeline_model_parallel_size{args.pipeline_model_parallel_size}.")
        layer_num = 0
        for vpp_layers in pipe_layers:
            if args.virtual_pipeline_model_parallel_size is None:
                if len(vpp_layers) != 1:
                    raise AssertionError(f"vpp_layers' length: {len(vpp_layers)} should equal to 1.")
            elif len(vpp_layers) == 0 or len(vpp_layers) != args.virtual_pipeline_model_parallel_size:
                raise AssertionError(f"vpp_layers' length: {len(vpp_layers)} in pipeline_num_transformer_layers"
                                     f"should equal to virtual_pipeline_model_parallel_size: {args.virtual_pipeline_model_parallel_size}.")
            for num in vpp_layers:
                if num < 0:
                    raise AssertionError(f"vpp_layers' number{num} in pipeline_num_transformer_layers"
                                         f"should be bigger than 0.")
                layer_num += num
        if layer_num > args.num_layers:
            raise AssertionError(f"pipeline_num_transformer_layers' sum{layer_num}"
                                 f"should be smaller than num_layers{args.num_layers}")
        args.pipeline_num_transformer_layers = pipe_layers

    def register_patches(self, patch_manager, args):
        if args.pipeline_num_transformer_layers is None:
            return
        from mindspeed.core.pipeline_parallel.unaligned.unaligned_pipeline import get_layer_offset_unaligned, \
            get_num_layers_to_build_unaligned
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch(
                "megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset",
                get_layer_offset_unaligned)
            patch_manager.register_patch('megatron.core.transformer.transformer_block.get_num_layers_to_build',
                                         get_num_layers_to_build_unaligned)
