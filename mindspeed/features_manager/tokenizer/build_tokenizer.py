# Copyright 2025 Huawei Technologies Co., Ltd.
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
import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class BuildTokenizerFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('tokenizer-type', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        self.add_parser_argument_choices_value(
            parser,
            "--tokenizer-type",
            'PretrainedFromHF'
        )

        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                           help="Name or path of the huggingface tokenizer.")
        group.add_argument("--tokenizer-not-use-fast", action='store_false',
                           help="HuggingFace tokenizer not use the fast version.")

    def validate_args(self, args):
        tokenizer_path = getattr(args, 'tokenizer_name_or_path', None)
        if tokenizer_path:
            args.tokenizer_name_or_path = os.path.realpath(tokenizer_path)
        if args.tokenizer_name_or_path and (not os.path.exists(args.tokenizer_name_or_path)):
            raise AssertionError(f'{args.tokenizer_name_or_path} not exists!')

    def register_patches(self, patch_manager, args):
        if args.tokenizer_type == "PretrainedFromHF":
            from mindspeed.tokenizer.build_tokenizer.adaptor import build_tokenizer_HF
            patch_manager.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_HF)
