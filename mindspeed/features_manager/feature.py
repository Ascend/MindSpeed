"""Define base class for mind speed features."""

import argparse

from megatron_adaptor.features_manager.mindspeed_feature import (
    MindSpeedFeature as MAFeature,
)


class MindSpeedFeature(MAFeature):
    """MindSpeed feature base with MindSpeed-only parser helpers."""

    @property
    def patch_manager(self):
        """MindSpeed patches are owned by MindSpeedPatchesManager.

        This keeps MindSpeed patch ownership separate from MA, so that
        removing the MindSpeed layer restores the MA implementation.
        """
        from mindspeed.patch_utils import MindSpeedPatchesManager

        return MindSpeedPatchesManager

    @staticmethod
    def _is_arg_registered(parser, option_string):
        """Check if an argument with the given option string is already registered."""
        for action in parser._actions:
            if option_string in action.option_strings:
                return True
        return False

    @staticmethod
    def add_parser_argument_choices_value(parser, argument_name, new_choice):
        """Add a choice while preserving the parser's choices container type."""
        for action in parser._actions:
            exist_arg = isinstance(action, argparse.Action) and argument_name in action.option_strings
            if exist_arg and action.choices is not None and new_choice not in action.choices:
                action.choices = type(action.choices)(list(action.choices) + [new_choice])
