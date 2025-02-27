import argparse


class MindSpeedFeature:
    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0

    def register_args(self, parser):
        pass

    def pre_validate_args(self, args):
        pass

    def validate_args(self, args):
        pass

    def post_validate_args(self, args):
        pass

    def register_patches(self, patch_manager, args):
        ...

    def incompatible_check(self, global_args, check_args):
        if getattr(global_args, self.feature_name, None) and getattr(global_args, check_args, None):
            raise AssertionError('{} and {} are incompatible.'.format(self.feature_name, check_args))

    def dependency_check(self, global_args, check_args):
        if getattr(global_args, self.feature_name, None) and not getattr(global_args, check_args, None):
            raise AssertionError('{} requires {}.'.format(self.feature_name, check_args))

    @staticmethod
    def add_parser_argument_choices_value(parser, argument_name, new_choice):
        for action in parser._actions:
            exist_arg = isinstance(action, argparse.Action) and argument_name in action.option_strings
            if exist_arg and action.choices is not None and new_choice not in action.choices:
                action.choices.append(new_choice)
