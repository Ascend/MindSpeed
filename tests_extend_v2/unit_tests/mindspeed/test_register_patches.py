from functools import wraps
import pytest
from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
from tests_extend_v2.unit_tests.common import DistributedTest


def function1():
    return 'this is function1'


def function2():
    return 'this is function2'


def function3():
    return 'this is function3'


def function_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) + ' wrapper'

    return wrapper


def function_second_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) + ' wrapper2'

    return wrapper


def function_third_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) + ' wrapper3'

    return wrapper


class TestClass:
    test_variable = 1


class TestRegisterPatches(DistributedTest):
    world_size = 1

    def test_replace_class_variable(self):
        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import TestClass
        assert TestClass.test_variable == 1
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.TestClass.test_variable', 2)
        aspm.apply_patches()
        assert TestClass.test_variable == 2

    def test_import_no_exist_function(self):
        aspm.register_patch('no_exist_module.module.no_exist_function', create_dummy=True)
        aspm.apply_patches()

        from no_exist_module.module import no_exist_function
        with pytest.raises(RuntimeError, match='function no_exist_module.module.no_exist_function no exist'):
            no_exist_function()

    def test_import_no_exist_module(self):
        aspm.register_patch('no_exist_module', create_dummy=True)
        aspm.apply_patches()
        import no_exist_module


class TestRegisterPatchesResetEnv(DistributedTest):
    world_size = 1
    reuse_dist_env = False

    def test_replace_function(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function2)
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function2'

    def test_wrapper_function(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function_wrapper)
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function1 wrapper'

    def test_multi_patch(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function2)
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function_wrapper)
        aspm.register_patch(
            'tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1',
            function_second_wrapper)
        aspm.register_patch(
            'tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1',
            function_third_wrapper)
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function2 wrapper wrapper2 wrapper3'

    def test_double_patch(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function2)

        with pytest.raises(RuntimeError, match='the patch of function1 exist !'):
            aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function3)

    def test_force_double_patch(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function2)
        aspm.register_patch(
            'tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1',
            function3,
            force_patch=True)
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function3'

    def test_double_wrapper(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function_wrapper)
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function_wrapper)
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function1 wrapper'

    def test_remove_wrapper(self):
        aspm.register_patch('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', function_wrapper)
        aspm.remove_wrappers('tests_extend_v2.unit_tests.mindspeed.test_register_patches.function1', 'function_wrapper')
        aspm.apply_patches()

        from tests_extend_v2.unit_tests.mindspeed.test_register_patches import function1

        assert function1() == 'this is function1'
