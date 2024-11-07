from mindspeed.op_builder.builder import MindSpeedOpBuilder


class AdaptiveCpOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "adaptive_cp"

    def __init__(self):
        super(AdaptiveCpOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/algorithm/adaptive_cp/adaptive_cp.cpp']

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-Wl,--disable-new-dtags,--rpath', '-s', '-O2']
        return args

