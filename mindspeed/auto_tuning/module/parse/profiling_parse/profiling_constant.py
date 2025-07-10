# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
class OperatorDetails:
    def __init__(self, name, type_, input_shapes, output_shapes, duration_us, wait_time_us, accelerator_core):
        self.name: str = name
        self.type: str = type_
        self.input_shapes: str = input_shapes
        self.output_shapes: str = output_shapes
        self.duration_us: float = duration_us
        self.wait_time_us: float = wait_time_us
        self.accelerator_core: str = accelerator_core


class SpecialOperatorName:
    EMBEDDING = 'embedding'
    FW_RMS_NORM_TYPE = 'RmsNorm'
    BW_RMS_NORM_TYPE = 'RmsNormGrad'
    FW_LAYER_NORM_TYPE = 'LayerNormV3WithImplMode'
    BW_LAYER_NORM_TYPE = 'LayerNormBetaGammaBackpropV2'
    RMS_NORM = 'rms_norm'
    LAYER_NORM = 'layer_norm'
    BACKWARD = 'backward'


class SpecialKeyName:
    NAME = 'Name'
    COMPONENT = 'Component'
    TOTAL_RESERVED = 'Total Reserved(MB)'
    ALLOCATED_MEMORY = 'Allocation Total Allocated(MB)'
    ACCELERATOR_CORE = 'Accelerator Core'
    DURATION_US = 'Duration(us)'
    START_TIME_US = 'Start Time(us)'
    ELAPSE_TIME_MS = 'Elapse Time(ms)'
    WAIT_TIME_MS = 'Wait Time(ms)'
    IDLE_TIME_MS = 'Idle Time(ms)'
    TRANSIT_TIME_MS = 'Transit Time(ms)'
