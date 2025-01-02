# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os

import torch
import numpy as np
import pandas as pd


# profiling data filed
class KeyField:
    OpType = 'Type'
    InputShapes = 'Input Shapes'
    OutputShapes = 'Output Shapes'
    Duration = 'Duration(us)'
    FwdTime = 'fwd_time'
    BwdTime = 'bwd_time'


class SampleCache:
    def __init__(self):
        self.MatMul = {}
        self.RmsNorm = {}
        self.RmsNormGrad = {}
        self.BatchMatMul = {}
        self.Add = {}
        self.LayerNorm = {}
        self.LayerNormGrad = {}
        self.ScaledMaskedSoftmax = {}
        self.ScaledMaskedSoftmaxGrad = {}
        self.FastGeluGrad = {}
        self.FastGelu = {}
        self.Mul = {}
        self.Softmax = {}
        self.SoftmaxGrad = {}
        self.FlashAttentionScore = {}
        self.FlashAttentionScoreGrad = {}

    def clear_cache(self):
        for attr in self.__dict__:
            setattr(self, attr, {})


class ModelManager:
    def __init__(self, npu_type='A2'): 
        self.models = {}
        self.npu_type = npu_type

    def cache_model(self, model, op):
        self.models[op] = model

    def get_cached_model(self, model_name: str):
        return self.models.get(model_name, None)

    def load_model(self, model, op, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Can't find '{model_dir}'.")
        path = os.path.join(model_dir, f"{op}_{self.npu_type}.pth")
        weight = torch.load(path)
        model.set_model_info(weight.popitem()[1])
        model.load_state_dict(weight)
        # if use model to predict,need to set training=False,otherwise require inputs dims==model_train_inputs dims
        # during fit,after clear model cache(self.train()),training's value will be reset True
        model.training = False
        self.models[op] = model

    def save_model(self, model, op, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=False)
        weight = model.state_dict()
        weight['model_info'] = model.get_model_info()
        torch.save(weight, f'{model_dir}/{op}_{self.npu_type}.pth')

    def save_models(self, model_dir):
        for op, op_model in self.models.items():
            self.save_model(op_model, op, model_dir)


class OperateProfileCache:
    def __init__(self):
        self.data_frame = pd.DataFrame(
            columns=[KeyField.OpType, KeyField.InputShapes, KeyField.OutputShapes, KeyField.FwdTime, KeyField.BwdTime]
        )

    def record(self, op_type: str, input_shapes: list, output_shapes: list, fwd_time: float, bwd_time: float):
        _, _, exist = self.find(op_type, input_shapes)
        if not exist:
            input_shapes_str = OperateProfileCache.shapes_to_str(input_shapes)
            output_shape_str = OperateProfileCache.shapes_to_str(output_shapes)
            self.data_frame.loc[len(self.data_frame.index)] = [
                op_type, input_shapes_str, output_shape_str, fwd_time, bwd_time
            ]

    def find(self, op_type: str, input_shapes: list):
        input_shapes_str = OperateProfileCache.shapes_to_str(input_shapes)
        data = self.data_frame[
            (self.data_frame[KeyField.OpType] == op_type) &
            (self.data_frame[KeyField.InputShapes] == input_shapes_str)
        ]
        fwd_time = data[KeyField.FwdTime].mean()
        bwd_time = data[KeyField.BwdTime].mean()
        from_cache = False if np.isnan(fwd_time) and np.isnan(bwd_time) else True
        return fwd_time, bwd_time, from_cache

    @staticmethod
    def shapes_to_str(shapes):
        result = ''
        index = 0
        for shape in shapes:
            result += ','.join(map(lambda x: str(x), shape)) if isinstance(shape, list) else str(shape)
            if index < len(shapes) - 1:
                result += ';' if isinstance(shape, list) else ','
            index += 1
        result = '"' + result
        result = result + '"'
        return result
    

# init singleton instance
model_manager = ModelManager()
sample_cache = SampleCache()
operator_cache = OperateProfileCache()