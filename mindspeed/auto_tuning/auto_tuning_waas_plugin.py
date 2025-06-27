# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import codecs
import pickle

from tuner import Tuner

from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import MindSpeedSettings, MindSpeedSettingsPKL
from mindspeed.auto_tuning.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_tuning.config.generate_profiling_configs import generate_profiling_configs


class AutoTuning(Tuner):
    def __init__(self, dependency):
        super().__init__(dependency)
        self.log = self.logger()
        self.workspace = self.workspace_name()
        self.kv = self.kv_operator()
        self.uuid: str = None  # type: ignore
        self.settings: MindSpeedSettingsPKL = None  # type: ignore

    def on_tune(self, ctx):
        self.uuid = ctx.getRequestProperty("uuid")
        self.log.logInfo(self.uuid)
        self.settings = pickle.loads(codecs.decode(ctx.getRequestData().encode(), "base64"))
        self.log.logInfo(self.settings.model_cfg)
        MindSpeedSettings().load_settings_from_pkl(self.settings)

        MemoryModeling.set_model_cfg(self.settings.model_cfg)
        static_mem, dynamic_mem = MemoryModeling.generate_mem_modeling_profiling_list()
        performance = generate_profiling_configs(self.settings.model_cfg)

        self.kv.put(f"{self.uuid}_static_mem", codecs.encode(pickle.dumps(static_mem), "base64").decode())
        self.kv.put(f"{self.uuid}_dynamic_mem", codecs.encode(pickle.dumps(dynamic_mem), "base64").decode())
        self.kv.put(f"{self.uuid}_performance", codecs.encode(pickle.dumps(performance), "base64").decode())

        return 0
