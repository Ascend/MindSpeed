from collections import defaultdict
from typing import Iterable

from mindspeed.auto_settings.module.communication.linear_models import CommProtocolModel
from mindspeed.auto_settings.module.communication.linear_models import CrossDomainModel
from mindspeed.auto_settings.module.communication.linear_models import HCCSDomainModel
from mindspeed.auto_settings.module.communication.linear_models import LinearModel
from mindspeed.auto_settings.module.communication.linear_models import ROCEDomainModel


class CommPerfLinearModelFactory:
    _instance_table = defaultdict(dict)
    """
    {
    "dp_mlp_ag":{'roce': roce_instance, 'hccs': hccs_instance, 'cross': cross_instance},
    "dp_mlp_rs":{'roce': roce_instance, 'hccs': hccs_instance, 'cross': cross_instance},
    "dp_attn_ag":{'roce': roce_instance, 'hccs': hccs_instance, 'cross': cross_instance},
    "dp_attn_rs":{'roce': roce_instance, 'hccs': hccs_instance, 'cross': cross_instance},
    
    "pp":{'roce': roce_instance, 'hccs': hccs_instance},# each vpp time
    
    # no domain diff
    "cp_vector":common_protocol_instance,
    
    "cp_time":{'roce': roce_instance, 'hccs': hccs_instance,'cross': cross_instance}, # total comm
    
    # no  domain diff
    "cp_attn_fwd":common_protocol_instance,# overlap
    
    # no domain diff
    "cp_attn_bwd":common_protocol_instance, # overlap
    
    "ep":{'roce': roce_instance, 'hccs': hccs_instance, 'cross': cross_instance},# EP time
    }
    """

    @staticmethod
    def get_or_create_model(
        module_name, min_rank_num, max_rank_num, max_hccs_dev_num
    ) -> LinearModel:
        if module_name in ["cp_vector", "cp_attn_fwd", "cp_attn_bwd"]:
            module_model = CommPerfLinearModelFactory._instance_table.get(module_name)
            if not module_model:
                module_model = CommProtocolModel(module_name)
                CommPerfLinearModelFactory._instance_table[module_name] = module_model
                return module_model
            else:
                return module_model

        # HCCS domain model
        if max_rank_num <= max_hccs_dev_num:
            module_hccs_model = CommPerfLinearModelFactory._instance_table.get(module_name, {}).get(
                "hccs"
            )
            if not module_hccs_model:
                module_hccs_model = HCCSDomainModel()
                CommPerfLinearModelFactory._instance_table[module_name]["hccs"] = module_hccs_model
                return module_hccs_model
            else:
                return module_hccs_model

        # CrossDomain model
        if min_rank_num < max_hccs_dev_num < max_rank_num:
            module_cross_model = CommPerfLinearModelFactory._instance_table.get(
                module_name, {}
            ).get("cross")
            if not module_cross_model:
                cross_hccs_model = CommPerfLinearModelFactory._instance_table.get(
                    module_name, {}
                ).get("hccs")
                cross_roce_model = CommPerfLinearModelFactory._instance_table.get(
                    module_name, {}
                ).get("roce")
                if not cross_hccs_model:
                    cross_hccs_model = HCCSDomainModel()
                    CommPerfLinearModelFactory._instance_table[module_name][
                        "hccs"
                    ] = cross_hccs_model

                if not cross_roce_model:
                    cross_roce_model = ROCEDomainModel()
                    CommPerfLinearModelFactory._instance_table[module_name][
                        "roce"
                    ] = cross_roce_model

                cross_model = CrossDomainModel(cross_hccs_model, cross_roce_model)
                CommPerfLinearModelFactory._instance_table[module_name]["cross"] = cross_model
                return cross_model
            else:
                return module_cross_model

        # ROCE domain model
        module_roce_model = CommPerfLinearModelFactory._instance_table.get(module_name, {}).get(
            "roce"
        )
        if not module_roce_model:
            module_roce_model = ROCEDomainModel()
            CommPerfLinearModelFactory._instance_table[module_name]["roce"] = module_roce_model
            return module_roce_model
        else:
            return module_roce_model

    @staticmethod
    def get_models_by_module_name(module_name: str) -> Iterable[CommProtocolModel]:
        if module_name not in ["cp_vector", "cp_attn_fwd", "cp_attn_bwd"]:
            hccs_model = CommPerfLinearModelFactory._instance_table.get(module_name).get('hccs')
            roce_model = CommPerfLinearModelFactory._instance_table.get(module_name).get('roce')
            cross_model = CommPerfLinearModelFactory._instance_table.get(module_name).get('cross')
            re_models = []
            if hccs_model:
                re_models.append(hccs_model)
            if roce_model:
                re_models.append(roce_model)
            if cross_model:
                re_models.append(cross_model)
            return re_models
        else:
            return [
                CommPerfLinearModelFactory.get_or_create_model(
                    module_name, min_rank_num=0, max_rank_num=0, max_hccs_dev_num=0
                )
            ]
