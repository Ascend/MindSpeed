from typing import List

from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.module.operator.operator_database import OperatorHistory
try:
    from waas_sdk.waas_client import WaasClient
    from waas_sdk.api.krb_options import KrbOptions
    from waas_sdk.api.tls_options import TlsOptions
    from waas_sdk.api.data_options import DataOptions
except ImportError:
    WaasClient = None
    KrbOptions = None
    TlsOptions = None
    DataOptions = None


class WaasDataBase(object):
    def __init__(self, ip_address: str, ip_port: int):
        self.ip_address = ip_address
        self.ip_port = ip_port
        self.waas_client = WaasClient()
        self.krb_options = KrbOptions()
        self.tls_options = TlsOptions()
        self.data_option = DataOptions()

        self.krb_options.set_enable(False)
        self.tls_options.set_enable(False)
        self.waas_client.set_krb(self.krb_options)
        self.waas_client.set_tls(self.tls_options)
        self.connection = True
        try:
            self.waas_client.connect(ip_address, ip_port, "AutoTuning")
            self.data_option.set_request_timeout(60)
            self.data_client = self.waas_client.get_kv_data_client(self.data_option)
        except Exception as e:
            self.connection = False
        self.keys = []
        self.values = []
        self.attributes_set = []
        self.attributes_exclusive_set = []
        self.key_prefix = ""
        self._logger = get_logger('WaasDataBase')

    def insert_data(self, data_key: List, data_value: List, batch_size=100):
        if len(data_key) != len(data_value):
            raise ValueError("The length of data_key and data_value must be the same.")

        self.keys = data_key
        self.values = data_value
        total_items = len(data_key)
        for index in range(0, total_items, batch_size):
            end_index = min(index + batch_size, total_items)
            key_batch = self.keys[index:index + end_index]
            value_batch = self.values[index:index + end_index]
            self.update_data(key_batch, value_batch)

    def update_data(self, key, value):
        batch_length = len(key)
        for index in range(batch_length):
            exist_value = self.get_data(key[index])
            update_key, update_value = key[index], value[index]
            if exist_value:
                exist_operator = self.unmerge_get_attributes(key[index], exist_value)
                new_operator = self.unmerge_get_attributes(key[index], value[index])
                duration = (float(exist_operator['duration']) + float(new_operator['duration'])) / 2
                new_operator['duration'] = str(duration)
                update_key, update_value = self.merge_insert_attributes_dict(new_operator)
            self.data_client.put(update_key, update_value)

    def get_data(self, key):
        temp_value = self.data_client.get(key)
        return temp_value

    def get_all_data(self, keys):
        self.key_prefix = keys
        temp_key, temp_value = [], []
        self.data_client.get_all(self.key_prefix, temp_key, temp_value)
        return temp_key, temp_value

    def delete_data(self, data_key: List):
        for item in data_key:
            self.data_client.delete_all(item)

    def convert_level_db_format(self, operators):
        insert_list = {'key': [], 'value': []}
        for operator in operators:
            insert_key, insert_value = self.merge_insert_attributes(operator)
            insert_list['key'].append(insert_key)
            insert_list['value'].append(insert_value)
        return insert_list

    def merge_insert_attributes(self, operator):
        selected_values = []
        remaining_values = []
        for attr in self.attributes_set:
            try:
                value = getattr(operator, attr)
                selected_values.append(str(value))
            except AttributeError:
                self._logger.warning(f"{attr} is not in operator object")
                selected_values.append("")
        for attr in self.attributes_exclusive_set:
            value = getattr(operator, attr)
            remaining_values.append(str(value))
        separator = '-'
        key = separator.join(selected_values)
        value = separator.join(remaining_values)
        return key, value

    def merge_insert_attributes_dict(self, operator):
        selected_values = []
        remaining_values = []

        for attr in self.attributes_set:
            value = operator.get(attr, "")
            selected_values.append(str(value))

        for attr in self.attributes_exclusive_set:
            value = operator.get(attr, "")
            remaining_values.append(str(value))

        separator = '-'
        key = separator.join(selected_values)
        value = separator.join(remaining_values)

        return key, value

    @staticmethod
    def merge_operator_cal(operator, input_shape=None, output_shape=None):
        class_name = type(operator).__name__
        if class_name == 'DictShape':
            name = operator.type
            if not name:
                name = operator.types
        else:
            name = operator.types
        accelerator_core = operator.accelerator_core
        search_key = [accelerator_core, name]
        if input_shape:
            search_key.append(input_shape)
        if output_shape:
            search_key.append(output_shape)
        separator = '-'
        key = separator.join(search_key)
        return key

    def unmerge_get_attributes(self, key, value):
        separator = '-'
        selected_values = key.split(separator)
        key_attr_values = dict(zip(self.attributes_set, selected_values))

        remaining_values = value.split(separator)
        value_attr_values = dict(zip(self.attributes_exclusive_set, remaining_values))

        attr_values = {**key_attr_values, **value_attr_values}
        return attr_values

    def restore_all_data(self, operator):
        keys = self.merge_operator_cal(operator)
        key_list, value_list = self.get_all_data(keys)
        key_length = len(key_list)
        operator_list = []
        for index in range(key_length):
            dict_operator = self.unmerge_get_attributes(key_list[index], value_list[index])
            operators = self.restore_attributes_to_operator(
                OperatorHistory(
                    types='',
                    accelerator_core='',
                    input_shape='',
                    output_shape='',
                    duration=0,
                    device='',
                    jit='',
                    cann='',
                    driver='',
                    dtype=''
                ),
                dict_operator
            )
            operator_list.append(operators)
        return operator_list

    @staticmethod
    def restore_attributes_to_operator(operator, attr_values):
        for attr, value in attr_values.items():
            if attr == 'duration':
                value = float(value)
            setattr(operator, attr, value)
        return operator

    def attribute_separator(self, operator, attributes_list=None):
        if attributes_list is None:
            attributes_list = ['accelerator_core', 'types', 'input_shape']
        total_attributes = list(vars(operator).keys())
        attributes_set = list(attributes_list)
        attributes_exclusive_set = [attr for attr in total_attributes if attr not in attributes_set]
        if '_sa_instance_state' in attributes_exclusive_set:
            attributes_exclusive_set.remove('_sa_instance_state')
        self.attributes_set = attributes_set
        self.attributes_exclusive_set = attributes_exclusive_set
