import json
from decimal import Decimal
from mindspeed.mindspore.auto_settings.utils.file_reader import tolerant_json_parse


def get_module_info(file_path, key, sub_key=None):
    try:
        with open(file_path, 'r') as file:
            content = json.loads(file.read())
            if sub_key is None:
                return content[key]
            else:
                return content[key][sub_key]
    except Exception:
        content_data = tolerant_json_parse(file_path)
        content = convert_decimals_to_float(content_data)
        if sub_key is None:
            return content[key]
        else:
            return content[key][sub_key]


def convert_decimals_to_float(data):
    """convert Decimal to float"""
    if isinstance(data, dict):
        return {k: convert_decimals_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_decimals_to_float(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data