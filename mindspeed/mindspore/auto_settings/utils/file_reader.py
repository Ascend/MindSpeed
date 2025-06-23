import demjson3


def tolerant_json_parse(file_path):
    """tolerant JSON parser"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        return demjson3.decode(content)
    except demjson3.JSONDecodeError as e:
        print(f"parse fail: {e}")
        return tolerant_json_parse_partial(content)


def tolerant_json_parse_partial(content):
    """tolerant JSON parse partial"""
    start = content.find('{')
    end = content.rfind('}')

    if start == -1 or end == -1:
        return None
    for i in range(end, start, -1):
        try:
            return demjson3.decode(content[start:i + 1])
        except demjson3.JSONDecodeError:
            continue

    return None
