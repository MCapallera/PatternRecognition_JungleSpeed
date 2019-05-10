import ast
import re
is_string = re.compile(r"^[a-z]+", flags=re.IGNORECASE)


def subset(dict, prefix):
    return {key[len(prefix):]: value for key, value in dict.items() if key.startswith(prefix)}


def dynamic_cast(dict):
    for key, value in dict.items():
        if isinstance(value, str) and not is_string.match(value):
            dict[key] = ast.literal_eval(value)
    return dict


def create_and_set(data, key, sub_key, value):
    if key in data:
        data[key][sub_key] = value
    else:
        data[key] = {sub_key: value}
