import json

def sort_nested_dict(d):
    if not isinstance(d, dict):
        return d
    sorted_items = sorted(d.items())
    sorted_dict = {}
    for key, value in sorted_items:
        sorted_dict[key] = sort_nested_dict(value)
    return sorted_dict

def convert_dict_to_natural_language(d, prefix=""):
    if not isinstance(d, dict):
        return d

    words = ""
    for key, value in d.items():
        if prefix == "":
            new_prefix = f"{key}"
        else:
            new_prefix = f"{prefix} - {key}"
        words += f"{new_prefix} - {convert_dict_to_natural_language(value, prefix=new_prefix)}\n"

    return words

def preprocess_wazuh_event(event_str):
    event_dict = json.loads(event_str, strict=False)
    event_dict = event_dict["_source"]
    del event_dict["@timestamp"]
    event_dict = sort_nested_dict(event_dict)
    return convert_dict_to_natural_language(event_dict)
