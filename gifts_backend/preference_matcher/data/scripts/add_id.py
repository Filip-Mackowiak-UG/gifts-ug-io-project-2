from model_training import utils
import json


def add_ids_to_json(json_list):
    for i, item in enumerate(json_list):
        item["id"] = i
    return json_list


products_json = utils.read_json_file("../products.json")
products_with_ids = add_ids_to_json(products_json)

with open("../products.json", 'w') as outfile:
    json.dump(products_with_ids, outfile)

