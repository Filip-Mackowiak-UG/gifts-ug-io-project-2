import json


def read_json_file(filename):
    with open(filename, "r") as file:
        file_contents = json.load(file)
    return file_contents

