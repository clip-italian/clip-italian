import json

with open('../data/train_dataset.json', 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    # print(f"result: {result}")
    if not isinstance(result, dict):
        print(json_str)
