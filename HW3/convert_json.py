import json
import os

data_path = './Spoken-SQuAD'
json_files = os.listdir(data_path)
json_files = [f for f in json_files if f.endswith('.json')]

for f in json_files:
    fpath = os.path.join(data_path, f)
    with open(fpath, 'r') as file:
        data = json.load(file)
    with open(fpath, 'w') as file:
        json.dump(data, file, indent=4)