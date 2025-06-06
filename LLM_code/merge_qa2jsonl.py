

import json
import random

# Assume these are your two input JSONL files
input_file_1 = '../task-1/task1_train_dataset_new.jsonl'  # The first JSONL filename
input_file_2 = '../task-2/task2_train_dataset.jsonl'  # The second JSONL filename
output_file = '../train_dataset_new.jsonl'    # The filename of the merged JSONL output

# Read JSONL file and store it into a list
def read_jsonl(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Read two JSONL files
data1 = read_jsonl(input_file_1)
data2 = read_jsonl(input_file_2)

# Merge data
merged_data = data1 + data2

# Shuffle the merged data randomly
random.shuffle(merged_data)

# Write the shuffled data to a new JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in merged_data:
        f.write(json.dumps(entry) + '\n')

print(f'{len(merged_data)} records have been merged, shuffled, and saved to {output_file}')