import json

def round_floats(data, digits=4):
    if isinstance(data, float):
        return round(data, digits)
    elif isinstance(data, dict):
        return {key: round_floats(value, digits) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_floats(item, digits) for item in data]
    else:
        return data

def save_json_with_5_digits(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    print(f"begin to round")
    rounded_data = round_floats(data, digits=5)

    print(f"dump to {output_file_path} begin...")
    with open(output_file, 'w') as f:
        json.dump(rounded_data, f, indent=None, separators=(',', ': '), ensure_ascii=False, allow_nan=False)

# Example usage:
input_file_path = 'baked_dnn.json'
print(f"load {input_file_path}")
output_file_path = 'baked_dnn_reduced.json'
save_json_with_5_digits(input_file_path, output_file_path)