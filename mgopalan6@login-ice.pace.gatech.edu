import json
from datasets import Dataset 

def convert_to_dpo_format(raw_data_path, output_path):
    """
    Data should look like this:
    {
        "algorithm_description": "...",
        "preferred_code": "...",
        "rejected_code": "...",
        "par2_preferred": float,
        "par2_rejected": float
    }
    """

    dpo_data = []

    with open(raw_data_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    for item in raw_data:
        dpo_example = {
            "prompt": f"Implement the following Kissat restart heuristic algorithm:\n\n{item['algorithm_description']}\n\nProvide the complete restart.c implementation:",
            "chosen": item['preferred_code'],
            "rejected": item['rejected_code']
        }
        dpo_data.append(dpo_example)

    dataset = Dataset.from_list(dpo_data)
    dataset.save_to_disk(output_path)

    print(f"Converted {len(dpo_data)} examples")
    print(f"Saved to {output_path}")

    return dataset

if __name__ == "__main__":
    raw_data = "data/dpo_training_data.jsonl"
    output_dir = "data/dpo_formatted"

    dataset = convert_to_dpo_format(raw_data, output_dir)

    print("\nExample Entry:")
    print(f"Prompt len: {len(dataset[0]['prompt'])} chars")
    print(f"Chosen len: {len(dataset[0]['chosen'])} chars")
    print(f"Rejected len: {len(dataset[0]['rejected'])} chars")

