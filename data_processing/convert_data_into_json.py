import os
import json
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert HuggingFace datasets to JSON format")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory for JSON files")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading MONDAY from HuggingFace Hub...")

    # Load the dataset from HuggingFace
    dataset_dict = load_dataset("runamu/MONDAY")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    output_paths = {}

    # Process each split
    for split_name, dataset in dataset_dict.items():
        print(f"Converting split '{split_name}' with {len(dataset)} examples...")

        # Convert the dataset to a list of dictionaries
        episode_list = []
        for row in tqdm(dataset):
            episode = []
            for i in range(len(row['actions'])):
                step = {}
                step['ep_id'] = row['video_id']
                step['goal'] = row['title']
                step['img_filename'] = f"{row['video_id']}/frame_{i:04d}"
                step['action_list'] = row['actions'][i]
                episode.append(step)
            episode_list.append(episode)
        ours_data = {'ours': episode_list}

        # Define the output path
        output_path = os.path.join(args.output_dir, f"ours_data_{split_name}.json")
        output_paths[split_name] = output_path

        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ours_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(episode_list)} examples to {output_path}")

if __name__ == "__main__":
    main()
