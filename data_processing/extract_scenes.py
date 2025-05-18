from datasets import load_dataset
import os
import concurrent.futures
import time
from functools import partial
import argparse
from tqdm import tqdm

from utils import *

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process video dataset")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for output")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

# Process a single row from the dataset, corresponding to a video
def process_row(my_row, split, video_dir, output_dir, verbose=False):
    try:
        video_file = os.path.join(video_dir, my_row["video_id"] + ".mp4")
        # Skip if the video file does not exist
        if not os.path.isfile(video_file):
            return False
        scene_timestamps = my_row["scene_timestamps_in_sec"]
        screen_bboxes = my_row["screen_bboxes"]
        output_folder = os.path.join(output_dir, split, my_row['video_id'])
        os.makedirs(output_folder, exist_ok=True)

        # Extract and crop frames based on timestamps and screen bounding boxes
        extract_frames(video_file, scene_timestamps, output_folder, verbose=verbose)
        crop_and_save_images(output_folder, screen_bboxes)
        if verbose:
            print(f"Processed {my_row.get('video_id', 'unknown')} finished.")
        return True
    except Exception as e:
        print(f"Error processing {my_row.get('video_id', 'unknown')}: {str(e)}")
        return False

# Process an entire dataset split using multiple workers
def process_dataset(dataset, split, video_dir, output_dir, num_workers, verbose):
    process_fn = partial(process_row, split=split, video_dir=video_dir, output_dir=output_dir, verbose=verbose)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_fn, dataset),
            total=len(dataset),
            desc=f"Processing {split}",
        ))
    return sum(results)  # Count of successfully processed items

def main():
    args = parse_args()

    # Check if the input video directory exists
    if not os.path.isdir(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MONDAY from HuggingFace Hub...")
    dataset_dict = load_dataset("runamu/MONDAY")
    start_time = time.time()

    total_processed = 0
    for split in dataset_dict.keys():
        print(f"Processing split '{split}' with {len(dataset_dict[split])} examples...")
        dataset = dataset_dict[split]
        processed = process_dataset(
            dataset,
            split,
            args.video_dir,
            args.output_dir,
            args.workers,
            args.verbose
        )
        total_processed += processed
        print(f"Processed {processed}/{len(dataset)} items in '{split}' split")

    end_time = time.time()
    print(f"Total processed: {total_processed} items")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
