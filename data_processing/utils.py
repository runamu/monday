import numpy as np
import os
import subprocess
from PIL import Image

# Format seconds into HH:MM:SS.mmm for ffmpeg
def format_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

# Extract frames from a video at specific timestamps using ffmpeg
def extract_frames(video_path, timestamps, output_dir, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, sec in enumerate(timestamps):
        ts = format_seconds(sec)
        output_path = os.path.join(output_dir, f"frame_{i:04}_temp.png")

        if os.path.exists(output_path) or os.path.exists(output_path.replace('_temp.png', '.png')):
            continue

        if verbose and i % 5 == 0:  # Print progress every 5 frames
            print(f"  Processing frame {i}/{len(timestamps)} at {ts}")

        cmd = [
            "ffmpeg",
            "-ss", ts,
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",  # high quality
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Crop images in-place to bounding boxes, then refine to pure content region
def crop_and_save_images(output_dir, crop_boxes):
    image_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_temp.png')])

    if len(image_paths) != len(crop_boxes):
        raise ValueError(f"Mismatch between number of images ({len(image_paths)}) and crop boxes ({len(crop_boxes)})")
    if len(image_paths) == 0:
        return

    for idx, (img_path, box) in enumerate(zip(image_paths, crop_boxes)):
        image = Image.open(img_path)

        # Crop to the provided bounding box
        cropped_image = image.crop(box)

        # Further crop to exclude black borders
        frame = np.array(cropped_image)
        H, W, _ = frame.shape
        p_x, p_y, p_w, p_h = get_pure_size(frame)
        x1, y1, x2, y2 = int(p_x * W), int(p_y * H), int((p_x + p_w) * W), int((p_y + p_h) * H)
        pure_image = cropped_image.crop((x1, y1, x2, y2))
        pure_image.save(img_path.replace('_temp.png', '.png'))

        # Remove the temporary file
        os.remove(img_path)

# Compute the bounding box of non-black content in an image
def get_pure_size(image_np: np.ndarray):
    H, W, _ = image_np.shape

    # Generate mask of non-black pixels
    if image_np.ndim == 3:
        mask = image_np.max(axis=-1) > 0
    else:
        mask = image_np > 0

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return 0.0, 0.0, 1.0, 1.0  # Return full image if completely black

    if coords.shape[1] != 2:
        raise ValueError(f"Unexpected coordinates shape: {coords.shape}")

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    return y0 / W, x0 / H, (y1 - y0) / W, (x1 - x0) / H
