import os
import shutil
from PIL import Image
import torch
from utils import WaifuScorer
import decimal

def round_half_up(n):
    return int(decimal.Decimal(n).to_integral_value(rounding=decimal.ROUND_HALF_UP))

SCORER = None

def get_scorer():
    global SCORER
    if SCORER is None:
        SCORER = WaifuScorer(
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True
        )
    return SCORER


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def process_directory(folder):
    scorer = get_scorer()
    files = [f for f in os.listdir(folder) if is_image_file(f)]

    for filename in files:
        path = os.path.join(folder, filename)
        try:
            image = Image.open(path).convert("RGB")
            score = scorer([image])[0]  # returns a list of scores
            bucket = str(min(max(round_half_up(score), 0), 10)) # clamp 0–10, rounding half up

            target_dir = os.path.join(folder, bucket)
            os.makedirs(target_dir, exist_ok=True)

            shutil.move(path, os.path.join(target_dir, filename))
            print(f"Moved {filename} → {bucket} (Score: {score:.2f})")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    while True:
        folder = input("\nEnter folder path (or press Enter to quit): ").strip()
        if not folder:
            break
        if not os.path.isdir(folder):
            print("Invalid path. Please enter a valid folder.")
            continue
        process_directory(folder)
