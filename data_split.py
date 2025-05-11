# data_split.py
import os, shutil, random
from pathlib import Path

src_dir = Path("flowers_5")  # Source data
dst_base = Path("data")      # Output directory
splits = {"train": 0.7, "val": 0.2, "test": 0.1}
random.seed(42)

def split_class(cls_path):
    images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
    random.shuffle(images)
    n = len(images)
    train_end = int(n * splits["train"])
    val_end = int(n * (splits["train"] + splits["val"]))
    
    for i, img in enumerate(images):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
        # Split images into train, validation, and test sets
        # and copy them into respective directories
        dst = dst_base / split / cls_path.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst / img.name)

if __name__ == "__main__":
    for cls in src_dir.iterdir():
        if cls.is_dir():
            split_class(cls)
    print("Data splitting completed, output to ./data directory")
