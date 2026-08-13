import os
import shutil
from pathlib import Path

def prepare_subset(src_dir="dataset", dst_dir="dataset_subset", count=100):
    for split in ['train', 'val']:
        n = count if split == 'train' else count // 5
        src_img = Path(src_dir) / "images" / split
        dst_img = Path(dst_dir) / "images" / split
        src_lbl = Path(src_dir) / "labels" / split
        dst_lbl = Path(dst_dir) / "labels" / split
        
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        files = sorted(list(src_img.glob("*.jpg")))[:n]
        for f in files:
            shutil.copy(f, dst_img / f.name)
            lbl = src_lbl / (f.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dst_lbl / lbl.name)
    print(f"Copied {count} images to {dst_dir}")

if __name__ == "__main__":
    prepare_subset()
