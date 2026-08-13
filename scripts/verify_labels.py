import cv2
import os
from pathlib import Path

def verify_labels(img_dir="dataset_subset/images/train", lbl_dir="dataset_subset/labels/train", out_dir="verify_labels"):
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = list(img_dir.glob("*.jpg"))[:10]
    for img_p in img_files:
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        lbl_p = lbl_dir / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
            
        with open(lbl_p, "r") as f:
            for line in f:
                cls, xc, yc, nw, nh = map(float, line.split())
                x1 = int((xc - nw/2) * w)
                y1 = int((yc - nh/2) * h)
                x2 = int((xc + nw/2) * w)
                y2 = int((yc + nh/2) * h)
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(int(cls)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(out_dir / img_p.name), img)
    print(f"Verified 10 labels in {out_dir}")

if __name__ == "__main__":
    verify_labels()
