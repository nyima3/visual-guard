from ultralytics import YOLO
import cv2
import os

import argparse

def quick_test(weights="models/best.pt", img_path=None):
    if img_path is None:
        img_path = "dataset_subset/images/train/synth_train_0000.jpg"
    if not os.path.exists(weights):
        print(f"Error: Weights not found at {weights}")
        return
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    model = YOLO(weights)
    results = model(img_path, conf=0.1)
    for r in results:
        print(f"Detections: {len(r.boxes)}")
        for box in r.boxes:
            print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}")

if __name__ == "__main__":
    quick_test()
