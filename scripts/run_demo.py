import cv2
from ultralytics import YOLO
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import draw_boxes

def run_demo():
    # Load model
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = YOLO(model_path)
    
    # Pick a synthetic image
    test_img_path = 'dataset/images/val/synthetic_val_0000.jpg'
    if not os.path.exists(test_img_path):
        print(f"Error: {test_img_path} not found.")
        return
        
    img = cv2.imread(test_img_path)
    results = model(img, conf=0.25)[0]
    
    boxes = []
    labels = []
    scores = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2))
        scores.append(float(box.conf[0]))
        labels.append(model.names[int(box.cls[0])])
        
    # Draw results
    annotated = draw_boxes(img, boxes, labels, scores)
    
    # Save output
    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/demo_test_result.jpg'
    cv2.imwrite(out_path, annotated)
    print(f"✅ Demo result saved to {out_path}")
    print(f"Found {len(boxes)} objects.")

if __name__ == "__main__":
    run_demo()
