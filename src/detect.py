"""Optimized detection and tracking utilities."""
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from typing import Optional, List, Tuple
from .utils import draw_boxes, save_image

def load_model(weights: str = 'models/best.pt', device: Optional[str] = None):
    """Load YOLOv8 model with auto-device selection."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)
    model.to(device)
    return model

def detect_image_optimized(model, img: np.ndarray, conf: float = 0.25):
    """Run inference on an image and return structured results."""
    res = model(img, conf=conf)[0]
    boxes = []
    labels = []
    scores = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2))
        scores.append(float(box.conf[0]))
        labels.append(model.names[int(box.cls[0])])
    return boxes, labels, scores

def detect_video_with_tracking(model, video_path: str, out_path: str = 'outputs/tracked_video.mp4', conf: float = 0.25):
    """Process video with YOLOv8 tracking (ByteTrack/BoT-SORT)."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    outv = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    # Use track() for multi-object tracking
    results = model.track(source=video_path, conf=conf, persist=True, stream=True)
    
    for r in results:
        frame = r.orig_img
        boxes = []
        labels = []
        scores = []
        
        if r.boxes:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
                scores.append(float(box.conf[0]))
                cls_id = int(box.cls[0])
                # If tracking is active, include track ID
                track_id = int(box.id[0]) if box.id is not None else ""
                label = f"{model.names[cls_id]} {track_id}"
                labels.append(label)
        
        annotated_frame = draw_boxes(frame, boxes, labels, scores)
        outv.write(annotated_frame)
        
    cap.release()
    outv.release()
    return out_path
