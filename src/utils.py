import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import logging

# Setup standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logger(log_path: str = "runs/train.log"):
    """Placeholder for backward compatibility"""
    return logger


def load_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    return img


def save_image(img: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def draw_boxes(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]=None, scores: List[float]=None) -> np.ndarray:
    """Draw premium bounding boxes with semi-transparent labels."""
    out = img.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = labels[i] if labels and i < len(labels) else "Object"
        score = scores[i] if scores and i < len(scores) else None
        
        # Color palette (modern)
        if 'logo' in label.lower():
            color_bgr = (254, 242, 0) # Cyan-ish
        elif 'watermark' in label.lower():
            color_bgr = (254, 172, 79) # Blue-ish
        else:
            color_bgr = (0, 255, 0)
            
        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Prepare label text
        txt = label.upper()
        if score is not None:
            txt += f" {score:.2f}"
            
        # Label background
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
        
        # Draw label background rectangle
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1), color_bgr, -1)
        # Draw text
        cv2.putText(out, txt, (x1 + 5, y1 - 7), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out
