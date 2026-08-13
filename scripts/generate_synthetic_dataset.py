import cv2
import numpy as np
import os
import random
from pathlib import Path

def generate_synthetic_dataset(output_dir="dataset", num_images=1000):
    """
    Generates realistic synthetic images with varying logos and watermarks.
    """
    classes = ['logo', 'watermark']
    
    # Text samples for logos and watermarks
    logo_texts = ["NIKE", "ADIDAS", "APPLE", "GOOGLE", "SAMSUNG", "SONY", "DELL", "HP", "INTEL", "LOGO", "BRAND", "CORP"]
    wm_texts = ["CONFIDENTIAL", "DRAFT", "COPYRIGHT 2026", "SAMPLE", "PREVIEW", "DO NOT COPY", "WATERMARK", "SECURITY"]

    for split in ['train', 'val']:
        img_dir = Path(output_dir) / "images" / split
        lbl_dir = Path(output_dir) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        count = num_images if split == 'train' else max(20, num_images // 5)
        
        for i in range(count):
            # 1. Generate a more realistic background
            width, height = 640, 640
            bg_type = random.choice(['noise', 'gradient', 'solid', 'pattern', 'mixed'])
            
            if bg_type == 'noise':
                img = np.random.randint(100, 180, (height, width, 3), dtype=np.uint8)
                # Add some larger blobs of color
                for _ in range(5):
                    cv2.circle(img, (random.randint(0, width), random.randint(0, height)), random.randint(50, 200), (random.randint(50, 200),)*3, -1)
            elif bg_type == 'gradient':
                img = np.zeros((height, width, 3), dtype=np.uint8)
                c1 = [random.randint(0, 100) for _ in range(3)]
                c2 = [random.randint(150, 255) for _ in range(3)]
                for c in range(3):
                    img[:, :, c] = np.linspace(c1[c], c2[c], width)
            elif bg_type == 'pattern':
                img = np.full((height, width, 3), random.randint(150, 255), dtype=np.uint8)
                step = random.randint(20, 60)
                for k in range(0, 640, step):
                    cv2.line(img, (0, k), (640, k), (random.randint(100, 150),)*3, 1)
                    cv2.line(img, (k, 0), (k, 640), (random.randint(100, 150),)*3, 1)
            else:
                img = np.full((height, width, 3), random.randint(100, 200), dtype=np.uint8)

            img = cv2.GaussianBlur(img, (3, 3), 0)
            labels = []
            
            # 2. Add 1-5 objects
            num_objects = random.randint(1, 5)
            for _ in range(num_objects):
                cls_id = random.randint(0, 1)
                
                # Random text and size
                text = random.choice(logo_texts if cls_id == 0 else wm_texts)
                font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC])
                scale = random.uniform(0.5, 2.5)
                thickness = random.randint(1, 3)
                
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                pad = 10
                w, h = tw + 2*pad, th + 2*pad
                
                # Ensure width and height are within bounds
                if w >= width:
                    scale *= (width - 2*pad) / w
                    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                    w, h = tw + 2*pad, th + 2*pad
                if h >= height:
                    scale *= (height - 2*pad) / h
                    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                    w, h = tw + 2*pad, th + 2*pad

                x = random.randint(0, max(0, width - w))
                y = random.randint(max(th + pad, 0), max(th + pad, height - pad))
                
                # Create a transparent overlay for rotation
                overlay = np.zeros((height, width, 4), dtype=np.uint8)
                
                if cls_id == 0: # logo
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                    # Sometimes add a shape
                    if random.random() > 0.4:
                        shape_type = random.choice(['rect', 'circle', 'tri'])
                        if shape_type == 'rect':
                            cv2.rectangle(overlay, (x, y - th - pad), (x + tw + pad, y + pad), color, -1)
                        elif shape_type == 'circle':
                            cv2.circle(overlay, (x + w//2, y - th//2), max(w,h)//2, color, -1)
                    cv2.putText(overlay, text, (x + pad, y), font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
                else: # watermark
                    color = (255, 255, 255, random.randint(50, 150))
                    # Tiled or single?
                    if random.random() > 0.7: # Tiled
                        for tx in range(0, width, w + 100):
                            for ty in range(th, height, h + 100):
                                cv2.putText(overlay, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(overlay, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

                # Random rotation
                angle = random.randint(-30, 30)
                M = cv2.getRotationMatrix2D((x + w/2, y - h/2), angle, 1.0)
                overlay = cv2.warpAffine(overlay, M, (width, height))
                
                # Blend overlay with image
                alpha_mask = overlay[:, :, 3] / 255.0
                for c in range(3):
                    img[:, :, c] = ((1.0 - alpha_mask) * img[:, :, c] + alpha_mask * overlay[:, :, c]).astype(np.uint8)

                # --- ACCURATE BOUNDING BOX CALCULATION ---
                # 1. Define corners of the original unrotated box
                corners = np.array([
                    [x, y - th - pad],
                    [x + tw + pad, y - th - pad],
                    [x + tw + pad, y + pad],
                    [x, y + pad]
                ], dtype=np.float32)
                
                # 2. Transform corners using the same rotation matrix
                ones = np.ones(shape=(len(corners), 1))
                points_ones = np.concatenate([corners, ones], axis=1)
                transformed_corners = M.dot(points_ones.T).T
                
                # 3. Find the axis-aligned bounding box (min/max)
                min_x = np.min(transformed_corners[:, 0])
                max_x = np.max(transformed_corners[:, 0])
                min_y = np.min(transformed_corners[:, 1])
                max_y = np.max(transformed_corners[:, 1])
                
                # 4. Clip to image boundaries
                min_x = max(0, min_x)
                max_x = min(width, max_x)
                min_y = max(0, min_y)
                max_y = min(height, max_y)
                
                # 5. Normalized YOLO BBox
                nw_box = (max_x - min_x) / width
                nh_box = (max_y - min_y) / height
                x_center = (min_x + max_x) / 2 / width
                y_center = (min_y + max_y) / 2 / height
                
                if nw_box > 0 and nh_box > 0:
                    labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw_box:.6f} {nh_box:.6f}")
            
            # Save
            img_name = f"synth_{split}_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / img_name), img)
            with open(lbl_dir / f"synth_{split}_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))

    print(f"DONE: High-fidelity synthetic dataset generated: {num_images} images.")

if __name__ == "__main__":
    generate_synthetic_dataset()
