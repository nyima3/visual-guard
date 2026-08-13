"""Watermark removal options: OpenCV inpainting and refined mask logic.
"""
import cv2
import numpy as np
from typing import Tuple, List

def inpaint_opencv(img: np.ndarray, mask: np.ndarray, method='telea', dilate_iter=3) -> np.ndarray:
    """
    Perform inpainting using OpenCV.
    - mask: binary mask where 255 indicates the region to remove.
    - dilate_iter: number of iterations to dilate the mask to ensure coverage.
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype('uint8')
        
    # Dilate the mask to cover edges better
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        
    if method == 'telea':
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

def enhance_image(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0, sharpness: float = 0.0, denoise: bool = False) -> np.ndarray:
    """
    Apply image enhancement filters.
    - brightness: multiplier for pixel values.
    - contrast: multiplier for contrast.
    - sharpness: 0 to 1 scale for sharpening.
    - denoise: whether to apply denoising (can be slow).
    """
    out = img.copy()
    
    # Brightness & Contrast
    if brightness != 1.0 or contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=contrast, beta=int((brightness - 1.0) * 100))
        
    # Denoising
    if denoise:
        # Using a faster version if possible, but fastNlMeans is the standard
        out = cv2.fastNlMeansDenoisingColored(out, None, 10, 10, 7, 21)
        
    # Sharpness (Unsharp Masking)
    if sharpness > 0:
        gaussian_3 = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.0 + sharpness, gaussian_3, -sharpness, 0)
        
    return out

def remove_watermark_regions(img: np.ndarray, boxes: List[Tuple[int, int, int, int]], dilate_iter=3) -> np.ndarray:
    """
    Apply inpainting to specific bounding box regions.
    """
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return inpaint_opencv(out, mask, dilate_iter=dilate_iter)

# --- Future Extension: U-Net / GAN Integration ---
# One can load a pre-trained U-Net here for more complex watermark removal
# that preserves texture better than simple inpainting.
