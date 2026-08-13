"""Data augmentation pipeline using albumentations.
This script augments images and correspondingly transforms YOLO bboxes.
"""
import os
from glob import glob
import cv2
import albumentations as A
from pathlib import Path
import argparse

AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))


def load_label_txt(txt_path):
    bboxes = []
    labels = []
    if not os.path.exists(txt_path):
        return bboxes, labels
    with open(txt_path,'r') as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            cls = int(vals[0])
            bbox = list(map(float, vals[1:5]))
            bboxes.append(bbox)
            labels.append(cls)
    return bboxes, labels


def save_label_txt(txt_path, bboxes, labels):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path,'w') as f:
        for cls,b in zip(labels,bboxes):
            f.write(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")


def augment_image(img_path, label_path, out_img_path, out_lbl_path, n=3):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    bboxes, labels = load_label_txt(label_path)
    for i in range(n):
        try:
            augmented = AUG(image=img, bboxes=bboxes, category_ids=labels)
        except Exception:
            continue
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        save_img_p = out_img_path.replace('{i}',str(i))
        save_lbl_p = out_lbl_path.replace('{i}',str(i))
        cv2.imwrite(save_img_p, aug_img)
        save_label_txt(save_lbl_p, aug_bboxes, augmented['category_ids'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='dataset/images/train')
    parser.add_argument('--labels', default='dataset/labels/train')
    parser.add_argument('--out', default='dataset_aug')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    img_files = glob(os.path.join(args.src,'*.jpg')) + glob(os.path.join(args.src,'*.png'))
    for img_path in img_files:
        stem = Path(img_path).stem
        lbl_path = os.path.join(args.labels, stem + '.txt')
        out_img = os.path.join(args.out, 'images', stem + '_aug_{i}.jpg')
        out_lbl = os.path.join(args.out, 'labels', stem + '_aug_{i}.txt')
        augment_image(img_path, lbl_path, out_img, out_lbl, n=args.n)
