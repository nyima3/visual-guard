"""Create a tiny sample dataset with one train/val image and YOLO label files.
Useful for quick smoke-testing training pipelines.
"""
import os
from PIL import Image, ImageDraw


def ensure_dirs():
    for p in ['dataset/images/train','dataset/images/val','dataset/labels/train','dataset/labels/val']:
        os.makedirs(p, exist_ok=True)


def create_image(path, size=(640,480), color=(200,200,200)):
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    # draw a small rectangle to serve as a 'logo' at center
    w,h = size
    box = (int(w*0.4), int(h*0.4), int(w*0.6), int(h*0.6))
    draw.rectangle(box, fill=(30,144,255))
    img.save(path, quality=95)


def write_label(path):
    # class 0, x_center y_center width height (normalized)
    with open(path,'w') as f:
        f.write('0 0.5 0.5 0.2 0.2\n')


def main():
    ensure_dirs()
    train_img = 'dataset/images/train/0001.jpg'
    train_lbl = 'dataset/labels/train/0001.txt'
    val_img = 'dataset/images/val/0001.jpg'
    val_lbl = 'dataset/labels/val/0001.txt'
    create_image(train_img)
    write_label(train_lbl)
    create_image(val_img)
    write_label(val_lbl)
    print('Created sample images and labels:')
    print(' ', train_img, train_lbl)
    print(' ', val_img, val_lbl)
    print('\nNow run:')
    print('  python src/validate_dataset.py')
    print('Then retry training:')
    print('  python src/train.py --data data.yaml --epochs 1 --imgsz 640 --batch 2')

if __name__ == '__main__':
    main()
