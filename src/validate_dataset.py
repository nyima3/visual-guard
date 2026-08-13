"""Validate YOLO dataset layout and labels against `data.yaml`."""
import os
import sys
import yaml
from glob import glob


def load_data_yaml(p='data.yaml'):
    if not os.path.exists(p):
        print(f'Missing {p}')
        sys.exit(1)
    with open(p,'r') as f:
        return yaml.safe_load(f)


def find_missing_labels(img_files, label_dir):
    missing = []
    for img in img_files:
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(label_dir, stem + '.txt')
        if not os.path.exists(lbl):
            missing.append((img, lbl))
    return missing


def main():
    data = load_data_yaml('data.yaml')
    train_p = data.get('train')
    val_p = data.get('val')
    print('Train:', train_p)
    print('Val:  ', val_p)

    for name, path in [('train', train_p), ('val', val_p)]:
        img_dir = path
        if not os.path.isdir(img_dir):
            print(f'ERROR: {name} images path not found: {img_dir}')
            continue
        imgs = glob(os.path.join(img_dir, '*.*'))
        imgs = [p for p in imgs if p.lower().endswith(('.jpg','.jpeg','.png','bmp','tif','tiff','webp'))]
        print(f'{name} images found: {len(imgs)}')
        lbl_dir = img_dir.replace('images', 'labels')
        if not os.path.isdir(lbl_dir):
            print(f'WARNING: labels dir not found for {name}: {lbl_dir}')
        else:
            missing = find_missing_labels(imgs, lbl_dir)
            if missing:
                print(f'Missing {len(missing)} label files in {lbl_dir}. Example:')
                for m in missing[:5]:
                    print('  ', m)
            else:
                print('All images have corresponding label files.')

    print('\nIf there are zero images, add images to the dataset/images/* folders or run the sample generator script:')
    print('  python scripts/create_sample_dataset.py')

if __name__ == '__main__':
    main()
