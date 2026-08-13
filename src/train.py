"""Train YOLOv8 model using ultralytics package."""
import argparse
from ultralytics import YOLO
import os


def train(data: str, epochs: int=100, imgsz: int=640, batch: int=16, device: str=None, save_dir: str='runs'):
    model = YOLO('yolov8s.pt')  # Use standard detection model for bounding box training
    os.makedirs(save_dir, exist_ok=True)
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, save=True, project=save_dir, name='logo_watermark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='runs')
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device, args.save_dir)
