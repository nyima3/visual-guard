"""Evaluation metrics for YOLOv8 model."""
from ultralytics import YOLO
import argparse
import os

def evaluate_model(weights: str, data: str):
    """
    Evaluate the model on the test/val set and print mAP, Precision, Recall, etc.
    """
    model = YOLO(weights)
    results = model.val(data=data)
    
    print("\n--- Evaluation Results ---")
    print(f"mAP@50:    {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall:    {results.results_dict['metrics/recall(B)']:.4f}")
    print("--------------------------\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt", help="Path to best.pt")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    args = parser.parse_args()
    
    if os.path.exists(args.weights):
        evaluate_model(args.weights, args.data)
    else:
        print(f"Error: Weights not found at {args.weights}")
