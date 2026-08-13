#!/usr/bin/env bash
# Helper script for Unix/macOS to set up env and run Streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "To train: python src/train.py --data data.yaml --epochs 100 --imgsz 640 --batch 16"
echo "To run UI: cd app && streamlit run app.py"
