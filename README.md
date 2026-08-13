# 🛡️ VisionGuard AI: Logo & Watermark Detection Suite

VisionGuard is a production-ready system for the detection and removal of logos and watermarks from images, videos, and real-time feeds. Powered by YOLOv8 and built with a premium Streamlit UI.

---

## 🚀 How to Run the Project

Follow these steps to get the system running perfectly on your local machine.

### 1. Prerequisites
Ensure you have Python 3.9+ installed. This project does **not** require Docker.

### 2. Setup Environment
Open your terminal (PowerShell on Windows or Bash on Mac/Linux) and run:

```bash
# Navigate to the project directory
cd watermarklogo

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Data (Optional)
If you don't have a dataset yet, you can generate synthetic images to test the pipeline:
```bash
python scripts/generate_synthetic_dataset.py
```

### 4. Train the Model
To train the model on your dataset (or the synthetic one):
```bash
python src/train.py --data data.yaml --epochs 100 --imgsz 640 --batch 16
```
*Note: A pre-trained model will be saved in `models/best.pt`.*

### 5. Launch the Premium Dashboard
This is the main application interface.
```bash
streamlit run app/app.py
```

---

## 🛠️ Advanced Usage

### Run a Detection Demo
To quickly verify that the detection logic is working perfectly:
```bash
python scripts/run_demo.py
```
Check the output in the `outputs/` folder.

### Evaluate Performance
To see metrics like mAP, Precision, and Recall:
```bash
python src/evaluate.py --weights models/best.pt --data data.yaml
```

---

## 📂 Project Structure
- **`app/`**: Premium Streamlit UI.
- **`src/`**: Core detection, training, and removal logic.
- **`scripts/`**: Helper scripts for data generation and demos.
- **`models/`**: Stores your trained `.pt` weights.
- **`dataset/`**: Your images and labels.

---
© 2026 VisionGuard AI | Built with YOLOv8 & Streamlit
