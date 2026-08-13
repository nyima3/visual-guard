# PowerShell helper to set up environment and run the Streamlit app (Windows)

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

echo "To train: python src/train.py --data data.yaml --epochs 100 --imgsz 640 --batch 16"
echo "To run UI: cd app && streamlit run app.py"
