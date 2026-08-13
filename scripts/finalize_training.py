import shutil
import os
from pathlib import Path

def finalize():
    # Find the latest run
    runs_dir = Path("runs/detect/runs")
    if not runs_dir.exists():
        print("No runs found.")
        return
    
    latest_run = sorted(list(runs_dir.glob("logo_watermark*")), key=os.path.getmtime)[-1]
    best_pt = latest_run / "weights" / "best.pt"
    
    if best_pt.exists():
        dst = Path("models/best.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dst)
        print(f"Successfully copied {best_pt} to {dst}")
    else:
        print(f"Could not find {best_pt}")

if __name__ == "__main__":
    finalize()
