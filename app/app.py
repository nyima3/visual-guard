import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from src.detect import load_model
    from src.watermark_removal import inpaint_opencv, enhance_image
    from src.utils import draw_boxes
except ImportError:
    st.error("Could not import source modules. Please ensure you are running from the project root.")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VisionGuard | AI Logo & Watermark Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI CUSTOMIZATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2091/2091665.png", width=100)
    st.header("UI Customization")
    theme_choice = st.selectbox("Appearance Theme", 
                                ["Midnight Dark", "Arctic White", "Deep Ocean", "Forest Edge", "Sunset Glow"])
    st.markdown("---")

# --- THEME DEFINITIONS ---
themes = {
    "Midnight Dark": {
        "bg": "linear-gradient(135deg, #0e1117 0%, #161b22 100%)",
        "text": "#ffffff",
        "sub_text": "#8b949e",
        "card_bg": "rgba(255, 255, 255, 0.05)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "tab_color": "#8b949e"
    },
    "Arctic White": {
        "bg": "linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%)",
        "text": "#1f2937",
        "sub_text": "#4b5563",
        "card_bg": "rgba(0, 0, 0, 0.05)",
        "card_border": "rgba(0, 0, 0, 0.1)",
        "tab_color": "#4b5563"
    },
    "Deep Ocean": {
        "bg": "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        "text": "#f8fafc",
        "sub_text": "#94a3b8",
        "card_bg": "rgba(30, 41, 59, 0.7)",
        "card_border": "rgba(51, 65, 85, 1)",
        "tab_color": "#94a3b8"
    },
    "Forest Edge": {
        "bg": "linear-gradient(135deg, #064e3b 0%, #065f46 100%)",
        "text": "#ecfdf5",
        "sub_text": "#a7f3d0",
        "card_bg": "rgba(255, 255, 255, 0.05)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "tab_color": "#a7f3d0"
    },
    "Sunset Glow": {
        "bg": "linear-gradient(135deg, #4c1d95 0%, #7c3aed 100%)",
        "text": "#f5f3ff",
        "sub_text": "#ddd6fe",
        "card_bg": "rgba(255, 255, 255, 0.05)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "tab_color": "#ddd6fe"
    }
}

# Get selected theme
selected_theme = themes.get(theme_choice, themes["Midnight Dark"])

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {selected_theme['text']};
    }}
    
    .stApp {{
        background: {selected_theme['bg']};
    }}
    
    /* Title Styling */
    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}
    
    .sub-title {{
        color: {selected_theme['sub_text']};
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }}
    
    /* Card-like containers */
    .metric-card {{
        background: {selected_theme['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid {selected_theme['card_border']};
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        font-weight: 600;
        color: {selected_theme['tab_color']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: transparent;
        color: #4facfe !important;
        border-bottom: 2px solid #4facfe !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<h1 class="main-title">VisionGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Production-grade Detection & Removal for Watermarks and Logos</p>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    
    model_path = st.text_input("Model Path", value="models/best.pt", help="Path to your YOLOv8 weights (.pt)")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.markdown("---")
    st.header("Image & Video Enhancement")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    sharpness = st.slider("Sharpness", 0.0, 1.0, 0.0, 0.1)
    denoise = st.checkbox("Enable AI Denoising (Slower)", value=False)
    
    st.markdown("---")
    st.header("Advanced Options")
    enable_removal = st.checkbox("Enable Watermark Removal", value=False)
    dilation_iter = st.slider("Mask Dilation (Edge Coverage)", 1, 10, 3)
    show_labels = st.checkbox("Show Labels", value=True)
    save_outputs = st.checkbox("Auto-save Outputs", value=True)
    
    st.info("Running on CPU/GPU based on availability.")
    
    st.markdown("---")
    st.subheader("Model Status")
    if os.path.exists(model_path):
        st.success(f"Model loaded: {os.path.basename(model_path)}")
    else:
        st.error("Weights not found! Please run training.")
        if st.button("Generate Data & Train (Local)"):
            st.info("Starting background training task...")
            # Note: In a real app, this would be a background process
            os.system("python scripts/generate_synthetic_dataset.py --num_images 1000")
            os.system("python src/train.py --data data.yaml --epochs 50")
            st.success("Training initiated!")
            
    if st.button("🔄 Force Reload Model"):
        st.cache_resource.clear()
        st.success("Model cache cleared. Reloading...")
        st.rerun()

# --- UTILS ---
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.warning(f"Weights not found at {path}. Using default yolov8n.pt")
        return YOLO('yolov8s.pt')
    return YOLO(path)

model = load_yolo_model(model_path)

def process_frame(frame, conf, removal=False, enhance=False):
    results = model(frame, conf=conf)[0]
    boxes = []
    labels = []
    scores = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2))
        scores.append(float(box.conf[0]))
        labels.append(model.names[int(box.cls[0])])
        
    annotated_frame = frame.copy()
    if show_labels:
        annotated_frame = draw_boxes(annotated_frame, boxes, labels, scores)
    else:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    processed_frame = frame.copy()
    if removal:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            if 'watermark' in lab.lower():
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        if mask.sum() > 0:
            processed_frame = inpaint_opencv(processed_frame, mask, dilate_iter=dilation_iter)
            
    if enhance:
        processed_frame = enhance_image(processed_frame, brightness, contrast, sharpness, denoise)
            
    return annotated_frame, processed_frame, list(zip(labels, scores))

# --- MAIN INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "📹 Live Webcam"])

# --- TAB 1: IMAGE ---
with tab1:
    col_up, col_res = st.columns([1, 1])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", width='stretch')
            
    with col_res:
        if uploaded_file:
            if st.button("Process Image", key="img_proc"):
                with st.spinner("Analyzing..."):
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    ann, proc, dets = process_frame(frame, conf_threshold, enable_removal, enhance=True)
                    
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Detections", width='stretch')
                    
                    if enable_removal or (brightness != 1.0 or contrast != 1.0 or sharpness > 0 or denoise):
                        st.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), caption="Processed Image", width='stretch')
                        
                        # Download button
                        _, img_encoded = cv2.imencode('.png', proc)
                        st.download_button(
                            label="📥 Download Processed Image",
                            data=img_encoded.tobytes(),
                            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        
                    if dets:
                        st.subheader("Detections Found")
                        for lab, sc in dets:
                            st.write(f"✅ **{lab.upper()}**: {sc:.2%}")
                    else:
                        st.success("No logos or watermarks detected above threshold.")

# --- TAB 2: VIDEO ---
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        st.video(uploaded_video, width='stretch')
        
        col_v1, col_v2 = st.columns(2)
        
        if col_v1.button("Preview Processing (10 frames)"):
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            for _ in range(10):
                ret, frame = cap.read()
                if not ret: break
                ann, proc, _ = process_frame(frame, conf_threshold, enable_removal, enhance=True)
                st_frame.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), width='stretch')
            cap.release()
            st.info("Preview complete. Use the button below for full processing.")

        if col_v2.button("🚀 Process & Download Full Video"):
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            st_frame = st.empty()
            
            curr_frame = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                _, proc, _ = process_frame(frame, conf_threshold, enable_removal, enhance=True)
                out.write(proc)
                
                curr_frame += 1
                if curr_frame % 5 == 0:
                    progress_bar.progress(curr_frame / frame_count)
                    elapsed = time.time() - start_time
                    fps_proc = curr_frame / elapsed
                    status_text.text(f"Processing frame {curr_frame}/{frame_count} ({fps_proc:.1f} FPS)")
                    st_frame.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), width='stretch', caption="Processing Stream...")
                
            cap.release()
            out.release()
            
            st.success("Video processing complete!")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )

# --- TAB 3: WEBCAM ---
with tab3:
    st.write("Real-time detection via your webcam.")
    run_webcam = st.toggle("Start Webcam")
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret: break
            
            ann, _, _ = process_frame(frame, conf_threshold, False)
            st_frame.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), width='stretch')
            
            # Use a small sleep to allow streamlit to refresh
            time.sleep(0.01)
            
            if not run_webcam:
                break
        
        cap.release()

# --- FOOTER ---
st.markdown("---")
st.markdown(f"© {datetime.now().year} VisionGuard AI | Built with YOLOv8 & Streamlit")
