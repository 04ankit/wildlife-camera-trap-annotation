import os
import tempfile
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Wildlife Camera-Trap Detector", page_icon="🐾", layout="wide")
st.title("🐾 Wildlife Camera-Trap Species Detector")
st.markdown("AI-assisted wildlife monitoring & anti-poaching detection powered by **YOLOv8**.")

MODEL_PATH = "results/best.pt" if os.path.exists("results/best.pt") else "yolov8n.pt"

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.15, 0.90, 0.35, 0.05)
st.sidebar.info(f"Loaded Weights: `{MODEL_PATH}`")

tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Camera-Trap Detection"])

with tab1:
    uploaded_img = st.file_uploader("Upload a camera-trap image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        results = model.predict(img, conf=conf_thresh)[0]
        annotated = results.plot()[:, :, ::-1]  # BGR to RGB

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Camera-Trap Frame", use_container_width=True)
        col2.image(annotated, caption="YOLOv8 Species Detection", use_container_width=True)

        # Count species detected
        names = results.names
        counts = {}
        for box in results.boxes:
            cls_name = names[int(box.cls[0])]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        st.subheader("📊 Species Count Summary")
        if counts:
            st.json(counts)
        else:
            st.warning("No wildlife detected — classified as `empty` trigger.")

with tab2:
    uploaded_vid = st.file_uploader("Upload a camera-trap video clip", type=["mp4", "avi", "mov"])
    if uploaded_vid and st.button("Run Video Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = model.predict(frame, conf=conf_thresh, verbose=False)[0]
            stframe.image(res.plot()[:, :, ::-1], use_container_width=True)
        cap.release()