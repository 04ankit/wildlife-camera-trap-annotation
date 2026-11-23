# 🐾 Wildlife Camera-Trap Image & Video Labelling

An end-to-end AI-assisted data annotation and species detection pipeline designed for automated wildlife monitoring and anti-poaching alerts (modeled for Similipal / Nandankanan conservation workflows, trained on public LILA BC WCS camera-trap data).

---

## 📋 Project Overview & Schema
- **Dataset:** 1,236 cleaned camera-trap images & extracted video frames sourced from LILA BC (Wildlife Conservation Society) with EXIF/GPS metadata stripped for anti-poaching security.
- **Annotation Tool:** CVAT (`app.cvat.ai`) with YOLOv8 (`yolov8n.pt`) automated pre-labelling.
- **Target Classes (10):** `elephant`, `tiger`, `leopard`, `deer`, `wild_boar`, `bear`, `other_animal`, `human`, `vehicle`, `empty`.
- **Dataset Split:** 70% Train (865) / 15% Validation (185) / 15% Test (186).

---

## ⚡ Annotation Efficiency & Quality Control
- **Manual Baseline (Pilot 40 images):** 9 min 27 sec (**14.18 sec/image**).
- **AI-Assisted Pre-Label + CVAT Review (1,236 images):** Significant reduction in per-image annotation time (see `results/time_saved_chart.png`).
- **Quality Control:** 2-member pilot verification (`pilot_40`) evaluated via Intersection over Union (IoU) and Cohen's Kappa, followed by manual bounding-box refinement in CVAT.

---

## 🔍 Model Evaluation & Error Analysis
Trained with **YOLOv8** on Google Colab (T4 GPU) and deployed via an interactive **Streamlit** web app (`app.py`):
- **Pose & Rear-View Confusion:** Animals facing away from the camera trap (e.g., black bears) can default to `other_animal` when diagnostic facial features are occluded.
- **Severe Camouflage:** Heavily camouflaged secondary animals blending into forest floor litter require lowering the confidence threshold (`0.20–0.35`) to improve recall.
- **Class Imbalance:** Rare species (`bear`, `tiger`, `leopard`) have fewer training instances than `other_animal`, representing a key area for future targeted data augmentation.

---

## 📂 Repository Structure
```text
wildlife-camera-trap/
├── 0_auto_fetch_data.py       # Fetch LILA BC WCS camera-trap metadata & images
├── 1_extract_and_clean.py     # Video frame extraction, MD5 deduplication & EXIF/GPS removal
├── 2_ai_prelabel.py           # YOLOv8 automated pre-annotation & pilot split
├── 3_pack_for_cvat.py         # Package images & Ultralytics YOLO labels for CVAT
├── 4_qc_and_split.py          # Time-saved chart, IoU/Kappa check & 70/15/15 split
├── app.py                     # Interactive Streamlit image & video detection demo
├── requirements.txt           # Python dependencies
└── results/
    ├── best.pt                # Trained YOLOv8 weights
    ├── results.png            # Training & validation loss/mAP curves
    ├── confusion_matrix.png   # Test set confusion matrix
    └── time_saved_chart.png   # Manual vs. AI-assisted speed comparison