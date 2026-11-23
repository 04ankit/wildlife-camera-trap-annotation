import os
import time
import shutil
import random
import pandas as pd
from ultralytics import YOLO

CLEAN_DIR = "data/clean_images"
OUT_IMG_DIR = "data/prelabeled_yolo/images"
OUT_LBL_DIR = "data/prelabeled_yolo/labels"
PILOT_DIR = "data/pilot_40"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)
os.makedirs(PILOT_DIR, exist_ok=True)

# 1. Save your 10 mandatory project classes[cite: 3]
CLASSES = [
    "elephant", "tiger", "leopard", "deer", "wild_boar",
    "bear", "other_animal", "human", "vehicle", "empty"
]
with open("data/prelabeled_yolo/classes.txt", "w") as f:
    f.write("\n".join(CLASSES))

# Map COCO pretrained IDs -> Your 10 project class IDs
COCO_TO_PROJECT = {
    0: 7,   # person -> human
    1: 8,   # bicycle -> vehicle
    2: 8,   # car -> vehicle
    3: 8,   # motorcycle -> vehicle
    5: 8,   # bus -> vehicle
    7: 8,   # truck -> vehicle
    20: 0,  # elephant -> elephant
    21: 5,  # bear -> bear
    # Other COCO animals (bird, cat, dog, horse, sheep, cow, zebra, giraffe) -> other_animal (to be refined in review)
    14: 6, 15: 6, 16: 6, 17: 6, 18: 6, 19: 6, 22: 6, 23: 6
}

all_images = sorted([f for f in os.listdir(CLEAN_DIR) if f.endswith(".jpg")])
print(f"Found {len(all_images)} clean images. Loading pretrained YOLOv8n detector...")

model = YOLO("yolov8n.pt")

# 2. Run AI Pre-Annotation & measure speed
timing_logs = []
empty_count = 0
detected_count = 0

for img_name in all_images:
    src_path = os.path.join(CLEAN_DIR, img_name)
    dst_img_path = os.path.join(OUT_IMG_DIR, img_name)
    lbl_path = os.path.join(OUT_LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    shutil.copy(src_path, dst_img_path)

    t0 = time.time()
    results = model.predict(src_path, conf=0.25, verbose=False)[0]
    ai_sec = round(time.time() - t0, 4)

    lines = []
    for box in results.boxes:
        coco_id = int(box.cls[0])
        if coco_id in COCO_TO_PROJECT:
            proj_id = COCO_TO_PROJECT[coco_id]
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{proj_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))

    if len(lines) == 0:
        empty_count += 1
    else:
        detected_count += 1

    timing_logs.append({
        "filename": img_name,
        "ai_prelabel_sec": ai_sec,
        "boxes_predicted": len(lines)
    })

pd.DataFrame(timing_logs).to_csv("data/ai_prelabel_log.csv", index=False)

# 3. Copy 40 random images into data/pilot_40 for 2-member pilot agreement check
random.seed(42)
pilot_sample = random.sample(all_images, min(40, len(all_images)))
for p_img in pilot_sample:
    shutil.copy(os.path.join(CLEAN_DIR, p_img), os.path.join(PILOT_DIR, p_img))

print(f"\nAI Pre-Annotation Complete!")
print(f"  - Images with bounding boxes predicted: {detected_count}")
print(f"  - Empty frames auto-filtered: {empty_count}")
print(f"  - Pilot set (40 images) saved to: {PILOT_DIR}")