import os
import zipfile
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

# ==========================================
# 1. UPDATE YOUR ACTUAL CVAT TIMINGS HERE
# ==========================================
MANUAL_TOTAL_SEC = 567.0     # 9 min 27 sec for 40 pilot images (14.18 sec/img)
MANUAL_IMG_COUNT = 40

AI_REVIEW_TOTAL_SEC = 2850.0 # <-- Replace with your actual CVAT AI-review seconds!
TOTAL_AI_IMAGES = 1236       # <-- Replace with total images reviewed in CVAT!

os.makedirs("results", exist_ok=True)

# 2. Compute & Plot Time-Saved Comparison
manual_sec_per_img = MANUAL_TOTAL_SEC / MANUAL_IMG_COUNT
ai_sec_per_img = AI_REVIEW_TOTAL_SEC / TOTAL_AI_IMAGES
time_saved_pct = ((manual_sec_per_img - ai_sec_per_img) / manual_sec_per_img) * 100

print(f"Manual Speed:      {manual_sec_per_img:.2f} sec/image")
print(f"AI-Assisted Speed: {ai_sec_per_img:.2f} sec/image")
print(f"Review Time Saved: {time_saved_pct:.2f}%")

plt.figure(figsize=(7, 4.5))
bars = plt.bar(["Manual Annotation", "AI Pre-Label + Review"],
               [manual_sec_per_img, ai_sec_per_img],
               color=["#d9534f", "#2ca02c"], width=0.5)
plt.ylabel("Average Time per Image (Seconds)")
plt.title(f"Annotation Efficiency Benchmark (Time Saved: {time_saved_pct:.1f}%)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}s", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("results/time_saved_chart.png", dpi=200)
plt.close()

# 3. Pilot Inter-Annotator Agreement (IoU & Cohen's Kappa) if both pilot ZIPs exist
def compute_iou(b1, b2):
    # YOLO format: xc, yc, w, h
    x1_min, y1_min = b1[0] - b1[2]/2, b1[1] - b1[3]/2
    x1_max, y1_max = b1[0] + b1[2]/2, b1[1] + b1[3]/2
    x2_min, y2_min = b2[0] - b2[2]/2, b2[1] - b2[3]/2
    x2_max, y2_max = b2[0] + b2[2]/2, b2[1] + b2[3]/2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = (b1[2]*b1[3]) + (b2[2]*b2[3]) - inter
    return inter / union if union > 0 else 0.0

if os.path.exists("data/pilot_m1.zip") and os.path.exists("data/pilot_m2.zip"):
    with zipfile.ZipFile("data/pilot_m1.zip") as z1, zipfile.ZipFile("data/pilot_m2.zip") as z2:
        z1.extractall("data/pilot_eval/m1")
        z2.extractall("data/pilot_eval/m2")

    ious, y_m1, y_m2 = [], [], []
    for root, _, files in os.walk("data/pilot_eval/m1"):
        for f in files:
            if not f.endswith(".txt") or f == "train.txt":
                continue
            p1 = os.path.join(root, f)
            p2 = p1.replace("m1", "m2")
            l1 = [line.split() for line in open(p1) if line.strip()]
            l2 = [line.split() for line in open(p2) if line.strip()] if os.path.exists(p2) else []

            c1 = int(l1[0][0]) if l1 else 9  # 9 = empty
            c2 = int(l2[0][0]) if l2 else 9
            y_m1.append(c1)
            y_m2.append(c2)

            if l1 and l2:
                b1 = [float(x) for x in l1[0][1:5]]
                b2 = [float(x) for x in l2[0][1:5]]
                ious.append(compute_iou(b1, b2))

    kappa = cohen_kappa_score(y_m1, y_m2)
    mean_iou = np.mean(ious) if ious else 0.0
    print(f"Pilot Inter-Annotator Agreement -> Mean IoU: {mean_iou:.3f} | Cohen's Kappa: {kappa:.3f}")

# 4. Unpack final_cvat_labels.zip & Create 70/15/15 Train/Val/Test Split
EXTRACT_DIR = "data/cvat_extracted"
if os.path.exists("data/final_cvat_labels.zip"):
    with zipfile.ZipFile("data/final_cvat_labels.zip", "r") as z:
        z.extractall(EXTRACT_DIR)

    # Collect all extracted .txt labels
    lbl_map = {}
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".txt") and f not in ("train.txt", "val.txt", "test.txt"):
                lbl_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    IMG_SRC = "data/prelabeled_yolo/images"
    all_imgs = sorted([f for f in os.listdir(IMG_SRC) if f.endswith(".jpg")])
    random.seed(42)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    splits = {
        "train": all_imgs[:int(0.70 * n)],
        "val":   all_imgs[int(0.70 * n):int(0.85 * n)],
        "test":  all_imgs[int(0.85 * n):]
    }

    FINAL_ROOT = "data/final_dataset"
    for split_name, img_list in splits.items():
        os.makedirs(f"{FINAL_ROOT}/{split_name}/images", exist_ok=True)
        os.makedirs(f"{FINAL_ROOT}/{split_name}/labels", exist_ok=True)
        for img_file in img_list:
            stem = os.path.splitext(img_file)[0]
            shutil.copy(os.path.join(IMG_SRC, img_file), f"{FINAL_ROOT}/{split_name}/images/{img_file}")
            dst_lbl = f"{FINAL_ROOT}/{split_name}/labels/{stem}.txt"
            if stem in lbl_map:
                shutil.copy(lbl_map[stem], dst_lbl)
            else:
                open(dst_lbl, "w").close()

    # Write clean data.yaml for YOLOv8 training
    yaml_text = """path: /content/final_dataset
train: train/images
val: val/images
test: test/images

names:
  0: elephant
  1: tiger
  2: leopard
  3: deer
  4: wild_boar
  5: bear
  6: other_animal
  7: human
  8: vehicle
  9: empty
"""
    with open(f"{FINAL_ROOT}/data.yaml", "w") as f:
        f.write(yaml_text)

    # Zip final_dataset for Google Colab upload
    shutil.make_archive("final_dataset_yolo", "zip", FINAL_ROOT)
    print(f"\nCreated 70/15/15 split ({len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test)")
    print("Packaged 'final_dataset_yolo.zip' — ready to upload to Google Colab!")
