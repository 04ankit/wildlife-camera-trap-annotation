import os
import hashlib
import cv2
import pandas as pd

VIDEO_DIR = "data/raw_videos"
RAW_IMG_DIR = "data/raw_images"
CLEAN_IMG_DIR = "data/clean_images"
os.makedirs(CLEAN_IMG_DIR, exist_ok=True)

seen_hashes = set()
records = []

def is_duplicate_or_corrupt(img):
    if img is None or img.size == 0:
        return True
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    if img_hash in seen_hashes:
        return True
    seen_hashes.add(img_hash)
    return False

# 1. Extract frames from camera-trap videos
frame_count = 0
for vid_file in os.listdir(VIDEO_DIR):
    if not vid_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, vid_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(fps * 1.5)  # 1 frame every 1.5 seconds
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and not is_duplicate_or_corrupt(frame):
            out_name = f"vid_{os.path.splitext(vid_file)[0]}_f{frame_count:04d}.jpg"
            cv2.imwrite(os.path.join(CLEAN_IMG_DIR, out_name), frame)  # Strips EXIF automatically
            records.append({"filename": out_name, "source": "video_frame"})
            frame_count += 1
        idx += 1
    cap.release()

# 2. Clean & deduplicate static camera-trap images
img_count = 0
for root, _, files in os.walk(RAW_IMG_DIR):
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(root, f)
        img = cv2.imread(path)
        if not is_duplicate_or_corrupt(img):
            out_name = f"img_{img_count:05d}.jpg"
            cv2.imwrite(os.path.join(CLEAN_IMG_DIR, out_name), img)  # Saves clean JPG without EXIF
            records.append({"filename": out_name, "source": "static_image"})
            img_count += 1

df = pd.DataFrame(records)
df.to_csv("data/dataset_manifest.csv", index=False)
print(f"Done! Extracted {frame_count} video frames + {img_count} clean images = {len(df)} total images.")