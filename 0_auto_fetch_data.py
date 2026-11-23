import os
import json
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW_DIR = "data/raw_images"
META_DIR = "data/meta"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# 1. Verified LILA BC Google Cloud metadata URL
meta_zip = os.path.join(META_DIR, "wcs_camera_traps.json.zip")
meta_url = "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip"

if not os.path.exists(meta_zip):
    print("Downloading LILA BC WCS metadata index (~35 MB)...")
    urllib.request.urlretrieve(meta_url, meta_zip)

with zipfile.ZipFile(meta_zip, "r") as z:
    z.extractall(META_DIR)

json_file = [f for f in os.listdir(META_DIR) if f.endswith(".json")][0]
print(f"Parsing {json_file}...")
with open(os.path.join(META_DIR, json_file), "r") as f:
    data = json.load(f)

# 2. Map WCS species names to our 10 project classes
cat_map = {}
for c in data["categories"]:
    name = c["name"].lower()
    if "elephant" in name or "loxodonta" in name or "elephas" in name:
        cat_map[c["id"]] = "elephant"
    elif "tiger" in name or "tigris" in name:
        cat_map[c["id"]] = "tiger"
    elif "leopard" in name or "panthera" in name or "jaguar" in name or "onca" in name:
        cat_map[c["id"]] = "leopard"
    elif "deer" in name or "cervus" in name or "muntjac" in name or "mazama" in name or "odocoileus" in name:
        cat_map[c["id"]] = "deer"
    elif "boar" in name or "pig" in name or "sus" in name or "peccary" in name or "tayassu" in name:
        cat_map[c["id"]] = "wild_boar"
    elif "bear" in name or "ursus" in name or "tremarctos" in name:
        cat_map[c["id"]] = "bear"
    elif "empty" in name:
        cat_map[c["id"]] = "empty"
    else:
        cat_map[c["id"]] = "other_animal"

# 3. Select up to 150 images per class (~1,150 images total)
img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
selected = {k: [] for k in ["elephant", "tiger", "leopard", "deer", "wild_boar", "bear", "other_animal", "empty"]}

for ann in data["annotations"]:
    cls = cat_map.get(ann["category_id"])
    img_id = ann["image_id"]
    if cls in selected and len(selected[cls]) < 150 and img_id in img_id_to_file:
        selected[cls].append((cls, img_id_to_file[img_id]))

for k, v in selected.items():
    print(f"  Class '{k}': {len(v)} images queued")

download_list = [item for sublist in selected.values() for item in sublist]
print(f"\nDownloading {len(download_list)} real LILA BC camera-trap images...")

# 4. Verified Google Cloud base URL for unzipped WCS images
BASE_URL = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"

def download_one(idx_item):
    idx, (cls, rel_path) = idx_item
    out_path = os.path.join(RAW_DIR, f"{cls}_{idx:04d}.jpg")
    if os.path.exists(out_path):
        return True
    try:
        urllib.request.urlretrieve(BASE_URL + rel_path, out_path)
        return True
    except Exception:
        return False

with ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(download_one, enumerate(download_list)))

print(f"\nDone! Successfully downloaded {sum(results)} images into {RAW_DIR}!")