import os
import zipfile

IMG_DIR = "data/prelabeled_yolo/images"
LBL_DIR = "data/prelabeled_yolo/labels"
OUT_DIR = "data/cvat_upload"
os.makedirs(OUT_DIR, exist_ok=True)

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

# 1. Create data.yaml with train.txt reference required by CVAT
yaml_content = """path: .
train: train.txt
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

# 2. Build train.txt listing every image path
train_txt_content = "\n".join([f"images/train/{img}" for img in images])

lbl_zip_path = os.path.join(OUT_DIR, "cvat_prelabels.zip")
with zipfile.ZipFile(lbl_zip_path, "w", zipfile.ZIP_DEFLATED) as z_lbl:
    z_lbl.writestr("data.yaml", yaml_content)
    z_lbl.writestr("train.txt", train_txt_content)

    for img_name in images:
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(LBL_DIR, lbl_name)
        if os.path.exists(lbl_path):
            z_lbl.write(lbl_path, arcname=f"labels/train/{lbl_name}")
        else:
            z_lbl.writestr(f"labels/train/{lbl_name}", "")

print(f"Fixed {lbl_zip_path} with train.txt for {len(images)} images!")