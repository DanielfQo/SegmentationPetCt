# make_datalist.py
import json
import glob
import os
import random

# We search for the dataset directory in multiple possible relative paths,
# checking both from the current working directory and the script's file location.
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

relative_dirs = [
    "HECKTOR2025_raw",
    "../HECKTOR2025_raw",
    "../../HECKTOR2025_raw",
    "../../../HECKTOR2025_raw",
    "HECKTOR 2025 Training Data",
    "../HECKTOR 2025 Training Data",
    "../../HECKTOR 2025 Training Data",
    "data",
    "../data",
    "../../data",
]

dataset_root = None

# Search relative to current working directory first, then relative to the script
for r_dir in relative_dirs:
    # Check relative to CWD
    if os.path.exists(r_dir) and os.path.isdir(r_dir):
        dataset_root = os.path.abspath(r_dir)
        break
    # Check relative to script directory
    script_rel = os.path.join(script_dir, r_dir)
    if os.path.exists(script_rel) and os.path.isdir(script_rel):
        dataset_root = os.path.abspath(script_rel)
        break

if dataset_root is None:
    # Fallback default
    dataset_root = os.path.abspath("../../HECKTOR2025_raw")

IMG_DIR = os.path.join(dataset_root, "imagesTr")
LBL_DIR = os.path.join(dataset_root, "labelsTr")
print(f"Buscando dataset en: {dataset_root}")

random.seed(0)


# Check for HECKTOR 2025 suffix first, fallback to 2022
is_hecktor_2025 = len(glob.glob(f"{IMG_DIR}/*_0000.nii.gz")) > 0

if is_hecktor_2025:
    suffix_ct = "_0000.nii.gz"
    suffix_pt = "_0001.nii.gz"
    split_char = "_"
else:
    suffix_ct = "__CT.nii.gz"
    suffix_pt = "__PT.nii.gz"
    split_char = "__"

cases = sorted({os.path.basename(p).split(split_char)[0]
                for p in glob.glob(f"{IMG_DIR}/*{suffix_ct}")})
random.shuffle(cases)

n_val = int(len(cases) * 0.2)
val, train = cases[:n_val], cases[n_val:]

def entry(c):
    return {
        "image": [f"{IMG_DIR}/{c}{suffix_ct}", f"{IMG_DIR}/{c}{suffix_pt}"],
        "label": f"{LBL_DIR}/{c}.nii.gz",
    }

datalist = {
    "training":   [{**entry(c), "fold": 0} for c in val]   # fold 0 = validación
                + [{**entry(c), "fold": 1} for c in train], # fold 1 = entrenamiento
}
json.dump(datalist, open("datalist_1fold.json", "w"), indent=2)
print(f"{len(cases)} casos -> train {len(train)} / val {len(val)}")
