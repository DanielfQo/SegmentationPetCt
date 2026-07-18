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

print(f"Buscando dataset en: {dataset_root}")

# Check both flat structure (imagesTr/labelsTr) and per-patient subfolder structure.
cases_dict = {}

# 1. Try scanning per-patient subfolders
for item in os.listdir(dataset_root):
    item_path = os.path.join(dataset_root, item)
    if os.path.isdir(item_path) and item not in ["imagesTr", "labelsTr", "persistent_cache_train", "persistent_cache_val", "venv", ".git"]:
        # Check for 2022 format: item__CT.nii.gz, item__PT.nii.gz, item.nii.gz
        ct_2022 = os.path.join(item_path, f"{item}__CT.nii.gz")
        pt_2022 = os.path.join(item_path, f"{item}__PT.nii.gz")
        lbl_2022 = os.path.join(item_path, f"{item}.nii.gz")
        
        # Check for 2025 format: item_0000.nii.gz, item_0001.nii.gz, item.nii.gz
        ct_2025 = os.path.join(item_path, f"{item}_0000.nii.gz")
        pt_2025 = os.path.join(item_path, f"{item}_0001.nii.gz")
        lbl_2025 = os.path.join(item_path, f"{item}.nii.gz")
        
        if os.path.exists(ct_2022) and os.path.exists(pt_2022) and os.path.exists(lbl_2022):
            cases_dict[item] = {
                "image": [ct_2022, pt_2022],
                "label": lbl_2022
            }
        elif os.path.exists(ct_2025) and os.path.exists(pt_2025) and os.path.exists(lbl_2025):
            cases_dict[item] = {
                "image": [ct_2025, pt_2025],
                "label": lbl_2025
            }

# 2. If no cases found via subfolders, fall back to flat structure
if not cases_dict:
    IMG_DIR = os.path.join(dataset_root, "imagesTr")
    LBL_DIR = os.path.join(dataset_root, "labelsTr")
    if os.path.exists(IMG_DIR) and os.path.exists(LBL_DIR):
        is_hecktor_2025 = len(glob.glob(os.path.join(IMG_DIR, "*_0000.nii.gz"))) > 0
        if is_hecktor_2025:
            suffix_ct = "_0000.nii.gz"
            suffix_pt = "_0001.nii.gz"
            split_char = "_"
        else:
            suffix_ct = "__CT.nii.gz"
            suffix_pt = "__PT.nii.gz"
            split_char = "__"
            
        ct_files = glob.glob(os.path.join(IMG_DIR, f"*{suffix_ct}"))
        for p in ct_files:
            c = os.path.basename(p).split(split_char)[0]
            ct_path = p
            pt_path = os.path.join(IMG_DIR, f"{c}{suffix_pt}")
            lbl_path = os.path.join(LBL_DIR, f"{c}.nii.gz")
            if os.path.exists(pt_path) and os.path.exists(lbl_path):
                cases_dict[c] = {
                    "image": [ct_path, pt_path],
                    "label": lbl_path
                }

# Shuffle and split
cases = sorted(list(cases_dict.keys()))
random.seed(0)
random.shuffle(cases)

n_val = int(len(cases) * 0.2)
val, train = cases[:n_val], cases[n_val:]

datalist = {
    "training":   [{**cases_dict[c], "fold": 0} for c in val]   # fold 0 = validación
                + [{**cases_dict[c], "fold": 1} for c in train], # fold 1 = entrenamiento
}
json.dump(datalist, open("datalist_1fold.json", "w"), indent=2)
print(f"Total detectados: {len(cases)} casos -> train {len(train)} / val {len(val)}")

