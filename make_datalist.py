# make_datalist.py
import json
import glob
import os
import random

# Candidate paths for the dataset root
candidates = [
    "../../HECKTOR2025_raw",

]

dataset_root = None
for c in candidates:
    if os.path.exists(c):
        dataset_root = c
        break

if dataset_root is None:
    # Fallback to HECKTOR2025_raw if none found
    dataset_root = "../../HECKTOR2025_raw"

IMG_DIR = f"{dataset_root}/imagesTr"
LBL_DIR = f"{dataset_root}/labelsTr"
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
