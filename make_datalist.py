# make_datalist.py
import json
import glob
import os
import random

# Adjust these paths as needed. For HECKTOR 2025, filenames are _0000.nii.gz and _0001.nii.gz.
# For HECKTOR 2022, they are __CT.nii.gz and __PT.nii.gz.
IMG_DIR = "../../HECKTOR 2025 Training Data/imagesTr"
LBL_DIR = "../../HECKTOR 2025 Training Data/labelsTr"
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
